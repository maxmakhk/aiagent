import cv2
import numpy as np

def calculate_iou(box1, box2):
    """
    Calculates Intersection over Union (IoU) of two bounding boxes.
    Boxes are in [x1, y1, x2, y2] format.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0.0
        
    return inter_area / float(union_area)

def calculate_containment(box1, box2):
    """
    Calculates how much of the smaller box is contained within the larger box.
    Returns the containment ratio (0.0 to 1.0).
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    min_area = min(box1_area, box2_area)
    if min_area == 0:
        return 0.0
        
    return inter_area / float(min_area)

def merge_boxes(boxes, iou_threshold=0.4, containment_threshold=0.8):
    """
    Iteratively merges bounding boxes that have high IoU or high containment.
    Returns a list of merged bounding boxes [x1, y1, x2, y2].
    """
    merged = True
    active_boxes = list(boxes)
    
    while merged:
        merged = False
        new_boxes = []
        skip_indices = set()
        
        for i in range(len(active_boxes)):
            if i in skip_indices:
                continue
                
            box_a = active_boxes[i]
            merged_this_step = False
            
            for j in range(i + 1, len(active_boxes)):
                if j in skip_indices:
                    continue
                    
                box_b = active_boxes[j]
                
                iou = calculate_iou(box_a, box_b)
                containment = calculate_containment(box_a, box_b)
                
                if iou > iou_threshold or containment > containment_threshold:
                    # Merge them: take the bounding envelope
                    box_a = [
                        min(box_a[0], box_b[0]),
                        min(box_a[1], box_b[1]),
                        max(box_a[2], box_b[2]),
                        max(box_a[3], box_b[3])
                    ]
                    skip_indices.add(j)
                    merged = True
                    merged_this_step = True
            
            new_boxes.append(box_a)
            
        if merged:
            active_boxes = new_boxes
            
    return active_boxes

def generate_grid_proposals(width, height):
    """
    Generates a fallback grid of overlapping candidate boxes at multiple scales.
    - Full image
    - 2x2 large quadrants (65% width/height, overlapping)
    - 3x3 medium grids (40% width/height, overlapping)
    """
    proposals = []
    
    # 1. Full image crop
    proposals.append([0, 0, width, height])
    
    # 2. Large overlapping quadrants (2x2)
    qw = int(width * 0.65)
    qh = int(height * 0.65)
    q_x_steps = [0, width - qw]
    q_y_steps = [0, height - qh]
    for x in q_x_steps:
        for y in q_y_steps:
            proposals.append([x, y, x + qw, y + qh])
            
    # 3. Medium overlapping grids (3x3)
    mw = int(width * 0.40)
    mh = int(height * 0.40)
    m_x_steps = [0, (width - mw) // 2, width - mw]
    m_y_steps = [0, (height - mh) // 2, height - mh]
    for x in m_x_steps:
        for y in m_y_steps:
            proposals.append([x, y, x + mw, y + mh])
            
    return proposals

def detect_candidate_crops(image_cv, min_area_ratio=0.015, max_area_ratio=0.85):
    """
    Extracts candidate bounding boxes using OpenCV contours & custom box merging.
    Falls back to a grid search if too few proposals are found.
    
    Parameters:
    - image_cv: OpenCV image in BGR/RGB format.
    - min_area_ratio: Min box area as a percentage of total image area.
    - max_area_ratio: Max box area as a percentage of total image area.
    
    Returns:
    - candidate_boxes: List of bounding boxes in [x1, y1, x2, y2] format.
    - debug_info: Dict containing contour counts, fallback usage, etc.
    """
    height, width = image_cv.shape[:2]
    total_area = width * height
    min_area = int(total_area * min_area_ratio)
    max_area = int(total_area * max_area_ratio)
    
    # Convert to Grayscale & Blur
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Bounding boxes from different segmentation channels
    raw_boxes = []
    
    # 1. Otsu Thresholding Contours
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours_thresh, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_thresh:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if min_area <= area <= max_area:
            raw_boxes.append([x, y, x + w, y + h])
            
    # 2. Canny Edge Detection Contours
    edges = cv2.Canny(blurred, 30, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    contours_edges, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_edges:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if min_area <= area <= max_area:
            raw_boxes.append([x, y, x + w, y + h])
            
    # Merge overlapping boxes
    contour_boxes = merge_boxes(raw_boxes, iou_threshold=0.35, containment_threshold=0.8)
    
    debug_info = {
        "raw_contour_proposals": len(raw_boxes),
        "merged_contour_proposals": len(contour_boxes),
        "using_fallback_grid": False
    }
    
    # Check if contour detection was insufficient (e.g. low contrast tabletop)
    # We want at least 3 candidate regions. If not, inject grid proposals to guarantee coverage.
    if len(contour_boxes) < 3:
        debug_info["using_fallback_grid"] = True
        grid_proposals = generate_grid_proposals(width, height)
        # Combine contours and grid, then merge them
        all_proposals = contour_boxes + grid_proposals
        final_boxes = merge_boxes(all_proposals, iou_threshold=0.45, containment_threshold=0.85)
    else:
        final_boxes = contour_boxes
        
    # Ensure all boxes are within image boundaries
    clamped_boxes = []
    for box in final_boxes:
        x1 = max(0, min(box[0], width - 1))
        y1 = max(0, min(box[1], height - 1))
        x2 = max(x1 + 10, min(box[2], width))
        y2 = max(y1 + 10, min(box[3], height))
        clamped_boxes.append([x1, y1, x2, y2])
        
    return clamped_boxes, debug_info
