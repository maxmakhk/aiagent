import os
import re
import cv2
import numpy as np

def sanitize_label(label_name):
    """
    Sanitizes label names to be safe for filenames and paths.
    Converts spaces to underscores, lowercases, and removes non-alphanumeric chars except dashes.
    """
    label_name = label_name.strip().lower()
    label_name = re.sub(r'\s+', '_', label_name)
    label_name = re.sub(r'[^a-z0-9_\-]', '', label_name)
    return label_name if label_name else "unnamed_object"

def draw_annotations(image_path, detections, output_path):
    """
    Reads an image, draws labeled bounding boxes, and saves the annotated result.
    
    Parameters:
    - image_path: Path to the original scene image.
    - detections: List of detection dicts with "bbox", "label", "score".
    - output_path: Where to save the annotated output image.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
        
    H, W = img.shape[:2]
    
    # Palette of premium matching colors (BGR format)
    # Indigo for detected objects, Orange-Red for unknown, Slate for text outlines
    color_known = (255, 100, 0)      # BGR for Neon Blue/Cyan
    color_unknown = (80, 80, 240)    # BGR for Soft Red/Orange
    color_bg = (30, 30, 30)          # Slate gray background for labels
    
    for det in detections:
        box = det["bbox"]
        label = det["label"]
        score = det["score"]
        
        x1, y1, x2, y2 = box
        
        # Determine color based on classification
        is_known = label != "unknown"
        color = color_known if is_known else color_unknown
        
        # 1. Draw bounding box rectangle
        # Thicker border (3px) for clear presentation
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        # 2. Draw outer glowing corners for a premium tech aesthetic
        corner_len = min(20, int(min(x2-x1, y2-y1) * 0.2))
        if corner_len > 5:
            # Top-Left corner
            cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, 5)
            cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, 5)
            # Top-Right corner
            cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, 5)
            cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, 5)
            # Bottom-Left corner
            cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, 5)
            cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, 5)
            # Bottom-Right corner
            cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, 5)
            cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, 5)

        # 3. Create text label string
        text = f"{label} ({score:.2f})" if is_known else f"unknown ({score:.2f})"
        
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1
        
        # Calculate size of text box
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        
        # Determine text background position (draw text inside box at the top left)
        # Handle cases where label is close to top border
        label_y1 = max(0, y1 - text_h - 12)
        label_y2 = y1
        label_x1 = x1
        label_x2 = x1 + text_w + 16
        
        # Clamp text background inside image boundaries
        if label_y1 < 0:
            label_y1 = y1
            label_y2 = y1 + text_h + 12
            
        # Draw text background banner
        cv2.rectangle(img, (label_x1, label_y1), (label_x2, label_y2), color_bg, -1)
        # Draw border around banner
        cv2.rectangle(img, (label_x1, label_y1), (label_x2, label_y2), color, 1)
        
        # Write text
        text_y = label_y2 - 6 if label_y1 == y1 - text_h - 12 else label_y2 - baseline
        cv2.putText(img, text, (label_x1 + 8, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        
    # Create output directory if not exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the final annotated image
    cv2.imwrite(output_path, img)
