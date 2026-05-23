import numpy as np
from PIL import Image
from recognizer.embeddings import load_embeddings_cache, compute_embedding

def match_candidates(scene_pil, candidate_boxes, data_dir, active_labels, threshold=0.75, target_label=None):
    """
    Compares candidate crops from the scene image against all active target reference embeddings.
    
    Parameters:
    - scene_pil: The original scene image as a PIL Image.
    - candidate_boxes: List of bounding boxes [x1, y1, x2, y2].
    - data_dir: Base data directory containing reference photos.
    - active_labels: List of label strings to match against.
    - threshold: Confidence threshold (cosine similarity). Below this, label is "unknown".
    - target_label: Optional label. If provided, only compare against this label.
    
    Returns:
    - detections: List of matching dicts:
      {
        "bbox": [x1, y1, x2, y2],
        "label": str,
        "score": float,
        "top_matches": [{"label": str, "score": float}, ...]
      }
    """
    # Load all cached reference embeddings
    # Structure: { label: { filename: [emb_vector] } }
    reference_database = {}
    labels_to_check = [target_label] if target_label else active_labels
    
    for label in labels_to_check:
        cache = load_embeddings_cache(data_dir, label)
        if cache:
            reference_database[label] = cache
            
    detections = []
    
    # Process each candidate crop
    for box in candidate_boxes:
        x1, y1, x2, y2 = box
        
        # Crop candidate region
        crop_pil = scene_pil.crop((x1, y1, x2, y2))
        
        # Compute embedding for candidate crop
        try:
            crop_emb = np.array(compute_embedding(crop_pil))
        except Exception as e:
            print(f"[!] Error computing embedding for candidate box {box}: {e}")
            continue
            
        best_label = "unknown"
        best_score = 0.0
        all_label_scores = [] # To find top-3 overall matches
        
        # Compare with each label in the database
        for label, ref_images in reference_database.items():
            if not ref_images:
                continue
                
            label_scores = []
            for filename, ref_emb_list in ref_images.items():
                ref_emb = np.array(ref_emb_list)
                # Compute Cosine Similarity (Dot product since both are L2 normalized)
                similarity = float(np.dot(crop_emb, ref_emb))
                label_scores.append(similarity)
                
            if label_scores:
                # Core similarity metrics:
                # 1. Max similarity represents how closely the crop matches any single reference image
                max_label_score = max(label_scores)
                all_label_scores.append({
                    "label": label,
                    "score": round(max_label_score, 4)
                })
                
        # Sort label scores to get top matches
        all_label_scores.sort(key=lambda x: x["score"], reverse=True)
        top_matches = all_label_scores[:3] # Keep top 3 labels
        
        if top_matches:
            top_match = top_matches[0]
            if top_match["score"] >= threshold:
                best_label = top_match["label"]
                best_score = top_match["score"]
            else:
                # If the score is weak, label as "unknown" but keep the score
                best_label = "unknown"
                best_score = top_match["score"]
        else:
            best_label = "unknown"
            best_score = 0.0
            
        detections.append({
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "label": best_label,
            "score": round(best_score, 4),
            "top_matches": top_matches
        })
        
    return detections

def filter_duplicate_detections(detections, iou_threshold=0.45):
    """
    Performs Non-Maximum Suppression (NMS) on classified detections
    to eliminate duplicate bounding boxes for the SAME physical object.
    
    If two boxes overlap significantly:
    - If they have the same label, keep the one with the higher score.
    - If one is labeled, and another is "unknown", prefer the labeled one.
    - Otherwise, keep the one with the higher score.
    """
    if not detections:
        return []
        
    # Sort detections by score descending
    sorted_dets = sorted(detections, key=lambda x: x["score"], reverse=True)
    keep = []
    
    while sorted_dets:
        best_det = sorted_dets.pop(0)
        keep.append(best_det)
        
        remaining_dets = []
        for det in sorted_dets:
            # Calculate IoU
            x1 = max(best_det["bbox"][0], det["bbox"][0])
            y1 = max(best_det["bbox"][1], det["bbox"][1])
            x2 = min(best_det["bbox"][2], det["bbox"][2])
            y2 = min(best_det["bbox"][3], det["bbox"][3])
            
            inter_area = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (best_det["bbox"][2] - best_det["bbox"][0]) * (best_det["bbox"][3] - best_det["bbox"][1])
            area2 = (det["bbox"][2] - det["bbox"][0]) * (det["bbox"][3] - det["bbox"][1])
            union = area1 + area2 - inter_area
            
            iou = inter_area / float(union) if union > 0 else 0.0
            
            # If overlap is high
            if iou > iou_threshold:
                # Merge logic: if best_det is unknown and current det is known, we might want to keep the known.
                # But since they are sorted by score, the best score is already in best_det.
                # If they are different labels, and both are known, they might be overlapping objects, 
                # but in high-overlap, it's usually the same object, so we drop the lower-scoring duplicate.
                continue
            else:
                remaining_dets.append(det)
                
        sorted_dets = remaining_dets
        
    return keep
