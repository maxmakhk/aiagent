#!/usr/bin/env python3
"""
Standalone object detection test without heavy dependencies
"""
import cv2
import numpy as np

def detect_objects_test(image_np):
    """
    Test object detection algorithm
    """
    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        h_img, w_img = image_np.shape[:2]
        
        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Try Otsu's automatic threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # If Otsu doesn't work well, fall back to manual threshold
        if np.sum(thresh) < 100:  # Almost no white pixels
            _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
        elif np.sum(thresh) > (w_img * h_img * 255 * 0.95):  # Almost all white
            _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        
        # Use morphological operations
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_large)
        
        # Find contours with hierarchy
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        min_area = (w_img * h_img) * 0.002  # 0.2% of image
        max_area = (w_img * h_img) * 0.85
        
        print(f"[Object Detection] Image size: {w_img}x{h_img}")
        print(f"[Object Detection] Area thresholds: min={min_area:.0f}, max={max_area:.0f}")
        print(f"[Object Detection] Found {len(contours)} contours")
        print(f"[Object Detection] Threshold white pixels: {np.sum(thresh)//255}")
        
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            x, y, cw, ch = cv2.boundingRect(contour)
            x2, y2 = x + cw, y + ch
            
            if area < min_area:
                print(f"  Contour {idx}: SKIP (area {area:.0f} < min)")
                continue
            if area > max_area:
                print(f"  Contour {idx}: SKIP (area {area:.0f} > max)")
                continue
            if cw < 8 or ch < 8:
                print(f"  Contour {idx}: SKIP (size {cw}x{ch} too small)")
                continue
            
            confidence = min((area / (cw * ch)) * 0.9, 1.0)
            if confidence < 0.2:
                print(f"  Contour {idx}: SKIP (confidence {confidence:.2f} too low)")
                continue
            
            print(f"  ✓ Contour {idx}: DETECT area={area:.0f}, bbox=({x},{y},{x2},{y2}), conf={confidence:.2f}")
            objects.append({
                "bbox": [x, y, x2, y2],
                "category": "Object",
                "confidence": float(confidence)
            })
        
        print(f"[Object Detection] Total detected: {len(objects)} objects\n")
        return objects
    except Exception as e:
        print(f"[Object Detection] Error: {e}")
        import traceback
        traceback.print_exc()
        return []


# Test 1: Synthetic white rectangle
print("=" * 70)
print("TEST 1: White rectangle on black background")
print("=" * 70)
img1 = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.rectangle(img1, (100, 100), (300, 300), (255, 255, 255), -1)
objects1 = detect_objects_test(img1)

# Test 2: Synthetic multiple objects
print("=" * 70)
print("TEST 2: Multiple white objects")
print("=" * 70)
img2 = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.rectangle(img2, (50, 50), (200, 200), (255, 255, 255), -1)
cv2.circle(img2, (400, 350), 80, (200, 200, 200), -1)
objects2 = detect_objects_test(img2)

# Test 3: Hand-like image (complex contours)
print("=" * 70)
print("TEST 3: Complex scene (hand + object simulation)")
print("=" * 70)
img3 = np.zeros((480, 640, 3), dtype=np.uint8)
# Draw a large hand-like region (filled polygon)
hand_points = np.array([
    [100, 150], [150, 100], [200, 120], [220, 180],
    [210, 250], [150, 300], [80, 280]
], np.int32)
cv2.fillPoly(img3, [hand_points], (200, 200, 200))

# Draw an object inside (phone)
cv2.rectangle(img3, (120, 180), (200, 280), (150, 150, 150), -1)

objects3 = detect_objects_test(img3)

print("=" * 70)
print("Summary")
print("=" * 70)
print(f"Test 1 (rectangle): {len(objects1)} objects")
print(f"Test 2 (multiple): {len(objects2)} objects")
print(f"Test 3 (complex): {len(objects3)} objects")
