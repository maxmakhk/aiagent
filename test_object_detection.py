#!/usr/bin/env python3
"""
Quick test script to diagnose object detection issues
"""
import cv2
import numpy as np
from PIL import Image
import sys

# Test 1: Simple synthetic image with white rectangle
print("=" * 60)
print("TEST 1: Synthetic image with white rectangle")
print("=" * 60)

img_synthetic = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.rectangle(img_synthetic, (100, 100), (300, 300), (255, 255, 255), -1)

gray = cv2.cvtColor(img_synthetic, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print(f"✓ Found {len(contours)} contours")
print(f"✓ White pixels: {np.sum(thresh)//255}")
print(f"✓ Image shape: {img_synthetic.shape}")

for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    print(f"  Contour {i}: area={area:.0f}, bbox=({x},{y},{x+w},{y+h})")

# Test 2: Load sample image if exists
print("\n" + "=" * 60)
print("TEST 2: Real sample image (if available)")
print("=" * 60)

try:
    img_path = "images/sample.jpg"
    if cv2.os.path.exists(img_path):
        img_real = cv2.imread(img_path)
        print(f"✓ Loaded image: {img_path}")
        print(f"  Shape: {img_real.shape}")
        
        gray = cv2.cvtColor(img_real, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"✓ Found {len(contours)} contours")
        print(f"✓ White pixels: {np.sum(thresh)//255}")
        
        h_img, w_img = img_real.shape[:2]
        min_area = (w_img * h_img) * 0.002
        max_area = (w_img * h_img) * 0.85
        
        detected = 0
        for i, cnt in enumerate(contours[:10]):  # Show first 10
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            
            if min_area <= area <= max_area and w >= 8 and h >= 8:
                detected += 1
                print(f"  ✓ Contour {i}: area={area:.0f}, bbox=({x},{y},{x+w},{y+h})")
            else:
                reason = ""
                if area < min_area: reason = "too small"
                elif area > max_area: reason = "too large"
                elif w < 8 or h < 8: reason = "thin"
                print(f"  ✗ Contour {i}: area={area:.0f} ({reason})")
        
        print(f"\n✓ Valid contours (would be detected): {detected}")
    else:
        print(f"✗ Sample image not found at {img_path}")
except Exception as e:
    print(f"✗ Error loading sample image: {e}")

print("\n" + "=" * 60)
print("TEST 3: Check detect_objects function")
print("=" * 60)

try:
    from aiagent import detect_objects
    
    # Use synthetic image
    objects = detect_objects(img_synthetic)
    print(f"✓ detect_objects() returned {len(objects)} objects")
    for obj in objects:
        print(f"  - {obj['category']} ({obj['confidence']*100:.0f}%) @ {obj['bbox']}")
        
except Exception as e:
    print(f"✗ Error testing detect_objects: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Diagnosis complete!")
print("=" * 60)
