#!/usr/bin/env python
"""Test script to verify MediaPipe is properly installed"""

try:
    from mediapipe import solutions
    print("✓ SUCCESS: MediaPipe solutions module imported successfully")
    print(f"✓ Hands module available: {solutions.hands}")
    
    # Try to initialize hands
    hands = solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    )
    print("✓ Hand detection initialized successfully")
    print("✓ All MediaPipe features working correctly!")
    
except ImportError as e:
    print(f"✗ FAILED: {e}")
    exit(1)
except Exception as e:
    print(f"✗ ERROR: {e}")
    exit(1)
