import os

# Toggle offline cache-only mode with HF_OFFLINE=1
hf_offline = os.environ.get('HF_OFFLINE', '0') == '1'
if hf_offline:
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
else:
    os.environ.pop('TRANSFORMERS_OFFLINE', None)
    os.environ.pop('HF_HUB_OFFLINE', None)

from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import time
import torch
import json
import re
import io
import requests
from datetime import datetime
import uuid
import shutil
from mediapipe import solutions
import numpy as np
import cv2

# Allow skipping heavy model load for quick testing (set SKIP_MODEL_LOAD=1)
skip_model_load = os.environ.get('SKIP_MODEL_LOAD') == '1'

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

print("GPU:", torch.cuda.is_available())
print("HF offline mode:", hf_offline)

# ===== Load Model =====
if not skip_model_load:
    print("Loading model (one time)...")
    start = time.time()
    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-base",
        trust_remote_code=True,
        local_files_only=hf_offline,
        clean_up_tokenization_spaces=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-base",
        trust_remote_code=True,
        local_files_only=hf_offline,
        torch_dtype=torch.float16
    ).to("cuda")
    print(f"Model loaded in {time.time()-start:.2f}s\n")
else:
    print("SKIP_MODEL_LOAD=1 detected; skipping heavy model initialization.")
    processor = None
    model = None

# ===== Utility Functions =====
def run_model(img, prompt, max_tokens=150):
    img.thumbnail((800, 800))
    inputs = processor(text=prompt, images=img, return_tensors="pt").to("cuda")
    if 'pixel_values' in inputs:
        inputs['pixel_values'] = inputs['pixel_values'].half()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    result = processor.decode(output[0], skip_special_tokens=True)
    return result

def parse_gauges_to_json(ocr_text, caption_text):
    """
    Dynamically extract all gauge information, not fixed term
    Auto-identify: number + unit + label
    """
    import re
    
    gauges = []
    
    # Combine OCR and Caption text
    combined_text = f"{ocr_text}\n{caption_text}"
    
    print(f"[DEBUG] Combined text: {combined_text[:200]}...")  # debug
    
    # Pattern 1: Label + Number + Unit (e.g., "Chamber 912 mBar", "Temperature 67°C")
    # Match format: <label> <number> <unit>
    pattern_labeled = r'([A-Za-z_]+)\s*[:\s]+(\d+\.?\d*)\s*([A-Za-z°℃℉%]+)'
    matches_labeled = re.findall(pattern_labeled, combined_text, re.IGNORECASE)
    
    for label, value, unit in matches_labeled:
        gauges.append({
            "type": label.lower().strip(),
            "value": f"{value} {unit}",
            "numeric_value": float(value),
            "unit": unit,
            "source": "labeled_pattern"
        })
    
    # Pattern 2: Pure number + unit (no label)
    # Match format: <number> <unit>
    if len(gauges) == 0:
        pattern_unlabeled = r'(\d+\.?\d*)\s*([A-Za-z°℃℉%]+)'
        matches_unlabeled = re.findall(pattern_unlabeled, combined_text)
        
        for i, (value, unit) in enumerate(matches_unlabeled):
            gauges.append({
                "type": f"gauge_{i+1}",
                "value": f"{value} {unit}",
                "numeric_value": float(value),
                "unit": unit,
                "source": "unlabeled_pattern"
            })
    
    # Pattern 3: Extract from complete sentence (using caption)
    # Example: "The first gauge is labeled 'Chamber 912 mBar'"
    if len(gauges) == 0:
        caption_lower = caption_text.lower()
        
        # Find all "xxx yyy mbar" or "xxx yyy °c" formats
        pattern_sentence = r'[\"\']?([a-z_]+)\s+(\d+\.?\d*)\s*(mbar|°c|bar|℃|%|℉)[\"\']?'
        matches_sentence = re.findall(pattern_sentence, caption_lower)
        
        for label, value, unit in matches_sentence:
            gauges.append({
                "type": label.strip(),
                "value": f"{value} {unit}",
                "numeric_value": float(value),
                "unit": unit,
                "source": "sentence_pattern"
            })
    
    # Pattern 4: Fallback - extract all numbers only (last resort)
    if len(gauges) == 0:
        all_numbers = re.findall(r'\d+\.?\d*', combined_text)
        print(f"[DEBUG] Fallback numbers: {all_numbers}")
        
        for i, num in enumerate(all_numbers[:10]):  # Max 10 numbers
            try:
                numeric_val = float(num)
                gauges.append({
                    "type": f"unknown_{i+1}",
                    "value": num,
                    "numeric_value": numeric_val,
                    "unit": "unknown",
                    "source": "fallback_numbers"
                })
            except:
                continue
    
    # Deduplicate (keep only first occurrence of each type)
    seen_types = set()
    unique_gauges = []
    for gauge in gauges:
        if gauge["type"] not in seen_types:
            seen_types.add(gauge["type"])
            unique_gauges.append(gauge)
    
    print(f"[DEBUG] Final gauges: {unique_gauges}")
    
    return unique_gauges

# ===== Flask API =====
app = Flask(__name__)
# Enable permissive CORS for all routes and ensure headers on all responses
CORS(app, resources={r"/*": {"origins": "*"}})


@app.after_request
def add_cors(response):
    response.headers.setdefault('Access-Control-Allow-Origin', '*')
    response.headers.setdefault('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    response.headers.setdefault('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    return response


@app.route('/local-image/<path:filename>')
def local_image(filename):
    base_dir = os.path.dirname(__file__) or '.'
    return send_from_directory(base_dir, filename)


@app.route('/images/<path:filename>')
def serve_image(filename):
    print(f"Serving image: {filename}")
    base_dir = os.path.dirname(__file__) or '.'
    images_dir = os.path.join(base_dir, 'images')
    return send_from_directory(images_dir, filename)


@app.route("/", methods=["GET", "POST"])
def index():
    """Homepage test interface. Also accepts raw POST uploads from ESP32 camera.
    If a POST with image bytes is received, save it to `images/` and return JSON
    containing the public URL so the ESP32 (or caller) can consume it.
    """
    if request.method == 'GET':
        return render_template("index.html")

    # Handle POST (raw image bytes) from devices like the ESP32
    try:
        data = request.get_data()
        if not data:
            return jsonify({"status": "error", "message": "No data received"}), 400

        base_dir = os.path.dirname(__file__) or '.'
        images_dir = os.path.join(base_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)

        timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        unique = uuid.uuid4().hex[:8]
        filename = f"esp32_{timestamp.replace(':', '')}_{unique}.jpg"
        dest = os.path.join(images_dir, filename)

        with open(dest, 'wb') as f:
            f.write(data)

        url = f"https://aiagent.maxsolo.co.uk/images/{filename}"

        return jsonify({
            "status": "success",
            "message": "Image captured from ESP32",
            "input": {"prompt": request.args.get('prompt', ''), "data": {}},
            "timestamp": timestamp,
            "output": {"filename": filename, "url": url}
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "gpu": torch.cuda.is_available(),
        "model": "Florence-2-base"
    })

@app.route("/analyze", methods=["POST"])
def api_analyze():
    try:
        if 'image' in request.files:
            img_file = request.files['image']
            img = Image.open(img_file.stream)
        else:
            return jsonify({"error": "No image field"}), 400

        task = request.form.get("task", "<MORE_DETAILED_CAPTION>")
        t0 = time.time()
        result = run_model(img, task, max_tokens=150)
        return jsonify({
            "success": True,
            "task": task,
            "result": result,
            "elapsed": time.time() - t0
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/analyze_json", methods=["POST"])
def api_analyze_json():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image field"}), 400
        img_file = request.files['image']
        img = Image.open(img_file.stream)

        t0 = time.time()
        ocr = run_model(img.copy(), "<OCR_WITH_REGION>", max_tokens=200)
        caption = run_model(img.copy(), "<MORE_DETAILED_CAPTION>", max_tokens=200)
        data = parse_gauges_to_json(ocr, caption)
        return jsonify({
            "success": True,
            "data": data,
            "elapsed": time.time() - t0
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/analyze_json_monitor", methods=["POST"])
def api_analyze_json_monitor():
    """JSON Mode + Health Check: Analyze equipment and check if operating normally"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image field"}), 400
        img_file = request.files['image']
        img = Image.open(img_file.stream)

        t0 = time.time()
        ocr = run_model(img.copy(), "<OCR>", max_tokens=300)
        caption = run_model(img.copy(), "<MORE_DETAILED_CAPTION>", max_tokens=200)
        
        # Parse OCR to structured object
        ocr_structured = parse_ocr_to_structured(ocr)
        
        # Extract gauge data
        data = parse_gauges_to_json(ocr, caption)
        
        # Ensure data is a list
        if isinstance(data, str):
            try:
                import json
                data = json.loads(data)
            except:
                data = []
        
        if not isinstance(data, list):
            data = []
        
        # Health Check Logic
        status = check_machine_health(data)
        
        return jsonify({
            "success": True,
            "data": data,
            "health_check": status,
            "raw_ocr": ocr,
            "raw_ocr_array": ocr_structured,  # New: Structured OCR array
            "raw_caption": caption,
            "elapsed": time.time() - t0
        })
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


def split_ocr_smart(ocr_text):
    """
    Smart split OCR text into tokens
    """
    import re
    
    tokens = []
    
    # Step 1: Identify all patterns
    # Pattern A: "Label Number Unit" (e.g., "Chamber 983 mbar")
    pattern_a = r'([A-Z][a-z]+)\s*(\d+\.?\d*)\s*(mbar|bar|℃|°C|%)'
    
    # Pattern B: "Number Unit" (e.g., "983 mbar", "13.3℃")
    pattern_b = r'(\d+\.?\d*)\s*(mbar|bar|℃|°C|%)'
    
    # Pattern C: "Label" (e.g., "Menu", "Trend")
    pattern_c = r'\b([A-Z][a-z]+)\b'
    
    # Pattern D: "Date/Time" (e.g., "28/01/26", "11:46:56")
    pattern_d = r'(\d{2}/\d{2}/\d{2}|\d{2}:\d{2}:\d{2})'
    
    # Find Date/Time first
    for match in re.finditer(pattern_d, ocr_text):
        tokens.append({
            "type": "datetime",
            "value": match.group(1),
            "raw": match.group(0)
        })
    
    # Remove processed date/time
    text_remaining = ocr_text
    for token in tokens:
        text_remaining = text_remaining.replace(token["raw"], " ", 1)
    
    # Find Label + Number + Unit
    for match in re.finditer(pattern_a, text_remaining, re.IGNORECASE):
        tokens.append({
            "type": "measurement",
            "label": match.group(1),
            "value": float(match.group(2)),
            "unit": match.group(3).lower(),
            "raw": match.group(0)
        })
        text_remaining = text_remaining.replace(match.group(0), " ", 1)
    
    # Find Number + Unit
    for match in re.finditer(pattern_b, text_remaining, re.IGNORECASE):
        tokens.append({
            "type": "measurement",
            "label": None,
            "value": float(match.group(1)),
            "unit": match.group(2).lower(),
            "raw": match.group(0)
        })
        text_remaining = text_remaining.replace(match.group(0), " ", 1)
    
    # Find pure Label
    for match in re.finditer(pattern_c, text_remaining):
        tokens.append({
            "type": "label",
            "label": match.group(1),
            "value": None,
            "unit": None,
            "raw": match.group(0)
        })
    
    return tokens


def parse_ocr_to_structured(ocr_text):
    """
    Parse raw OCR to structured array
    """
    tokens = split_ocr_smart(ocr_text)
    
    print(f"[DEBUG] Parsed {len(tokens)} tokens from OCR")
    
    return tokens


def check_machine_health(data):
    """
    Dynamic health check - supports any gauge type
    Automatically determine range based on unit
    """
    status = {
        "overall": "unknown",
        "warnings": [],
        "errors": [],
        "info": [],
        "detected_values": []
    }
    
    if not isinstance(data, list):
        status["errors"].append("Invalid data format")
        status["overall"] = "error"
        return status
    
    if len(data) == 0:
        status["warnings"].append("No gauge data detected")
        status["overall"] = "warning"
        return status
    
    # Dynamic standard ranges (auto-determined by unit)
    UNIT_STANDARDS = {
        "mbar": {"min": 800, "max": 1100, "critical_min": 700, "critical_max": 1200},
        "bar": {"min": 0.8, "max": 1.1, "critical_min": 0.7, "critical_max": 1.2},
        "°c": {"min": 0, "max": 150, "critical_min": -10, "critical_max": 200},
        "℃": {"min": 0, "max": 150, "critical_min": -10, "critical_max": 200},
        "%": {"min": 0, "max": 100, "critical_min": 0, "critical_max": 100},
    }
    
    all_ok = True
    
    for item in data:
        gauge_type = item.get("type", "unknown")
        value = item.get("numeric_value")
        unit = item.get("unit", "").lower()
        value_str = item.get("value", "")
        
        status["detected_values"].append(f"{gauge_type}: {value_str}")
        
        if value is None:
            continue
        
        # Auto-select standard based on unit
        std = UNIT_STANDARDS.get(unit)
        
        if std:
            # Critical error (exceeds extreme range)
            if value < std["critical_min"] or value > std["critical_max"]:
                status["errors"].append(
                    f"❌ {gauge_type.upper()}: {value}{unit} is OUT OF RANGE "
                    f"(critical: {std['critical_min']}-{std['critical_max']}{unit})"
                )
                all_ok = False
            # Warning (exceeds normal range but within extreme range)
            elif value < std["min"] or value > std["max"]:
                status["warnings"].append(
                    f"⚠️ {gauge_type.upper()}: {value}{unit} is outside normal range "
                    f"(normal: {std['min']}-{std['max']}{unit})"
                )
                all_ok = False
            # OK
            else:
                status["info"].append(f"✅ {gauge_type.upper()}: {value}{unit} (OK)")
        else:
            # Unknown unit, only record without judgment
            status["info"].append(f"ℹ️ {gauge_type.upper()}: {value}{unit} (unknown unit)")
    
    # Determine overall status
    if status["errors"]:
        status["overall"] = "❌ ERROR"
        status["summary"] = "Critical issues detected!"
    elif status["warnings"]:
        status["overall"] = "⚠️ WARNING"
        status["summary"] = "Some values outside normal range"
    elif all_ok and status["info"]:
        status["overall"] = "✅ HEALTHY"
        status["summary"] = "All systems normal"
    else:
        status["overall"] = "❓ UNKNOWN"
        status["summary"] = "Unable to determine status"
    
    return status

# ===== Hand Detection Functions =====

# MediaPipe hand landmark connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

def draw_hand_landmarks(image_np, hand_landmarks_list, handedness_list):
    """
    Draw hand landmarks and connections on the image
    
    Args:
        image_np: numpy array image in BGR format
        hand_landmarks_list: list of MediaPipe hand landmarks
        handedness_list: list of hand classifications (Left/Right)
        
    Returns:
        Annotated numpy array image
    """
    annotated_image = image_np.copy()
    h, w = annotated_image.shape[:2]
    
    # Colors for drawing
    landmark_color = (0, 255, 0)  # Green for landmarks
    connection_color = (255, 0, 0)  # Blue for connections
    landmark_radius = 5
    connection_thickness = 2
    
    if hand_landmarks_list and handedness_list:
        for hand_landmarks, handedness in zip(hand_landmarks_list, handedness_list):
            # Draw connections
            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                start = hand_landmarks.landmark[start_idx]
                end = hand_landmarks.landmark[end_idx]

                # Draw connection regardless of visibility (visibility may be 0 for MediaPipe Hands)
                start_pos = (int(start.x * w), int(start.y * h))
                end_pos = (int(end.x * w), int(end.y * h))
                cv2.line(annotated_image, start_pos, end_pos, connection_color, connection_thickness)
            
            # Draw landmarks
            for idx, landmark in enumerate(hand_landmarks.landmark):
                # Draw all landmarks; some MediaPipe models set visibility=0 for hands
                pos = (int(landmark.x * w), int(landmark.y * h))
                cv2.circle(annotated_image, pos, landmark_radius, landmark_color, -1)
                # Draw landmark index
                cv2.putText(annotated_image, str(idx), (pos[0] + 10, pos[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return annotated_image

def image_to_base64(image_np):
    """
    Convert numpy array image to base64 string
    
    Args:
        image_np: numpy array image in BGR format
        
    Returns:
        base64 encoded string
    """
    import base64
    _, buffer = cv2.imencode('.jpg', image_np)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return base64_image

def detect_objects(image_np):
    """
    Detect objects in image and return bounding boxes with confidence scores.
    Uses MediaPipe ObjectDetector if available, otherwise uses simple contour detection.
    
    Args:
        image_np: numpy array image in BGR format
        
    Returns:
        List of detected objects with bounding boxes and labels:
        [{"bbox": [x1, y1, x2, y2], "category": "object_name", "confidence": score}, ...]
    """
    try:
        # Try using MediaPipe ObjectDetector (if available)
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        import os as mp_os
        
        # Path to object detection model
        model_path = os.path.join(os.path.dirname(__file__), 'efficientdet_lite0.tflite')
        
        if os.path.exists(model_path):
            detector = vision.ObjectDetector.create_from_model_path(model_path)
            rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            mp_image = vision.Image(image_format=vision.ImageFormat.SRGB, data=rgb_image)
            detection_result = detector.detect(mp_image)
            
            objects = []
            h, w = image_np.shape[:2]
            
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                x1 = int(bbox.origin_x * w)
                y1 = int(bbox.origin_y * h)
                x2 = int((bbox.origin_x + bbox.width) * w)
                y2 = int((bbox.origin_y + bbox.height) * h)
                
                # Get category label
                category = "Object"
                if detection.categories:
                    category = detection.categories[0].category_name
                    confidence = detection.categories[0].score
                    objects.append({
                        "bbox": [x1, y1, x2, y2],
                        "category": category,
                        "confidence": float(confidence)
                    })
            
            return objects
    except Exception as e:
        # MediaPipe ObjectDetector not available or failed
        print(f"MediaPipe ObjectDetector unavailable: {e}")
    
    # Fallback: Advanced contour-based object detection with ROI focus
    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        h_img, w_img = image_np.shape[:2]
        
        # Define ROI (Region of Interest) - center 60% of image
        roi_margin_h = int(h_img * 0.2)
        roi_margin_w = int(w_img * 0.2)
        roi_y1, roi_y2 = roi_margin_h, h_img - roi_margin_h
        roi_x1, roi_x2 = roi_margin_w, w_img - roi_margin_w
        
        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Enhance contrast for low-contrast images
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Try Otsu's automatic threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Fallback thresholds
        white_pixel_ratio = np.sum(thresh) / (w_img * h_img * 255)
        if white_pixel_ratio < 0.001:
            _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        elif white_pixel_ratio > 0.95:
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_large, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small)

        # Mild separation step - erosions can help split merged hand+object blobs
        try:
            sep_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            thresh = cv2.erode(thresh, sep_kernel, iterations=1)
            thresh = cv2.dilate(thresh, sep_kernel, iterations=1)
        except Exception:
            # If any morphological op fails, keep original thresh
            pass

        # Find contours with hierarchy
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        min_area = (w_img * h_img) * 0.002  # 0.2% of image area (lowered from 0.5%)
        max_area = (w_img * h_img) * 0.85  # Max 85% (raised from 50%)
        
        print(f"[Object Detection] Image: {w_img}x{h_img}, ROI: ({roi_x1},{roi_y1}) to ({roi_x2},{roi_y2}), Found {len(contours)} contours")
        print(f"[Object Detection] Area filter: {min_area:.0f} - {max_area:.0f}")
        
        # Find the best-scoring object in ROI using multiple heuristics
        largest_obj = None
        largest_score = -1e9
        valid_contours = []

        # Image center for center-score calculation
        img_cx = w_img / 2.0
        img_cy = h_img / 2.0
        max_dist = (img_cx**2 + img_cy**2)**0.5

        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            # Skip too small or too large
            if area < min_area or area > max_area:
                continue

            x, y, cw, ch = cv2.boundingRect(contour)
            x2, y2 = x + cw, y + ch

            # Calculate center of object
            cx = (x + x2) / 2
            cy = (y + y2) / 2

            # Prefer objects in center ROI
            in_roi = (roi_x1 <= cx <= roi_x2) and (roi_y1 <= cy <= roi_y2)
            roi_bonus_score = 2.0 if in_roi else 0.0

            # Skip very small boxes (lowered from 15 to 10 for debugging)
            if cw < 10 or ch < 10:
                continue

            # Basic measurements
            extent = (area / float(cw * ch)) if (cw * ch) > 0 else 0.0
            # Convex hull solidity
            try:
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = (area / hull_area) if hull_area > 0 else 0.0
            except Exception:
                solidity = 0.0

            # Confidence (keeps previous heuristic but used as one feature)
            area_ratio = extent
            confidence = min(max(area_ratio - 0.1, 0.15), 1.0)

            # Center score: higher if object is near image center
            dist = ((cx - img_cx)**2 + (cy - img_cy)**2)**0.5
            center_score = max(0.0, 1.0 - (dist / max_dist))

            # Combined object score (weights tuned for center + solidity + extent)
            object_score = (
                0.40 * confidence +
                0.30 * solidity +
                0.20 * extent +
                0.10 * center_score
            )

            # Apply ROI boost
            adjusted_score = object_score + roi_bonus_score

            valid_obj = {
                "idx": idx,
                "bbox": [x, y, x2, y2],
                "area": area,
                "confidence": float(confidence),
                "in_roi": in_roi,
                "box_size": (cw, ch),
                "center": (cx, cy),
                "solidity": float(solidity),
                "extent": float(extent),
                "score": float(object_score)
            }
            valid_contours.append(valid_obj)

            # Track best object by adjusted score
            if adjusted_score > largest_score:
                largest_score = adjusted_score
                largest_obj = valid_obj
        
        # Log all valid contours for debugging
        print(f"[Object Detection] Valid contours: {len(valid_contours)}")
        for vc in valid_contours:
            print(f"  Contour {vc['idx']}: area={vc['area']:.0f}, conf={vc['confidence']:.2f}, roi={vc['in_roi']}, box={vc['box_size']}, center=({vc['center'][0]:.0f},{vc['center'][1]:.0f})")
        
        # Only return the single largest/most important object
        if largest_obj:
            print(f"[Object Detection] Main object selected: area={largest_obj['area']:.0f}, conf={largest_obj['confidence']:.2f}, in_ROI={largest_obj['in_roi']}")
            objects.append({
                "bbox": largest_obj['bbox'],
                "category": "Object",
                "confidence": largest_obj['confidence'],
                "type": "main"
            })
        
        print(f"[Object Detection] Total objects returned: {len(objects)}")
        return objects
    except Exception as e:
        print(f"[Object Detection] Fallback detection failed: {e}")
        import traceback
        traceback.print_exc()
        return []

def detect_hand_pose(image_input):
    """
    Detect hand landmarks and describe the hand pose
    
    Args:
        image_input: PIL Image or numpy array
        
    Returns:
        Dictionary with hand detection results including:
        - success: bool
        - hands_detected: int
        - hand_details: list of hand info with pose descriptions
        - pose_description: string describing the overall pose
        - annotated_image: base64 encoded image with landmarks drawn
        - objects_detected: list of detected objects with bounding boxes
    """
    try:
        # Convert PIL Image to numpy array if needed
        if isinstance(image_input, Image.Image):
            image_np = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        else:
            image_np = image_input
        
        # Store original dimensions for coordinate scaling
        orig_h, orig_w = image_np.shape[:2]
        scale_factor = 1.0
        
        # Upscale small images to improve detection of small/remote hands
        h, w = image_np.shape[:2]
        max_dim = max(h, w)
        if max_dim < 800:
            scale_factor = 800.0 / float(max_dim)
            new_w, new_h = int(w * scale_factor), int(h * scale_factor)
            image_np = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # Initialize MediaPipe Hands with higher capacity and lower confidence thresholds
        # (model_complexity=1 for better accuracy; increase max_num_hands)
        hands = solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=10,
            model_complexity=1,
            min_detection_confidence=0.1,
            min_tracking_confidence=0.1
        )

        # Process image
        results = hands.process(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        
        hands_detected = 0
        hand_details = []
        
        if results.multi_hand_landmarks and results.multi_handedness:
            hands_detected = len(results.multi_hand_landmarks)
            
            for hand_idx, (hand_landmarks, handedness) in enumerate(zip(
                results.multi_hand_landmarks, 
                results.multi_handedness
            )):
                hand_label = handedness.classification[0].label  # "Left" or "Right"
                confidence = handedness.classification[0].score
                
                # Extract landmarks (normalized coordinates 0-1, independent of scale)
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append({
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z,
                        "visibility": landmark.visibility
                    })
                
                # Analyze pose from landmarks
                pose_description = analyze_hand_pose(landmarks, hand_label)
                
                hand_details.append({
                    "hand_number": hand_idx + 1,
                    "hand_type": hand_label,
                    "confidence": float(confidence),
                    "landmarks_count": len(landmarks),
                    "pose_description": pose_description,
                    "landmarks": landmarks
                })
        
        # Generate overall description
        if hands_detected == 0:
            overall_description = "No hands detected in the image."
        elif hands_detected == 1:
            hand = hand_details[0]
            overall_description = f"One {hand['hand_type'].lower()} hand detected. {hand['pose_description']}"
        else:
            descriptions = [f"{h['hand_type'].lower()}: {h['pose_description']}" for h in hand_details]
            overall_description = f"Two hands detected. Left hand: {descriptions[0] if hand_details[0]['hand_type'] == 'Left' else descriptions[1]}. Right hand: {descriptions[1] if hand_details[0]['hand_type'] == 'Left' else descriptions[0]}"
        
        # Detect objects in the image
        objects = detect_objects(image_np)
        # If hands were detected, compute hand bounding boxes (scaled coordinates)
        hand_bboxes = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                xs = [lm.x for lm in hand_landmarks.landmark]
                ys = [lm.y for lm in hand_landmarks.landmark]
                minx, maxx = min(xs), max(xs)
                miny, maxy = min(ys), max(ys)
                # Convert normalized landmarks to pixel coordinates on scaled image
                bx1 = int(minx * w)
                by1 = int(miny * h)
                bx2 = int(maxx * w)
                by2 = int(maxy * h)
                hand_bboxes.append([bx1, by1, bx2, by2])

        # Mark object overlap with hands (if any) on the scaled coordinates
        def _iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interW = max(0, xB - xA)
            interH = max(0, yB - yA)
            interArea = interW * interH
            boxAArea = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
            boxBArea = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))
            union = boxAArea + boxBArea - interArea
            return (interArea / union) if union > 0 else 0.0

        for obj in objects:
            obj['overlaps_hand'] = False
            # obj['bbox'] are in scaled image coordinates already
            for hb in hand_bboxes:
                try:
                    iou_val = _iou(obj['bbox'], hb)
                    if iou_val > 0.25:
                        obj['overlaps_hand'] = True
                        break
                except Exception:
                    continue

        # Scale object coordinates back to original image size if image was upscaled
        if scale_factor != 1.0:
            for obj in objects:
                obj['bbox'] = [
                    int(obj['bbox'][0] / scale_factor),
                    int(obj['bbox'][1] / scale_factor),
                    int(obj['bbox'][2] / scale_factor),
                    int(obj['bbox'][3] / scale_factor)
                ]
        
        # Draw landmarks on image (using scaled version for better quality)
        annotated_image_np = draw_hand_landmarks(image_np, results.multi_hand_landmarks, results.multi_handedness)
        annotated_image_base64 = image_to_base64(annotated_image_np)
        
        return {
            "success": True,
            "hands_detected": hands_detected,
            "hand_details": hand_details,
            "pose_description": overall_description,
            "annotated_image": annotated_image_base64,
            "objects_detected": objects
        }
    
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def analyze_hand_pose(landmarks, hand_label):
    """
    Analyze hand landmarks and return a description of the pose
    
    Args:
        landmarks: list of landmark dicts with x, y, z coordinates
        hand_label: "Left" or "Right"
        
    Returns:
        String describing the hand pose
    """
    # MediaPipe hand landmark indices
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20
    
    THUMB_PIP = 3
    INDEX_PIP = 6
    MIDDLE_PIP = 10
    RING_PIP = 14
    PINKY_PIP = 18
    
    def distance(p1, p2):
        """Calculate Euclidean distance between two points"""
        return ((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2 + (p1['z'] - p2['z'])**2)**0.5
    
    def is_finger_extended(tip_idx, pip_idx):
        """Check if a finger is extended (tip above PIP joint)"""
        return landmarks[tip_idx]['y'] < landmarks[pip_idx]['y']
    
    def is_finger_folded(tip_idx, pip_idx):
        """Check if a finger is folded (tip below PIP joint)"""
        return landmarks[tip_idx]['y'] > landmarks[pip_idx]['y']
    
    # Analyze each finger
    thumb_extended = is_finger_extended(THUMB_TIP, THUMB_PIP)
    index_extended = is_finger_extended(INDEX_TIP, INDEX_PIP)
    middle_extended = is_finger_extended(MIDDLE_TIP, MIDDLE_PIP)
    ring_extended = is_finger_extended(RING_TIP, RING_PIP)
    pinky_extended = is_finger_extended(PINKY_TIP, PINKY_PIP)
    
    fingers_extended = sum([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended])
    
    # Detect common hand poses
    if fingers_extended == 5:
        return "Open hand, all fingers extended."
    elif fingers_extended == 0:
        return "Closed fist, all fingers folded."
    elif index_extended and middle_extended and not thumb_extended and not ring_extended and not pinky_extended:
        return "Victory/Peace sign (index and middle fingers extended)."
    elif thumb_extended and index_extended and middle_extended and not ring_extended and not pinky_extended:
        return "Three-finger gesture (thumb, index, middle extended)."
    elif index_extended and not thumb_extended and not middle_extended and not ring_extended and not pinky_extended:
        return "Pointing gesture (index finger extended)."
    elif thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
        return "Thumbs up gesture."
    elif index_extended and middle_extended and ring_extended and not thumb_extended and not pinky_extended:
        return "Three-finger salute or similar gesture."
    elif index_extended and middle_extended and ring_extended and pinky_extended and not thumb_extended:
        return "Four fingers extended, thumb folded."
    else:
        # Generic description based on number of extended fingers
        return f"Hand gesture with {fingers_extended} fingers extended."


@app.route("/verify", methods=["POST"])
def api_verify():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image field"}), 400
        img_file = request.files['image']
        img = Image.open(img_file.stream)
        target = request.form.get("target", "chair")

        img.thumbnail((600, 600))
        inputs = processor(text="<CAPTION>", images=img, return_tensors="pt").to("cuda")
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].half()
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        result = processor.decode(output[0], skip_special_tokens=True)
        passed = target.lower() in result.lower()

        return jsonify({
            "success": True,
            "passed": passed,
            "result": result,
            "target": target
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/hand_detection", methods=["POST"])
def api_hand_detection():
    """
    Detect hand gestures/poses and objects in an image
    """
    try:
        if 'image' not in request.files:
            return jsonify({
                "success": False,
                "error": "No image field provided"
            }), 400
        
        img_file = request.files['image']
        img = Image.open(img_file.stream)
        
        # Get hand detection results (includes object detection)
        t0 = time.time()
        results = detect_hand_pose(img)
        elapsed = time.time() - t0
        
        if results.get("success"):
            return jsonify({
                "success": True,
                "hands_detected": results["hands_detected"],
                "pose_description": results["pose_description"],
                "hand_details": results["hand_details"],
                "objects_detected": results.get("objects_detected", []),
                "annotated_image": results.get("annotated_image"),
                "elapsed": elapsed
            })
        else:
            return jsonify({
                "success": False,
                "error": results.get("error", "Unknown error during hand detection")
            }), 500
        
    except Exception as e:
        import traceback
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route("/snapshot", methods=["GET"])
def api_snapshot():
    print("Received snapshot request with args:", request.args)

    try:
        base_dir = os.path.dirname(__file__) or '.'
        images_dir = os.path.join(base_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)

        # Try to find an existing sample image to clone
        candidate_paths = [
            os.path.join(base_dir, 'sample.jpg'),
            os.path.join(base_dir, 'sample.png'),
            os.path.join(images_dir, 'sample.jpg'),
            os.path.join(base_dir, 'images', 'sample.jpg'),
        ]
        src = None
        for p in candidate_paths:
            if os.path.exists(p):
                src = p
                break

        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        unique = uuid.uuid4().hex[:8]
        filename = f"snapshot_{timestamp}_{unique}.jpg"
        dest = os.path.join(images_dir, filename)

        if src:
            shutil.copy(src, dest)
        else:
            # If no sample image exists, create a simple placeholder image
            img = Image.new('RGB', (640, 480), color=(73, 109, 137))
            img.save(dest, 'JPEG')

        # Return the externally-accessible URL (as requested)
        url = f"https://aiagent.maxsolo.co.uk/images/{filename}"
        print(f"Snapshot saved to {dest}, accessible at {url}")
        return jsonify({"success": True, "filename": filename, "url": url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting Flask on 0.0.0.0:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
