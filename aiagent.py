#python aiagent.py
import os
import sys

# Append RefVision project folder so recognizer can be loaded properly
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_object_identification"))

# Import recognizer package
try:
    from recognizer import (
        load_clip_model,
        compute_embedding,
        update_label_embeddings,
        load_embeddings_cache,
        detect_candidate_crops,
        match_candidates,
        filter_duplicate_detections,
        sanitize_label,
        draw_annotations
    )
except ImportError as e:
    print(f"[!] Warning: Could not import recognizer modules: {e}")

from werkzeug.utils import secure_filename

# Toggle offline cache-only mode with HF_OFFLINE=1
hf_offline = os.environ.get('HF_OFFLINE', '0') == '1'
if hf_offline:
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
else:
    os.environ.pop('TRANSFORMERS_OFFLINE', None)
    os.environ.pop('HF_HUB_OFFLINE', None)

# Base Directories for RefVision
REFVISION_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_object_identification")
REFVISION_DATA_DIR = os.path.join(REFVISION_BASE_DIR, "data")
REFVISION_REF_DIR = os.path.join(REFVISION_DATA_DIR, "references")
REFVISION_SCENES_DIR = os.path.join(REFVISION_DATA_DIR, "scenes")
REFVISION_RESULTS_DIR = os.path.join(REFVISION_DATA_DIR, "results")

# Ensure required directories exist for RefVision
for path in [REFVISION_DATA_DIR, REFVISION_REF_DIR, REFVISION_SCENES_DIR, REFVISION_RESULTS_DIR]:
    os.makedirs(path, exist_ok=True)

# Allowed image formats for RefVision
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

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

print("python aiagent.py")
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
app = Flask(__name__, static_folder="ai_object_identification/static", template_folder="templates")
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

def draw_objects_grasp(image_np, objects):
    """
    Draw object bounding boxes and grasp indicators on the image.
    - Normal object  : blue box  labelled "Object <conf>"
    - Grasped object : orange box labelled "GRASP <conf>"
    """
    annotated = image_np.copy()
    for obj in objects:
        x1, y1, x2, y2 = obj['bbox']
        is_grasp = obj.get('overlaps_hand', False)
        confidence = obj.get('confidence', 0.0)
        category = obj.get('category', 'Object')

        if is_grasp:
            color = (0, 165, 255)   # Orange (BGR)
            label = f"GRASP {confidence:.2f}"
            thickness = 3
        else:
            color = (255, 80, 0)    # Blue (BGR)
            label = f"{category} {confidence:.2f}"
            thickness = 2

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Label background
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y0 = max(y1 - lh - baseline - 4, 0)
        cv2.rectangle(annotated, (x1, label_y0), (x1 + lw, label_y0 + lh + baseline + 4), color, -1)
        cv2.putText(annotated, label, (x1, label_y0 + lh + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return annotated

# COCO-compatible category groups for convenient filtering
CATEGORY_GROUPS = {
    "vehicle": {"car", "truck", "bus", "motorcycle", "bicycle", "train",
                "airplane", "boat", "vehicle"},
    "person":  {"person", "pedestrian", "man", "woman", "child"},
    "animal":  {"cat", "dog", "horse", "sheep", "cow", "elephant",
                "bear", "zebra", "giraffe", "bird"},
    "furniture":   {"chair", "couch", "bed", "dining table", "toilet"},
    "electronics": {"tv", "laptop", "cell phone", "keyboard", "mouse",
                    "remote", "microwave", "oven"},
}

# Per-category display colours (BGR)
_CATEGORY_COLOURS = {
    "person":     (0,   255, 128),
    "car":        (0,   128, 255),
    "truck":      (0,   80,  200),
    "bus":        (0,   60,  180),
    "motorcycle": (0,   200, 255),
    "bicycle":    (100, 200, 255),
    "train":      (50,  50,  255),
    "airplane":   (200, 100, 255),
    "boat":       (255, 150, 0),
    "default":    (255, 80,  0),
}

def _category_colour(category):
    name = category.lower()
    for key, colour in _CATEGORY_COLOURS.items():
        if key in name:
            return colour
    return _CATEGORY_COLOURS["default"]

def _expand_filter(filter_categories):
    """Expand group aliases (e.g. 'vehicle') into individual category names."""
    if not filter_categories:
        return None
    expanded = set()
    for c in filter_categories:
        cl = c.lower().strip()
        if cl in CATEGORY_GROUPS:
            expanded |= CATEGORY_GROUPS[cl]
        else:
            expanded.add(cl)
    return expanded

def detect_objects(image_np, hand_bboxes=None, filter_categories=None, hint_bbox=None):
    """
    Detect objects in image and return bounding boxes with confidence scores.
    Uses MediaPipe ObjectDetector if available, otherwise uses contour detection.

    Args:
        image_np: numpy array image in BGR format
        hand_bboxes: optional list of hand bboxes [[x1,y1,x2,y2], ...]
        filter_categories: optional list of category names or group aliases.
            e.g. ["person", "vehicle"] — pass None to return all.
        hint_bbox: optional [x1,y1,x2,y2] to bias scoring toward a tracked object.

    Returns:
        [{"bbox": [x1,y1,x2,y2], "category": str, "confidence": float}, ...]
    """
    allowed = _expand_filter(filter_categories)

    try:
        # Try using MediaPipe ObjectDetector (if available)
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        # Path to object detection model
        model_path = os.path.join(os.path.dirname(__file__), 'efficientdet_lite0.tflite')

        if os.path.exists(model_path):
            detector = vision.ObjectDetector.create_from_model_path(model_path)
            rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            mp_image = vision.Image(image_format=vision.ImageFormat.SRGB, data=rgb_image)
            detection_result = detector.detect(mp_image)

            objects = []
            for detection in detection_result.detections:
                if not detection.categories:
                    continue
                category = detection.categories[0].category_name
                confidence = detection.categories[0].score

                # Apply category filter
                if allowed and category.lower() not in allowed:
                    print(f"[Object Detection] Skipping '{category}' (not in filter)")
                    continue

                bbox = detection.bounding_box
                objects.append({
                    "bbox": [int(bbox.origin_x), int(bbox.origin_y),
                             int(bbox.origin_x + bbox.width),
                             int(bbox.origin_y + bbox.height)],
                    "category": category,
                    "confidence": float(confidence)
                })

            print(f"[Object Detection] MediaPipe: {len(objects)} objects"
                  + (f" (filter={filter_categories})" if filter_categories else ""))
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
        min_area = (w_img * h_img) * 0.002   # 0.2% of image area
        max_area = (w_img * h_img) * 0.40    # Max 40%: reject background/desk blobs
        
        print(f"[Object Detection] Image: {w_img}x{h_img}, ROI: ({roi_x1},{roi_y1}) to ({roi_x2},{roi_y2}), Found {len(contours)} contours")
        print(f"[Object Detection] Area filter: {min_area:.0f} - {max_area:.0f}")
        
        # Find the best-scoring object using center-priority scoring
        largest_obj = None
        largest_score = -1e9
        valid_contours = []

        # Image center for center-score calculation
        img_cx = w_img / 2.0
        img_cy = h_img / 2.0
        # Use half-diagonal as normalisation (so center_score=1 at center, 0 at corner)
        max_dist = (img_cx**2 + img_cy**2)**0.5

        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            # Skip too small or too large
            if area < min_area or area > max_area:
                continue

            x, y, cw, ch = cv2.boundingRect(contour)
            x2, y2 = x + cw, y + ch

            if cw < 10 or ch < 10:
                continue

            # Object center
            cx = (x + x2) / 2.0
            cy = (y + y2) / 2.0

            # --- CENTER SCORE (primary): objects closer to image centre score higher ---
            dist = ((cx - img_cx)**2 + (cy - img_cy)**2)**0.5
            center_score = max(0.0, 1.0 - (dist / max_dist))

            # --- SIZE SCORE: prefer medium-sized objects, penalise very large blobs ---
            area_ratio = area / float(w_img * h_img)
            # Peak at ~8% of image; falls off for tiny (<1%) or huge (>30%) objects
            size_score = max(0.0, 1.0 - abs(area_ratio - 0.08) / 0.25)

            # --- SHAPE SCORE: compact/solid objects preferred over thin noise ---
            extent = area / float(cw * ch) if (cw * ch) > 0 else 0.0
            try:
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0.0
            except Exception:
                solidity = 0.0
            shape_score = 0.5 * extent + 0.5 * solidity

            # --- ROI bonus: extra boost when centre is inside the middle 60% zone ---
            in_roi = (roi_x1 <= cx <= roi_x2) and (roi_y1 <= cy <= roi_y2)
            roi_bonus = 1.5 if in_roi else 0.0

            # Combined score: centre is the dominant factor (0.55 weight)
            object_score = (
                0.55 * center_score +
                0.25 * size_score +
                0.20 * shape_score
            )
            adjusted_score = object_score + roi_bonus

            # Confidence exposed in result (derived from shape quality)
            confidence = min(max(shape_score - 0.1, 0.15), 1.0)

            # Hand proximity boost: objects overlapping a hand are likely being grasped
            if hand_bboxes:
                for _hb in hand_bboxes:
                    _xA = max(x, _hb[0]);  _yA = max(y, _hb[1])
                    _xB = min(x2, _hb[2]); _yB = min(y2, _hb[3])
                    _inter = max(0, _xB - _xA) * max(0, _yB - _yA)
                    _union = (cw * ch) + max(0, _hb[2] - _hb[0]) * max(0, _hb[3] - _hb[1]) - _inter
                    _iou_hand = _inter / _union if _union > 0 else 0.0
                    if _iou_hand > 0.02:
                        adjusted_score += 4.0 * min(_iou_hand * 5, 1.0)
                        break

            # Hint bbox boost: heavily favour the region being tracked
            if hint_bbox:
                _hxA = max(x, hint_bbox[0]); _hyA = max(y, hint_bbox[1])
                _hxB = min(x2, hint_bbox[2]); _hyB = min(y2, hint_bbox[3])
                _hint_inter = max(0, _hxB - _hxA) * max(0, _hyB - _hyA)
                _hint_w = max(0, hint_bbox[2] - hint_bbox[0])
                _hint_h = max(0, hint_bbox[3] - hint_bbox[1])
                _hint_union = (cw * ch) + _hint_w * _hint_h - _hint_inter
                _iou_hint = _hint_inter / _hint_union if _hint_union > 0 else 0.0
                if _iou_hint > 0.1:
                    adjusted_score += 8.0 * _iou_hint

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

def detect_hand_pose(image_input, hint_bbox=None):
    """
    Detect hand landmarks and describe the hand pose
    
    Args:
        image_input: PIL Image or numpy array
        hint_bbox: optional [x1,y1,x2,y2] in original image coords to bias
                   object detection toward a tracked region.
        
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

        # Scale hint_bbox to match (possibly upscaled) image coordinates
        scaled_hint_bbox = None
        if hint_bbox and len(hint_bbox) == 4:
            if scale_factor != 1.0:
                scaled_hint_bbox = [int(c * scale_factor) for c in hint_bbox]
            else:
                scaled_hint_bbox = list(hint_bbox)

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
        
        # Compute hand bounding boxes (scaled pixel coords) BEFORE object detection
        # so detect_objects can prioritise objects near/touching the hand.
        hand_bboxes = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                xs = [lm.x for lm in hand_landmarks.landmark]
                ys = [lm.y for lm in hand_landmarks.landmark]
                minx, maxx = min(xs), max(xs)
                miny, maxy = min(ys), max(ys)
                bx1 = int(minx * w)
                by1 = int(miny * h)
                bx2 = int(maxx * w)
                by2 = int(maxy * h)
                hand_bboxes.append([bx1, by1, bx2, by2])

        # Detect the single best object with hand-aware scoring (scaled image coords)
        objects = detect_objects(image_np, hand_bboxes=hand_bboxes if hand_bboxes else None,
                                 hint_bbox=scaled_hint_bbox)

        # IoU helper for overlap marking
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

        # Mark objects that overlap a hand bounding box (potential grasp)
        for obj in objects:
            obj['overlaps_hand'] = False
            for hb in hand_bboxes:
                try:
                    if _iou(obj['bbox'], hb) > 0.15:
                        obj['overlaps_hand'] = True
                        break
                except Exception:
                    continue

        # ── Draw on the SCALED image (best quality) ──────────────────────────
        # 1. Hand landmarks
        annotated_image_np = draw_hand_landmarks(
            image_np, results.multi_hand_landmarks, results.multi_handedness)
        # 2. Object box + GRASP indicator (still in scaled coords)
        annotated_image_np = draw_objects_grasp(annotated_image_np, objects)
        annotated_image_base64 = image_to_base64(annotated_image_np)
        # ─────────────────────────────────────────────────────────────────────

        # Scale object coordinates back to original image size
        if scale_factor != 1.0:
            for obj in objects:
                obj['bbox'] = [
                    int(obj['bbox'][0] / scale_factor),
                    int(obj['bbox'][1] / scale_factor),
                    int(obj['bbox'][2] / scale_factor),
                    int(obj['bbox'][3] / scale_factor)
                ]

        # Grasp = at least one detected hand overlapping the selected object
        grasped_objects = [obj for obj in objects if obj.get('overlaps_hand', False)]
        grasp_detected = len(grasped_objects) > 0 and hands_detected > 0

        return {
            "success": True,
            "hands_detected": hands_detected,
            "hand_details": hand_details,
            "pose_description": overall_description,
            "annotated_image": annotated_image_base64,
            "objects_detected": objects,
            "grasp_detected": grasp_detected,
            "grasped_objects": grasped_objects
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


@app.route("/detect_objects", methods=["POST"])
def api_detect_objects():
    """
    General object detection with optional category filter.

    Form fields:
        image          : image file (required)
        categories     : comma-separated names or group aliases.
                         Groups: vehicle, person, animal, furniture, electronics
                         e.g. "person,vehicle" or "car,bus,person"
                         Omit to detect all objects.
        min_confidence : float 0..1, default 0.3
    """
    try:
        if 'image' not in request.files:
            return jsonify({"success": False, "error": "No image field provided"}), 400

        img_pil = Image.open(request.files['image'].stream)
        image_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        raw_cats = request.form.get("categories", "").strip()
        filter_categories = [c.strip() for c in raw_cats.split(",") if c.strip()] or None

        try:
            min_conf = float(request.form.get("min_confidence", 0.3))
        except ValueError:
            min_conf = 0.3

        t0 = time.time()
        objects = detect_objects(image_np, filter_categories=filter_categories)
        elapsed = time.time() - t0

        objects = [o for o in objects if o.get("confidence", 1.0) >= min_conf]

        # Draw coloured boxes per category
        annotated = image_np.copy()
        for obj in objects:
            x1, y1, x2, y2 = obj['bbox']
            colour = _category_colour(obj.get('category', 'Object'))
            label = f"{obj['category']} {obj['confidence']:.0%}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
            (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ly0 = max(y1 - lh - bl - 4, 0)
            cv2.rectangle(annotated, (x1, ly0), (x1 + lw, ly0 + lh + bl + 4), colour, -1)
            cv2.putText(annotated, label, (x1, ly0 + lh + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return jsonify({
            "success": True,
            "filter": filter_categories,
            "min_confidence": min_conf,
            "objects_detected": objects,
            "count": len(objects),
            "annotated_image": image_to_base64(annotated),
            "elapsed": elapsed
        })

    except Exception as e:
        import traceback
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route("/detect_grasped", methods=["POST"])
def api_detect_grasped():
    """
    Return only objects that appear to be grasped (overlap with detected hand).

    Form fields:
      image          : image file (required)
      min_confidence : optional float 0..1 to filter returned grasped objects
    """
    try:
        if 'image' not in request.files:
            return jsonify({"success": False, "error": "No image field provided"}), 400

        img_pil = Image.open(request.files['image'].stream)

        try:
            min_conf = float(request.form.get("min_confidence", 0.0))
        except Exception:
            min_conf = 0.0

        t0 = time.time()
        results = detect_hand_pose(img_pil)
        elapsed = time.time() - t0

        if not results.get("success"):
            return jsonify({"success": False, "error": results.get("error", "detection failed")}), 500

        grasped = results.get("grasped_objects", []) or []
        # apply confidence filter
        grasped = [o for o in grasped if o.get("confidence", 0.0) >= min_conf]

        return jsonify({
            "success": True,
            "grasp_detected": results.get("grasp_detected", False),
            "grasped_count": len(grasped),
            "grasped_objects": grasped,
            "annotated_image": results.get("annotated_image"),
            "elapsed": elapsed
        })

    except Exception as e:
        import traceback
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


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

        # Optional tracking hint: [x1,y1,x2,y2] in original image coords
        hint_bbox = None
        hint_bbox_raw = request.form.get('hint_bbox')
        if hint_bbox_raw:
            try:
                hint_bbox = json.loads(hint_bbox_raw)
            except Exception:
                hint_bbox = None

        # Get hand detection results (includes object detection)
        t0 = time.time()
        results = detect_hand_pose(img, hint_bbox=hint_bbox)
        elapsed = time.time() - t0
        
        if results.get("success"):
            return jsonify({
                "success": True,
                "hands_detected": results["hands_detected"],
                "pose_description": results["pose_description"],
                "hand_details": results["hand_details"],
                "objects_detected": results.get("objects_detected", []),
                "grasp_detected": results.get("grasp_detected", False),
                "grasped_objects": results.get("grasped_objects", []),
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

# ===== RefVision (CLIP Object Recognizer) API Endpoints =====

# Route: Serve RefVision Homepage
@app.route("/refvision")
def refvision_home():
    return render_template("refvision.html")

# Route: Serving Static Files from the RefVision Data Folder
@app.route("/data/<path:filename>")
def serve_data(filename):
    return send_from_directory(REFVISION_DATA_DIR, filename)

# Route: RefVision Backend Health & Status
@app.route("/api/health", methods=["GET"])
def refvision_health():
    try:
        # Check if CLIP is pre-loaded or load it lazily
        model, _ = load_clip_model()
        device = str(model.device)
        
        # Get active labels in references directory
        active_labels = []
        if os.path.exists(REFVISION_REF_DIR):
            active_labels = [d for d in os.listdir(REFVISION_REF_DIR) if os.path.isdir(os.path.join(REFVISION_REF_DIR, d))]
            
        return jsonify({
            "status": "healthy",
            "device": device,
            "clip_model": "openai/clip-vit-base-patch32",
            "active_labels_count": len(active_labels),
            "active_labels": active_labels
        }), 200
    except Exception as e:
        return jsonify({
            "status": "degraded",
            "error": str(e)
        }), 500

# Route: Create a new Reference Object Label
@app.route("/api/reference/create", methods=["POST"])
def create_reference_label():
    data = request.get_json() or {}
    raw_label = data.get("label", "").strip()
    
    if not raw_label:
        return jsonify({"error": "Label name is required"}), 400
        
    label = sanitize_label(raw_label)
    label_dir = os.path.join(REFVISION_REF_DIR, label)
    
    if os.path.exists(label_dir):
        return jsonify({
            "message": f"Label '{label}' already exists",
            "label": label,
            "created": False
        }), 200
        
    os.makedirs(label_dir, exist_ok=True)
    return jsonify({
        "message": f"Label '{label}' created successfully",
        "label": label,
        "created": True
    }), 201

# Route: List all registered target labels and image counts
@app.route("/api/reference/list", methods=["GET"])
def list_references():
    if not os.path.exists(REFVISION_REF_DIR):
        return jsonify([])
        
    labels = []
    for label_name in os.listdir(REFVISION_REF_DIR):
        label_path = os.path.join(REFVISION_REF_DIR, label_name)
        if os.path.isdir(label_path):
            # Count valid reference image files
            files = [f for f in os.listdir(label_path) if allowed_file(f)]
            
            # Find first thumbnail if any exist
            thumbnail_url = None
            if files:
                thumbnail_url = f"/data/references/{label_name}/{files[0]}"
                
            labels.append({
                "label": label_name,
                "photo_count": len(files),
                "thumbnail": thumbnail_url
            })
            
    return jsonify(labels), 200

# Route: Get details (photos) of a specific target label
@app.route("/api/reference/<label>", methods=["GET"])
def get_reference_details(label):
    label = sanitize_label(label)
    label_path = os.path.join(REFVISION_REF_DIR, label)
    
    if not os.path.exists(label_path) or not os.path.isdir(label_path):
        return jsonify({"error": f"Label '{label}' does not exist"}), 404
        
    files = [f for f in os.listdir(label_path) if allowed_file(f)]
    photo_urls = [f"/data/references/{label}/{f}" for f in files]
    
    # Check cache status
    cache = load_embeddings_cache(REFVISION_DATA_DIR, label)
    cached_count = len(cache)
    
    return jsonify({
        "label": label,
        "photo_count": len(files),
        "photos": photo_urls,
        "cached_embeddings_count": cached_count,
        "cache_synchronized": len(files) == cached_count
    }), 200

# Route: Upload Reference Images for a target label
@app.route("/api/reference/upload", methods=["POST"])
def upload_reference_photos():
    label = request.form.get("label", "").strip()
    if not label:
        return jsonify({"error": "Label is required"}), 400
        
    label = sanitize_label(label)
    label_path = os.path.join(REFVISION_REF_DIR, label)
    os.makedirs(label_path, exist_ok=True)
    
    if "files" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400
        
    files = request.files.getlist("files")
    saved_files = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Add unique prefix to avoid duplicate filenames overwriting each other
            unique_filename = f"{uuid.uuid4().hex[:8]}_{filename}"
            save_path = os.path.join(label_path, unique_filename)
            file.save(save_path)
            saved_files.append(unique_filename)
            
    # Trigger an incremental embedding calculation and cache update
    if saved_files:
        update_label_embeddings(REFVISION_DATA_DIR, label)
        
    return jsonify({
        "message": f"Successfully uploaded {len(saved_files)} reference files for label '{label}'",
        "label": label,
        "files": saved_files
    }), 200

# Route: Rebuild embeddings for all active labels
@app.route("/api/reference/rebuild", methods=["POST"])
def rebuild_embeddings():
    if not os.path.exists(REFVISION_REF_DIR):
        return jsonify({"error": "No labels found to rebuild"}), 400
        
    rebuilt_labels = {}
    for label_name in os.listdir(REFVISION_REF_DIR):
        label_path = os.path.join(REFVISION_REF_DIR, label_name)
        if os.path.isdir(label_path):
            cache = update_label_embeddings(REFVISION_DATA_DIR, label_name, force_rebuild=True)
            rebuilt_labels[label_name] = len(cache)
            
    return jsonify({
        "message": "Successfully rebuilt all reference embeddings",
        "rebuilt_labels": rebuilt_labels
    }), 200

# Route: Recognize objects in a scene photo
@app.route("/api/recognize", methods=["POST"])
def recognize():
    if "image" not in request.files:
        return jsonify({"error": "No scene image file uploaded"}), 400
        
    file = request.files["image"]
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Upload a valid image"}), 400
        
    # Get parameters with defaults
    threshold = float(request.form.get("threshold", 0.75))
    target_label = request.form.get("target_label", "").strip()
    if target_label == "all" or target_label == "":
        target_label = None
        
    # Save original scene image
    filename = secure_filename(file.filename)
    unique_name = f"scene_{uuid.uuid4().hex[:8]}_{filename}"
    scene_path = os.path.join(REFVISION_SCENES_DIR, unique_name)
    file.save(scene_path)
    
    # Process scene using PIL and OpenCV
    try:
        scene_pil = Image.open(scene_path)
        # Convert PIL to CV image (BGR) for contour extraction
        image_cv = cv2.imread(scene_path)
        if image_cv is None:
            raise ValueError("Could not read image in OpenCV format.")
    except Exception as e:
        return jsonify({"error": f"Failed to parse image file: {e}"}), 400
        
    # 1. Extract active target labels
    active_labels = []
    if os.path.exists(REFVISION_REF_DIR):
        active_labels = [d for d in os.listdir(REFVISION_REF_DIR) if os.path.isdir(os.path.join(REFVISION_REF_DIR, d))]
        
    if not active_labels:
        return jsonify({
            "error": "No reference object labels exist. Please teach the system an object first!"
        }), 400
        
    # Verify embeddings are updated
    for label in active_labels:
        update_label_embeddings(REFVISION_DATA_DIR, label)
        
    # 2. OpenCV candidate proposals
    candidate_boxes, debug_info = detect_candidate_crops(image_cv)
    
    # 3. Match candidate embeddings against reference embeddings
    raw_detections = match_candidates(
        scene_pil, 
        candidate_boxes, 
        REFVISION_DATA_DIR, 
        active_labels, 
        threshold=threshold, 
        target_label=target_label
    )
    
    # 4. Apply NMS / Box merging to filter out duplicate boxes for the same target object
    filtered_detections = filter_duplicate_detections(raw_detections, iou_threshold=0.45)
    
    # 5. Summarize object counts
    counts = {}
    for det in filtered_detections:
        lbl = det["label"]
        if lbl != "unknown": # Only count positive matches
            counts[lbl] = counts.get(lbl, 0) + 1
            
    # 6. Generate annotated scene drawing
    result_name = f"result_{unique_name}"
    result_path = os.path.join(REFVISION_RESULTS_DIR, result_name)
    draw_annotations(scene_path, filtered_detections, result_path)
    
    # Relative URLs for frontend mapping
    scene_url = f"/data/scenes/{unique_name}"
    result_url = f"/data/results/{result_name}"
    
    return jsonify({
        "status": "success",
        "scene_url": scene_url,
        "result_url": result_url,
        "counts": counts,
        "detections": filtered_detections,
        "debug": {
            "proposals_count": debug_info["raw_contour_proposals"],
            "merged_contour_count": debug_info["merged_contour_proposals"],
            "using_fallback_grid": debug_info["using_fallback_grid"],
            "final_analyzed_candidates": len(candidate_boxes)
        }
    }), 200

if __name__ == "__main__":
    # Pre-warm models on startup
    if not skip_model_load:
        print("[*] Pre-warming CLIP model...")
        try:
            load_clip_model()
        except Exception as e:
            print(f"[!] Warning: CLIP model failed to load during pre-warm: {e}")
            
    print("Starting Flask on 0.0.0.0:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
