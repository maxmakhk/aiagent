import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import time
import torch
import json
import re
import io

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

print("GPU:", torch.cuda.is_available())

# ===== Load Model =====
print("Loading model (one time)...")
start = time.time()
processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-base",
    trust_remote_code=True,
    local_files_only=True,
    clean_up_tokenization_spaces=True
)
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-base",
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=torch.float16
).to("cuda")
print(f"Model loaded in {time.time()-start:.2f}s\n")

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
CORS(app)


@app.route("/", methods=["GET"])
def index():
    """Homepage test interface"""
    return render_template("index.html")

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

if __name__ == "__main__":
    print("Starting Flask on 0.0.0.0:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=True)
