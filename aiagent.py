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
from collections import Counter
from difflib import SequenceMatcher
from mediapipe import solutions
import numpy as np
import cv2

# Vision2 JSON storage
VISION2_DATA_DIR = os.path.join(REFVISION_DATA_DIR, "vision2")
VISION2_BARCODE_DB_PATH = os.path.join(VISION2_DATA_DIR, "barcode_db.json")
VISION2_QRCODE_DB_PATH = os.path.join(VISION2_DATA_DIR, "qrcode_db.json")
os.makedirs(VISION2_DATA_DIR, exist_ok=True)

# Optional Ollama chat bridge for short-name extraction
OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434/api/chat')
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'gemma4:31b-cloud')

# Allow skipping heavy model load for quick testing (set SKIP_MODEL_LOAD=1)
skip_model_load = os.environ.get('SKIP_MODEL_LOAD') == '1'

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

print("python aiagent.py")
print("GPU:", torch.cuda.is_available())
print("HF offline mode:", hf_offline)

# ===== Load Model =====

model_name = "MiaoshouAI/Florence-2-base-PromptGen"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    local_files_only=hf_offline,   # 這個你原本就有
    torch_dtype=torch.float16      # 建議放在這邊，而不是 processor 那裡
).to("cuda")

processor = AutoProcessor.from_pretrained(
    model_name,
    trust_remote_code=True,
    local_files_only=hf_offline,
    # 這裡不要放 torch_dtype
    # 如果你真的想調整空白，可以在 decode 時控制
)


if not skip_model_load:
    print("Loading model (one time)...")
    start = time.time()
    '''
    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-base-PromptGen",
        trust_remote_code=True,
        local_files_only=hf_offline,
        clean_up_tokenization_spaces=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-base-PromptGen",
        trust_remote_code=True,
        local_files_only=hf_offline,
        torch_dtype=torch.float16
    ).to("cuda")
    print(f"Model loaded in {time.time()-start:.2f}s\n")
    '''
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


def _vision2_load_db(path):
    if not os.path.exists(path):
        return {"version": 1, "updated_at": datetime.utcnow().isoformat() + "Z", "records": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Invalid db format")
        data.setdefault("version", 1)
        data.setdefault("updated_at", datetime.utcnow().isoformat() + "Z")
        data.setdefault("records", [])
        if not isinstance(data["records"], list):
            data["records"] = []
        return data
    except Exception:
        return {"version": 1, "updated_at": datetime.utcnow().isoformat() + "Z", "records": []}


def _vision2_save_db(path, payload):
    payload["updated_at"] = datetime.utcnow().isoformat() + "Z"
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _vision2_upsert_record(path, kind, record):
    code = str(record.get("code", "")).strip()
    if not code:
        raise ValueError("'code' is required")

    db = _vision2_load_db(path)
    normalized = {
        "kind": kind,
        "code": code,
        "name": str(record.get("name", "")).strip(),
        "code_number": str(record.get("code_number", "")).strip(),
        "simple_description": str(record.get("simple_description", "")).strip(),
        "description": str(record.get("description", "")).strip(),
        "tags": record.get("tags", []),
        "updated_at": datetime.utcnow().isoformat() + "Z"
    }
    if not isinstance(normalized["tags"], list):
        normalized["tags"] = []

    replaced = False
    for idx, existing in enumerate(db["records"]):
        if str(existing.get("code", "")).strip() == code:
            db["records"][idx] = {**existing, **normalized}
            replaced = True
            break

    if not replaced:
        db["records"].append(normalized)

    _vision2_save_db(path, db)
    return normalized


def _vision2_delete_record(path, code):
    db = _vision2_load_db(path)
    original_count = len(db["records"])
    db["records"] = [r for r in db["records"] if str(r.get("code", "")).strip() != str(code).strip()]
    removed = original_count - len(db["records"])
    if removed > 0:
        _vision2_save_db(path, db)
    return removed


def _vision2_db_by_kind(kind):
    kind_norm = str(kind).strip().lower()
    if kind_norm == "barcode":
        return "barcode", VISION2_BARCODE_DB_PATH
    if kind_norm in ("qrcode", "qr", "qr_code"):
        return "qrcode", VISION2_QRCODE_DB_PATH
    raise ValueError("kind must be 'barcode' or 'qrcode'")


def _vision2_rotate_image(image, angle):
    """
    Rotates an image by a given angle and returns the rotated image along with the inverse
    transformation matrix to map coordinates back to the original image.
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Adjust matrix to prevent clipping
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    rotated = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(255, 255, 255))
    M_inv = cv2.invertAffineTransform(M)
    return rotated, M_inv


def _vision2_detect_candidate_regions(image_np):
    """
    Locates candidate regions for barcodes/QR codes in the image using morphological operations.
    Returns list of tuples (kind, (x1, y1, x2, y2)) in original image coordinates.
    """
    candidates = []
    if image_np is None:
        return candidates

    h, w = image_np.shape[:2]
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_np.copy()

    # Precalculate Sobel gradients with kernel size 3
    try:
        grad_x = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
        grad_y = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
        grad_x = cv2.convertScaleAbs(grad_x)
        grad_y = cv2.convertScaleAbs(grad_y)
    except Exception as e:
        print(f"[Vision2] Sobel gradient computation failed: {e}")
        return candidates

    # --- 1. Barcode Region Proposals ---
    try:
        # Subtract vertical edges from horizontal gradients to isolate barcode stripes
        gradient = cv2.subtract(grad_x, grad_y)
        _, thresh = cv2.threshold(gradient, 25, 255, cv2.THRESH_BINARY)

        # Close horizontally (horizontal kernel merges vertical stripes)
        kernel_bc = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_bc)

        # Clean noise
        closed = cv2.erode(closed, None, iterations=2)
        closed = cv2.dilate(closed, None, iterations=4)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            if cw > 0 and ch > 0:
                area = cw * ch
                aspect_ratio = cw / float(ch)
                # Barcodes are typically wide-ish
                if area > 1000 and (0.3 < aspect_ratio < 10.0) and cw > 50 and ch > 20:
                    candidates.append(("barcode", (x, y, x + cw, y + ch)))
    except Exception as e:
        print(f"[Vision2] Barcode candidate detection failed: {e}")

    # --- 2. QR Code Region Proposals ---
    try:
        grad_abs = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
        _, thresh_qr = cv2.threshold(grad_abs, 25, 255, cv2.THRESH_BINARY)

        # Close with a square kernel
        kernel_qr = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
        closed_qr = cv2.morphologyEx(thresh_qr, cv2.MORPH_CLOSE, kernel_qr)
        closed_qr = cv2.erode(closed_qr, None, iterations=2)
        closed_qr = cv2.dilate(closed_qr, None, iterations=4)

        contours_qr, _ = cv2.findContours(closed_qr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours_qr:
            x, y, cw, ch = cv2.boundingRect(c)
            if cw > 0 and ch > 0:
                area = cw * ch
                aspect_ratio = cw / float(ch)
                # QR codes are square-ish
                if area > 1200 and (0.5 < aspect_ratio < 2.0) and cw > 40 and ch > 40:
                    candidates.append(("qrcode", (x, y, x + cw, y + ch)))
    except Exception as e:
        print(f"[Vision2] QR candidate detection failed: {e}")

    # --- 3. Merge overlapping candidates using IoA check ---
    candidates = sorted(candidates, key=lambda x: (x[1][2] - x[1][0]) * (x[1][3] - x[1][1]), reverse=True)
    merged = []
    
    for kind, box in candidates:
        bx1, by1, bx2, by2 = box
        b_area = (bx2 - bx1) * (by2 - by1)
        
        overlap = False
        for m_kind, m_box in merged:
            mx1, my1, mx2, my2 = m_box
            
            ix1 = max(bx1, mx1)
            iy1 = max(by1, my1)
            ix2 = min(bx2, mx2)
            iy2 = min(by2, my2)
            
            if ix2 > ix1 and iy2 > iy1:
                int_area = (ix2 - ix1) * (iy2 - iy1)
                # If smaller candidate is mostly inside an already accepted candidate region
                if int_area / float(b_area) > 0.6:
                    overlap = True
                    break
        if not overlap:
            merged.append((kind, box))
            
    return merged


def _vision2_get_image_variants(image_np):
    """
    Generates multiple preprocessed, scaled, and rotated variants of the input image
    to ensure extremely robust barcode and QR code detection under challenging conditions
    (e.g., high resolution, tilt/rotation, bad lighting, shadows).
    Returns:
        List of tuples: (name, image, scale_x, scale_y, offset_x, offset_y, M_inv)
    """
    variants = []
    if image_np is None:
        return variants

    height, width = image_np.shape[:2]
    
    # 1. Original (BGR)
    variants.append(("original", image_np, 1.0, 1.0, 0, 0, None))

    # 2. Grayscale
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_np.copy()
    variants.append(("gray", gray, 1.0, 1.0, 0, 0, None))

    # 3. CLAHE (Contrast Enhancement)
    clahe = None
    try:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        clahe_gray = clahe.apply(gray)
        variants.append(("clahe", clahe_gray, 1.0, 1.0, 0, 0, None))
    except Exception:
        clahe_gray = gray

    # 4. Otsu Threshold
    try:
        _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(("otsu", binary_otsu, 1.0, 1.0, 0, 0, None))
    except Exception:
        pass

    # 5. Adaptive Threshold
    try:
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 11)
        variants.append(("adaptive", adaptive, 1.0, 1.0, 0, 0, None))
    except Exception:
        pass

    # 6. High-Resolution Downscaled Variants (Crucial for 1D Barcodes in pyzbar!)
    for target_w in [1280, 800]:
        if width > target_w:
            try:
                scale = target_w / width
                target_h = int(height * scale)
                scaled_gray = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
                variants.append((f"downscaled_{target_w}", scaled_gray, scale, scale, 0, 0, None))
                
                scaled_clahe = cv2.resize(clahe_gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
                variants.append((f"downscaled_clahe_{target_w}", scaled_clahe, scale, scale, 0, 0, None))
            except Exception:
                pass

    # 7. Rotated Variants (Crucial for tilted/rotated barcodes!)
    for angle in [-30, -15, 15, 30, 90]:
        try:
            rot_gray, M_inv = _vision2_rotate_image(gray, angle)
            variants.append((f"rotated_{angle}", rot_gray, 1.0, 1.0, 0, 0, M_inv))
            
            rot_clahe, M_inv_clahe = _vision2_rotate_image(clahe_gray, angle)
            variants.append((f"rotated_clahe_{angle}", rot_clahe, 1.0, 1.0, 0, 0, M_inv_clahe))
        except Exception:
            pass

    # 8. Upscaled 2x (for small/distant codes)
    try:
        scaled_2x = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        variants.append(("scaled_2x", scaled_2x, 2.0, 2.0, 0, 0, None))
    except Exception:
        pass

    # 9. Cropped top 88%
    try:
        crop_ratio = 0.88
        crop_h = max(1, int(height * crop_ratio))
        cropped_top = image_np[:crop_h, :]
        variants.append(("cropped_top", cropped_top, 1.0, 1.0, 0, 0, None))
        
        cropped_top_2x = cv2.resize(cropped_top, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        variants.append(("cropped_top_2x", cropped_top_2x, 2.0, 2.0, 0, 0, None))
    except Exception:
        pass

    # 10. Region of Interest (ROI) Candidate crops with 20% padding
    roi_candidates = _vision2_detect_candidate_regions(image_np)
    for idx, (kind, (bx1, by1, bx2, by2)) in enumerate(roi_candidates):
        try:
            cw = bx2 - bx1
            ch = by2 - by1
            
            # Add 20% padding
            pad_x = int(cw * 0.20)
            pad_y = int(ch * 0.20)
            
            px1 = max(0, bx1 - pad_x)
            py1 = max(0, by1 - pad_y)
            px2 = min(width, bx2 + pad_x)
            py2 = min(height, by2 + pad_y)
            
            cropped_roi = image_np[py1:py2, px1:px2]
            if cropped_roi.size == 0:
                continue
                
            roi_w = px2 - px1
            scale = 1.0
            
            # Upscale if the candidate region is small
            if roi_w < 400:
                scale = 2.0
                cropped_roi = cv2.resize(cropped_roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                
            # Yield ROI BGR
            variants.append((f"roi_{kind}_{idx}", cropped_roi, scale, scale, px1, py1, None))
            
            # Yield ROI Grayscale
            if len(cropped_roi.shape) == 3:
                roi_gray = cv2.cvtColor(cropped_roi, cv2.COLOR_BGR2GRAY)
            else:
                roi_gray = cropped_roi.copy()
            variants.append((f"roi_{kind}_{idx}_gray", roi_gray, scale, scale, px1, py1, None))
            
            # Yield ROI CLAHE
            if clahe is not None:
                try:
                    roi_clahe = clahe.apply(roi_gray)
                    variants.append((f"roi_{kind}_{idx}_clahe", roi_clahe, scale, scale, px1, py1, None))
                except Exception:
                    pass
        except Exception as e:
            print(f"[Vision2] Failed to process candidate ROI variant {idx}: {e}")

    return variants


def _vision2_detect_qrcode(image_np):
    records = []
    seen_codes = set()

    def _add_record(text, pts, decoder_name, scale_x=1.0, scale_y=1.0, offset_x=0, offset_y=0, M_inv=None):
        if not text:
            return
        code_text = str(text).strip()
        if not code_text or code_text in seen_codes:
            return
        bbox = None
        if pts is not None:
            try:
                pts_arr = np.array(pts, dtype=np.float32).reshape(-1, 2)
                
                # Transform rotated coordinates back first if applicable
                if M_inv is not None:
                    pts_reshaped = pts_arr.reshape(-1, 1, 2)
                    pts_orig = cv2.transform(pts_reshaped, M_inv)
                    pts_arr = pts_orig.reshape(-1, 2)
                    
                if M_inv is None:
                    x1 = int(np.min(pts_arr[:, 0]) / scale_x + offset_x)
                    y1 = int(np.min(pts_arr[:, 1]) / scale_y + offset_y)
                    x2 = int(np.max(pts_arr[:, 0]) / scale_x + offset_x)
                    y2 = int(np.max(pts_arr[:, 1]) / scale_y + offset_y)
                else:
                    x1 = int(np.min(pts_arr[:, 0]))
                    y1 = int(np.min(pts_arr[:, 1]))
                    x2 = int(np.max(pts_arr[:, 0]))
                    y2 = int(np.max(pts_arr[:, 1]))
                    
                bbox = [x1, y1, x2, y2]
            except Exception:
                pass
        seen_codes.add(code_text)
        records.append({
            "kind": "qrcode",
            "code": code_text,
            "bbox": bbox,
            "confidence": 1.0,
            "decoder": decoder_name
        })

    # Get preprocessed variants
    variants = _vision2_get_image_variants(image_np)

    # 1. Primary: Try PyZbar on all variants
    try:
        from pyzbar.pyzbar import decode as pyzbar_decode
        for var_name, variant, scale_x, scale_y, offset_x, offset_y, M_inv in variants:
            decoded_items = pyzbar_decode(variant)
            for item in decoded_items:
                item_type = str(item.type).upper()
                if item_type != "QRCODE":
                    continue
                code_text = item.data.decode("utf-8", errors="ignore").strip()
                if not code_text or code_text in seen_codes:
                    continue
                x, y, w, h = item.rect
                
                # Transform points back to original image coordinates
                pts = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                pts_arr = np.array(pts, dtype=np.float32).reshape(-1, 2)
                
                if M_inv is not None:
                    pts_reshaped = pts_arr.reshape(-1, 1, 2)
                    pts_orig = cv2.transform(pts_reshaped, M_inv)
                    pts_arr = pts_orig.reshape(-1, 2)
                    x1 = int(np.min(pts_arr[:, 0]))
                    y1 = int(np.min(pts_arr[:, 1]))
                    x2 = int(np.max(pts_arr[:, 0]))
                    y2 = int(np.max(pts_arr[:, 1]))
                else:
                    x1 = int(x / scale_x + offset_x)
                    y1 = int(y / scale_y + offset_y)
                    x2 = int((x + w) / scale_x + offset_x)
                    y2 = int((y + h) / scale_y + offset_y)
                    
                bbox = [x1, y1, x2, y2]
                seen_codes.add(code_text)
                records.append({
                    "kind": "qrcode",
                    "code": code_text,
                    "bbox": bbox,
                    "confidence": 1.0,
                    "decoder": f"pyzbar_{var_name}"
                })
    except Exception as e:
        print(f"[Vision2] pyzbar qrcode fallback unavailable or failed: {e}")

    if records:
        return records

    # 2. Secondary: Try OpenCV WeChat QR Code Detector
    try:
        wechat_detector = cv2.wechat_qrcode.WeChatQRCode()
        for var_name, variant, scale_x, scale_y, offset_x, offset_y, M_inv in variants:
            if len(variant.shape) == 2:
                variant_bgr = cv2.cvtColor(variant, cv2.COLOR_GRAY2BGR)
            else:
                variant_bgr = variant
            res, points = wechat_detector.detectAndDecode(variant_bgr)
            if res:
                for text, quad in zip(res, points):
                    _add_record(text, quad, f"wechat_{var_name}", scale_x, scale_y, offset_x, offset_y, M_inv)
            if records:
                return records
    except Exception as e:
        print(f"[Vision2] WeChat QR Code detector failed or unavailable: {e}")

    # 3. Fallback: Try standard OpenCV QRCodeDetector (Multi)
    try:
        opencv_detector = cv2.QRCodeDetector()
        for var_name, variant, scale_x, scale_y, offset_x, offset_y, M_inv in variants:
            retval, decoded_info, points, _ = opencv_detector.detectAndDecodeMulti(variant)
            if retval and points is not None and decoded_info is not None:
                for text, quad in zip(decoded_info, points):
                    _add_record(text, quad, f"opencv_multi_{var_name}", scale_x, scale_y, offset_x, offset_y, M_inv)
            if records:
                return records
    except Exception:
        pass

    # 4. Fallback: Try standard OpenCV QRCodeDetector (Single)
    try:
        opencv_detector = cv2.QRCodeDetector()
        for var_name, variant, scale_x, scale_y, offset_x, offset_y, M_inv in variants:
            text, pts, _ = opencv_detector.detectAndDecode(variant)
            _add_record(text, pts, f"opencv_single_{var_name}", scale_x, scale_y, offset_x, offset_y, M_inv)
            if records:
                return records
    except Exception:
        pass

    return records


def _vision2_detect_barcode_with_pyzbar(image_np):
    records = []
    seen_codes = set()

    def _add_record(text, pts, decoder_name, scale_x=1.0, scale_y=1.0, offset_x=0, offset_y=0, M_inv=None, code_type="barcode"):
        if not text:
            return
        code_text = str(text).strip()
        if not code_text or code_text in seen_codes:
            return
        bbox = None
        if pts is not None:
            try:
                pts_arr = np.array(pts, dtype=np.float32).reshape(-1, 2)
                if M_inv is not None:
                    pts_reshaped = pts_arr.reshape(-1, 1, 2)
                    pts_orig = cv2.transform(pts_reshaped, M_inv)
                    pts_arr = pts_orig.reshape(-1, 2)
                    x1 = int(np.min(pts_arr[:, 0]))
                    y1 = int(np.min(pts_arr[:, 1]))
                    x2 = int(np.max(pts_arr[:, 0]))
                    y2 = int(np.max(pts_arr[:, 1]))
                else:
                    x1 = int(np.min(pts_arr[:, 0]) / scale_x + offset_x)
                    y1 = int(np.min(pts_arr[:, 1]) / scale_y + offset_y)
                    x2 = int(np.max(pts_arr[:, 0]) / scale_x + offset_x)
                    y2 = int(np.max(pts_arr[:, 1]) / scale_y + offset_y)
                bbox = [x1, y1, x2, y2]
            except Exception:
                pass
        seen_codes.add(code_text)
        records.append({
            "kind": "barcode",
            "code": code_text,
            "code_type": code_type,
            "bbox": bbox,
            "confidence": 1.0,
            "decoder": decoder_name
        })

    # Get preprocessed variants
    variants = _vision2_get_image_variants(image_np)

    # 1. Primary: Try PyZbar on all variants
    try:
        from pyzbar.pyzbar import decode as pyzbar_decode
        for var_name, variant, scale_x, scale_y, offset_x, offset_y, M_inv in variants:
            decoded_items = pyzbar_decode(variant)
            for item in decoded_items:
                item_type = str(item.type).upper()
                if item_type == "QRCODE":
                    continue
                code_text = item.data.decode("utf-8", errors="ignore").strip()
                if not code_text or code_text in seen_codes:
                    continue
                x, y, w, h = item.rect
                
                # Transform points back to original image coordinates
                pts = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                pts_arr = np.array(pts, dtype=np.float32).reshape(-1, 2)
                
                if M_inv is not None:
                    pts_reshaped = pts_arr.reshape(-1, 1, 2)
                    pts_orig = cv2.transform(pts_reshaped, M_inv)
                    pts_arr = pts_orig.reshape(-1, 2)
                    x1 = int(np.min(pts_arr[:, 0]))
                    y1 = int(np.min(pts_arr[:, 1]))
                    x2 = int(np.max(pts_arr[:, 0]))
                    y2 = int(np.max(pts_arr[:, 1]))
                else:
                    x1 = int(x / scale_x + offset_x)
                    y1 = int(y / scale_y + offset_y)
                    x2 = int((x + w) / scale_x + offset_x)
                    y2 = int((y + h) / scale_y + offset_y)
                    
                bbox = [x1, y1, x2, y2]
                seen_codes.add(code_text)
                records.append({
                    "kind": "barcode",
                    "code": code_text,
                    "code_type": item_type,
                    "bbox": bbox,
                    "confidence": 1.0,
                    "decoder": f"pyzbar_{var_name}"
                })
    except Exception as e:
        print(f"[Vision2] pyzbar barcode detection failed or unavailable: {e}")

    if records:
        return records

    # 2. Secondary: Try OpenCV BarcodeDetector (Multi) as fallback
    try:
        barcode_detector = cv2.barcode.BarcodeDetector()
        for var_name, variant, scale_x, scale_y, offset_x, offset_y, M_inv in variants:
            retval, decoded_info, points, format_types = barcode_detector.detectAndDecodeMulti(variant)
            if retval and points is not None and decoded_info is not None:
                for text, quad, fmt in zip(decoded_info, points, format_types):
                    _add_record(text, quad, f"opencv_barcode_multi_{var_name}", scale_x, scale_y, offset_x, offset_y, M_inv, code_type=str(fmt))
            if records:
                return records
    except Exception as e:
        print(f"[Vision2] OpenCV BarcodeDetector Multi failed or unavailable: {e}")

    # 3. Fallback: Try OpenCV BarcodeDetector (Single) as fallback
    try:
        barcode_detector = cv2.barcode.BarcodeDetector()
        for var_name, variant, scale_x, scale_y, offset_x, offset_y, M_inv in variants:
            retval, points, format_type = barcode_detector.detect(variant)
            if retval and points is not None:
                text, points, format_type = barcode_detector.decode(variant, points)
                if text:
                    _add_record(text, points, f"opencv_barcode_single_{var_name}", scale_x, scale_y, offset_x, offset_y, M_inv, code_type=str(format_type))
            if records:
                return records
    except Exception as e:
        print(f"[Vision2] OpenCV BarcodeDetector Single failed or unavailable: {e}")

    return records


def _vision2_detect_codes(image_np, include_qrcode=True, include_barcode=True):
    t0_qr = time.time()
    qr_records = _vision2_detect_qrcode(image_np) if include_qrcode else []
    t1_qr = time.time()

    t0_bar = time.time()
    barcode_records = _vision2_detect_barcode_with_pyzbar(image_np) if include_barcode else []
    t1_bar = time.time()

    seen = set()
    merged = []
    for rec in qr_records + barcode_records:
        key = (rec.get("kind"), rec.get("code"))
        if key in seen:
            continue
        seen.add(key)
        merged.append(rec)

    return {
        "qrcode": qr_records,
        "barcode": barcode_records,
        "all_codes": merged,
        "counts": {
            "qrcode": len(qr_records),
            "barcode": len(barcode_records),
            "total": len(merged)
        },
        "timings_ms": {
            "qrcode_detection_ms": round((t1_qr - t0_qr) * 1000, 2),
            "barcode_detection_ms": round((t1_bar - t0_bar) * 1000, 2)
        }
    }


def _vision2_reference_detect(image_pil, threshold=0.75, target_label=None):
    image_rgb = image_pil.convert("RGB")
    image_cv = cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2BGR)

    active_labels = []
    if os.path.exists(REFVISION_REF_DIR):
        active_labels = [d for d in os.listdir(REFVISION_REF_DIR) if os.path.isdir(os.path.join(REFVISION_REF_DIR, d))]

    if not active_labels:
        return {
            "success": False,
            "error": "No reference labels available"
        }

    for label in active_labels:
        update_label_embeddings(REFVISION_DATA_DIR, label)

    candidate_boxes, debug_info = detect_candidate_crops(image_cv)
    raw_detections = match_candidates(
        image_rgb,
        candidate_boxes,
        REFVISION_DATA_DIR,
        active_labels,
        threshold=threshold,
        target_label=target_label
    )
    filtered_detections = filter_duplicate_detections(raw_detections, iou_threshold=0.45)

    counts = {}
    labels = []
    for det in filtered_detections:
        lbl = det.get("label", "unknown")
        labels.append(lbl)
        if lbl != "unknown":
            counts[lbl] = counts.get(lbl, 0) + 1

    return {
        "success": True,
        "active_labels": active_labels,
        "counts": counts,
        "detections": filtered_detections,
        "labels": labels,
        "debug": {
            "proposals_count": debug_info.get("raw_contour_proposals", 0),
            "merged_contour_count": debug_info.get("merged_contour_proposals", 0),
            "using_fallback_grid": debug_info.get("using_fallback_grid", False),
            "final_analyzed_candidates": len(candidate_boxes)
        }
    }


def _vision2_attach_db_metadata(codes_payload):
    barcode_db = _vision2_load_db(VISION2_BARCODE_DB_PATH)
    qrcode_db = _vision2_load_db(VISION2_QRCODE_DB_PATH)

    barcode_map = {str(r.get("code", "")).strip(): r for r in barcode_db.get("records", [])}
    qrcode_map = {str(r.get("code", "")).strip(): r for r in qrcode_db.get("records", [])}

    all_codes = []
    for rec in codes_payload.get("all_codes", []):
        code = str(rec.get("code", "")).strip()
        db_hit = None
        if rec.get("kind") == "barcode":
            db_hit = barcode_map.get(code)
        elif rec.get("kind") == "qrcode":
            db_hit = qrcode_map.get(code)

        all_codes.append({
            **rec,
            "db_record": db_hit
        })

    codes_payload["all_codes"] = all_codes
    codes_payload["matched_in_db"] = sum(1 for r in all_codes if r.get("db_record") is not None)
    return codes_payload


def _vision2_parse_local_decoded(source):
    """Normalize local barcode/QR decode payload from request form/json."""
    if source is None:
        return None

    text = str(source.get("local_decoded_text", "")).strip()
    if not text:
        return None

    raw_kind = str(source.get("local_decoded_kind", "")).strip().lower()
    if raw_kind in ("qr", "qr_code"):
        kind = "qrcode"
    elif raw_kind in ("barcode", "qrcode"):
        kind = raw_kind
    else:
        kind = "unknown"

    return {
        "code": text,
        "kind": kind,
        "code_type": str(source.get("local_decoded_format", "") or "unknown").strip(),
        "decoder": str(source.get("local_decoded_engine", "") or "local").strip(),
        "polygon": [],
        "center": None,
        "source": "local"
    }


def _vision2_merge_local_code_result(codes_payload, local_rec, include_qrcode=True, include_barcode=True):
    """Merge local-decoded code into server detections (dedupe by kind+code)."""
    if not local_rec:
        return codes_payload

    kind = local_rec.get("kind")
    if kind == "qrcode" and not include_qrcode:
        return codes_payload
    if kind == "barcode" and not include_barcode:
        return codes_payload

    all_codes = list(codes_payload.get("all_codes", []))
    code_text = str(local_rec.get("code", "")).strip()
    if not code_text:
        return codes_payload

    existed = any(
        str(rec.get("code", "")).strip() == code_text and
        str(rec.get("kind", "")).strip().lower() == str(kind).strip().lower()
        for rec in all_codes
    )

    if existed:
        return codes_payload

    all_codes.append(local_rec)
    codes_payload["all_codes"] = all_codes

    counts = dict(codes_payload.get("counts", {}))
    counts.setdefault("qrcode", 0)
    counts.setdefault("barcode", 0)
    if kind == "qrcode":
        counts["qrcode"] = int(counts.get("qrcode", 0)) + 1
    elif kind == "barcode":
        counts["barcode"] = int(counts.get("barcode", 0)) + 1
    counts["total"] = int(counts.get("qrcode", 0)) + int(counts.get("barcode", 0))
    codes_payload["counts"] = counts
    return codes_payload


def _vision2_extract_code_from_upload(image_file, expected_kind=None):
    if image_file is None:
        return None, [], []

    try:
        image_file.stream.seek(0)
    except Exception:
        pass
    image_pil = Image.open(image_file.stream).convert("RGB")

    # Server-side resize check: limit long edge to 1000px
    try:
        w, h = image_pil.size
        long_edge = max(w, h)
        if long_edge > 1000:
            scale = 1000.0 / long_edge
            try:
                resample_method = Image.Resampling.LANCZOS
            except AttributeError:
                try:
                    resample_method = Image.LANCZOS
                except AttributeError:
                    resample_method = Image.ANTIALIAS
            image_pil = image_pil.resize((int(w * scale), int(h * scale)), resample_method)
    except Exception as e:
        print(f"[Vision2] Server-side resize in extract upload failed: {e}")

    image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    detected = _vision2_detect_codes(image_np)
    all_codes = detected.get("all_codes", [])

    if expected_kind:
        filtered = [item for item in all_codes if item.get("kind") == expected_kind]
    else:
        filtered = all_codes

    code_value = filtered[0].get("code") if filtered else None
    return code_value, filtered, all_codes


def _vision2_model_understanding(image_pil, keyword_tags, enable_model_keyword_check=True, enable_ocr_check=True, short_product_name=False):
    """
    Use loaded Florence-2 model to build semantic context (caption + optional OCR)
    and perform keyword matching against model-generated content.
    """
    if not enable_model_keyword_check:
        return {
            "enabled": False,
            "available": True,
            "reason": "model_keyword_check_disabled",
            "caption": None,
            "ocr": None,
            "matched_keywords": [],
            "keyword_details": []
        }

    if skip_model_load or processor is None or model is None:
        return {
            "enabled": True,
            "available": False,
            "reason": "model_not_loaded",
            "caption": None,
            "ocr": None,
            "matched_keywords": [],
            "keyword_details": []
        }

    caption_text = None
    ocr_text = None
    keyword_details = []

    try:
        # Generate caption (or short product name prompt) using the Florence model
        if short_product_name:
            prompt = "<MORE_DETAILED_CAPTION>"
            caption_text = run_model(image_pil.copy(), prompt, max_tokens=28)
        else:
            prompt = "<MORE_DETAILED_CAPTION>"
            caption_text = run_model(image_pil.copy(), prompt, max_tokens=180)

        # Protect against prompt-echo from model; treat as no caption
        print(f"[Vision2] Model caption output: {caption_text}")
        if isinstance(caption_text, str):
            ct_str = caption_text.strip()
            if not ct_str or ct_str == prompt or ct_str.startswith(prompt):
                caption_text = None

        product_name = None
        # Only call Ollama when short product name was requested and we have some caption text
        if short_product_name and caption_text:
            try:
                payload = {
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {"role": "user", "content": f"Extract a concise product/item name from the following description. Return only the name (one short phrase): {caption_text}"}
                    ],
                    "temperature": 0
                }
                print(f"[Vision2] Calling Ollama: {OLLAMA_URL} model={OLLAMA_MODEL}")
                # Use streaming request to handle Ollama's line-delimited or chunked JSON
                resp = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=8)
                name = None
                if resp.ok:
                    # First try: parse as a single JSON blob
                    try:
                        j = resp.json()
                        parsed = True
                    except ValueError:
                        parsed = False

                    content_acc = ""
                    # If single JSON parsed, try extracting usual fields
                    if parsed and isinstance(j, dict):
                        choices = j.get('choices')
                        if choices and isinstance(choices, list):
                            first = choices[0]
                            msg = first.get('message') or first.get('content') or first.get('text')
                            if isinstance(msg, dict):
                                content = msg.get('content') or msg.get('text')
                                if isinstance(content, dict):
                                    name = content.get('text') or content.get('output_text')
                                elif isinstance(content, str):
                                    name = content
                            elif isinstance(msg, str):
                                name = msg
                        if not name:
                            name = j.get('text') or j.get('output') or j.get('result')

                    # If we couldn't parse single JSON, or want to support streaming, iterate lines
                    if not name:
                        try:
                            for line in resp.iter_lines(decode_unicode=True):
                                if not line:
                                    continue
                                line = line.strip()
                                # Try to parse this line as JSON
                                try:
                                    chunk = json.loads(line)
                                except Exception:
                                    # treat as raw text
                                    try:
                                        content_acc += line
                                    except Exception:
                                        pass
                                    continue

                                # Extract message content from common chunk shapes
                                piece = ""
                                if isinstance(chunk, dict):
                                    # Ollama streaming may include {'message': {'content': '...'}}
                                    piece = chunk.get('message', {}).get('content') or ""
                                    if not piece:
                                        choices = chunk.get('choices')
                                        if choices and isinstance(choices, list):
                                            first = choices[0]
                                            msg = first.get('message') or first.get('content') or first.get('text') or ""
                                            if isinstance(msg, dict):
                                                piece = msg.get('content') or msg.get('text') or ""
                                            elif isinstance(msg, str):
                                                piece = msg
                                if piece:
                                    content_acc += piece

                            if content_acc:
                                name = content_acc
                        except Exception as e:
                            print(f"[Vision2] Ollama streaming parse failed: {e}")

                    if name:
                        name = str(name).strip().strip(' "\'\n')
                        if name and name != prompt and not name.startswith(prompt):
                            product_name = name
                            caption_text = product_name
                else:
                    print(f"[Vision2] Ollama HTTP error: {resp.status_code} {resp.text}")
            except Exception as ex:
                print(f"[Vision2] Ollama call failed: {ex}")
    except Exception as e:
        caption_text = f"[caption_error] {e}"

    if enable_ocr_check:
        try:
            ocr_prompt = "<OCR_WITH_REGION>"
            ocr_text = run_model(image_pil.copy(), ocr_prompt, max_tokens=260)
            if isinstance(ocr_text, str):
                ot = ocr_text.strip()
                if not ot or ot == ocr_prompt or ot.startswith(ocr_prompt):
                    ocr_text = None
        except Exception as e:
            ocr_text = f"[ocr_error] {e}"

    corpus_parts = [caption_text or ""]
    if enable_ocr_check:
        corpus_parts.append(ocr_text or "")
    corpus = "\n".join(corpus_parts).lower()

    corpus_tokens = re.findall(r"[a-z0-9_\-]+", corpus)
    matched_keywords = []

    for kw in keyword_tags:
        kw_l = str(kw).strip().lower()
        if not kw_l:
            continue

        direct_match = kw_l in corpus
        token_match = all(token in corpus for token in kw_l.split()) if kw_l.split() else False
        best_fuzzy = 0.0
        best_token = None

        for token in corpus_tokens:
            score = SequenceMatcher(None, kw_l, token).ratio()
            if score > best_fuzzy:
                best_fuzzy = score
                best_token = token

        fuzzy_match = best_fuzzy >= 0.82
        is_match = direct_match or token_match or fuzzy_match
        if is_match:
            matched_keywords.append(kw_l)

        keyword_details.append({
            "keyword": kw_l,
            "matched": is_match,
            "match_type": (
                "direct" if direct_match else
                "token" if token_match else
                "fuzzy" if fuzzy_match else
                "none"
            ),
            "best_fuzzy_score": round(best_fuzzy, 4),
            "best_fuzzy_token": best_token
        })

    return {
        "enabled": True,
        "available": True,
        "reason": None,
        "caption": caption_text,
        "product_name": product_name,
        "ocr": ocr_text,
        "matched_keywords": sorted(list(set(matched_keywords))),
        "keyword_details": keyword_details,
        "ocr_check_enabled": bool(enable_ocr_check)
    }

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
        "model": "Florence-2-base-PromptGen"
    })

@app.route("/analyze", methods=["POST"])
def api_analyze():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image field"}), 400

        imgfile = request.files["image"]
        img = Image.open(imgfile.stream).convert("RGB")

        task = request.form.get("task", "caption")
        engine = request.form.get("engine", "florence")  # florence / gemma

        t0 = time.time()

        if engine == "gemma":
            if task == "item_name":
                prompt = "Identify the main item in the image. Return only the item name. No explanation."
            elif task == "caption":
                prompt = "Describe this image briefly and clearly."
            elif task == "json_gauges":
                prompt = (
                    "Analyze this industrial instrument image. "
                    "Extract visible gauge names, values, and units. "
                    "Return JSON array only. Example: "
                    "[{\"type\":\"pressure\",\"value\":\"12\",\"unit\":\"bar\"}]"
                )
            else:
                prompt = task

            result = run_ollama_vision(img, prompt, model=OLLAMA_MODEL, timeout=45)
        else:
            florence_task = "MORE_DETAILED_CAPTION" if task == "caption" else task
            result = runmodel(img, florence_task, maxtokens=150)

        return jsonify({
            "success": True,
            "engine": engine,
            "task": task,
            "result": result,
            "elapsed": time.time() - t0
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
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


@app.route("/readorder", methods=["POST"])
def api_readorder():
    """Proxy endpoint for front-end to call Ollama (or configured OLLAMA_URL).
    Expects the same JSON body the front-end currently sends to the Ollama API
    (eg. { model, messages, images: [base64], stream:false, options:{...} }).
    """
    try:
        data = None
        # Accept JSON body
        if request.is_json:
            data = request.get_json()
        else:
            # Try to parse raw body as json
            try:
                data = json.loads(request.get_data(as_text=True) or '{}')
            except Exception:
                data = None

        if not data:
            return jsonify({"error": "No JSON payload provided"}), 400

        # Forward to configured Ollama URL
        try:
            resp = requests.post(OLLAMA_URL, json=data, timeout=60)
            # Try to return Ollama's JSON response directly
            try:
                return jsonify(resp.json()), resp.status_code
            except Exception:
                return (resp.text, resp.status_code, {"Content-Type": "text/plain"})
        except requests.RequestException as ex:
            return jsonify({"error": "Failed to contact OLLAMA endpoint", "detail": str(ex)}), 502

    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
    if image_np is not None:
        try:
            h, w = image_np.shape[:2]
            long_edge = max(h, w)
            max_dim = 480
            if long_edge > max_dim:
                scale = max_dim / float(long_edge)
                new_w = int(w * scale)
                new_h = int(h * scale)
                image_np = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(f"[Vision2] Resizing annotated image failed: {e}")
            
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

# Route: Serve RefVision Homepage
@app.route("/localscanner")
def localscanner_home():
    return render_template("local_scanner.html")



@app.route("/vision2")
def vision2_home():
    return render_template("vision2.html")

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


@app.route("/vision2/api/analyze", methods=["POST"])
def vision2_api_analyze():
    """
    One-shot camera analysis for Vision2 user flow:
    1) Reference-based object detection
    2) Hand pose + grasp detection
    3) Barcode / QR-code detection
    """
    try:
        if "image" not in request.files:
            return jsonify({"success": False, "error": "No image field provided"}), 400

        image_file = request.files["image"]
        image_pil = Image.open(image_file.stream).convert("RGB")

        # Server-side resize check: limit long edge to 1000px
        try:
            w, h = image_pil.size
            long_edge = max(w, h)
            if long_edge > 1000:
                scale = 1000.0 / long_edge
                try:
                    resample_method = Image.Resampling.LANCZOS
                except AttributeError:
                    try:
                        resample_method = Image.LANCZOS
                    except AttributeError:
                        resample_method = Image.ANTIALIAS
                image_pil = image_pil.resize((int(w * scale), int(h * scale)), resample_method)
        except Exception as e:
            print(f"[Vision2] Server-side resize in analyze API failed: {e}")

        image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        try:
            threshold = float(request.form.get("threshold", 0.75))
        except Exception:
            threshold = 0.75

        target_label = str(request.form.get("target_label", "")).strip()
        if target_label in ("", "all"):
            target_label = None

        raw_keywords = str(request.form.get("keywords", "")).strip()
        keyword_tags = [k.strip().lower() for k in raw_keywords.split(",") if k.strip()]

        model_keyword_check = str(request.form.get("model_keyword_check", "1")).strip().lower() not in ("0", "false", "no")
        ocr_check = str(request.form.get("ocr_check", "1")).strip().lower() not in ("0", "false", "no")
        ref_object_check = str(request.form.get("ref_object_check", "1")).strip().lower() not in ("0", "false", "no")
        hand_check = str(request.form.get("hand_check", "1")).strip().lower() not in ("0", "false", "no")
        code_check = str(request.form.get("code_check", "1")).strip().lower() not in ("0", "false", "no")
        raw_qrcode_check = request.form.get("qrcode_check")
        raw_barcode_check = request.form.get("barcode_check")
        qrcode_check = (
            (str(raw_qrcode_check).strip().lower() not in ("0", "false", "no"))
            if raw_qrcode_check is not None else code_check
        )
        barcode_check = (
            (str(raw_barcode_check).strip().lower() not in ("0", "false", "no"))
            if raw_barcode_check is not None else code_check
        )
        code_check = qrcode_check or barcode_check
        local_decoded = _vision2_parse_local_decoded(request.form)

        t0 = time.time()
        performance = {}

        if ref_object_check:
            t_ref0 = time.time()
            reference_result = _vision2_reference_detect(image_pil, threshold=threshold, target_label=target_label)
            performance["reference_detection_ms"] = round((time.time() - t_ref0) * 1000, 2)
        else:
            reference_result = {"success": False, "reason": "disabled", "counts": {}, "detections": [], "labels": []}
            performance["reference_detection_ms"] = 0.0

        if hand_check:
            t_hand0 = time.time()
            hand_result = detect_hand_pose(image_pil)
            performance["hand_tracking_ms"] = round((time.time() - t_hand0) * 1000, 2)
        else:
            hand_result = {"success": False, "reason": "disabled", "hands_detected": 0, "objects_detected": [], "grasped_objects": []}
            performance["hand_tracking_ms"] = 0.0

        if code_check:
            t_code0 = time.time()
            code_result = _vision2_detect_codes(image_np, include_qrcode=qrcode_check, include_barcode=barcode_check)
            code_result = _vision2_merge_local_code_result(
                code_result,
                local_decoded,
                include_qrcode=qrcode_check,
                include_barcode=barcode_check
            )
            code_result = _vision2_attach_db_metadata(code_result)
            performance["code_detection_ms"] = round((time.time() - t_code0) * 1000, 2)
            timings = code_result.get("timings_ms", {})
            performance["qrcode_detection_ms"] = timings.get("qrcode_detection_ms", 0.0)
            performance["barcode_detection_ms"] = timings.get("barcode_detection_ms", 0.0)
        else:
            code_result = {
                "success": False,
                "reason": "disabled",
                "all_codes": [],
                "counts": {"qrcode": 0, "barcode": 0, "total": 0},
                "timings_ms": {"qrcode_detection_ms": 0.0, "barcode_detection_ms": 0.0}
            }
            performance["code_detection_ms"] = 0.0
            performance["qrcode_detection_ms"] = 0.0
            performance["barcode_detection_ms"] = 0.0

        t_model0 = time.time()
        short_name_only = str(request.form.get('short_name_only', '0')).strip().lower() in ('1', 'true', 'yes')
        model_understanding = _vision2_model_understanding(
            image_pil,
            keyword_tags,
            enable_model_keyword_check=model_keyword_check,
            enable_ocr_check=ocr_check,
            short_product_name=short_name_only
        )
        performance["model_understanding_ms"] = round((time.time() - t_model0) * 1000, 2)

        detected_terms = []
        if reference_result.get("success"):
            detected_terms.extend([str(lbl).lower() for lbl in reference_result.get("labels", [])])
        detected_terms.extend([str(c.get("code", "")).lower() for c in code_result.get("all_codes", [])])

        if hand_result.get("success"):
            for obj in hand_result.get("objects_detected", []):
                detected_terms.append(str(obj.get("category", "")).lower())

        matched_keywords = [kw for kw in keyword_tags if any(kw in term for term in detected_terms)]
        keyword_counter = Counter([kw for kw in keyword_tags if kw in matched_keywords])

        model_matched = model_understanding.get("matched_keywords", []) if model_understanding else []
        combined_keyword_matches = sorted(list(set(matched_keywords + model_matched)))

        return jsonify({
            "success": True,
            "elapsed": time.time() - t0,
            "performance": {
                **performance,
                "total_elapsed_ms": round((time.time() - t0) * 1000, 2)
            },
            "input": {
                "threshold": threshold,
                "target_label": target_label,
                "keyword_tags": keyword_tags,
                "model_keyword_check": model_keyword_check,
                "ocr_check": ocr_check,
                "ref_object_check": ref_object_check,
                "hand_check": hand_check,
                "code_check": code_check,
                "qrcode_check": qrcode_check,
                "barcode_check": barcode_check,
                "local_decoded": local_decoded
            },
            "user_view": {
                "reference_detection": reference_result,
                "hand_tracking": hand_result,
                "code_detection": code_result,
                "keyword_tags": {
                    "requested": keyword_tags,
                    "matched": combined_keyword_matches,
                    "matched_count": len(combined_keyword_matches),
                    "detector_only_matched": matched_keywords,
                    "model_matched": model_matched,
                    "detector_only_matched_count": sum(keyword_counter.values())
                },
                "model_understanding": model_understanding,
                "ocr_check": {
                    "enabled": ocr_check,
                    "text": model_understanding.get("ocr") if model_understanding else None
                }
            },
            "editor_summary": {
                "barcode_db_records": len(_vision2_load_db(VISION2_BARCODE_DB_PATH).get("records", [])),
                "qrcode_db_records": len(_vision2_load_db(VISION2_QRCODE_DB_PATH).get("records", []))
            }
        }), 200

    except Exception as e:
        import traceback
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

def run_ollama_vision(img, prompt, model=None, timeout=30):
    if model is None:
        model = OLLAMA_MODEL

    img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64]
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0
        }
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    if "message" in data and isinstance(data["message"], dict):
        return (data["message"].get("content") or "").strip()

    return (data.get("response") or "").strip()

@app.route("/vision2/api/db/<kind>", methods=["GET", "POST", "DELETE"])
def vision2_api_db(kind):
    """CRUD for barcode/qrcode JSON databases."""
    try:
        kind_norm, db_path = _vision2_db_by_kind(kind)
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    if request.method == "GET":
        db = _vision2_load_db(db_path)
        return jsonify({"success": True, "kind": kind_norm, "db": db}), 200

    if request.method == "POST":
        if request.files or request.form:
            payload = {
                "code": str(request.form.get("code", "")).strip(),
                "name": str(request.form.get("name", "")).strip(),
                "code_number": str(request.form.get("code_number", "")).strip(),
                "simple_description": str(request.form.get("simple_description", "")).strip(),
                "description": str(request.form.get("description", "")).strip()
            }
            local_decoded = _vision2_parse_local_decoded(request.form)
            if not payload["code"] and local_decoded:
                payload["code"] = str(local_decoded.get("code", "")).strip()

            if local_decoded and local_decoded.get("kind") in ("barcode", "qrcode"):
                local_kind = local_decoded.get("kind")
                if local_kind != kind_norm:
                    return jsonify({
                        "success": False,
                        "error": f"Detected {local_kind} from local decode, but current type is {kind_norm}",
                        "detected_items": [local_decoded],
                        "suggested_kind": local_kind,
                        "detected_code": local_decoded.get("code")
                    }), 400

            image_file = request.files.get("image")
            detected_items = []
            if image_file and not payload["code"]:
                detected_code, detected_items, all_detected_items = _vision2_extract_code_from_upload(image_file, expected_kind=kind_norm)
                if detected_code:
                    payload["code"] = detected_code
                elif all_detected_items:
                    first_detected = all_detected_items[0]
                    detected_kind = first_detected.get("kind")
                    detected_code_value = first_detected.get("code")
                    return jsonify({
                        "success": False,
                        "error": f"Detected {detected_kind} in uploaded image, but current type is {kind_norm}",
                        "detected_items": all_detected_items,
                        "suggested_kind": detected_kind,
                        "detected_code": detected_code_value
                    }), 400
            if image_file and not payload["code"]:
                return jsonify({
                    "success": False,
                    "error": f"No {kind_norm} detected in uploaded image",
                    "detected_items": detected_items,
                    "suggested_kind": None,
                    "detected_code": None
                }), 400
        else:
            payload = request.get_json(silent=True) or {}
            detected_items = []
        try:
            saved = _vision2_upsert_record(db_path, kind_norm, payload)
            return jsonify({
                "success": True,
                "kind": kind_norm,
                "record": saved,
                "detected_items": detected_items
            }), 200
        except ValueError as e:
            return jsonify({"success": False, "error": str(e)}), 400

    code = request.args.get("code", "")
    if not code:
        body = request.get_json(silent=True) or {}
        code = str(body.get("code", "")).strip()

    if not code:
        return jsonify({"success": False, "error": "'code' is required for delete"}), 400

    removed = _vision2_delete_record(db_path, code)
    return jsonify({"success": True, "kind": kind_norm, "removed": removed, "code": code}), 200

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
