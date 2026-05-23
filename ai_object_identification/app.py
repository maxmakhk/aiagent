import os
import cv2
import uuid
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image

# Import recognizer package
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

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app) # Enable CORS for development flexibility

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
REF_DIR = os.path.join(DATA_DIR, "references")
SCENES_DIR = os.path.join(DATA_DIR, "scenes")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

# Ensure required directories exist
for path in [DATA_DIR, REF_DIR, SCENES_DIR, RESULTS_DIR]:
    os.makedirs(path, exist_ok=True)

# Allowed image formats
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Route: Serve Homepage
@app.route("/")
def home():
    return render_template("index.html")

# Route: Serving Static Files from the Data Folder
@app.route("/data/<path:filename>")
def serve_data(filename):
    return send_from_directory(DATA_DIR, filename)

# Route: Backend Health & Status
@app.route("/api/health", methods=["GET"])
def health():
    try:
        # Check if CLIP is pre-loaded or load it lazily
        model, _ = load_clip_model()
        device = str(model.device)
        
        # Get active labels in references directory
        active_labels = []
        if os.path.exists(REF_DIR):
            active_labels = [d for d in os.listdir(REF_DIR) if os.path.isdir(os.path.join(REF_DIR, d))]
            
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
    label_dir = os.path.join(REF_DIR, label)
    
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
    if not os.path.exists(REF_DIR):
        return jsonify([])
        
    labels = []
    for label_name in os.listdir(REF_DIR):
        label_path = os.path.join(REF_DIR, label_name)
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
    label_path = os.path.join(REF_DIR, label)
    
    if not os.path.exists(label_path) or not os.path.isdir(label_path):
        return jsonify({"error": f"Label '{label}' does not exist"}), 404
        
    files = [f for f in os.listdir(label_path) if allowed_file(f)]
    photo_urls = [f"/data/references/{label}/{f}" for f in files]
    
    # Check cache status
    cache = load_embeddings_cache(DATA_DIR, label)
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
    label_path = os.path.join(REF_DIR, label)
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
        update_label_embeddings(DATA_DIR, label)
        
    return jsonify({
        "message": f"Successfully uploaded {len(saved_files)} reference files for label '{label}'",
        "label": label,
        "files": saved_files
    }), 200

# Route: Rebuild embeddings for all active labels (Nice-to-have)
@app.route("/api/reference/rebuild", methods=["POST"])
def rebuild_embeddings():
    if not os.path.exists(REF_DIR):
        return jsonify({"error": "No labels found to rebuild"}), 400
        
    rebuilt_labels = {}
    for label_name in os.listdir(REF_DIR):
        label_path = os.path.join(REF_DIR, label_name)
        if os.path.isdir(label_path):
            cache = update_label_embeddings(DATA_DIR, label_name, force_rebuild=True)
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
    scene_path = os.path.join(SCENES_DIR, unique_name)
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
    if os.path.exists(REF_DIR):
        active_labels = [d for d in os.listdir(REF_DIR) if os.path.isdir(os.path.join(REF_DIR, d))]
        
    if not active_labels:
        return jsonify({
            "error": "No reference object labels exist. Please teach the system an object first!"
        }), 400
        
    # Verify embeddings are updated
    for label in active_labels:
        update_label_embeddings(DATA_DIR, label)
        
    # 2. OpenCV candidate proposals
    candidate_boxes, debug_info = detect_candidate_crops(image_cv)
    
    # 3. Match candidate embeddings against reference embeddings
    raw_detections = match_candidates(
        scene_pil, 
        candidate_boxes, 
        DATA_DIR, 
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
    result_path = os.path.join(RESULTS_DIR, result_name)
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
    # Pre-warm model on startup so the first request is ultra-fast
    print("[*] Pre-warming CLIP model...")
    try:
        load_clip_model()
    except Exception as e:
        print(f"[!] Warning: CLIP model failed to load during pre-warm: {e}")
        
    print("[*] Running Flask server on http://127.0.0.1:5000...")
    app.run(host="0.0.0.0", port=5000, debug=True)
