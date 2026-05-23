import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Global model container for lazy loading
_clip_model = None
_clip_processor = None

def get_device():
    """Returns the best available device (cuda or cpu)."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_clip_model(model_name="openai/clip-vit-base-patch32"):
    """Loads and caches the CLIP model and processor."""
    global _clip_model, _clip_processor
    if _clip_model is None or _clip_processor is None:
        device = get_device()
        print(f"[*] Loading CLIP model '{model_name}' on device: {device}...")
        _clip_processor = CLIPProcessor.from_pretrained(model_name)
        _clip_model = CLIPModel.from_pretrained(model_name).to(device)
        _clip_model.eval()
        print("[*] CLIP model loaded successfully.")
    return _clip_model, _clip_processor

def compute_embedding(pil_image):
    """Computes a normalized embedding vector (512-dim) for a single PIL image."""
    model, processor = load_clip_model()
    device = get_device()
    
    # Ensure image is in RGB format
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
        
    with torch.no_grad():
        inputs = processor(images=pil_image, return_tensors="pt").to(device)
        image_features = model.get_image_features(**inputs)
        # L2 Normalize the embedding vector
        normalized_features = F.normalize(image_features, p=2, dim=-1)
        embedding = normalized_features[0].cpu().numpy().tolist()
        
    return embedding

def get_label_cache_path(data_dir, label):
    """Returns the path to the embeddings cache file for a given label."""
    return os.path.join(data_dir, "references", label, "embeddings.json")

def load_embeddings_cache(data_dir, label):
    """Loads the cached embeddings for a specific label."""
    cache_path = get_label_cache_path(data_dir, label)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[!] Error loading cache for {label}: {e}")
    return {}

def save_embeddings_cache(data_dir, label, cache_data):
    """Saves the embeddings cache for a specific label."""
    cache_path = get_label_cache_path(data_dir, label)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    try:
        with open(cache_path, "w") as f:
            json.dump(cache_data, f, indent=2)
    except Exception as e:
        print(f"[!] Error saving cache for {label}: {e}")

def update_label_embeddings(data_dir, label, force_rebuild=False):
    """
    Computes embeddings for any new reference photos for a given label
    and updates the cache. If force_rebuild is True, all are recomputed.
    """
    ref_dir = os.path.join(data_dir, "references", label)
    if not os.path.exists(ref_dir):
        return {}
        
    cache = {} if force_rebuild else load_embeddings_cache(data_dir, label)
    updated = False
    
    # Find all image files
    valid_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    for filename in os.listdir(ref_dir):
        if filename.lower().endswith(valid_exts) and filename != "embeddings.json":
            if filename not in cache:
                img_path = os.path.join(ref_dir, filename)
                try:
                    print(f"[*] Computing embedding for reference: {label}/{filename}")
                    with Image.open(img_path) as img:
                        emb = compute_embedding(img)
                        cache[filename] = emb
                        updated = True
                except Exception as e:
                    print(f"[!] Failed to compute embedding for {filename}: {e}")
                    
    # Clean cache of deleted files
    existing_files = set(os.listdir(ref_dir))
    to_delete = [f for f in cache if f not in existing_files]
    if to_delete:
        for f in to_delete:
            del cache[f]
        updated = True
        
    if updated:
        save_embeddings_cache(data_dir, label, cache)
        
    return cache
