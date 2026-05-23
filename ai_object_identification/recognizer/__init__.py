# Recognizer Package
from recognizer.embeddings import (
    load_clip_model,
    compute_embedding,
    update_label_embeddings,
    load_embeddings_cache,
    save_embeddings_cache
)
from recognizer.detect_candidates import detect_candidate_crops
from recognizer.match import match_candidates, filter_duplicate_detections
from recognizer.utils import sanitize_label, draw_annotations
