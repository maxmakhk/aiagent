"""
Vision AI Agent - Flask Server
Third-Party Vision Integration via Local Model Server
Using Florence-2 Vision Model
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import io
import base64
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size
CORS(app)

# Global variables for model and processor
model = None
processor = None
device = None

def initialize_model():
    """Initialize Florence-2 model"""
    global model, processor, device
    
    print("Initializing Florence-2 model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model_name = "microsoft/Florence-2-base"
    
    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True
        ).to(device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Model will be downloaded on first use.")

def analyze_image(image, task_type="<MORE_DETAILED_CAPTION>"):
    """
    Analyze image using Florence-2 model
    
    Args:
        image: PIL Image object
        task_type: Task type for Florence-2 model
        
    Returns:
        Analysis result as string
    """
    global model, processor, device
    
    if model is None or processor is None:
        initialize_model()
    
    # Prepare inputs
    inputs = processor(text=task_type, images=image, return_tensors="pt").to(device)
    
    # Generate analysis
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    
    # Decode results
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_type, 
        image_size=(image.width, image.height)
    )
    
    return parsed_answer

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model": "Florence-2",
        "device": device if device else "not initialized"
    })

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Analyze image endpoint
    Expects JSON with base64 encoded image and task_type
    """
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({"error": "No image provided"}), 400
        
        # Get task type (default to detailed caption)
        task_type = data.get('task_type', '<MORE_DETAILED_CAPTION>')
        
        # Decode base64 image
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Analyze image
        result = analyze_image(image, task_type)
        
        return jsonify({
            "success": True,
            "result": result,
            "task_type": task_type
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/task-types', methods=['GET'])
def get_task_types():
    """Get available task types"""
    task_types = {
        "caption": {
            "id": "<CAPTION>",
            "name": "Caption",
            "description": "Generate a brief caption"
        },
        "detailed_caption": {
            "id": "<DETAILED_CAPTION>",
            "name": "Detailed Caption",
            "description": "Generate a detailed caption"
        },
        "more_detailed_caption": {
            "id": "<MORE_DETAILED_CAPTION>",
            "name": "More Detailed Caption",
            "description": "Generate a very detailed caption"
        },
        "object_detection": {
            "id": "<OD>",
            "name": "Object Detection",
            "description": "Detect objects in the image"
        },
        "dense_region_caption": {
            "id": "<DENSE_REGION_CAPTION>",
            "name": "Dense Region Caption",
            "description": "Generate captions for different regions"
        },
        "region_proposal": {
            "id": "<REGION_PROPOSAL>",
            "name": "Region Proposal",
            "description": "Propose regions of interest"
        }
    }
    return jsonify(task_types)

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    
    # Initialize model on startup
    initialize_model()
    
    # Check if debug mode should be enabled (via environment variable)
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Run server
    print("\n" + "="*60)
    print("🚀 Vision AI Agent Server Starting")
    print("="*60)
    print("📸 Vision Model: Florence-2")
    print("🔒 Local Secure Server")
    print("🌐 Access at: http://localhost:5000")
    if debug_mode:
        print("⚠️  Debug mode: ENABLED (for development only)")
    print("="*60 + "\n")
    
    # Debug mode disabled by default for security
    # Set environment variable FLASK_DEBUG=true to enable in development
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)
