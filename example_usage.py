"""
Example: Using the Vision AI Agent API
This script demonstrates how to use the API programmatically
"""

import requests
import base64
import json

# Configuration
API_BASE_URL = "http://localhost:5000/api"

def encode_image(image_path):
    """Encode image file to base64"""
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded}"

def check_health():
    """Check if the server is healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("✓ Server is healthy")
            print(f"  Model: {data['model']}")
            print(f"  Device: {data['device']}")
            return True
        else:
            print("✗ Server returned error:", response.status_code)
            return False
    except Exception as e:
        print(f"✗ Error connecting to server: {e}")
        return False

def analyze_image(image_path, task_type="<MORE_DETAILED_CAPTION>"):
    """
    Analyze an image using the Vision AI Agent
    
    Args:
        image_path: Path to the image file
        task_type: Florence-2 task type
        
    Returns:
        Analysis results or None on error
    """
    try:
        # Encode image
        print(f"\nAnalyzing image: {image_path}")
        print(f"Task type: {task_type}")
        
        image_data = encode_image(image_path)
        
        # Send request
        response = requests.post(
            f"{API_BASE_URL}/analyze",
            json={
                "image": image_data,
                "task_type": task_type
            },
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                print("✓ Analysis successful!")
                return data['result']
            else:
                print(f"✗ Analysis failed: {data.get('error', 'Unknown error')}")
                return None
        else:
            print(f"✗ Server returned error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"✗ Error during analysis: {e}")
        return None

def get_task_types():
    """Get available task types"""
    try:
        response = requests.get(f"{API_BASE_URL}/task-types")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"Error getting task types: {e}")
        return None

def main():
    """Main example"""
    print("=" * 60)
    print("Vision AI Agent - API Example")
    print("=" * 60)
    
    # Check server health
    print("\n1. Checking server health...")
    if not check_health():
        print("\nPlease start the server first:")
        print("  python app.py")
        return
    
    # Get available task types
    print("\n2. Getting available task types...")
    task_types = get_task_types()
    if task_types:
        print("✓ Available task types:")
        for key, value in task_types.items():
            print(f"  - {value['name']}: {value['description']}")
    
    # Example: Analyze an image
    print("\n3. Analyzing image...")
    print("\nNote: Replace 'example.jpg' with your actual image path")
    
    # Uncomment and modify this to use with an actual image:
    # result = analyze_image("example.jpg", "<MORE_DETAILED_CAPTION>")
    # if result:
    #     print("\nAnalysis Result:")
    #     print(json.dumps(result, indent=2))
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
