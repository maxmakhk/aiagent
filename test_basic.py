"""
Simple test to verify basic functionality of the Vision AI Agent
"""

def test_imports():
    """Test that all required imports work"""
    try:
        import flask
        import flask_cors
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM
        from PIL import Image
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_flask_routes():
    """Test Flask routes are defined"""
    try:
        from app import app
        
        # Check if routes exist
        routes = [rule.rule for rule in app.url_map.iter_rules()]
        
        expected_routes = ['/', '/api/health', '/api/analyze', '/api/task-types']
        
        for route in expected_routes:
            if route in routes:
                print(f"✓ Route {route} defined")
            else:
                print(f"✗ Route {route} not found")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Error testing routes: {e}")
        return False

def test_template_exists():
    """Test that the HTML template exists"""
    import os
    template_path = 'templates/index.html'
    
    if os.path.exists(template_path):
        print(f"✓ Template {template_path} exists")
        return True
    else:
        print(f"✗ Template {template_path} not found")
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("Vision AI Agent - Basic Tests")
    print("=" * 60)
    
    print("\nNote: This test checks basic structure without loading the model")
    print("The actual model will be downloaded on first use.\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Flask Routes Test", test_flask_routes),
        ("Template Test", test_template_exists)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    if all(results):
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("=" * 60)
