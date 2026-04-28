#!/usr/bin/env python
"""Run aiagent.py with SKIP_MODEL_LOAD to enable hand detection without loading Florence model"""

import os
import sys

# Set skip model load to avoid CUDA memory issues
os.environ['SKIP_MODEL_LOAD'] = '1'

# Change to project directory
os.chdir(r'e:\ai_vision')
sys.path.insert(0, r'e:\ai_vision')

# Import and run Flask app
from aiagent import app

print("\n" + "="*60)
print("✓ Vision AI Agent - Hand Detection Mode")
print("✓ Florence Model: SKIPPED (CUDA memory optimization)")
print("✓ Hand Detection: ENABLED")
print("="*60)
print("✓ Starting Flask server on http://localhost:5000")
print("="*60 + "\n")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
