# Vision AI Agent - Quick Reference

## Starting the Server

### Option 1: Using the startup script
```bash
./start.sh
```

### Option 2: Manual start
```bash
pip install -r requirements.txt
python app.py
```

## Accessing the Application

Open your browser to: **http://localhost:5000**

## Using the Web Interface

### Upload an Image
1. Click **📂 Upload File**
2. Select an image from your computer
3. Image preview will appear

### Take a Photo
1. Click **📷 Take Photo**
2. Allow camera access when prompted
3. Position your subject
4. Click **📸 Capture Photo**

### Analyze
1. Select task type from dropdown (default: Detailed Caption)
2. Click **🚀 Analyze Image**
3. Wait for analysis to complete
4. View results below

## Task Types

| Task | Description | Use Case |
|------|-------------|----------|
| **Detailed Caption** | Comprehensive description | General understanding |
| **Caption** | Brief description | Quick summary |
| **Object Detection** | Identify objects | Find specific items |
| **Dense Region Caption** | Region-by-region analysis | Detailed scene understanding |
| **Region Proposal** | Areas of interest | Focus detection |

## API Usage

### Health Check
```bash
curl http://localhost:5000/api/health
```

### Analyze Image
```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,...", "task_type": "<MORE_DETAILED_CAPTION>"}'
```

### Get Task Types
```bash
curl http://localhost:5000/api/task-types
```

## Keyboard Shortcuts

- **Tab**: Navigate between elements
- **Enter**: Activate buttons
- **Esc**: Cancel camera/file dialogs

## Requirements

- Python 3.8+
- ~1GB free disk space (for model)
- Internet connection (first run only)
- Modern web browser
- Optional: CUDA-capable GPU for faster processing
- Optional: Webcam for photo capture

## Troubleshooting

### Server won't start
```bash
# Check Python version
python --version

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Camera not working
- Check browser permissions
- Use HTTPS or localhost
- Try different browser
- Check if camera is in use by another app

### Model download fails
- Check internet connection
- Check disk space
- Verify Hugging Face is accessible
- Try again (downloads resume automatically)

### Analysis is slow
- First analysis is slower (model initialization)
- Use GPU if available
- Reduce image size
- Consider smaller model variant

## Performance

| Component | Speed | Notes |
|-----------|-------|-------|
| **Model Load** | 10-30s | One-time on startup |
| **First Analysis** | 5-15s | Cache warming |
| **Subsequent** | 2-5s | With GPU |
| **Subsequent** | 10-30s | CPU only |

## File Size Limits

- **Images**: Recommended max 10MB
- **Model**: ~900MB download
- **Memory**: ~2-4GB RAM during inference

## Privacy

- ✓ Everything runs locally
- ✓ No data sent to external servers
- ✓ No tracking or analytics
- ✓ No account required
- ✓ Images processed in memory only

## Support

- GitHub Issues: Report bugs or request features
- README.md: Full documentation
- DEVELOPMENT.md: Developer guide
- UI_DOCUMENTATION.md: Interface details

## Quick Tips

💡 **Tip 1**: First run downloads the model (~900MB) - be patient!

💡 **Tip 2**: GPU makes analysis 5-10x faster

💡 **Tip 3**: Try different task types for different insights

💡 **Tip 4**: Larger images take longer to process

💡 **Tip 5**: Camera permissions are required for photo capture

---

**Need more help?** Check the full README.md or open an issue on GitHub.
