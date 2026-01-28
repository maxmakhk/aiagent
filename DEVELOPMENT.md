# Development Guide

## Project Structure

```
aiagent/
├── app.py                  # Main Flask application and AI model integration
├── templates/
│   └── index.html         # Web UI frontend
├── requirements.txt       # Python dependencies
├── start.sh              # Startup script
├── test_basic.py         # Basic functionality tests
├── README.md             # User documentation
└── .gitignore           # Git ignore rules
```

## Key Components

### Backend (app.py)

**Main Features:**
- Flask web server setup with CORS
- Florence-2 model initialization and management
- Image analysis API endpoints
- Health check endpoint
- Task type discovery endpoint

**Key Functions:**
- `initialize_model()`: Loads the Florence-2 model from Hugging Face
- `analyze_image()`: Processes images using the Florence-2 model
- API routes for health checks, analysis, and task types

### Frontend (templates/index.html)

**UI Components:**
- Image Source Selection (Upload/Camera)
- Analysis Mode Selector
- Task Type Dropdown
- Image Preview
- Camera Preview with Capture
- Results Display
- Loading Indicators

**JavaScript Functions:**
- `selectSource()`: Switch between upload and camera modes
- `startCamera()`: Initialize webcam access
- `capturePhoto()`: Capture image from webcam
- `handleFileSelect()`: Handle file uploads
- `analyzeImage()`: Send image to backend for analysis
- `displayResults()`: Show analysis results

## Development Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run in Development Mode**
   ```bash
   python app.py
   ```

3. **Access the Application**
   Open browser to http://localhost:5000

## Testing

### Manual Testing Checklist

- [ ] Server starts without errors
- [ ] Web UI loads correctly
- [ ] File upload works
- [ ] Camera capture works (requires camera access)
- [ ] Image preview displays correctly
- [ ] Task type selection works
- [ ] Analyze button is enabled after image selection
- [ ] Loading indicator shows during analysis
- [ ] Results display correctly
- [ ] Multiple analysis tasks work
- [ ] Different task types produce different results

### Running Basic Tests

```bash
python test_basic.py
```

## API Documentation

### GET /
Returns the main HTML interface

### GET /api/health
Health check endpoint
```json
{
  "status": "healthy",
  "model": "Florence-2",
  "device": "cuda" or "cpu"
}
```

### POST /api/analyze
Analyze an image
```json
Request:
{
  "image": "data:image/jpeg;base64,...",
  "task_type": "<MORE_DETAILED_CAPTION>"
}

Response:
{
  "success": true,
  "result": {...},
  "task_type": "<MORE_DETAILED_CAPTION>"
}
```

### GET /api/task-types
Get available task types
```json
{
  "caption": {...},
  "detailed_caption": {...},
  ...
}
```

## Model Information

**Florence-2 Model:**
- Source: microsoft/Florence-2-base
- Type: Vision-Language Model
- Tasks: Captioning, Object Detection, Region Analysis
- Size: ~900MB download on first use
- Framework: PyTorch + Transformers

## Configuration Options

### Model Selection
Change model in `app.py`:
```python
model_name = "microsoft/Florence-2-large"  # Larger model
model_name = "microsoft/Florence-2-base"   # Default
```

### Server Configuration
```python
app.run(
    host='0.0.0.0',  # Listen on all interfaces
    port=5000,       # Port number
    debug=True       # Debug mode (disable in production)
)
```

### Device Selection
Automatically selects CUDA if available, otherwise CPU:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

## Security Considerations

1. **Local Server Only**: Designed to run locally for privacy
2. **CORS Enabled**: For local development
3. **No Authentication**: Not meant for public deployment without adding auth
4. **File Size Limits**: Consider adding limits for uploaded images
5. **Input Validation**: Basic validation on image data

## Performance Tips

1. **Use GPU**: Significant speedup with CUDA-capable GPU
2. **Image Size**: Smaller images process faster
3. **Model Selection**: Base model is faster than large model
4. **Batch Processing**: Process multiple images if needed

## Troubleshooting

**Issue: Model download fails**
- Check internet connection
- Verify Hugging Face is accessible
- Check disk space (need ~1GB free)

**Issue: CUDA out of memory**
- Use CPU instead (automatic fallback)
- Reduce image size
- Use smaller model variant

**Issue: Camera not working**
- Check browser permissions
- Ensure HTTPS or localhost
- Verify camera is not in use

**Issue: Slow performance**
- First run is slower (model loading)
- Consider using GPU
- Check system resources

## Contributing

When contributing:
1. Follow existing code style
2. Test all features manually
3. Update documentation
4. Keep dependencies minimal
5. Maintain security best practices

## Future Enhancements

Potential improvements:
- [ ] Batch image processing
- [ ] Image preprocessing options
- [ ] More Florence-2 task types
- [ ] Result export functionality
- [ ] Image history/gallery
- [ ] Fine-tuning capabilities
- [ ] API authentication
- [ ] Docker containerization
- [ ] Model caching optimization
