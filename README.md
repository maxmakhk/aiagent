# Vision AI Agent 🤖

**Third-Party Vision Integration via Local Model Server**

A powerful Vision AI Agent powered by Microsoft's Florence-2 model, providing state-of-the-art vision understanding capabilities through a local secure server.

## ✨ Features

- 🧠 **AI Agent - Self-Modifiable**: Advanced AI capabilities with Florence-2 vision model
- 🔒 **Local Secure Server**: Run entirely on your local machine for privacy and security
- 👁️ **Vision Model: Florence-2**: Microsoft's powerful vision-language model
- 📸 **Flexible Image Input**: Upload files or capture photos directly from your webcam
- 🎯 **Multiple Analysis Modes**: Support for various vision tasks
- 📋 **Rich Task Types**: Captions, object detection, region proposals, and more

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- (Optional) CUDA-capable GPU for faster inference

### Installation

1. Clone the repository:
```bash
git clone https://github.com/maxmakhk/aiagent.git
cd aiagent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:5000
```

## 📖 Usage

### Image Source Options

1. **📂 Upload File**: Click to select and upload an image from your computer
2. **📷 Take Photo**: Use your webcam to capture a photo in real-time

### Analysis Modes

- **Standard Analysis**: Default analysis mode for general vision tasks

### Task Types

Choose from various vision analysis tasks:

- **Detailed Caption**: Generate comprehensive image descriptions
- **Caption**: Quick, concise image captions
- **Object Detection**: Identify and locate objects in images
- **Dense Region Caption**: Generate captions for different regions
- **Region Proposal**: Identify regions of interest

### Analyze Image

1. Select your image source (upload or camera)
2. Choose the desired task type
3. Click **🚀 Analyze Image**
4. View results displayed below

## 🛠️ Technical Details

### Architecture

- **Backend**: Flask web server (Python)
- **AI Model**: Florence-2 (microsoft/Florence-2-base)
- **Frontend**: HTML/CSS/JavaScript with modern UI
- **Model Framework**: PyTorch + Transformers (Hugging Face)

### API Endpoints

- `GET /`: Main web interface
- `GET /api/health`: Health check endpoint
- `POST /api/analyze`: Image analysis endpoint
- `GET /api/task-types`: Available task types

### Supported Task Types

| Task ID | Description |
|---------|-------------|
| `<CAPTION>` | Brief caption |
| `<DETAILED_CAPTION>` | Detailed caption |
| `<MORE_DETAILED_CAPTION>` | Very detailed caption |
| `<OD>` | Object detection |
| `<DENSE_REGION_CAPTION>` | Dense region captions |
| `<REGION_PROPOSAL>` | Region proposals |

## 🔧 Configuration

### Model Selection

The default model is `microsoft/Florence-2-base`. To use a different model variant, modify `app.py`:

```python
model_name = "microsoft/Florence-2-large"  # or Florence-2-large-ft
```

### Server Settings

Modify host and port in `app.py`:

```python
app.run(host='0.0.0.0', port=5000, debug=True)
```

## 📦 Dependencies

- Flask 3.0.0 - Web framework
- flask-cors 4.0.0 - CORS support
- transformers 4.36.0 - Hugging Face transformers
- torch 2.1.0 - PyTorch deep learning framework
- pillow 10.1.0 - Image processing
- einops 0.7.0 - Tensor operations
- timm 0.9.12 - Image models

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project uses the Florence-2 model from Microsoft, which has its own license terms. Please refer to the [Hugging Face model page](https://huggingface.co/microsoft/Florence-2-base) for details.

## 🙏 Acknowledgments

- Microsoft Research for the Florence-2 model
- Hugging Face for the transformers library
- The open-source AI community

## 📞 Support

For issues, questions, or contributions, please open an issue on GitHub.

---

**Built with ❤️ using Florence-2 and Flask**
