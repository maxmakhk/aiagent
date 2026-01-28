# Vision AI Agent - UI Screenshot Documentation

## Main Interface

The Vision AI Agent provides a modern, user-friendly web interface with the following sections:

### Header Section
```
🤖 Vision AI Agent
Third-Party Vision Integration via Local Model Server

[🧠 AI Agent - Self-Modifiable] [🔒 Local Secure Server] [👁️ Vision Model: Florence-2]
```

### 📸 Image Source
Two options for providing images:
- **📂 Upload File**: Click to select an image file from your computer
- **📷 Take Photo**: Use your webcam to capture a live photo

### Image Preview Area
- Displays uploaded or captured images
- Shows camera feed when using webcam
- Image scales to fit display area (max 400px height)

### 🎯 Analysis Mode
- **Standard Analysis**: Default analysis mode

### 📋 Task Type
Dropdown selector with options:
- Detailed Caption (default)
- Caption
- Detailed Caption (Short)
- Object Detection
- Dense Region Caption
- Region Proposal

### 🚀 Analyze Image Button
- Large, prominent button
- Disabled until an image is selected
- Shows loading spinner during analysis
- Green gradient color scheme

### Results Display
- Appears after analysis completes
- Shows formatted analysis results
- Smooth scroll-to animation
- Pretty-printed JSON for structured data

## Color Scheme

### Primary Colors
- Purple gradient background: `#667eea` to `#764ba2`
- Green analyze button: `#11998e` to `#38ef7d`
- White content area with subtle shadows

### UI Elements
- Rounded corners (8-20px border radius)
- Smooth transitions (0.3s)
- Box shadows for depth
- Hover effects on interactive elements

## Responsive Design
- Maximum width: 1200px
- Padding and margins adjust for mobile
- Flexible grid layout
- Touch-friendly button sizes

## Accessibility
- Semantic HTML structure
- Proper ARIA labels
- Keyboard navigation support
- High contrast colors
- Clear visual feedback

## User Flow

1. **Page Load**
   - Server health check runs automatically
   - UI initializes with upload mode selected
   - Analyze button is disabled

2. **Image Selection**
   - User clicks Upload File or Take Photo
   - Image preview appears
   - Analyze button becomes enabled

3. **Analysis**
   - User selects task type (optional)
   - User clicks Analyze Image
   - Loading spinner appears
   - Results display when complete

4. **View Results**
   - Results scroll into view
   - Data formatted appropriately
   - User can analyze another image

## Browser Compatibility
- Modern browsers (Chrome, Firefox, Safari, Edge)
- ES6+ JavaScript features
- CSS Grid and Flexbox
- MediaDevices API for camera access
- Canvas API for image capture
- Fetch API for HTTP requests

## Security Features
- Base64 image encoding for transmission
- Local processing (no external API calls)
- CORS configured for local access
- No cookies or tracking
- Camera access requires user permission
