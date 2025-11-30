# Image to Song Recommender

A beautiful web interface for the Image to Song Recommender project. Upload an image and get personalized song recommendations based on the detected emotion and scene!

## Features

- **Image Upload**: Drag and drop or click to upload images
- **AI Analysis**: Detects face emotions and scene context and detects objects
- **Music Recommendations**: Get top 20 song recommendatio
- **Spotify Links**: Direct links to play songs on Spotify

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   python app.py
   ```

3. **Open in Browser**:
   Navigate to `http://localhost:5000`

## Usage

1. Click the upload area or drag and drop an image
2. Click "Get Recommendations" button
3. View your personalized song recommendations
4. Navigate between pages using the Previous/Next buttons
5. Click "Play on Spotify" to open any song in Spotify

## Project Structure

```
SongRecommender/
├── app.py                 # Flask backend API
├── main.py                # Original CLI script
├── image_processor.py     # Image analysis module
├── music_engine.py        # Music recommendation engine
├── static/
│   ├── index.html         # Frontend HTML
│   ├── style.css          # Styling
│   └── script.js          # Frontend JavaScript
├── uploads/               # Temporary upload directory (auto-created)
└── requirements.txt       # Python dependencies
```

## API Endpoint

- **POST** `/api/recommend`: Upload an image and get song recommendations
  - Request: Form data with `image` field
  - Response: JSON with `emotion`, `scene`, `objects` and `playlist` (array of 20 songs)

