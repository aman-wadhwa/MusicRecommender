import os
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import uuid

# Suppress TensorFlow logs for a cleaner terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from image_processor import ImageAnalyzer
from music_engine import MusicRecommender

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize models once (they'll be reused for all requests)
analyzer = None
dj = None

def init_models():
    """Initialize the models once at startup"""
    global analyzer, dj
    if analyzer is None:
        analyzer = ImageAnalyzer()
    if dj is None:
        dj = MusicRecommender()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/recommend', methods=['POST'])
def recommend():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload an image (png, jpg, jpeg, gif, webp)'}), 400
        
        # Initialize models
        init_models()
        
        # Save uploaded file
        filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        try:
            # Analyze the image
            results = analyzer.get_image_data(filepath)
            
            emotion = results['face_emotion']
            scene = results['detected_scene']
            objects = results['detected_object']
            # Get music recommendations
            playlist = dj.get_recommendation(scene=scene, emotion=emotion, objects=objects)
            
            # Clean up uploaded file
            try:
                os.remove(filepath)
            except:
                pass
            
            return jsonify({
                'success': True,
                'emotion': emotion,
                'scene': scene,
                'objects' : objects,
                'playlist': playlist
            })
            
        except Exception as e:
            # Clean up uploaded file on error
            try:
                os.remove(filepath)
            except:
                pass
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting Song Recommender API...")
    print("Initializing models (this may take a moment)...")
    init_models()
    print("Models loaded! Server starting...")
    # port = int(os.environ.get('PORT', 5000))
    # app.run(host='0.0.0.0', port=port, debug=False)
    app.run(debug=True, port=5000)

