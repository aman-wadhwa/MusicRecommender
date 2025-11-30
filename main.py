import os
import logging
# Suppress TensorFlow logs for a cleaner terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from image_processor import ImageAnalyzer
from music_engine import MusicRecommender

def run_project(image_path):
    print(f"\nSTARTING MOOD MUSIC FOR: {image_path}")
    print("=" * 40)

    # 1. Initialize the AI Agents
    # The analyzer loads Intel model and DeepFace
    analyzer = ImageAnalyzer()
    
    # The DJ loads the playlists
    dj = MusicRecommender()

    # 2. Analyze the Image
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found!")
        return

    # This gets both Face Emotion AND Scene Vibe
    results = analyzer.get_image_data(image_path)
    
    emotion = results['face_emotion']
    scene = results['detected_scene']
    
    # Print what the AI found
    print(f"\n AI DIAGNOSIS:")
    if emotion:
        print(f"Face Detected:  {emotion.upper()}")
    else:
        print(f"Face Detected:  None")
    
    print(f"Scene Detected: {scene.upper()}")

    # 3. Get Music Recommendation
    playlist = dj.get_recommendation(scene=scene, emotion=emotion)

    # 4. Final Output
    print(f"\nRECOMMENDED PLAYLIST:")
    print("-" * 30)
    
    if not playlist:
        print("   (No songs found for this vibe yet!)")
    else:
        for i, song in enumerate(playlist):
            print(f"   {i+1}. {song['title']} - {song['artist']}")
            print(f"      {song['link']}")
        
            
    print("-" * 30)

if __name__ == "__main__":
    test_image = "test.jpg" 
    
    run_project(test_image)