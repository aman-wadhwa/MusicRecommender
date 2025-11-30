import os
import numpy as np
import keras
from keras.models import load_model
from keras.preprocessing import image as keras_image
from deepface import DeepFace
from collections import Counter

class ImageAnalyzer:
    def __init__(self):
        print("--- INITIALIZING ANALYZER ---")
        
        # 1. SETUP SCENE RECOGNITION (Custom Model)
        self.model_path = 'intel_scene_model.h5'
        self.classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
        
        if os.path.exists(self.model_path):
            print(f"Loading Custom Vibe Model: {self.model_path}...")
            self.scene_model = load_model(self.model_path)
            print("Scene Model Loaded!")
        else:
            print(f"ERROR: {self.model_path} not found.")
            self.scene_model = None

    def analyze_face_emotion(self, img_path):
        """
        Detects face and returns dominant emotion using DeepFace.
        Returns None if no face is found.
        """
        try:
            # enforce_detection=True ensures we only return an emotion if a real face exists
            analysis = DeepFace.analyze(img_path, actions=['emotion'], enforce_detection=True, silent=True)
            
            # print('asdad')
            # DeepFace returns a list of dicts
            if isinstance(analysis, list):
                all_emotions = [face['dominant_emotion'] for face in analysis]
                vote_counts = Counter(all_emotions)
                winner = vote_counts.most_common(1)[0][0]
                return winner
                # return analysis[0]['dominant_emotion']
            return analysis['dominant_emotion']
            
        except ValueError:  
            return None
        except Exception as e:
            print(f"Face Error: {e}")
            return None

    def analyze_scene_objects(self, img_path):
        """
        Detects scene context CUSTOM Intel Model.
        Returns the name of the scene
        """
        if self.scene_model is None:
            return "unknown"

        try:
            # 1. Resize to 150x150 (Match your training!)
            img = keras_image.load_img(img_path, target_size=(150, 150))
            
            # 2. Convert to Array
            x = keras_image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            
            # 3. Normalize (Divide by 255, just like in training!)
            x = x / 255.0

            # 4. Predict
            preds = self.scene_model.predict(x, verbose=0)
            
            # 5. Get the highest score
            class_idx = np.argmax(preds)
            confidence = np.max(preds)
            
            detected_scene = self.classes[class_idx]
            
            
            return detected_scene
            
        except Exception as e:
            print(f"Scene Error: {e}")
            return "unknown"

    def get_image_data(self, img_path):
        """
        function to get all data (Face + Scene).
        """
        print(f"\nAnalyzing Image: {img_path}...")
        
        # 1. Get Face Emotion
        emotion = self.analyze_face_emotion(img_path)
        
        # 2. Get Scene
        scene = self.analyze_scene_objects(img_path)

        return {
            "face_emotion": emotion,
            "detected_scene": scene
        }

if __name__ == "__main__":
    analyzer = ImageAnalyzer()
    
    test_img = "test.jpg"
    
    if os.path.exists(test_img):
        result = analyzer.get_image_data(test_img)
        
        print("\n--- FINAL RESULTS ---")
        print(f"Face Emotion: {result['face_emotion']}")
        print(f"Scene Vibe:   {result['detected_scene']}")
    else:
        print(f"'{test_img}' not found.")