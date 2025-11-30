import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class MusicRecommender:
    def __init__(self, csv_path='dataset2.csv'):
        print("Initializing Context-Aware Engine...")
        
        try:
            self.df = pd.read_csv(csv_path)
            self.df = self.df.dropna()
            self.df = self.df.drop_duplicates(subset=['track_name', 'artist_name'])
            
            # can add genre features
            
            self.feature_cols = ['valence', 'energy', 'danceability', 'acousticness']
            self.scaler = MinMaxScaler()
            self.feature_matrix = self.scaler.fit_transform(self.df[self.feature_cols])
            
            self.nn_model = NearestNeighbors(n_neighbors=20, algorithm='brute', metric='euclidean')
            self.nn_model.fit(self.feature_matrix)
            
            self.data_ready = True
            
        except Exception as e:
            print(f"Engine Error: {e}")
            self.data_ready = False

        # 1. SCENE VECTORS (The "Context")
        self.scene_vectors = {
            # [Valence, Energy, Danceability, Acousticness]
            'buildings': [0.5, 0.8, 0.8, 0.1], # Pop Base
            'street':    [0.5, 0.7, 0.9, 0.2], # HipHop Base
            'forest':    [0.5, 0.3, 0.4, 0.9], # Acoustic Base
            'mountain':  [0.5, 0.9, 0.4, 0.1], # Rock Base
            'glacier':   [0.2, 0.1, 0.1, 0.9], # Ambient Base
            'sea':       [0.8, 0.6, 0.7, 0.3]  # Tropical Base
        }

        # 2. EMOTION MODIFIERS Adjustment
        # These SHIFT the scene vector up or down.
        # eg "Sad" pulls Valence and Energy DOWN.
        self.emotion_modifiers = {
            'happy':     {'valence': +0.3, 'energy': +0.2}, 
            'sad':       {'valence': -0.4, 'energy': -0.3},
            'angry':     {'valence': -0.2, 'energy': +0.4}, # Angry = High Energy, Negative
            'fear':      {'valence': -0.2, 'energy': -0.2}, 
            'surprise':  {'valence': +0.1, 'energy': +0.3},
            'neutral':   {'valence': 0.0,  'energy': 0.0},   # No change
            'disgust':   {'valence': -0.3, 'energy': +0.1}
        }

    def get_recommendation(self, scene="unknown", emotion="neutral"):
        if not self.data_ready:
            return []

        # STEP A: Start with the Scene (Context)
        # Default to "Neutral" vector if scene is unknown
        base_vector = np.array(self.scene_vectors.get(scene, [0.5, 0.5, 0.5, 0.5]))
        
        # STEP B: Apply Emotion (The Modifier)
        modifier = self.emotion_modifiers.get(emotion, {'valence': 0, 'energy': 0})
        
        # Modify Valence (Index 0)
        base_vector[0] = np.clip(base_vector[0] + modifier['valence'], 0, 1)
        
        # Modify Energy (Index 1)
        base_vector[1] = np.clip(base_vector[1] + modifier['energy'], 0, 1)

        print(f"Logic: Scene '{scene}' set base. Emotion '{emotion}' shifted values.")
        print(f" Final Target Vector: {base_vector}")

        # STEP C: Find Songs
        distances, indices = self.nn_model.kneighbors([base_vector])
        
        playlist = []
        for idx in indices[0][:20]: # Top 20
            song_row = self.df.iloc[idx]
            
            t_id = song_row.get('track_id', '')
            link = f"http://open.spotify.com/track/{t_id}" if t_id else "#"
            
            playlist.append({
                "title": song_row['track_name'],
                "artist": song_row['artist_name'],
                "link": link
            })
            
        return playlist