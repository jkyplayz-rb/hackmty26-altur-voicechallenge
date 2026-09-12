import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from app.detection.spectral import extract_spectral_features_from_audio

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MANIFEST_PATH = os.path.join(DATA_DIR, 'manifest.csv')
AUDIO_DIR = os.path.join(DATA_DIR, 'audio')
MODEL_OUT = os.path.join(os.path.dirname(__file__), '..', 'app', 'detection', 'spectral_model.pkl')

def build_dataset(manifest, split):
    rows, labels = [], []
    split_data = manifest[manifest['split'] == split]
    
    for _, row in split_data.iterrows():
        audio_path = os.path.join(AUDIO_DIR, f"{row['anon_id']}.wav")
        try:
            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()
            # Suprimir advertencias durante el procesamiento en lote
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                feats = extract_spectral_features_from_audio(audio_bytes)
        except Exception:
            continue
            
        if feats is None:
            continue
            
        rows.append(feats)
        labels.append(1 if row['label'] == 'synthetic' else 0)
        
    X = pd.DataFrame(rows).fillna(0)
    y = np.array(labels)
    return X, y

def main():
    manifest = pd.read_csv(MANIFEST_PATH)

    print("Building training dataset (this will take longer due to Pitch/MFCC extraction)...")
    X_train, y_train = build_dataset(manifest, 'train')
    
    print("Building validation dataset...")
    X_val, y_val = build_dataset(manifest, 'val')

    print(f"Train: {len(X_train)} calls, Val: {len(X_val)} calls")

    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(32, 16), 
            activation='relu', 
            alpha=0.01,  # Aumento de regularización
            max_iter=1000, 
            random_state=42,
            early_stopping=True
        )
    )
    
    print("Training model...")
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)

    print(f"\nTrain accuracy: {accuracy_score(y_train, train_preds):.3f}")
    print(f"Val accuracy: {accuracy_score(y_val, val_preds):.3f}")
    
    print("\n=== Val classification report (Spectral NN) ===")
    print(classification_report(y_val, val_preds, target_names=['human', 'synthetic']))

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    with open(MODEL_OUT, 'wb') as f:
        pickle.dump({'model': model, 'feature_names': list(X_train.columns)}, f)
    print(f"\nModel saved to {MODEL_OUT}")

if __name__ == '__main__':
    main()