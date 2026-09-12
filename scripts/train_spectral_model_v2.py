import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score, f1_score

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


def best_threshold(y_true, proba, metric=f1_score):
    """Scan thresholds and return the one maximizing the given metric."""
    best_t, best_score = 0.5, -1
    for t in np.arange(0.20, 0.81, 0.01):
        preds = (proba >= t).astype(int)
        score = metric(y_true, preds)
        if score > best_score:
            best_score, best_t = score, t
    return best_t, best_score


def main():
    manifest = pd.read_csv(MANIFEST_PATH)

    print("Building training dataset (this takes a while, MFCC/pitch extraction is slow)...")
    X_train, y_train = build_dataset(manifest, 'train')
    print("Building validation dataset...")
    X_val, y_val = build_dataset(manifest, 'val')

    print(f"Train: {len(X_train)} calls, Val: {len(X_val)} calls, Features: {X_train.shape[1]}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Search over feature count (k) and regularization strength (alpha).
    # Small k + high alpha = less overfitting risk on ~280 training examples.
    k_options = [10, 15, 20, min(30, X_train.shape[1])]
    alpha_options = [0.001, 0.01, 0.05, 0.1]

    best_cfg = None
    best_cv_score = -1

    print("\nCross-validating feature count (k) x regularization (alpha)...")
    for k in k_options:
        for alpha in alpha_options:
            pipe = make_pipeline(
                StandardScaler(),
                SelectKBest(f_classif, k=k),
                MLPClassifier(
                    hidden_layer_sizes=(16, 8),
                    activation='relu',
                    alpha=alpha,
                    max_iter=1000,
                    random_state=42,
                    early_stopping=True,
                )
            )
            # cross_val_predict gives out-of-fold predictions -> honest CV estimate
            proba_oof = cross_val_predict(pipe, X_train, y_train, cv=skf, method='predict_proba')[:, 1]
            preds_oof = (proba_oof >= 0.5).astype(int)
            score = balanced_accuracy_score(y_train, preds_oof)
            print(f"  k={k:2d} alpha={alpha:<6} CV balanced_acc={score:.3f}")
            if score > best_cv_score:
                best_cv_score = score
                best_cfg = (k, alpha)

    k, alpha = best_cfg
    print(f"\nBest config: k={k}, alpha={alpha} (CV balanced_acc={best_cv_score:.3f})")

    final_pipe = make_pipeline(
        StandardScaler(),
        SelectKBest(f_classif, k=k),
        MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation='relu',
            alpha=alpha,
            max_iter=1000,
            random_state=42,
            early_stopping=True,
        )
    )

    # Tune decision threshold using out-of-fold CV predictions on train (never touches val)
    proba_oof = cross_val_predict(final_pipe, X_train, y_train, cv=skf, method='predict_proba')[:, 1]
    threshold, cv_score = best_threshold(y_train, proba_oof, metric=balanced_accuracy_score)
    print(f"Tuned threshold: {threshold:.2f} (CV balanced_acc at this threshold: {cv_score:.3f})")

    # Now fit on the full train set for the model we'll actually ship
    final_pipe.fit(X_train, y_train)

    train_proba = final_pipe.predict_proba(X_train)[:, 1]
    val_proba = final_pipe.predict_proba(X_val)[:, 1]

    train_preds = (train_proba >= threshold).astype(int)
    val_preds = (val_proba >= threshold).astype(int)

    print(f"\nTrain accuracy (tuned threshold): {accuracy_score(y_train, train_preds):.3f}")
    print(f"Val accuracy (tuned threshold):   {accuracy_score(y_val, val_preds):.3f}")
    print("\n=== Val classification report (Spectral NN v2) ===")
    print(classification_report(y_val, val_preds, target_names=['human', 'synthetic']))

    # Also report at default 0.5 threshold for comparison
    val_preds_default = (val_proba >= 0.5).astype(int)
    print(f"(For comparison, val accuracy at default 0.5 threshold: {accuracy_score(y_val, val_preds_default):.3f})")

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    with open(MODEL_OUT, 'wb') as f:
        pickle.dump({
            'model': final_pipe,
            'feature_names': list(X_train.columns),
            'threshold': float(threshold),
        }, f)
    print(f"\nModel saved to {MODEL_OUT}")


if __name__ == '__main__':
    main()