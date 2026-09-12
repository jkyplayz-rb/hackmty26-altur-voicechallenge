import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MANIFEST_PATH = os.path.join(DATA_DIR, 'manifest.csv')
TURNS_DIR = os.path.join(DATA_DIR, 'turns')
MODEL_OUT = os.path.join(os.path.dirname(__file__), '..', 'app', 'detection', 'behavioral_model.pkl')


def load_turns(anon_id):
    path = os.path.join(TURNS_DIR, f'{anon_id}.json')
    with open(path) as f:
        return json.load(f)['turns']


def caller_response_gaps(turns):
    gaps = []
    last_agent_end = None
    for turn in turns:
        if turn['channel'] == 1:
            last_agent_end = turn['end']
        elif turn['channel'] == 0 and last_agent_end is not None:
            gap = turn['start'] - last_agent_end
            if gap >= 0:
                gaps.append(gap)
            last_agent_end = None
    return gaps


def extract_features(anon_id):
    """Returns a feature vector, or None if the call has no usable data."""
    try:
        turns = load_turns(anon_id)
    except FileNotFoundError:
        return None
    gaps = caller_response_gaps(turns)
    if not gaps:
        return None
    gaps = np.array(gaps)
    return {
        'mean_gap': gaps.mean(),
        'min_gap': gaps.min(),
        'max_gap': gaps.max(),
        'std_gap': gaps.std() if len(gaps) > 1 else 0.0,
        'n_gaps': len(gaps),
    }


def build_dataset(manifest, split):
    rows = []
    labels = []
    for _, row in manifest[manifest['split'] == split].iterrows():
        feats = extract_features(row['anon_id'])
        if feats is None:
            continue
        rows.append(feats)
        labels.append(1 if row['label'] == 'synthetic' else 0)
    X = pd.DataFrame(rows)
    y = np.array(labels)
    return X, y


def main():
    manifest = pd.read_csv(MANIFEST_PATH)

    X_train, y_train = build_dataset(manifest, 'train')
    X_val, y_val = build_dataset(manifest, 'val')

    print(f"Train: {len(X_train)} calls, Val: {len(X_val)} calls")

    model = LogisticRegression()
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)

    print(f"\nTrain accuracy: {accuracy_score(y_train, train_preds):.3f}")
    print(f"Val accuracy: {accuracy_score(y_val, val_preds):.3f}")
    print("\n=== Val classification report ===")
    print(classification_report(y_val, val_preds, target_names=['human', 'synthetic']))

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    with open(MODEL_OUT, 'wb') as f:
        pickle.dump({'model': model, 'feature_names': list(X_train.columns)}, f)
    print(f"\nModel saved to {MODEL_OUT}")


if __name__ == '__main__':
    main()