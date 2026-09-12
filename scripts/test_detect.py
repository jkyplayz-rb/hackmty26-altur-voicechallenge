import base64
import csv
import os
import sys

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
MANIFEST_PATH = os.path.join(DATA_DIR, 'manifest.csv')
AUDIO_DIR = os.path.join(DATA_DIR, 'audio')
DETECT_URL = 'http://127.0.0.1:5000/detect'


def load_val_rows(limit):
    rows = []
    with open(MANIFEST_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['split'] == 'val':
                rows.append(row)
    return rows[:limit]


def predict_call(anon_id):
    audio_path = os.path.join(AUDIO_DIR, f'{anon_id}.wav')
    with open(audio_path, 'rb') as f:
        audio_b64 = base64.b64encode(f.read()).decode('ascii')

    response = requests.post(DETECT_URL, json={'audio': audio_b64})
    response.raise_for_status()
    return response.json()


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 71
    rows = load_val_rows(limit)

    correct = 0
    confusion = {'human': {'human': 0, 'synthetic': 0}, 'synthetic': {'human': 0, 'synthetic': 0}}

    for i, row in enumerate(rows):
        true_label = row['label']
        result = predict_call(row['anon_id'])
        predicted_label = 'synthetic' if result['is_synthetic'] else 'human'

        confusion[true_label][predicted_label] += 1
        if predicted_label == true_label:
            correct += 1

        print(f"{i+1}/{len(rows)} {row['anon_id']} true={true_label} pred={predicted_label} conf={result['confidence']:.2f}")

    total = len(rows)
    print(f"\nAccuracy: {correct}/{total} = {correct/total:.3f}")
    print("Confusion matrix (rows=true, cols=predicted):")
    print("          human  synthetic")
    print(f"human     {confusion['human']['human']:5d}  {confusion['human']['synthetic']:5d}")
    print(f"synthetic {confusion['synthetic']['human']:5d}  {confusion['synthetic']['synthetic']:5d}")


if __name__ == '__main__':
    main()