import io
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import soundfile as sf
import librosa

from .vad import detect_turns

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'spectral_model.pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        _saved = pickle.load(f)
        _model = _saved['model']
        _feature_names = _saved['feature_names']
        _threshold = _saved.get('threshold', 0.5)
except FileNotFoundError:
    _model = None
    _threshold = 0.5

def extract_spectral_features_from_audio(audio_bytes):
    data, sample_rate = sf.read(io.BytesIO(audio_bytes))
    if data.ndim != 2 or data.shape[1] < 2:
        return None

    channel_caller = data[:, 0]

    turns_caller = detect_turns(channel_caller, sample_rate)
    if not turns_caller:
        return None

    active_speech = []
    for turn in turns_caller:
        start_idx = int(turn['start'] * sample_rate)
        end_idx = int(turn['end'] * sample_rate)
        active_speech.extend(channel_caller[start_idx:end_idx])

    active_speech = np.array(active_speech, dtype=np.float32)
    if len(active_speech) == 0:
        return None

    centroid = librosa.feature.spectral_centroid(y=active_speech, sr=sample_rate)[0]
    flatness = librosa.feature.spectral_flatness(y=active_speech)[0]
    rolloff = librosa.feature.spectral_rolloff(y=active_speech, sr=sample_rate, roll_percent=0.85)[0]

    stft = np.abs(librosa.stft(active_speech))
    freqs = librosa.fft_frequencies(sr=sample_rate)
    band1_mask = (freqs >= 0) & (freqs < 2000)
    band2_mask = (freqs >= 2000) & (freqs <= 4000)
    energy_band1 = np.sum(stft[band1_mask, :], axis=0)
    energy_band2 = np.sum(stft[band2_mask, :], axis=0)

    mfccs = librosa.feature.mfcc(y=active_speech, sr=sample_rate, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(y=active_speech)[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f0, _, _ = librosa.pyin(
            active_speech,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C6'),
            sr=sample_rate,
            fill_na=None
        )

    f0_voiced = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    pitch_mean = np.mean(f0_voiced) if len(f0_voiced) > 0 else 0.0
    pitch_std = np.std(f0_voiced) if len(f0_voiced) > 0 else 0.0

    features = {
        'centroid_mean': np.mean(centroid), 'centroid_std': np.std(centroid),
        'flatness_mean': np.mean(flatness), 'flatness_std': np.std(flatness),
        'rolloff_mean': np.mean(rolloff),   'rolloff_std': np.std(rolloff),
        'eband1_mean': np.mean(energy_band1), 'eband1_std': np.std(energy_band1),
        'eband2_mean': np.mean(energy_band2), 'eband2_std': np.std(energy_band2),
        'zcr_mean': np.mean(zcr), 'zcr_std': np.std(zcr),
        'pitch_mean': pitch_mean, 'pitch_std': pitch_std
    }

    for i in range(1, 14):
        features[f'mfcc_{i}_mean'] = np.mean(mfccs[i-1])
        features[f'mfcc_{i}_std'] = np.std(mfccs[i-1])

    return features

def predict_spectral(audio_bytes):
    if _model is None:
        return None, None

    feats = extract_spectral_features_from_audio(audio_bytes)
    if feats is None:
        return None, None

    X = pd.DataFrame([[feats.get(name, 0.0) for name in _feature_names]], columns=_feature_names)
    X = X.fillna(0)

    proba_synthetic = _model.predict_proba(X)[0][1]
    is_synthetic = bool(proba_synthetic >= _threshold)
    confidence = float(proba_synthetic if is_synthetic else (1.0 - proba_synthetic))

    return is_synthetic, confidence

def _warmup():
    try:
        sr = 8000
        dummy = (np.random.randn(sr * 2) * 0.5).astype(np.float32)
        librosa.feature.mfcc(y=dummy, sr=sr, n_mfcc=13)
        librosa.feature.spectral_centroid(y=dummy, sr=sr)
        librosa.feature.spectral_flatness(y=dummy)
        librosa.feature.spectral_rolloff(y=dummy, sr=sr, roll_percent=0.85)
        librosa.stft(dummy)
        librosa.feature.zero_crossing_rate(y=dummy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            librosa.pyin(dummy, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'), sr=sr, fill_na=None)
    except Exception:
        pass

if _model is not None:
    _warmup()