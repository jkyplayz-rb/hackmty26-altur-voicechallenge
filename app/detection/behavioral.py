import io
import pickle
import os

import numpy as np
import pandas as pd
import soundfile as sf

from .vad import detect_turns

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'behavioral_model.pkl')

with open(MODEL_PATH, 'rb') as f:
    _saved = pickle.load(f)
    _model = _saved['model']
    _feature_names = _saved['feature_names']


def caller_response_gaps(turns_caller, turns_agent):
    """Same logic as training: gap between agent turn ending and next caller turn starting."""
    all_turns = sorted(
        [{'channel': 0, **t} for t in turns_caller] + [{'channel': 1, **t} for t in turns_agent],
        key=lambda t: t['start']
    )
    gaps = []
    last_agent_end = None
    for turn in all_turns:
        if turn['channel'] == 1:
            last_agent_end = turn['end']
        elif turn['channel'] == 0 and last_agent_end is not None:
            gap = turn['start'] - last_agent_end
            if gap >= 0:
                gaps.append(gap)
            last_agent_end = None
    return gaps


def extract_features_from_audio(audio_bytes):
    """
    Takes raw stereo WAV bytes, returns a feature dict (or None if
    not enough data to compute gaps).
    """
    data, sample_rate = sf.read(io.BytesIO(audio_bytes))
    if data.ndim != 2 or data.shape[1] < 2:
        return None

    channel_caller = data[:, 0]
    channel_agent = data[:, 1]

    turns_caller = detect_turns(channel_caller, sample_rate)
    turns_agent = detect_turns(channel_agent, sample_rate)

    gaps = caller_response_gaps(turns_caller, turns_agent)
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


def predict(audio_bytes):
    """
    Returns (is_synthetic: bool, confidence: float), or (None, None)
    if there wasn't enough data to make a call.
    """
    feats = extract_features_from_audio(audio_bytes)
    if feats is None:
        return None, None

    X = pd.DataFrame([[feats[name] for name in _feature_names]], columns=_feature_names)
    proba = _model.predict_proba(X)[0]
    is_synthetic = bool(_model.predict(X)[0])
    confidence = float(proba[1] if is_synthetic else proba[0])

    return is_synthetic, confidence