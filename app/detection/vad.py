import numpy as np


def detect_turns(samples, sample_rate, energy_threshold=0.02, min_silence_s=0.3, min_turn_s=0.15):
    """
    Simple energy-based voice activity detection for one audio channel.
    Returns a list of {'start': float, 'end': float} segments in seconds.
    """
    frame_ms = 20
    frame_len = int(sample_rate * frame_ms / 1000)
    n_frames = len(samples) // frame_len

    # RMS energy per frame
    energies = []
    for i in range(n_frames):
        frame = samples[i * frame_len:(i + 1) * frame_len]
        rms = np.sqrt(np.mean(frame.astype(np.float64) ** 2)) if len(frame) else 0
        energies.append(rms)
    energies = np.array(energies)

    if energies.max() > 0:
        energies = energies / energies.max()

    is_speech = energies > energy_threshold

    # Merge frames into turns, bridging short silences
    turns = []
    start = None
    silence_frames = 0
    min_silence_frames = int(min_silence_s * 1000 / frame_ms)

    for i, speech in enumerate(is_speech):
        t = i * frame_ms / 1000
        if speech:
            if start is None:
                start = t
            silence_frames = 0
        else:
            if start is not None:
                silence_frames += 1
                if silence_frames >= min_silence_frames:
                    end = t - (silence_frames * frame_ms / 1000)
                    if end - start >= min_turn_s:
                        turns.append({'start': start, 'end': end})
                    start = None
                    silence_frames = 0

    if start is not None:
        end = n_frames * frame_ms / 1000
        if end - start >= min_turn_s:
            turns.append({'start': start, 'end': end})

    return turns