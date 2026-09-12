import base64
import io

import soundfile as sf

ACCEPTED_AUDIO_KEYS = ('audio', 'audio_base64', 'wav_base64')
EXPECTED_SAMPLE_RATE = 8000
EXPECTED_CHANNELS = 2
MAX_AUDIO_BYTES = 10 * 1024 * 1024


class AudioValidationError(Exception):
    pass


def extract_audio_field(payload):
    present = [k for k in ACCEPTED_AUDIO_KEYS if k in payload]
    if not present:
        raise AudioValidationError(f'Missing audio field, expected one of {ACCEPTED_AUDIO_KEYS}')
    if len(present) > 1:
        raise AudioValidationError(f'Multiple audio fields present ({present}), expected exactly one')
    return payload[present[0]]


def decode_audio(payload):
    audio_b64 = extract_audio_field(payload)

    try:
        audio_bytes = base64.b64decode(audio_b64, validate=True)
    except Exception:
        raise AudioValidationError('Invalid base64 audio data')

    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise AudioValidationError(f'Audio exceeds size limit of {MAX_AUDIO_BYTES} bytes')

    try:
        data, sample_rate = sf.read(io.BytesIO(audio_bytes))
    except Exception:
        raise AudioValidationError('Could not read WAV audio')

    if data.ndim != 2 or data.shape[1] != EXPECTED_CHANNELS:
        raise AudioValidationError(f'Expected {EXPECTED_CHANNELS}-channel audio, got shape {data.shape}')

    if sample_rate != EXPECTED_SAMPLE_RATE:
        raise AudioValidationError(f'Expected {EXPECTED_SAMPLE_RATE}Hz audio, got {sample_rate}Hz')

    return audio_bytes