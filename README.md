# Altur Voice Deepfake Detector

Real-time detection of synthetic (AI-generated) voices on bank support calls, built for Altur's **"Defend the Bank Against Voice Deepfakes"** challenge at HackMTY 2026.

**Live demo:** http://155.138.208.163/

## The problem

Voice deepfakes are becoming convincing enough to fool call center agents and automated phone banking systems, opening the door to account takeover, fraudulent transfers, and social engineering at scale. Existing authentication (PINs, security questions) doesn't verify that the *voice itself* is real — it can be phished or leaked independently. This project adds that missing layer: a drop-in API that flags whether the caller side of a recorded conversation is human or synthetic, in real time.

## Approach

Each call is a stereo WAV recording: channel 0 is the caller (interlocutor) being classified, channel 1 is the bank agent. Two independent models each score the caller's channel, and their scores are combined with equal weighting:

**Behavioral model** — analyzes conversational turn-taking timing between the caller and the agent. Synthetic voices tend to produce response-timing patterns that differ subtly but measurably from natural human conversation rhythm.

**Spectral/acoustic model** — extracts pitch (via `librosa.pyin`), MFCCs, spectral centroid, spectral flatness, and spectral rolloff directly from the caller's audio, then classifies with a trained scikit-learn model.

The two models' scores are averaged (50/50) into a single confidence value; if only one model produces a valid score (e.g. a channel is too short for reliable pitch tracking), that score is used alone.

## Results

| Model | Accuracy |
|---|---|
| Behavioral only | 93.0% |
| Spectral only | 91.5% |
| **Ensemble (both combined)** | **95.8%** |

### Independent verification

The 95.8% figure isn't just our own benchmark script — we ran the organizers' own official test client (`check_endpoint.py` from the `alturio/hackmty26` repo) directly against our live deployed server, using the exact request/response contract the real judging system uses:

- Balanced accuracy: **0.958**
- AUC: **0.998**
- Brier score: 0.068
- Mean latency: 2.75s per call, max 9.1s
- 71/71 calls answered, 0 errors

## API

### `POST /detect`

Request body:

```json
{
  "audio": "<base64-encoded WAV, 8kHz, 2-channel (stereo)>"
}
```

Also accepts `audio_base64` or `wav_base64` as the key name, matching the exact contract used by the official judging system.

Response:

```json
{
  "is_synthetic": true,
  "confidence": 0.86
}
```

`is_synthetic` is always present. `confidence` (0.0–1.0) reflects how strongly the ensemble leans toward its answer.

### `GET /health`

```json
{ "status": "ok" }
```

## Frontend

A browser UI lives at `app/static/` — served directly by Flask, no build step or framework required. Drag in a WAV file to see:

- A live waveform view of both channels (caller / agent)
- The classification result with confidence score
- Raw endpoint response (for debugging/transparency)
- A downloadable JSON report of the analysis

Covered by 36 unit tests (`node --test tests/frontend/`) validating WAV parsing, payload construction, and error handling — no npm install needed.

## Running locally

```bash
pip install -r requirements.txt
python3 app/main.py
```

The server listens on port 5000 by default. Visit `http://localhost:5000/` for the UI, or POST directly to `/detect`.

**Note on cold starts:** the spectral model's first request after a fresh server start can take 20–30s, because `librosa`'s pitch-tracking functions are JIT-compiled by numba on first use. The server pre-warms these functions at startup (bypassing the normal audio pipeline to force compilation early), so real requests in production stay in the 1–3s range.

## Deployment

Currently deployed on a Vultr VPS running Gunicorn behind a 120s worker timeout (to comfortably clear the cold-start JIT compilation window on first boot). See `scripts/check_endpoint.py` for the exact client used to validate the live deployment end-to-end.

## Project structure

```
app/
  main.py                    # Flask app: /detect, /health, and static frontend routes
  validation.py               # Audio decoding + request contract validation
  detection/
    behavioral.py             # Turn-taking timing model
    spectral.py                # Acoustic/spectral model + startup warmup
    spectral_model.pkl         # Trained spectral classifier
  static/                     # Frontend (HTML/CSS/JS), no build step
scripts/
  train_spectral_model_v2.py   # Spectral model training pipeline
  test_detect.py                # Local validation script
  check_endpoint.py             # Official organizer test client
tests/
  frontend/                    # Frontend unit tests (node --test)
```

## Team

- **Juan Carlos Livas Reyes (Juanky)** — backend, ML models, deployment
- **Erick Morales Najera** — frontend development
- **Jose Luis Jasso Caballero** — pitch
- **Juan Pablo De la Cruz Rangel** — original detection model concept and prototype

---

Built for HackMTY 2026 / Altur Challenge.