import base64
import io

from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json(silent=True)
    if not data or 'audio' not in data:
        return jsonify({'error': 'Missing "audio" field (base64 WAV)'}), 400

    try:
        audio_bytes = base64.b64decode(data['audio'])
    except Exception:
        return jsonify({'error': 'Invalid base64 audio data'}), 400

    # --- Placeholder logic ---
    is_synthetic = False
    confidence = 0.5

    return jsonify({
        'is_synthetic': is_synthetic,
        'confidence': confidence,
    })


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(debug=True, port=5000)