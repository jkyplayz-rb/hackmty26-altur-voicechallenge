from flask import Flask, request, jsonify

from detection.behavioral import predict
from validation import decode_audio, AudioValidationError

app = Flask(__name__)


@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': 'Missing or invalid JSON body'}), 400

    try:
        audio_bytes = decode_audio(data)
    except AudioValidationError as e:
        return jsonify({'error': str(e)}), 400

    is_synthetic, confidence = predict(audio_bytes)

    if is_synthetic is None:
        return jsonify({
            'is_synthetic': False,
            'confidence': 0.5,
        })

    return jsonify({
        'is_synthetic': is_synthetic,
        'confidence': confidence,
    })


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(debug=True, port=5000)