import os
from flask import Flask, request, jsonify

from detection.behavioral import predict as predict_behavioral
from detection.spectral import predict_spectral
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

    is_synth_beh, conf_beh = predict_behavioral(audio_bytes)
    is_synth_spec, conf_spec = predict_spectral(audio_bytes)

    score_beh = (conf_beh if is_synth_beh else 1.0 - conf_beh) if conf_beh is not None else None
    score_spec = (conf_spec if is_synth_spec else 1.0 - conf_spec) if conf_spec is not None else None

    if score_beh is not None and score_spec is not None:
        final_score = 0.5 * score_beh + 0.5 * score_spec
    elif score_beh is not None:
        final_score = score_beh
    elif score_spec is not None:
        final_score = score_spec
    else:
        final_score = 0.5

    final_is_synthetic = final_score >= 0.5
    final_confidence = final_score if final_is_synthetic else (1.0 - final_score)

    return jsonify({
        'is_synthetic': final_is_synthetic,
        'confidence': round(final_confidence, 4)
    })

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=5000)