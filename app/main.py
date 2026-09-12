from flask import Flask, request, jsonify

# Importar ambos modelos
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

    # 1. Predicción basada en tiempos de respuesta
    is_synth_beh, conf_beh = predict_behavioral(audio_bytes)
    
    # 2. Predicción basada en red neuronal espectral
    is_synth_spec, conf_spec = predict_spectral(audio_bytes)

    # Lógica de ensamble
    score_beh = conf_beh if is_synth_beh else (1.0 - (conf_beh or 0.5))
    score_spec = conf_spec if is_synth_spec else (1.0 - (conf_spec or 0.5))
    
    # Promediar probabilidades si ambos modelos retornaron datos válidos
    if is_synth_beh is not None and is_synth_spec is not None:
        final_score = (score_beh + score_spec) / 2.0
    else:
        # Fallback al modelo que haya funcionado
        final_score = score_beh if is_synth_beh is not None else (score_spec if is_synth_spec is not None else 0.5)

    final_is_synthetic = final_score > 0.5
    final_confidence = final_score if final_is_synthetic else (1.0 - final_score)

    return jsonify({
        'is_synthetic': final_is_synthetic,
        'confidence': round(final_confidence, 4)
    })

    if __name__ == '__main__':
        app.run(debug=True, port=5000)