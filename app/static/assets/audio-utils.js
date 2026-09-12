/* Pure helpers shared by the browser and dependency-free Node tests. */
(function (root, factory) {
  "use strict";
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.AlturAudio = Object.freeze(factory());
})(typeof window === "undefined" ? {} : window, function () {
  "use strict";
  const MAX_AUDIO_BYTES = 10 * 1024 * 1024;
  const REQUEST_TIMEOUT_MS = 150000;

  function parseWav(buffer) {
    if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 44) {
      throw new Error("El archivo está vacío o no contiene una cabecera WAV completa.");
    }
    if (buffer.byteLength > MAX_AUDIO_BYTES) throw new Error("El archivo supera el límite de 10 MiB del servidor.");
    const view = new DataView(buffer);
    const tag = (offset) => String.fromCharCode(...new Uint8Array(buffer, offset, 4));
    if (tag(0) !== "RIFF" || tag(8) !== "WAVE") {
      throw new Error("El contenido no es un WAV RIFF válido. Cambiar la extensión no convierte el audio.");
    }
    const end = view.getUint32(4, true) + 8;
    if (end > buffer.byteLength || end < 44) throw new Error("La cabecera indica un WAV incompleto o dañado.");
    let format = null;
    let data = null;
    let offset = 12;
    while (offset + 8 <= end) {
      const id = tag(offset);
      const size = view.getUint32(offset + 4, true);
      const start = offset + 8;
      if (start + size > end) throw new Error("El WAV contiene un bloque incompleto.");
      if (id === "fmt ") {
        if (format || size < 16) throw new Error("El formato del WAV no es válido.");
        let encoding = view.getUint16(start, true);
        const channels = view.getUint16(start + 2, true);
        const sampleRate = view.getUint32(start + 4, true);
        const byteRate = view.getUint32(start + 8, true);
        const blockAlign = view.getUint16(start + 12, true);
        const bitsPerSample = view.getUint16(start + 14, true);
        if (encoding === 0xfffe) {
          if (size < 40 || view.getUint16(start + 16, true) < 22) throw new Error("El WAV extensible está incompleto.");
          const subtype = view.getUint32(start + 24, true);
          const tail = [0, 0, 16, 0, 128, 0, 0, 170, 0, 56, 155, 113];
          if (!tail.every((byte, i) => view.getUint8(start + 28 + i) === byte)) {
            throw new Error("Esta demo no admite ese códec WAV. Usa el PCM de los archivos del reto.");
          }
          encoding = subtype;
        }
        if (channels !== 2) throw new Error(`Se necesitan 2 canales: interlocutor y agente. El archivo tiene ${channels}.`);
        if (sampleRate !== 8000) throw new Error(`Se necesitan 8 kHz. El archivo está a ${sampleRate.toLocaleString("es-MX")} Hz. No se remuestrea automáticamente.`);
        const supported = encoding === 1 && [8, 16, 24, 32].includes(bitsPerSample)
          || encoding === 3 && [32, 64].includes(bitsPerSample);
        if (!supported) throw new Error("Esta demo admite WAV PCM o IEEE float sin compresión. Usa el WAV original del reto.");
        const expectedAlign = channels * bitsPerSample / 8;
        if (blockAlign !== expectedAlign || byteRate !== sampleRate * blockAlign) {
          throw new Error("La cabecera del WAV contiene tamaños de muestra inconsistentes.");
        }
        format = { encoding, channels, sampleRate, bitsPerSample, blockAlign };
      } else if (id === "data") {
        if (data) throw new Error("El WAV contiene varios bloques de audio. Usa un WAV convencional de un solo bloque.");
        data = { dataOffset: start, dataBytes: size };
      }
      offset = start + size + (size % 2);
    }
    if (!format || !data) throw new Error("Falta el formato o el bloque de audio del WAV.");
    if (data.dataBytes % format.blockAlign !== 0) throw new Error("El audio termina con una muestra incompleta.");
    const frames = data.dataBytes / format.blockAlign;
    if (frames < 160) throw new Error("El clip necesita al menos 20 ms de audio. Para clasificarlo, carga una conversación completa.");
    return { ...format, ...data, frames, duration: frames / format.sampleRate };
  }

  function extractPeaks(buffer, metadata, bins = 280) {
    const view = new DataView(buffer);
    const peaks = [new Float32Array(bins), new Float32Array(bins)];
    const bytes = metadata.bitsPerSample / 8;
    let maximum = 0;
    const read = (offset) => {
      if (metadata.encoding === 3) return bytes === 4 ? view.getFloat32(offset, true) : view.getFloat64(offset, true);
      if (bytes === 1) return (view.getUint8(offset) - 128) / 128;
      if (bytes === 2) return view.getInt16(offset, true) / 32768;
      if (bytes === 4) return view.getInt32(offset, true) / 2147483648;
      let n = view.getUint8(offset) | (view.getUint8(offset + 1) << 8) | (view.getUint8(offset + 2) << 16);
      if (n & 0x800000) n -= 0x1000000;
      return n / 8388608;
    };
    for (let frame = 0; frame < metadata.frames; frame++) {
      const bin = Math.min(bins - 1, Math.floor(frame * bins / metadata.frames));
      for (let channel = 0; channel < 2; channel++) {
        const sample = read(metadata.dataOffset + frame * metadata.blockAlign + channel * bytes);
        if (!Number.isFinite(sample)) throw new Error("El WAV contiene muestras no finitas y no se enviará al servidor.");
        const magnitude = Math.abs(sample);
        maximum = Math.max(maximum, magnitude);
        peaks[channel][bin] = Math.max(peaks[channel][bin], magnitude);
      }
    }
    return { channels: peaks, maximum };
  }

  function validateResponse(payload) {
    if (!payload || Array.isArray(payload) || typeof payload !== "object" || typeof payload.is_synthetic !== "boolean") {
      throw new Error("La respuesta del servidor no contiene is_synthetic como booleano.");
    }
    const confidence = payload.confidence;
    if (confidence !== undefined && confidence !== null
      && (typeof confidence !== "number" || !Number.isFinite(confidence) || confidence < 0 || confidence > 1)) {
      throw new Error("El servidor devolvió una confianza inválida. Debe ser un número entre 0 y 1.");
    }
    return { is_synthetic: payload.is_synthetic, confidence: confidence == null ? null : confidence };
  }

  function toBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    const chunks = [];
    for (let i = 0; i < bytes.length; i += 16384) {
      chunks.push(String.fromCharCode(...bytes.subarray(i, i + 16384)));
    }
    return btoa(chunks.join(""));
  }

  async function postAudio(buffer, options = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), options.timeoutMs ?? REQUEST_TIMEOUT_MS);
    try {
      const response = await (options.fetchImpl || fetch)("/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json", "Accept": "application/json" },
        body: JSON.stringify({ audio: toBase64(buffer) }),
        signal: controller.signal,
        cache: "no-store"
      });
      const text = await response.text();
      let payload;
      try { payload = JSON.parse(text); } catch {
        throw new Error(`El servidor devolvió ${response.ok ? "una respuesta que no es JSON" : `un error HTTP ${response.status}`}. Revisa la disponibilidad del backend.`);
      }
      if (!response.ok) {
        const detail = typeof payload?.error === "string" ? ` ${payload.error.slice(0, 240)}` : "";
        throw new Error(`No se pudo analizar la llamada (HTTP ${response.status}).${detail}`);
      }
      const prediction = validateResponse(payload);
      return { raw: payload, prediction };
    } catch (error) {
      if (error.name === "AbortError") {
        throw new Error("Se agotó el tiempo de espera del navegador. El servidor podría seguir procesando; comprueba su estado antes de reintentar.");
      }
      if (error instanceof TypeError) throw new Error("No se pudo conectar con el servidor. Revisa la conexión y que la página se sirva desde el mismo Flask.");
      throw error;
    } finally { clearTimeout(timer); }
  }

  function formatTime(seconds) {
    const n = Number.isFinite(seconds) ? Math.max(0, Math.floor(seconds)) : 0;
    return `${Math.floor(n / 60)}:${String(n % 60).padStart(2, "0")}`;
  }

  return { MAX_AUDIO_BYTES, REQUEST_TIMEOUT_MS, parseWav, extractPeaks, validateResponse, toBase64, postAudio, formatTime };
});
