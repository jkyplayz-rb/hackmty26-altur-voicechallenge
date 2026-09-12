(function () {
  "use strict";
  const api = window.AlturAudio;
  const $ = (id) => document.getElementById(id);
  const input = $("file-input");
  const dropzone = $("dropzone");
  const player = $("audio-player");
  const resultCard = $("result-card");
  const onServer = /^https?:$/.test(location.protocol) && document.documentElement.dataset.preview !== "true";
  let selected = null;
  let fileURL = null;
  let reading = false;
  let analyzing = false;
  let selectionVersion = 0;
  let lastReport = null;
  let healthChecking = false;

  function syncControls() {
    const locked = reading || analyzing;
    input.disabled = locked;
    dropzone.setAttribute("aria-disabled", String(locked));
    dropzone.tabIndex = locked ? -1 : 0;
    $("change-file").disabled = locked;
    $("remove-file").disabled = locked;
    $("analyze-button").disabled = locked || !selected || !onServer;
    $("analyze-label").textContent = analyzing ? "Analizando…" : reading ? "Leyendo archivo…" : "Analizar llamada";
    $("play-button").disabled = !selected;
    $("seek").disabled = !selected;
  }

  function showNotice(id, message) {
    $(id).textContent = message;
    $(id).hidden = !message;
  }

  function clearResult() {
    lastReport = null;
    resultCard.dataset.state = "empty";
    resultCard.setAttribute("aria-busy", "false");
    $("result-badge").textContent = "En espera";
    $("result-eyebrow").textContent = "EL OTRO LADO DE LA LÍNEA";
    $("result-title").textContent = "Una llamada. Una nueva lectura.";
    $("result-description").textContent = selected ? "Tu audio está listo. Inicia el análisis cuando quieras." : "Carga un audio para explorar las señales de una voz humana o sintética.";
    $("result-symbol").querySelector("use").setAttribute("href", "#i-signal");
    $("confidence-value").textContent = "—";
    $("confidence-fill").style.width = "0%";
    $("confidence-meter").removeAttribute("aria-valuenow");
    $("confidence-meter").removeAttribute("role");
    $("confidence-meter").setAttribute("aria-hidden", "true");
    $("confidence-meter").setAttribute("aria-valuetext", "Sin resultado");
    $("request-time").textContent = "—";
    $("time-unit").textContent = "";
    $("result-details").hidden = true;
    $("response-json").textContent = "";
    $("result-details").querySelector("details").open = false;
  }

  function releaseAudio() {
    player.pause();
    player.removeAttribute("src");
    player.load();
    if (fileURL) URL.revokeObjectURL(fileURL);
    fileURL = null;
    $("play-button").querySelector("use").setAttribute("href", "#i-play");
    $("play-button").setAttribute("aria-label", "Reproducir llamada");
    $("current-time").textContent = "0:00";
    $("total-time").textContent = "0:00";
    $("seek").value = "0";
    $("seek").setAttribute("aria-valuetext", "Sin audio");
  }

  function resetSelection() {
    selected = null;
    selectionVersion++;
    releaseAudio();
    input.value = "";
    $("selected-file").hidden = true;
    dropzone.hidden = false;
    $("waveforms").classList.remove("has-audio");
    $("signal-source").textContent = "Sin archivo";
    showNotice("file-error", "");
    showNotice("player-notice", "");
    clearResult();
    syncControls();
    drawWaveforms();
  }

  function chooseFile() { if (!reading && !analyzing) input.click(); }
  dropzone.addEventListener("click", chooseFile);
  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") { event.preventDefault(); chooseFile(); }
  });
  $("change-file").addEventListener("click", chooseFile);
  $("remove-file").addEventListener("click", () => {
    if (reading || analyzing) return;
    resetSelection();
    dropzone.focus();
  });
  input.addEventListener("change", () => {
    const files = Array.from(input.files || []);
    if (files.length) loadFiles(files);
  });

  // The whole audio card accepts drops, including replacing an existing file.
  const audioCard = document.querySelector(".audio-card");
  document.addEventListener("dragover", (event) => event.preventDefault());
  document.addEventListener("drop", (event) => event.preventDefault());
  audioCard.addEventListener("dragover", (event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = reading || analyzing ? "none" : "copy";
    if (!reading && !analyzing) dropzone.classList.add("drag-over");
  });
  audioCard.addEventListener("dragleave", (event) => {
    if (!audioCard.contains(event.relatedTarget)) dropzone.classList.remove("drag-over");
  });
  audioCard.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("drag-over");
    if (reading || analyzing) return;
    loadFiles(Array.from(event.dataTransfer.files || []));
  });

  async function loadFiles(files) {
    if (reading || analyzing || !files.length) return;
    resetSelection();
    if (files.length !== 1) { showNotice("file-error", "Carga una llamada a la vez. No se ha seleccionado ningún archivo."); return; }
    const file = files[0];
    if (file.size > api.MAX_AUDIO_BYTES) { showNotice("file-error", "El archivo supera el límite de 10 MiB. No se ha enviado al servidor."); return; }
    reading = true;
    syncControls();
    const version = selectionVersion;
    try {
      const buffer = await file.arrayBuffer();
      const metadata = api.parseWav(buffer);
      const peaks = api.extractPeaks(buffer, metadata);
      if (version !== selectionVersion) return;
      selected = { file, buffer, metadata, peaks };
      fileURL = URL.createObjectURL(new Blob([buffer], { type: "audio/wav" }));
      player.src = fileURL;
      player.load();
      $("file-name").textContent = file.name;
      $("file-name").title = file.name;
      const size = file.size >= 1024 * 1024 ? `${(file.size / (1024 * 1024)).toFixed(2)} MiB` : `${(file.size / 1024).toFixed(1)} KiB`;
      $("file-description").textContent = `${size} · ${metadata.bitsPerSample} bits · ${api.formatTime(metadata.duration)} min`;
      $("selected-file").hidden = false;
      dropzone.hidden = true;
      $("waveforms").classList.add("has-audio");
      $("signal-source").textContent = "Amplitud del WAV";
      $("total-time").textContent = api.formatTime(metadata.duration);
      $("seek").setAttribute("aria-valuetext", `0:00 de ${api.formatTime(metadata.duration)}`);
      clearResult();
      drawWaveforms();
    } catch (error) {
      if (version === selectionVersion) showNotice("file-error", error.message || "No se pudo leer el archivo.");
    } finally { reading = false; syncControls(); }
  }

  function drawWaveforms() {
    const progress = selected ? Math.min(1, player.currentTime / selected.metadata.duration) : 0;
    ["caller-wave", "agent-wave"].forEach((id, channel) => {
      const canvas = $(id);
      const { width, height } = canvas.getBoundingClientRect();
      if (!width || !height) return;
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      const context = canvas.getContext("2d");
      if (!context) return;
      context.scale(ratio, ratio);
      context.strokeStyle = "#daddce";
      context.lineWidth = 1;
      context.setLineDash(selected ? [] : [2, 3]);
      context.beginPath(); context.moveTo(0, height / 2); context.lineTo(width, height / 2); context.stroke();
      if (!selected) return;
      context.setLineDash([]);
      const peaks = selected.peaks.channels[channel];
      const maximum = selected.peaks.maximum || 1;
      const count = Math.max(1, Math.floor(width / 3));
      context.lineWidth = 1.6;
      context.lineCap = "round";
      for (let i = 0; i < count; i++) {
        const start = Math.floor(i * peaks.length / count);
        const end = Math.max(start + 1, Math.floor((i + 1) * peaks.length / count));
        let magnitude = 0;
        for (let p = start; p < end; p++) magnitude = Math.max(magnitude, peaks[p] || 0);
        if (magnitude === 0) continue;
        const amplitude = Math.min(height / 2 - 3, magnitude / maximum * (height / 2 - 3));
        const x = (i + .5) * width / count;
        context.strokeStyle = channel === 0 ? (x < progress * width ? "#a7462e" : "#c89882") : (x < progress * width ? "#596449" : "#a4af8e");
        context.beginPath(); context.moveTo(x, height / 2 - amplitude); context.lineTo(x, height / 2 + amplitude); context.stroke();
      }
      if (progress > 0 && progress < 1) {
        context.strokeStyle = "#555c4d"; context.lineWidth = 1;
        context.beginPath(); context.moveTo(progress * width, 0); context.lineTo(progress * width, height); context.stroke();
      }
    });
  }

  $("play-button").addEventListener("click", async () => {
    if (!selected) return;
    if (!player.paused) { player.pause(); return; }
    try { await player.play(); showNotice("player-notice", ""); }
    catch { showNotice("player-notice", "El navegador no pudo reproducir este WAV. Puedes enviarlo al detector de todas formas."); }
  });
  player.addEventListener("play", () => {
    $("play-button").querySelector("use").setAttribute("href", "#i-pause");
    $("play-button").setAttribute("aria-label", "Pausar llamada");
  });
  player.addEventListener("pause", () => {
    $("play-button").querySelector("use").setAttribute("href", "#i-play");
    $("play-button").setAttribute("aria-label", "Reproducir llamada");
  });
  player.addEventListener("error", () => {
    if (selected) showNotice("player-notice", "La reproducción no está disponible en este navegador; el análisis del WAV sigue habilitado.");
  });
  player.addEventListener("timeupdate", () => {
    if (!selected) return;
    $("current-time").textContent = api.formatTime(player.currentTime);
    $("seek").value = String(Math.min(1000, player.currentTime / selected.metadata.duration * 1000));
    $("seek").setAttribute("aria-valuetext", `${api.formatTime(player.currentTime)} de ${api.formatTime(selected.metadata.duration)}`);
    drawWaveforms();
  });
  $("seek").addEventListener("input", () => {
    if (selected && player.readyState > 0) player.currentTime = Number($("seek").value) / 1000 * selected.metadata.duration;
  });
  if (typeof ResizeObserver !== "undefined") new ResizeObserver(drawWaveforms).observe($("waveforms"));
  else window.addEventListener("resize", drawWaveforms);

  $("analyze-button").addEventListener("click", async () => {
    if (!selected || reading || analyzing || !onServer) return;
    const current = selected;
    clearResult();
    analyzing = true;
    syncControls();
    resultCard.dataset.state = "loading";
    resultCard.setAttribute("aria-busy", "true");
    $("result-badge").textContent = "Analizando";
    $("result-eyebrow").textContent = "LEYENDO LA CONVERSACIÓN";
    $("result-title").textContent = "Escuchando las señales.";
    $("result-description").textContent = "El servidor está procesando tu llamada. Los audios largos pueden tardar más.";
    const started = performance.now();
    const tick = () => {
      const seconds = (performance.now() - started) / 1000;
      $("request-time").textContent = seconds.toFixed(1);
      $("time-unit").textContent = "s";
      if (seconds >= 30) $("result-description").textContent = "Seguimos esperando al servidor. No hace falta volver a enviar el archivo.";
    };
    tick();
    const clock = setInterval(tick, 250);
    try {
      const response = await api.postAudio(current.buffer);
      const totalMs = performance.now() - started;
      const { is_synthetic, confidence } = response.prediction;
      resultCard.dataset.state = is_synthetic ? "synthetic" : "human";
      $("result-badge").textContent = "Completado";
      $("result-eyebrow").textContent = "CLASIFICACIÓN DEL INTERLOCUTOR";
      $("result-title").textContent = is_synthetic ? "Señales de voz sintética." : "Señales de voz humana.";
      $("result-symbol").querySelector("use").setAttribute("href", is_synthetic ? "#i-signal" : "#i-check");
      $("result-description").textContent = confidence === .5 ? "El modelo devolvió 50% de confianza: este resultado es ambiguo. No lo tomes como una identificación concluyente." : "La clasificación corresponde al canal 0. Es una lectura automática, no una verificación de identidad.";
      if (confidence !== null) {
        const percent = confidence * 100;
        $("confidence-value").textContent = `${percent.toLocaleString("es-MX", { maximumFractionDigits: 1 })}%`;
        $("confidence-fill").style.width = `${percent}%`;
        $("confidence-meter").setAttribute("aria-valuenow", String(percent));
        $("confidence-meter").setAttribute("role", "meter");
        $("confidence-meter").removeAttribute("aria-hidden");
        $("confidence-meter").setAttribute("aria-valuetext", `${percent.toFixed(1)} por ciento, puntuación del modelo`);
      } else {
        $("confidence-value").textContent = "No enviada";
        $("confidence-meter").setAttribute("aria-valuetext", "El servidor no envió confianza");
      }
      $("response-json").textContent = JSON.stringify(response.raw, null, 2);
      $("result-details").hidden = false;
      lastReport = {
        analyzed_at: new Date().toISOString(),
        file: { name: current.file.name, size_bytes: current.file.size, duration_s: current.metadata.duration, sample_rate_hz: current.metadata.sampleRate, channels: current.metadata.channels },
        browser_request_ms: Math.round(totalMs),
        response: response.raw,
        notes: "Tiempo medido en el navegador, incluida la preparación y la red; no es tiempo de inferencia. La confianza no se presenta como probabilidad calibrada. No contiene audio."
      };
    } catch (error) {
      resultCard.dataset.state = "error";
      $("result-badge").textContent = "Sin resultado";
      $("result-eyebrow").textContent = "EL ANÁLISIS NO SE COMPLETÓ";
      $("result-title").textContent = "No recibimos una lectura.";
      $("result-description").textContent = error.message || "Ocurrió un error inesperado. Revisa el servidor antes de reintentar.";
    } finally {
      clearInterval(clock);
      $("request-time").textContent = ((performance.now() - started) / 1000).toFixed(2);
      $("time-unit").textContent = "s";
      analyzing = false;
      resultCard.setAttribute("aria-busy", "false");
      syncControls();
    }
  });

  $("download-result").addEventListener("click", () => {
    if (!lastReport) return;
    const blob = new Blob([JSON.stringify(lastReport, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const safeName = lastReport.file.name.replace(/\.wav$/i, "").replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 90) || "llamada";
    a.download = `${safeName}_resultado.json`;
    document.body.appendChild(a);
    a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });

  async function checkHealth() {
    if (healthChecking || !onServer) return;
    healthChecking = true;
    $("connection").disabled = true;
    $("connection").dataset.status = "checking";
    $("connection-text").textContent = "Comprobando API";
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 5000);
    try {
      const response = await fetch("/health", { signal: controller.signal, cache: "no-store", headers: { "Accept": "application/json" } });
      const payload = await response.json();
      if (!response.ok || payload?.status !== "ok") throw new Error("Unhealthy");
      $("connection").dataset.status = "ok";
      $("connection-text").textContent = "API accesible";
      $("connection").title = "/health respondió correctamente. Esto no verifica que los modelos estén listos. Pulsa para comprobar otra vez.";
    } catch {
      $("connection").dataset.status = "error";
      $("connection-text").textContent = "API no confirmada";
      $("connection").title = "No se pudo confirmar /health. Puedes volver a comprobar o intentar el análisis.";
    } finally { clearTimeout(timeout); healthChecking = false; $("connection").disabled = false; }
  }
  $("connection").addEventListener("click", checkHealth);
  window.addEventListener("beforeunload", () => { if (fileURL) URL.revokeObjectURL(fileURL); });
  if (!onServer) {
    $("connection-text").textContent = "Vista local";
    $("connection").disabled = true;
    showNotice("environment-notice", "Vista previa sin servidor. Puedes cargar y escuchar un WAV, pero para analizarlo abre /static/index.html desde el mismo Flask que sirve /detect. No uses Live Server para la integración.");
  } else checkHealth();
  syncControls();
  drawWaveforms();
})();
