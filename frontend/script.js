(() => {
  "use strict";

  const MAX_UPLOAD_BYTES = 4 * 1024 * 1024;
  const ACCEPTED_TYPES = /^image\/(png|jpe?g|webp|bmp|gif)$/i;
  const HEALTH_RETRY_MS = 12000;

  const $ = (id) => document.getElementById(id);

  const el = {
    header: $("siteHeader"),
    menuButton: $("menuButton"),
    mobileMenu: $("mobileMenu"),
    heroStatus: $("heroStatus"),
    demoModelDot: $("demoModelDot"),
    demoModelStatus: $("demoModelStatus"),
    dropzone: $("dropzone"),
    fileInput: $("fileInput"),
    dropPrompt: $("dropPrompt"),
    previewImage: $("previewImage"),
    fileMeta: $("fileMeta"),
    fileName: $("fileName"),
    clearButton: $("clearButton"),
    sampleList: $("sampleList"),
    invertToggle: $("invertToggle"),
    predictButton: $("predictButton"),
    predictLabel: $("predictLabel"),
    predictSpinner: $("predictSpinner"),
    elapsedBadge: $("elapsedBadge"),
    resultEmpty: $("resultEmpty"),
    resultLoading: $("resultLoading"),
    resultError: $("resultError"),
    resultContent: $("resultContent"),
    errorMessage: $("errorMessage"),
    resultLabel: $("resultLabel"),
    resultRank: $("resultRank"),
    resultConfidence: $("resultConfidence"),
    preprocessedImage: $("preprocessedImage"),
    topPredictions: $("topPredictions"),
    probabilityBars: $("probabilityBars"),
    preprocessNote: $("preprocessNote"),
    anotherButton: $("anotherButton"),
    year: $("year"),
  };

  const metaBase = (document.querySelector('meta[name="api-base"]')?.content || "").trim();

  const state = {
    file: null,
    objectUrl: null,
    busy: false,
    apiVerified: false,
    apiBase: "",
    healthTimer: null,
    requestId: 0,
  };

  function apiCandidates() {
    if (metaBase) return [metaBase.replace(/\/+$/, "")];

    const bases = [];
    const { protocol, hostname, port } = window.location;

    if (protocol === "http:" || protocol === "https:") bases.push("");
    if (hostname === "localhost" || hostname === "127.0.0.1" || protocol === "file:") {
      bases.push("http://localhost:5000", "http://127.0.0.1:5000");
    }
    if (!bases.length) bases.push("http://localhost:5000", "http://127.0.0.1:5000");

    return [...new Set(bases)];
  }

  function fetchWithTimeout(url, options = {}, timeout = 20000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    return fetch(url, { ...options, signal: controller.signal }).finally(() => clearTimeout(timer));
  }

  async function probeHealth() {
    for (const base of apiCandidates()) {
      try {
        const response = await fetchWithTimeout(`${base}/api/health`, { headers: { Accept: "application/json" } }, 3000);
        if (!response.ok) continue;
        const data = await response.json();
        if (!data || data.success !== true) continue;
        state.apiBase = base;
        state.apiVerified = true;
        renderHealth(data);
        return data;
      } catch (_error) {
        continue;
      }
    }

    state.apiVerified = false;
    renderHealth(null);
    return null;
  }

  function scheduleHealthRetry() {
    if (state.healthTimer) clearInterval(state.healthTimer);
    state.healthTimer = setInterval(async () => {
      if (state.apiVerified) {
        clearInterval(state.healthTimer);
        state.healthTimer = null;
        return;
      }
      await probeHealth();
    }, HEALTH_RETRY_MS);
  }

  function renderHealth(data) {
    const reachable = Boolean(data);
    const ready = Boolean(data && data.model_loaded);

    const heroDot = el.heroStatus?.querySelector("i");
    const heroText = el.heroStatus?.querySelector("span");

    if (heroDot) {
      heroDot.className = `inline-block h-1.5 w-1.5 rounded-full ${
        ready ? "bg-mint" : reachable ? "bg-coral" : "bg-slate-500"
      }${ready ? " pulse-dot" : ""}`;
    }
    if (heroText) {
      heroText.textContent = ready ? "model ready" : reachable ? "model unavailable" : "api offline";
    }

    if (el.demoModelDot) {
      el.demoModelDot.className = `h-2 w-2 rounded-full ${
        ready ? "bg-mint" : reachable ? "bg-coral" : "bg-slate-400"
      }`;
    }
    if (el.demoModelStatus) {
      el.demoModelStatus.textContent = ready
        ? "CNN model ready · 10 classes"
        : reachable
          ? "Backend reachable, model not loaded"
          : "Backend offline — start Flask on port 5000";
    }

    const title = data?.model_error || (ready ? "Model loaded and warmed up." : "Backend is not reachable yet.");
    if (el.heroStatus) el.heroStatus.title = title;
    if (el.demoModelStatus) el.demoModelStatus.title = title;
  }

  function setStatus(message, tone) {
    if (!el.demoModelDot || !el.demoModelStatus) return;
    el.demoModelDot.className = `h-2 w-2 rounded-full ${
      tone === "ok" ? "bg-mint" : tone === "warn" ? "bg-coral" : "bg-slate-400"
    }`;
    el.demoModelStatus.textContent = message;
  }

  function showPanel(name) {
    const panels = [
      ["empty", el.resultEmpty],
      ["loading", el.resultLoading],
      ["error", el.resultError],
      ["content", el.resultContent],
    ];

    for (const [key, node] of panels) {
      if (!node) continue;
      const active = key === name;
      node.classList.toggle("hidden", !active);
      if (key === "loading" || key === "error") node.classList.toggle("flex", active);
    }
  }

  function setBusy(busy) {
    state.busy = busy;
    if (el.predictButton) el.predictButton.disabled = busy || !state.file;
    if (el.predictLabel) el.predictLabel.textContent = busy ? "Running inference…" : "Predict with CNN";
    if (el.predictSpinner) el.predictSpinner.classList.toggle("hidden", !busy);
    if (el.sampleList) {
      el.sampleList.querySelectorAll(".sample-button").forEach((button) => {
        button.disabled = busy;
        button.classList.toggle("opacity-60", busy);
        button.classList.toggle("cursor-not-allowed", busy);
      });
    }
    if (el.dropzone) el.dropzone.classList.toggle("pointer-events-none", busy);
  }

  function formatPercent(value) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return "—";
    if (numeric >= 10 || numeric === 0) return `${parseFloat(numeric.toFixed(1))}%`;
    return `${parseFloat(numeric.toFixed(2))}%`;
  }

  function clearSampleSelection() {
    if (!el.sampleList) return;
    el.sampleList.querySelectorAll(".sample-button.active").forEach((button) => {
      button.classList.remove("active");
      button.setAttribute("aria-pressed", "false");
    });
  }

  function revokeObjectUrl() {
    if (state.objectUrl) {
      URL.revokeObjectURL(state.objectUrl);
      state.objectUrl = null;
    }
  }

  function selectFile(file, label, options = {}) {
    if (!file) return;

    if (!ACCEPTED_TYPES.test(file.type || "")) {
      setStatus("Unsupported file type — use PNG, JPG or WebP", "warn");
      return;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      setStatus("Image is larger than 4 MB", "warn");
      return;
    }

    revokeObjectUrl();
    state.file = file;
    state.objectUrl = URL.createObjectURL(file);

    if (el.previewImage) {
      el.previewImage.src = state.objectUrl;
      el.previewImage.classList.remove("hidden");
    }
    if (el.dropPrompt) el.dropPrompt.classList.add("hidden");
    if (el.fileName) el.fileName.textContent = label || file.name || "image.png";
    if (el.fileMeta) {
      el.fileMeta.classList.remove("hidden");
      el.fileMeta.classList.add("flex");
    }
    if (el.predictButton) el.predictButton.disabled = false;

    if (options.autoPredict) {
      predict();
    } else {
      if (el.elapsedBadge) el.elapsedBadge.classList.add("hidden");
      showPanel("empty");
    }
  }

  function clearSelection() {
    revokeObjectUrl();
    state.file = null;

    if (el.fileInput) el.fileInput.value = "";
    if (el.previewImage) {
      el.previewImage.src = "";
      el.previewImage.classList.add("hidden");
    }
    if (el.dropPrompt) el.dropPrompt.classList.remove("hidden");
    if (el.fileMeta) {
      el.fileMeta.classList.add("hidden");
      el.fileMeta.classList.remove("flex");
    }
    if (el.predictButton) el.predictButton.disabled = true;
    if (el.elapsedBadge) el.elapsedBadge.classList.add("hidden");
    clearSampleSelection();
    showPanel("empty");
    setStatus(state.apiVerified ? "CNN model ready · 10 classes" : "Checking backend…", state.apiVerified ? "ok" : "idle");
  }

  function showError(message) {
    if (el.errorMessage) el.errorMessage.textContent = message;
    if (el.elapsedBadge) el.elapsedBadge.classList.add("hidden");
    showPanel("error");
  }

  function createBar(item, index, compact = false) {
    const row = document.createElement("div");
    row.className = compact ? "flex items-center gap-2 text-xs" : "flex items-center gap-2 text-[11px]";

    const name = document.createElement("span");
    name.className = `shrink-0 truncate ${
      compact ? "w-24" : "w-20"
    } ${index === 0 ? "font-semibold text-mint" : "text-slate-300"}`;
    name.textContent = item.label;

    const track = document.createElement("div");
    track.className = "h-2 min-w-0 flex-1 overflow-hidden rounded-full bg-white/10";

    const fill = document.createElement("div");
    fill.className = `bar-fill h-full rounded-full ${index === 0 ? "bg-mint" : "bg-slate-500"}`;
    fill.style.width = "0%";
    track.appendChild(fill);

    const value = document.createElement("span");
    value.className = `mono shrink-0 text-right ${compact ? "w-14 text-mint" : "w-12 text-slate-400"}`;
    value.textContent = formatPercent(item.percent);

    row.append(name, track, value);
    requestAnimationFrame(() => {
      fill.style.width = `${Math.max(Number(item.percent) || 0, 1)}%`;
    });
    return row;
  }

  function renderResult(data) {
    const prediction = data.prediction || {};
    const probabilities = Array.isArray(data.probabilities) ? data.probabilities : [];
    const ranked = [...probabilities].sort((a, b) => b.probability - a.probability);
    const top = ranked.slice(0, 3);
    const preprocessing = data.preprocessing || {};

    if (el.resultLabel) el.resultLabel.textContent = prediction.label || "Unknown";
    if (el.resultConfidence) el.resultConfidence.textContent = formatPercent(prediction.confidence_percent);

    const runnerUp = ranked[1];
    if (el.resultRank) {
      el.resultRank.textContent = runnerUp
        ? `Runner-up: ${runnerUp.label} · ${formatPercent(runnerUp.percent)}`
        : "Top-1 of 10 classes";
    }

    if (el.preprocessedImage && preprocessing.preview) {
      el.preprocessedImage.src = preprocessing.preview;
    }

    if (el.topPredictions) {
      el.topPredictions.replaceChildren();
      top.forEach((item, index) => el.topPredictions.appendChild(createBar(item, index, true)));
      if (!top.length) {
        const empty = document.createElement("p");
        empty.className = "text-xs text-slate-400";
        empty.textContent = "No probabilities returned.";
        el.topPredictions.appendChild(empty);
      }
    }

    if (el.probabilityBars) {
      el.probabilityBars.replaceChildren();
      ranked.forEach((item, index) => el.probabilityBars.appendChild(createBar(item, index)));
    }

    if (el.preprocessNote) {
      const confidence = Number(prediction.confidence || 0);
      const inverted = preprocessing.inverted
        ? "Background auto-inverted to match the light-on-dark training data."
        : "Kept as captured — no inversion needed.";
      const note =
        confidence < 0.5
          ? `${inverted} Low confidence: this image may sit outside the Fashion-MNIST distribution.`
          : inverted;
      el.preprocessNote.textContent = `Model input 28 × 28 grayscale · ${note}`;
    }

    if (el.elapsedBadge) {
      el.elapsedBadge.textContent = `${data.elapsed_ms ?? "—"} ms`;
      el.elapsedBadge.classList.remove("hidden");
    }

    showPanel("content");
  }

  async function ensureApi() {
    if (state.apiVerified) return true;
    const data = await probeHealth();
    return Boolean(data);
  }

  async function predict() {
    if (!state.file || state.busy) return;

    setBusy(true);
    showPanel("loading");
    setStatus("Running inference…", "idle");

    const requestId = ++state.requestId;

    try {
      const reachable = await ensureApi();
      if (!reachable) {
        throw new Error("Cannot reach the Flask backend. Start it with `python backend/app.py` and reload.");
      }

      const form = new FormData();
      form.append("image", state.file, state.file.name || "image.png");
      form.append("invert", el.invertToggle && el.invertToggle.checked ? "auto" : "false");

      const response = await fetchWithTimeout(
        `${state.apiBase}/api/predict`,
        { method: "POST", body: form, headers: { Accept: "application/json" } },
        45000,
      );

      let data = null;
      try {
        data = await response.json();
      } catch (_error) {
        data = null;
      }

      if (!response.ok || !data || data.success !== true) {
        throw new Error(data?.error || `Prediction request failed (HTTP ${response.status}).`);
      }

      if (requestId !== state.requestId) return;

      state.apiVerified = true;
      renderResult(data);
      setStatus("CNN model ready · 10 classes", "ok");
    } catch (error) {
      if (requestId !== state.requestId) return;
      const message =
        error?.name === "AbortError"
          ? "The request timed out. The model may still be warming up — try again."
          : error?.message || "Something went wrong while classifying the image.";
      showError(message);
      setStatus("Prediction failed — check the backend", "warn");
    } finally {
      if (requestId === state.requestId) setBusy(false);
    }
  }

  async function useSample(button) {
    if (state.busy) return;

    const src = button.dataset.src;
    const label = button.dataset.label || "sample";
    if (!src) return;

    clearSampleSelection();
    button.classList.add("active");
    button.setAttribute("aria-pressed", "true");

    setBusy(true);
    showPanel("loading");
    setStatus("Loading sample…", "idle");

    try {
      const response = await fetchWithTimeout(new URL(src, document.baseURI).href, { cache: "force-cache" }, 10000);
      if (!response.ok) throw new Error("sample");
      const blob = await response.blob();
      const fileName = src.split("/").pop() || "sample.png";
      const file = new File([blob], fileName, { type: blob.type || "image/png" });

      setBusy(false);
      selectFile(file, `${label} · ${fileName}`, { autoPredict: true });
    } catch (_error) {
      setBusy(false);
      showError("Could not load the sample image. Open the app through the Flask server at http://localhost:5000.");
      setStatus("Sample failed to load", "warn");
    }
  }

  function bindHeader() {
    const onScroll = () => {
      if (!el.header) return;
      el.header.classList.toggle("is-scrolled", window.scrollY > 24);
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });

    if (!el.menuButton || !el.mobileMenu) return;

    const closeMenu = () => {
      el.mobileMenu.classList.add("hidden");
      el.menuButton.setAttribute("aria-expanded", "false");
    };

    el.menuButton.addEventListener("click", () => {
      const willOpen = el.mobileMenu.classList.contains("hidden");
      el.mobileMenu.classList.toggle("hidden", !willOpen);
      el.menuButton.setAttribute("aria-expanded", String(willOpen));
    });

    el.mobileMenu.querySelectorAll("a").forEach((link) => link.addEventListener("click", closeMenu));
    window.addEventListener("resize", () => {
      if (window.innerWidth >= 768) closeMenu();
    });
  }

  function bindUpload() {
    if (!el.dropzone || !el.fileInput) return;

    el.fileInput.addEventListener("change", () => {
      const file = el.fileInput.files && el.fileInput.files[0];
      if (file) {
        clearSampleSelection();
        selectFile(file);
      }
    });

    el.dropzone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        el.fileInput.click();
      }
    });

    let dragDepth = 0;

    el.dropzone.addEventListener("dragenter", (event) => {
      event.preventDefault();
      dragDepth += 1;
      el.dropzone.classList.add("dropzone-active");
    });

    el.dropzone.addEventListener("dragover", (event) => {
      event.preventDefault();
      if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
    });

    el.dropzone.addEventListener("dragleave", (event) => {
      event.preventDefault();
      dragDepth = Math.max(0, dragDepth - 1);
      if (dragDepth === 0) el.dropzone.classList.remove("dropzone-active");
    });

    el.dropzone.addEventListener("drop", (event) => {
      event.preventDefault();
      dragDepth = 0;
      el.dropzone.classList.remove("dropzone-active");
      const file = event.dataTransfer?.files?.[0];
      if (file) {
        clearSampleSelection();
        selectFile(file);
      }
    });

    window.addEventListener("dragover", (event) => event.preventDefault());
    window.addEventListener("drop", (event) => event.preventDefault());

    el.clearButton?.addEventListener("click", () => {
      clearSelection();
      el.dropzone.focus();
    });
  }

  function bindDemo() {
    el.sampleList?.querySelectorAll(".sample-button").forEach((button) => {
      button.setAttribute("aria-pressed", "false");
      button.addEventListener("click", () => useSample(button));
    });

    el.predictButton?.addEventListener("click", predict);

    el.anotherButton?.addEventListener("click", () => {
      state.requestId += 1;
      clearSelection();
      el.dropzone?.scrollIntoView({ behavior: "smooth", block: "center" });
    });

    el.invertToggle?.addEventListener("change", () => {
      if (state.file) setStatus("Settings updated — run the prediction again", "idle");
    });
  }

  function init() {
    if (el.year) el.year.textContent = String(new Date().getFullYear());

    bindHeader();
    bindUpload();
    bindDemo();
    showPanel("empty");
    setBusy(false);
    setStatus("Checking backend…", "idle");

    probeHealth().then((data) => {
      if (!data) {
        setStatus("Backend offline — start Flask on port 5000", "warn");
        scheduleHealthRetry();
      }
    });

    window.addEventListener("beforeunload", revokeObjectUrl);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
