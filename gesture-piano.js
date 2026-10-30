const MP_VERSION = "1.2.0";
const MP_CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
const SCALE = [60, 62, 64, 65, 67, 69, 71];
const ROMAN = ["I", "ii", "iii", "IV", "V", "vi", "vii°"];
const TRIADS = ["C", "Dm", "Em", "F", "G", "Am", "Bdim"];
const QUALITIES = ["大三和弦", "小三和弦", "小三和弦", "大三和弦", "大三和弦", "小三和弦", "減三和弦"];
const PITCH_NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];
const SAMPLE_NAMES = ["C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B"];
const CONNECTIONS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
const GUITAR_SOURCES = [
  [45, "A2"], [59, "B3"], [71, "B4"], [85, "Cs6"], [36, "C2"], [48, "C3"],
  [74, "D5"], [52, "E3"], [64, "E4"], [41, "F2"], [80, "Gs5"], [55, "G3"], [67, "G4"],
];

const $ = (id) => document.getElementById(id);
const video = $("video");
const overlay = $("overlay");
const overlayCtx = overlay.getContext("2d");
const stage = $("stage");
const startButton = $("startButton");
const stopButton = $("stopButton");
const statusMessage = $("statusMessage");
const cameraState = $("cameraState");
const degreeDisplay = $("degreeDisplay");
const chordDisplay = $("chordDisplay");
const chordMeta = $("chordMeta");
const noteDisplay = $("noteDisplay");
const modeButton = $("modeButton");
const instrumentLabel = $("instrumentLabel");
const keyboard = $("keyboard");
const volumeSlider = $("volumeSlider");
const volumeValue = $("volumeValue");
const degreeList = $("degreeList");

let landmarker = null;
let stream = null;
let audioContext = null;
let masterGain = null;
let guitarBus = null;
let pianoSamples = new Map();
let guitarSamples = [];
let activeDegree = null;
let candidateDegree = null;
let candidateFrames = 0;
let lastValidGesture = 0;
let keyboardOverrideUntil = 0;
let instrument = "piano";
let pianoStep = 0;
let guitarStep = 0;
let nextNoteAt = 0;
let lastVideoTime = -1;
let animationId = 0;
let audioTimer = 0;
let running = false;
let lastRawNumber = null;

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function recognizeNumber(p) {
  if (!p || p.length < 21) return null;
  const wrist = p[0];
  const palmWidth = Math.max(distance(p[5], p[17]), 0.001);
  const thumbSpan = distance(p[4], p[5]) / palmWidth;
  const thumbReach = distance(p[4], wrist) / Math.max(distance(p[3], wrist), 0.001);
  const va = { x: p[2].x - p[3].x, y: p[2].y - p[3].y };
  const vb = { x: p[4].x - p[3].x, y: p[4].y - p[3].y };
  const denom = Math.max(Math.hypot(va.x, va.y) * Math.hypot(vb.x, vb.y), 1e-6);
  const thumbAngle = Math.acos(Math.max(-1, Math.min(1, (va.x * vb.x + va.y * vb.y) / denom))) * 180 / Math.PI;
  let thumbState = null;
  if (thumbAngle >= 150 && thumbSpan >= 0.52 && thumbReach >= 1.02) thumbState = true;
  else if (thumbAngle <= 135 || thumbSpan <= 0.42 || thumbReach <= 0.96) thumbState = false;

  const t = thumbState === true;
  const i = distance(p[8], wrist) > distance(p[6], wrist) * 1.10;
  const m = distance(p[12], wrist) > distance(p[10], wrist) * 1.10;
  const r = distance(p[16], wrist) > distance(p[14], wrist) * 1.08;
  const pinky = distance(p[20], wrist) > distance(p[18], wrist) * 1.06;

  const axis = { x: p[17].x - p[5].x, y: p[17].y - p[5].y };
  const axisLength = Math.max(Math.hypot(axis.x, axis.y), 0.001);
  axis.x /= axisLength;
  axis.y /= axisLength;
  const lateral = (point) => (point.x - p[5].x) * axis.x + (point.y - p[5].y) * axis.y;
  const thumbLateral = lateral(p[4]) / palmWidth;

  // Match the user's demonstrated Taiwan-style 3 and 9 hand shapes.
  if (i && m && r && !pinky && thumbLateral > 0.20) return 3;
  if (i && m && r && !pinky && thumbLateral < -0.20) return 9;

  const baseOrder = lateral(p[6]) - lateral(p[10]);
  const tipOrder = lateral(p[8]) - lateral(p[12]);
  const crossed = baseOrder * tipOrder < 0 && Math.abs(baseOrder) > palmWidth * 0.025 && Math.abs(tipOrder) > palmWidth * 0.04 && i && m;
  if (crossed && !r && !pinky) return 9;
  if (t && i && m && !r && !pinky) return 8;
  if (t && i && !m && !r && !pinky) return 7;
  if (t && pinky && !i && !m && !r) return 6;
  if (thumbState === true && i && m && r && pinky) return 5;
  if (thumbState === false && i && m && r && pinky) return 4;
  if (!t && i && m && !r && !pinky) return 2;
  if (!t && i && !m && !r && !pinky) return 1;
  if (thumbState === false && !i && !m && !r && !pinky) return 0;
  return null;
}

function chordForDegree(degree) {
  const start = degree - 1;
  return [0, 2, 4].map((offset) => {
    const index = start + offset;
    return SCALE[index % 7] + 12 * Math.floor(index / 7);
  });
}

function powerNotesForDegree(degree) {
  const root = SCALE[degree - 1];
  return [root, root + 7, root + 12];
}

function pitchLabel(midi) {
  return `${PITCH_NAMES[midi % 12]}${Math.floor(midi / 12) - 1}`;
}

function pianoSamplePath(midi) {
  const octave = Math.floor(midi / 12) - 1;
  return `./assets/piano/${SAMPLE_NAMES[midi % 12]}${octave}.ogg`;
}

async function fetchAndDecode(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`音色載入失敗：${url}`);
  return audioContext.decodeAudioData(await response.arrayBuffer());
}

function createAudioGraph() {
  audioContext = new AudioContext({ latencyHint: "interactive" });
  masterGain = audioContext.createGain();
  masterGain.gain.value = Number(volumeSlider.value) / 100;
  masterGain.connect(audioContext.destination);

  guitarBus = audioContext.createGain();
  guitarBus.gain.value = 1.0;
  const shaper = audioContext.createWaveShaper();
  const curve = new Float32Array(4096);
  for (let index = 0; index < curve.length; index++) {
    const x = (index * 2) / (curve.length - 1) - 1;
    curve[index] = Math.tanh(x * 3.8) / Math.tanh(3.8);
  }
  shaper.curve = curve;
  shaper.oversample = "4x";
  guitarBus.connect(shaper).connect(masterGain);
}

async function loadSounds() {
  const pianoEntries = [];
  for (let midi = 48; midi <= 83; midi++) pianoEntries.push([midi, pianoSamplePath(midi)]);
  const [loadedPiano, loadedGuitar] = await Promise.all([
    Promise.all(pianoEntries.map(async ([midi, path]) => [midi, await fetchAndDecode(path)])),
    Promise.all(GUITAR_SOURCES.map(async ([midi, file]) => ({ midi, buffer: await fetchAndDecode(`./assets/web_guitar/${file}.wav`) }))),
  ]);
  pianoSamples = new Map(loadedPiano);
  guitarSamples = loadedGuitar;
}

function closestGuitarSample(midi) {
  return guitarSamples.reduce((closest, sample) => Math.abs(sample.midi - midi) < Math.abs(closest.midi - midi) ? sample : closest, guitarSamples[0]);
}

function schedulePianoNote(midi, when, volume = 0.82, pan = 0) {
  const buffer = pianoSamples.get(midi);
  if (!buffer) return;
  const source = audioContext.createBufferSource();
  const gain = audioContext.createGain();
  const panner = audioContext.createStereoPanner();
  source.buffer = buffer;
  panner.pan.setValueAtTime(pan, when);
  gain.gain.setValueAtTime(volume, when);
  source.connect(gain).connect(panner).connect(masterGain);
  source.start(when);
  source.stop(when + Math.min(buffer.duration, 3.2));
}

function schedulePowerChord(degree, when, accent) {
  const notes = powerNotesForDegree(degree);
  const levels = [0.68, 0.52, 0.46];
  const pans = [-0.12, 0.08, 0.18];
  notes.forEach((midi, index) => {
    const sample = closestGuitarSample(midi);
    const source = audioContext.createBufferSource();
    const gain = audioContext.createGain();
    const panner = audioContext.createStereoPanner();
    const startAt = when + index * 0.01;
    source.buffer = sample.buffer;
    source.playbackRate.setValueAtTime(2 ** ((midi - sample.midi) / 12), startAt);
    panner.pan.setValueAtTime(pans[index], startAt);
    const peak = levels[index] * accent;
    gain.gain.setValueAtTime(0.0001, startAt);
    gain.gain.linearRampToValueAtTime(peak, startAt + 0.008);
    gain.gain.setValueAtTime(peak * 0.82, startAt + 0.22);
    gain.gain.linearRampToValueAtTime(0.0001, startAt + 0.43);
    source.connect(gain).connect(panner).connect(guitarBus);
    source.start(startAt);
    source.stop(startAt + 0.45);
  });
}

function scheduleMusic() {
  if (!audioContext || activeDegree === null) return;
  const now = audioContext.currentTime;
  const guitar = instrument === "guitar";
  const stepDuration = 60 / (guitar ? 96 : 112) / 2;
  while (nextNoteAt < now + 0.09) {
    if (nextNoteAt < now - 0.08) nextNoteAt = now + 0.015;
    if (guitar) {
      const accent = guitarStep % 2 === 0 ? 1.0 : 0.78;
      schedulePowerChord(activeDegree, nextNoteAt, accent);
      guitarStep = (guitarStep + 1) % 8;
    } else {
      const chord = chordForDegree(activeDegree);
      const pattern = [chord[0], chord[1], chord[2], chord[0] + 12, chord[2], chord[1]];
      const pans = [-0.12, 0.12, -0.05, 0.16, 0.04, -0.16];
      schedulePianoNote(pattern[pianoStep], nextNoteAt, 0.88, pans[pianoStep]);
      if (pianoStep === 0) schedulePianoNote(chord[0] - 12, nextNoteAt, 0.72, 0);
      pianoStep = (pianoStep + 1) % pattern.length;
    }
    nextNoteAt += stepDuration;
  }
}

function setStatus(message, isError = false) {
  statusMessage.textContent = message;
  statusMessage.style.color = isError ? "var(--danger)" : "";
}

function updateDegree(degree) {
  activeDegree = degree;
  pianoStep = 0;
  guitarStep = 0;
  if (audioContext) nextNoteAt = audioContext.currentTime + 0.025;
  renderMusicState();
}

function renderMusicState() {
  instrumentLabel.textContent = instrument === "piano" ? "鋼琴分解和弦" : "電吉他 Power Chord";
  modeButton.textContent = instrument === "piano" ? "切到電吉他" : "切到鋼琴";
  if (activeDegree === null) {
    degreeDisplay.textContent = "—";
    degreeDisplay.classList.add("idle");
    chordDisplay.textContent = "等待手勢";
    chordMeta.textContent = "保持一個 1–7 手勢開始演奏";
    noteDisplay.textContent = "C 大調 · WASM 手勢辨識";
    document.querySelectorAll(".degree-cell").forEach((cell) => cell.classList.remove("active"));
    drawKeyboard(null);
    return;
  }
  const index = activeDegree - 1;
  degreeDisplay.textContent = String(activeDegree);
  degreeDisplay.classList.remove("idle");
  if (instrument === "piano") {
    const chord = chordForDegree(activeDegree);
    chordDisplay.textContent = `${ROMAN[index]}   ${TRIADS[index]}`;
    chordMeta.textContent = `C 大調第 ${activeDegree} 級 · ${QUALITIES[index]}`;
    noteDisplay.textContent = chord.map(pitchLabel).concat(pitchLabel(chord[0] + 12)).join("  →  ");
    drawKeyboard(chord);
  } else {
    const notes = powerNotesForDegree(activeDegree);
    chordDisplay.textContent = `${ROMAN[index]}   ${PITCH_NAMES[SCALE[index] % 12]}5`;
    chordMeta.textContent = `C 大調第 ${activeDegree} 級 · POWER CHORD`;
    noteDisplay.textContent = `${pitchLabel(notes[0])}  →  ${pitchLabel(notes[1])}  →  ${pitchLabel(notes[2])}`;
    drawKeyboard(notes);
  }
  document.querySelectorAll(".degree-cell").forEach((cell) => cell.classList.toggle("active", Number(cell.dataset.degree) === activeDegree));
}

function createDegreeList() {
  degreeList.replaceChildren();
  for (let index = 0; index < 7; index++) {
    const cell = document.createElement("div");
    cell.className = "degree-cell";
    cell.dataset.degree = String(index + 1);
    cell.innerHTML = `<b>${index + 1} · ${ROMAN[index]}</b><span>${TRIADS[index]}</span>`;
    degreeList.append(cell);
  }
}

function createKeyboard() {
  keyboard.replaceChildren();
  let whiteCount = 0;
  const whitePitchClasses = new Set([0, 2, 4, 5, 7, 9, 11]);
  for (let midi = 60; midi <= 83; midi++) {
    const pitch = midi % 12;
    if (whitePitchClasses.has(pitch)) {
      const key = document.createElement("div");
      key.className = "white-key";
      key.dataset.pitch = String(pitch);
      if (pitch === 0) key.innerHTML = `<small>C${Math.floor(midi / 12) - 1}</small>`;
      keyboard.append(key);
      whiteCount++;
    } else {
      const black = document.createElement("div");
      black.className = "black-key";
      black.dataset.pitch = String(pitch);
      black.style.left = `${(whiteCount / 14) * 100}%`;
      keyboard.append(black);
    }
  }
}

function drawKeyboard(notes) {
  const pitchClasses = new Set((notes || []).map((note) => note % 12));
  keyboard.querySelectorAll("[data-pitch]").forEach((key) => key.classList.toggle("active", pitchClasses.has(Number(key.dataset.pitch))));
}

function drawLandmarks(landmarks) {
  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!width || !height) return;
  if (overlay.width !== width || overlay.height !== height) {
    overlay.width = width;
    overlay.height = height;
  }
  overlayCtx.clearRect(0, 0, width, height);
  overlayCtx.lineWidth = Math.max(2, width / 320);
  overlayCtx.strokeStyle = "#a8cba9";
  overlayCtx.fillStyle = "#f1efe7";
  overlayCtx.lineCap = "round";
  for (const [a, b] of CONNECTIONS) {
    overlayCtx.beginPath();
    overlayCtx.moveTo(landmarks[a].x * width, landmarks[a].y * height);
    overlayCtx.lineTo(landmarks[b].x * width, landmarks[b].y * height);
    overlayCtx.stroke();
  }
  for (const point of landmarks) {
    overlayCtx.beginPath();
    overlayCtx.arc(point.x * width, point.y * height, Math.max(3, width / 190), 0, Math.PI * 2);
    overlayCtx.fill();
  }
}

function animationLoop() {
  if (!running) return;
  if (landmarker && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const result = landmarker.detectForVideo(video, performance.now());
    const points = result.landmarks?.[0];
    const rawNumber = points ? recognizeNumber(points) : null;
    lastRawNumber = rawNumber;
    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
    if (points) drawLandmarks(points);

    const now = performance.now() / 1000;
    if (rawNumber !== null && rawNumber >= 1 && rawNumber <= 7) {
      keyboardOverrideUntil = 0;
      lastValidGesture = now;
      if (rawNumber === candidateDegree) candidateFrames++;
      else { candidateDegree = rawNumber; candidateFrames = 1; }
      if (candidateFrames >= 4 && activeDegree !== rawNumber) updateDegree(rawNumber);
    } else {
      candidateDegree = null;
      candidateFrames = 0;
      if (activeDegree !== null && now - lastValidGesture > 0.65 && now >= keyboardOverrideUntil) updateDegree(null);
    }
    if (rawNumber !== null) {
      $("gestureHint").textContent = `辨識手勢 ${rawNumber} · ${rawNumber >= 1 && rawNumber <= 7 ? `第 ${rawNumber} 級` : "此手勢不演奏"}`;
    } else {
      $("gestureHint").textContent = "手掌朝向鏡頭 · 1–7 選和弦 · Q 離開";
    }
  }
  animationId = requestAnimationFrame(animationLoop);
}

async function startApp() {
  if (running) return;
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("請透過 HTTPS 或 localhost 開啟此頁面以使用相機。", true);
    return;
  }
  startButton.disabled = true;
  setStatus("啟動相機並載入 WASM、音色…");
  try {
    if (!audioContext) createAudioGraph();
    await audioContext.resume();
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 960 }, height: { ideal: 720 } }, audio: false });
    video.srcObject = stream;
    await video.play();
    const { HandLandmarker, FilesetResolver } = await import(`${MP_CDN}/vision_bundle.mjs`);
    const vision = await FilesetResolver.forVisionTasks(`${MP_CDN}/wasm`);
    landmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: new URL("./hand_landmarker.task", document.baseURI).href },
      runningMode: "VIDEO",
      numHands: 1,
      minHandDetectionConfidence: 0.55,
      minHandPresenceConfidence: 0.55,
      minTrackingConfidence: 0.5,
    });
    setStatus("正在載入立式鋼琴與電吉他音色…");
    await loadSounds();
    running = true;
    lastVideoTime = -1;
    stage.classList.add("active");
    $("cameraBadge").classList.remove("hidden");
    cameraState.classList.add("on");
    cameraState.querySelector("span").textContent = "鏡頭運作中";
    startButton.classList.add("hidden");
    stopButton.classList.remove("hidden");
    setStatus("就緒 · 比出手勢開始演奏");
    animationId = requestAnimationFrame(animationLoop);
    audioTimer = window.setInterval(scheduleMusic, 25);
  } catch (error) {
    console.error(error);
    stream?.getTracks().forEach((track) => track.stop());
    stream = null;
    startButton.disabled = false;
    cameraState.classList.remove("on");
    const message = error.name === "NotAllowedError" ? "相機權限被拒絕，請在瀏覽器設定中允許相機。" : `啟動失敗：${error.message || error}`;
    setStatus(message, true);
  }
}

function stopApp() {
  running = false;
  cancelAnimationFrame(animationId);
  clearInterval(audioTimer);
  landmarker?.close();
  landmarker = null;
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;
  video.srcObject = null;
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  stage.classList.remove("active");
  $("cameraBadge").classList.add("hidden");
  cameraState.classList.remove("on");
  cameraState.querySelector("span").textContent = "尚未啟動";
  startButton.classList.remove("hidden");
  startButton.disabled = false;
  stopButton.classList.add("hidden");
  lastValidGesture = 0;
  keyboardOverrideUntil = 0;
  updateDegree(null);
  setStatus("相機已關閉");
}

function toggleInstrument(mode = null) {
  const next = mode || (instrument === "piano" ? "guitar" : "piano");
  if (next === instrument) return;
  instrument = next;
  pianoStep = 0;
  guitarStep = 0;
  if (audioContext) nextNoteAt = audioContext.currentTime + 0.025;
  renderMusicState();
}

function chooseDegree(degree) {
  if (!audioContext || !pianoSamples.size) {
    setStatus("先啟動相機，載入音色後即可試彈。", true);
    return;
  }
  keyboardOverrideUntil = performance.now() / 1000 + 5;
  lastValidGesture = performance.now() / 1000;
  updateDegree(degree);
}

startButton.addEventListener("click", startApp);
stopButton.addEventListener("click", stopApp);
modeButton.addEventListener("click", () => toggleInstrument());
volumeSlider.addEventListener("input", () => {
  const value = Number(volumeSlider.value);
  volumeValue.textContent = `${value}%`;
  if (masterGain) masterGain.gain.setTargetAtTime(value / 100, audioContext.currentTime, 0.035);
});
window.addEventListener("keydown", (event) => {
  if (event.target === volumeSlider) return;
  if (event.key === "Tab") { event.preventDefault(); toggleInstrument(); }
  else if (event.key.toLowerCase() === "p") toggleInstrument("piano");
  else if (event.key.toLowerCase() === "g") toggleInstrument("guitar");
  else if (event.key.toLowerCase() === "q" || event.key === "Escape") { if (running) stopApp(); }
  else if (/^[1-7]$/.test(event.key)) chooseDegree(Number(event.key));
});

createDegreeList();
createKeyboard();
renderMusicState();
