// ---- Maple Speak: 100% free, in-browser English practice -------------------
// The AI model runs entirely in your browser via WebLLM (WebGPU). No server
// AI, no API key, no account. Speech in/out uses the browser's Web Speech API.

import * as webllm from "https://esm.run/@mlc-ai/web-llm@0.2.79";

const $ = (id) => document.getElementById(id);

// Chat / control elements
const chatEl = $("chat");
const micBtn = $("micBtn");
const interimEl = $("interim");
const resetBtn = $("resetBtn");
const settingsToggle = $("settingsToggle");
const settingsEl = $("settings");
const textToggle = $("textToggle");
const textForm = $("textForm");
const textInput = $("textInput");
const voiceSelect = $("voiceSelect");
const controlsEl = document.querySelector(".controls");

// Setup / loading elements
const setupEl = $("setup");
const setupReady = $("setupReady");
const setupProgress = $("setupProgress");
const setupError = $("setupError");
const noWebGPU = $("noWebGPU");
const modelSelect = $("modelSelect");
const loadBtn = $("loadBtn");
const barFill = $("barFill");
const progressText = $("progressText");

const controls = {
  level: $("level"),
  style: $("style"),
  corrections: $("corrections"),
  scenario: $("scenario"),
  autoSpeak: $("autoSpeak"),
  autoListen: $("autoListen"),
};

// Curated small instruct models (verified against the WebLLM prebuilt list).
// All support a system role. f16 = smaller/faster but needs a GPU with
// shader-f16; f32 is the wider-compatibility fallback.
const MODELS = [
  { id: "Llama-3.2-1B-Instruct-q4f16_1-MLC", label: "Llama 3.2 1B — small & fast (recommended)", f16: true },
  { id: "Qwen2.5-1.5B-Instruct-q4f16_1-MLC", label: "Qwen 2.5 1.5B — small, a bit smarter", f16: true },
  { id: "Llama-3.2-3B-Instruct-q4f16_1-MLC", label: "Llama 3.2 3B — bigger & smarter (slower)", f16: true },
  { id: "Llama-3.2-1B-Instruct-q4f32_1-MLC", label: "Llama 3.2 1B (compatibility / older GPUs)", f16: false },
  { id: "Qwen2.5-1.5B-Instruct-q4f32_1-MLC", label: "Qwen 2.5 1.5B (compatibility / older GPUs)", f16: false },
];

// ---- State -----------------------------------------------------------------

let messages = []; // { role: 'user' | 'assistant', content }
let listening = false;
let busy = false;
let voices = [];

let engine = null;
let loadedModelId = null;
let supportsF16 = false;

const SpeechRecognition =
  window.SpeechRecognition || window.webkitSpeechRecognition;
const synth = window.speechSynthesis;
const recognition = SpeechRecognition ? new SpeechRecognition() : null;

// ---- Settings persistence --------------------------------------------------

const SETTINGS_KEY = "maple-speak-settings";

function loadSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}");
    for (const [k, el] of Object.entries(controls)) {
      if (saved[k] === undefined) continue;
      if (el.type === "checkbox") el.checked = saved[k];
      else el.value = saved[k];
    }
    if (saved.voiceURI) pendingVoiceURI = saved.voiceURI;
    if (saved.modelId) pendingModelId = saved.modelId;
  } catch {
    /* ignore */
  }
}

function saveSettings() {
  const data = {};
  for (const [k, el] of Object.entries(controls)) {
    data[k] = el.type === "checkbox" ? el.checked : el.value;
  }
  data.voiceURI = voiceSelect.value;
  data.modelId = modelSelect.value;
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(data));
}

let pendingVoiceURI = null;
let pendingModelId = null;

// ---- System prompt (built in the browser) ----------------------------------

const LEVELS = {
  beginner:
    "The learner is a BEGINNER. Use simple, common words and short sentences. Avoid idioms and slang. Be very encouraging.",
  intermediate:
    "The learner is INTERMEDIATE. Use everyday vocabulary and natural sentence length. You may use common idioms.",
  advanced:
    "The learner is ADVANCED. Speak naturally with rich vocabulary and idioms, as you would with a fluent speaker.",
};
const STYLES = {
  friendly: "You are warm, upbeat and adaptive. You keep things easy to follow while staying engaging.",
  casual: "You talk like a relaxed, real friend, using natural contractions and a conversational pace.",
  tutor: "You are a patient, gentle tutor. You speak clearly with simple vocabulary and lots of encouragement.",
};
const CORRECTIONS = {
  gentle:
    "When the learner makes a noticeable mistake, gently weave a short correction into your reply, then continue. Only correct what matters for being understood. Never lecture.",
  detailed:
    "After replying conversationally, ALWAYS add a short feedback note at the very end on its own line, in exactly this format:\n📝 Feedback: <one or two friendly tips, or praise if they did well>",
  none: "Do NOT correct the learner's mistakes. Just have a natural, enjoyable conversation.",
};

function buildSystemPrompt() {
  const level = LEVELS[controls.level.value] || LEVELS.intermediate;
  const style = STYLES[controls.style.value] || STYLES.friendly;
  const corr = CORRECTIONS[controls.corrections.value] || CORRECTIONS.gentle;
  const scenario = controls.scenario.value
    ? `\nTODAY'S SCENARIO: ${controls.scenario.value}. Stay in this scenario and play your role.`
    : "";

  return `You are "Maple", a friendly conversation partner helping someone practice spoken English. Your reply is read aloud, and the learner answers by speaking.

Keep the learner talking — that is how they improve.

PERSONALITY: ${style}
LEVEL: ${level}
CORRECTIONS: ${corr}

RULES:
- Keep replies SHORT and conversational: 1 to 3 sentences.
- End most turns with a friendly, open question.
- Sound like a real person speaking. Use plain spoken sentences with no markdown, lists, code, or emoji (the feedback note is the only exception).
- Never say you are an AI. You are just Maple.
- Reply only with what you would say out loud.${scenario}`;
}

// ---- Loading the in-browser model ------------------------------------------

function populateModelSelect() {
  modelSelect.innerHTML = "";
  // Default to an f16 model if the GPU supports it, otherwise an f32 one.
  for (const m of MODELS) {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.label;
    modelSelect.appendChild(opt);
  }
  const def =
    pendingModelId ||
    (supportsF16
      ? "Llama-3.2-1B-Instruct-q4f16_1-MLC"
      : "Llama-3.2-1B-Instruct-q4f32_1-MLC");
  modelSelect.value = def;
}

async function detectWebGPU() {
  if (!navigator.gpu) return false;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return false;
    supportsF16 = adapter.features?.has?.("shader-f16") ?? false;
    return true;
  } catch {
    return false;
  }
}

async function loadModel() {
  const modelId = modelSelect.value;
  setupReady.hidden = true;
  setupError.hidden = true;
  setupProgress.hidden = false;
  barFill.style.width = "0%";
  progressText.textContent = "Preparing…";

  try {
    engine = await webllm.CreateMLCEngine(modelId, {
      initProgressCallback: (report) => {
        const pct = Math.round((report.progress || 0) * 100);
        barFill.style.width = `${pct}%`;
        progressText.textContent = report.text || `${pct}%`;
      },
    });
    loadedModelId = modelId;
    saveSettings();
    finishSetup();
  } catch (err) {
    console.error("Model load failed:", err);
    setupProgress.hidden = true;
    setupReady.hidden = false;
    setupError.hidden = false;
    setupError.innerHTML =
      "<p>Couldn't load that model. If you have an older graphics card, try a " +
      "<strong>(compatibility / older GPUs)</strong> option. Otherwise check your " +
      "internet connection and try again.</p>";
  }
}

function finishSetup() {
  setupEl.hidden = true;
  controlsEl.hidden = false;
  showWelcome();
}

// ---- Voices ----------------------------------------------------------------

function populateVoices() {
  if (!synth) return;
  voices = synth.getVoices().filter((v) => v.lang.startsWith("en"));
  if (!voices.length) return;
  voiceSelect.innerHTML = "";
  for (const v of voices) {
    const opt = document.createElement("option");
    opt.value = v.voiceURI;
    opt.textContent = `${v.name} (${v.lang})`;
    voiceSelect.appendChild(opt);
  }
  const preferred =
    pendingVoiceURI ||
    voices.find((v) => /natural|google|samantha|aria|jenny/i.test(v.name))?.voiceURI ||
    voices[0].voiceURI;
  voiceSelect.value = preferred;
}

if (synth) {
  populateVoices();
  synth.onvoiceschanged = populateVoices;
}

function getSelectedVoice() {
  return voices.find((v) => v.voiceURI === voiceSelect.value) || voices[0];
}

// ---- Speaking (TTS) --------------------------------------------------------

function splitReply(text) {
  const idx = text.search(/\n-{2,}\s*\n?\s*📝|\n?📝\s*Feedback/i);
  if (idx === -1) return { spoken: text.trim(), feedback: null };
  const spoken = text.slice(0, idx).trim();
  const feedback = text
    .slice(idx)
    .replace(/^\s*\n?-{2,}\s*\n?/, "")
    .replace(/^\n?📝\s*Feedback:?\s*/i, "")
    .trim();
  return { spoken, feedback };
}

function speak(text, onEnd) {
  if (!synth || !controls.autoSpeak.checked || !text) {
    onEnd?.();
    return;
  }
  synth.cancel();
  const utter = new SpeechSynthesisUtterance(text);
  const voice = getSelectedVoice();
  if (voice) {
    utter.voice = voice;
    utter.lang = voice.lang;
  }
  utter.rate = controls.level.value === "beginner" ? 0.9 : 1.0;
  utter.onend = () => onEnd?.();
  utter.onerror = () => onEnd?.();
  synth.speak(utter);
}

// ---- Rendering -------------------------------------------------------------

function clearWelcome() {
  chatEl.querySelector(".welcome")?.remove();
}

function showWelcome() {
  chatEl.innerHTML = `
    <div class="welcome">
      <h2>Hi, I'm Maple 🍁</h2>
      <p>Tap the microphone and just start talking — about your day, your plans,
      anything. I'll chat back and gently help your English along the way.</p>
    </div>`;
}

function addMessage(role, text) {
  clearWelcome();
  const el = document.createElement("div");
  el.className = `msg ${role === "assistant" ? "maple" : "user"}`;
  el.textContent = text;
  chatEl.appendChild(el);
  chatEl.scrollTop = chatEl.scrollHeight;
  return el;
}

function addTyping() {
  clearWelcome();
  const el = document.createElement("div");
  el.className = "msg maple";
  el.innerHTML = `<span class="typing"><span></span><span></span><span></span></span>`;
  chatEl.appendChild(el);
  chatEl.scrollTop = chatEl.scrollHeight;
  return el;
}

function renderMapleMessage(el, fullText) {
  const { spoken, feedback } = splitReply(fullText);
  el.textContent = spoken || fullText;
  if (feedback) {
    const fb = document.createElement("div");
    fb.className = "feedback";
    fb.textContent = `📝 ${feedback}`;
    el.appendChild(fb);
  }
  const replay = document.createElement("button");
  replay.className = "speak-again";
  replay.innerHTML = "🔊 Hear again";
  replay.onclick = () => speak(spoken || fullText);
  el.appendChild(replay);
  chatEl.scrollTop = chatEl.scrollHeight;
}

// ---- Generation (in-browser, streaming) ------------------------------------

async function sendToMaple() {
  if (!engine) return;
  busy = true;
  micBtn.classList.add("thinking");
  const typingEl = addTyping();

  const chatMessages = [
    { role: "system", content: buildSystemPrompt() },
    ...messages.slice(-16).map((m) => ({ role: m.role, content: m.content })),
  ];

  let fullText = "";
  let firstChunk = true;

  try {
    const stream = await engine.chat.completions.create({
      messages: chatMessages,
      stream: true,
      temperature: 0.7,
      max_tokens: 300,
    });

    for await (const chunk of stream) {
      const delta = chunk.choices?.[0]?.delta?.content || "";
      if (!delta) continue;
      if (firstChunk) {
        typingEl.textContent = "";
        firstChunk = false;
      }
      fullText += delta;
      typingEl.textContent = splitReply(fullText).spoken || fullText;
      chatEl.scrollTop = chatEl.scrollHeight;
    }

    if (!fullText.trim()) throw new Error("Empty response");

    messages.push({ role: "assistant", content: fullText });
    renderMapleMessage(typingEl, fullText);

    const { spoken } = splitReply(fullText);
    speak(spoken || fullText, () => {
      if (controls.autoListen.checked && recognition && textForm.hidden) {
        startListening();
      }
    });
  } catch (err) {
    console.error(err);
    typingEl.className = "msg maple";
    typingEl.textContent =
      "Sorry, I had trouble responding just now. Please try again.";
  } finally {
    busy = false;
    micBtn.classList.remove("thinking");
  }
}

function handleUserUtterance(text) {
  const clean = text.trim();
  if (!clean || busy) return;
  addMessage("user", clean);
  messages.push({ role: "user", content: clean });
  sendToMaple();
}

// ---- Speech recognition ----------------------------------------------------

function setupRecognition() {
  if (!recognition) {
    $("unsupported").hidden = false;
    micBtn.querySelector(".mic-label").textContent = "Type to chat";
    controls.autoListen.checked = false;
    controls.autoListen.disabled = true;
    return;
  }

  recognition.lang = "en-US";
  recognition.continuous = false;
  recognition.interimResults = true;

  let finalText = "";

  recognition.onstart = () => {
    listening = true;
    finalText = "";
    micBtn.classList.add("listening");
    micBtn.querySelector(".mic-label").textContent = "Listening… tap to stop";
    interimEl.hidden = false;
    interimEl.textContent = "…";
  };

  recognition.onresult = (e) => {
    let interim = "";
    finalText = "";
    for (let i = 0; i < e.results.length; i++) {
      const t = e.results[i][0].transcript;
      if (e.results[i].isFinal) finalText += t;
      else interim += t;
    }
    interimEl.textContent = (finalText + " " + interim).trim() || "…";
  };

  recognition.onerror = (e) => {
    if (e.error === "no-speech") interimEl.textContent = "I didn't catch that — try again.";
  };

  recognition.onend = () => {
    listening = false;
    micBtn.classList.remove("listening");
    micBtn.querySelector(".mic-label").textContent = "Tap to speak";
    interimEl.hidden = true;
    if (finalText.trim()) handleUserUtterance(finalText);
  };
}

function startListening() {
  if (!recognition || listening || busy) return;
  synth?.cancel();
  try {
    recognition.start();
  } catch {
    /* already started */
  }
}

function stopListening() {
  if (recognition && listening) recognition.stop();
}

// ---- UI wiring -------------------------------------------------------------

loadBtn.addEventListener("click", loadModel);

micBtn.addEventListener("click", () => {
  if (!recognition) {
    toggleTextForm(true);
    return;
  }
  if (listening) stopListening();
  else startListening();
});

resetBtn.addEventListener("click", () => {
  synth?.cancel();
  stopListening();
  messages = [];
  showWelcome();
});

settingsToggle.addEventListener("click", () => {
  settingsEl.hidden = !settingsEl.hidden;
});

for (const el of Object.values(controls)) el.addEventListener("change", saveSettings);
voiceSelect.addEventListener("change", saveSettings);
modelSelect.addEventListener("change", saveSettings);

function toggleTextForm(show) {
  textForm.hidden = show === undefined ? !textForm.hidden : !show;
  if (!textForm.hidden) textInput.focus();
}
textToggle.addEventListener("click", () => toggleTextForm());
textForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = textInput.value;
  textInput.value = "";
  handleUserUtterance(text);
});

// ---- Init ------------------------------------------------------------------

async function init() {
  loadSettings();
  setupRecognition();

  const hasWebGPU = await detectWebGPU();
  if (!hasWebGPU) {
    setupReady.hidden = true;
    noWebGPU.hidden = false;
    return;
  }
  populateModelSelect();
}

init();
