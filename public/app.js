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
const rateSlider = $("rate");
const rateVal = $("rateVal");
const controlsEl = document.querySelector(".controls");
const reviewBtn = $("reviewBtn");
const autoBtn = $("autoBtn");
const stopBtn = $("stopBtn");
const scrollBtn = $("scrollBtn");
const streakBar = $("streakBar");
const streakText = $("streakText");

// Review modal
const reviewOverlay = $("reviewOverlay");
const reviewClose = $("reviewClose");
const reviewStats = $("reviewStats");
const reviewBody = $("reviewBody");
const reviewKeepGoing = $("reviewKeepGoing");
const reviewNew = $("reviewNew");

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
  length: $("length"),
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

// Hands-free auto mode: once on, the app loops listen → talk → reply → listen
// with no taps. Browsers require the FIRST mic start to come from a tap, so
// turning auto mode on also starts the first listen.
let autoMode = false;
let autoRestartTimer = null;

let engine = null;
let loadingModel = false;
let supportsF16 = false;

// Bumped whenever an in-flight reply should be discarded (e.g. the user resets
// or starts a new session mid-generation). A generation whose token no longer
// matches must not mutate shared state.
let generation = 0;

// Words the speech recognizer was unsure about, gathered across the session.
// These are good pronunciation-practice candidates.
let trickyWords = new Map(); // word -> count

// Friendly openers shown on the welcome screen (a random few each time).
const STARTERS = [
  "Tell me about your day so far.",
  "What did you have for lunch?",
  "What are your plans for the weekend?",
  "Describe your favorite place to relax.",
  "What's a movie or show you enjoyed recently?",
  "If you could travel anywhere, where would you go?",
  "What's a hobby you'd love to try?",
  "Tell me about a food you really love.",
  "What did you do last weekend?",
  "Who is someone you admire, and why?",
  "What's your dream job?",
  "Describe your hometown to me.",
  "What's a goal you're working on right now?",
  "What kind of music do you like?",
];

// Very common English words — used to surface less-common vocabulary the
// learner actually produced, for the session review.
const COMMON_WORDS = new Set(
  ("the be to of and a in that have i it for not on with he as you do at this but his by from they we say her she or an will my one all would there their what so up out if about who get which go me when make can like time no just him know take people into year your good some could them see other than then now look only come its over think also back after use two how our work first well way even new want because any these give day most us is are was were am been being has had did doing does i'm it's that's don't you're we're they're can't won't isn't really very okay yeah hi hello yes thanks thank please sorry hey oh well i've i'll i'd he's she's").split(
    /\s+/,
  ),
);

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
    if (saved.rate) {
      rateSlider.value = saved.rate;
      updateRateLabel();
    }
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
  data.rate = rateSlider.value;
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(data));
}

function updateRateLabel() {
  const r = parseFloat(rateSlider.value);
  rateVal.textContent = r < 0.85 ? "Slower" : r > 1.05 ? "Faster" : "Normal";
}

let pendingVoiceURI = null;
let pendingModelId = null;

// ---- Conversation persistence (resume after refresh) -----------------------

const SESSION_KEY = "maple-speak-session";

// Keep at most this many recent messages in storage (the model only sees the
// last 16 anyway). Prevents localStorage from growing without bound.
const MAX_STORED_MESSAGES = 100;

function saveSession() {
  try {
    localStorage.setItem(
      SESSION_KEY,
      JSON.stringify({
        messages: messages.slice(-MAX_STORED_MESSAGES),
        tricky: [...trickyWords.entries()].slice(-200),
        savedAt: Date.now(),
      }),
    );
  } catch {
    /* storage full or unavailable — non-fatal */
  }
}

function restoreSession() {
  try {
    const raw = localStorage.getItem(SESSION_KEY);
    if (!raw) return false;
    const data = JSON.parse(raw);
    if (!Array.isArray(data.messages) || data.messages.length === 0) return false;
    // Only resume recent sessions (within 24h) so it feels fresh otherwise.
    if (data.savedAt && Date.now() - data.savedAt > 24 * 60 * 60 * 1000) return false;
    messages = data.messages;
    trickyWords = new Map(data.tricky || []);
    return true;
  } catch {
    return false;
  }
}

function clearSession() {
  generation++; // invalidate any in-flight reply so it can't repopulate state
  busy = false;
  micBtn.classList.remove("thinking");
  messages = [];
  trickyWords = new Map();
  localStorage.removeItem(SESSION_KEY);
}

// ---- Daily practice streak (motivation) ------------------------------------

const STREAK_KEY = "maple-speak-streak";

function todayStamp() {
  // Local-date YYYY-MM-DD so streaks roll over at the user's midnight.
  const d = new Date();
  return `${d.getFullYear()}-${d.getMonth() + 1}-${d.getDate()}`;
}

function dayNumber(stamp) {
  const [y, m, d] = stamp.split("-").map(Number);
  return Math.floor(Date.UTC(y, m - 1, d) / 86400000);
}

function readStreak() {
  try {
    return JSON.parse(localStorage.getItem(STREAK_KEY) || "null");
  } catch {
    return null;
  }
}

// Call when the learner actually speaks/types a turn — that counts as practice.
function markPracticedToday() {
  const today = todayStamp();
  const prev = readStreak();
  if (prev && prev.last === today) return; // already counted today

  let count = 1;
  const totalDays = (prev?.totalDays || 0) + 1; // all-time days practiced
  if (prev) {
    const gap = dayNumber(today) - dayNumber(prev.last);
    if (gap === 1) count = (prev.count || 0) + 1; // consecutive day
    else if (gap === 0) count = prev.count || 1; // same day (shouldn't hit)
    // gap > 1 → streak broken, restart at 1
  }
  localStorage.setItem(
    STREAK_KEY,
    JSON.stringify({ last: today, count, totalDays }),
  );
  renderStreak();
}

function renderStreak() {
  const s = readStreak();
  if (!s || !s.count) {
    streakBar.hidden = true;
    return;
  }
  const today = todayStamp();
  const gap = dayNumber(today) - dayNumber(s.last);
  if (gap > 1) {
    // Streak lapsed — hide rather than show a stale number.
    streakBar.hidden = true;
    return;
  }
  const days = s.count;
  streakText.textContent =
    days === 1
      ? "🔥 Day 1 — nice, you started! Come back tomorrow to build a streak."
      : `🔥 ${days}-day streak! Keep it going.`;
  streakBar.hidden = false;
}

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
const LENGTHS = {
  short: "Keep replies very short: 1 sentence plus one short question.",
  medium: "Keep replies short: 1 to 2 sentences, then one short question.",
  long: "Replies can be a little fuller: 2 to 3 sentences, then one question.",
};

function buildSystemPrompt() {
  const level = LEVELS[controls.level.value] || LEVELS.intermediate;
  const style = STYLES[controls.style.value] || STYLES.friendly;
  const corr = CORRECTIONS[controls.corrections.value] || CORRECTIONS.gentle;
  const length = LENGTHS[controls.length?.value] || LENGTHS.medium;
  const scenario = controls.scenario.value
    ? `\nTODAY'S SCENARIO: ${controls.scenario.value}. Stay in this scenario and play your role.`
    : "";

  return `You are "Maple", a friendly conversation partner helping someone practice spoken English. Your reply is read aloud, and the learner answers by speaking.

Keep the learner talking — that is how they improve.

PERSONALITY: ${style}
LEVEL: ${level}
CORRECTIONS: ${corr}
LENGTH: ${length}

RULES:
- React to what the learner actually said, then move the conversation FORWARD with something new.
- Do NOT repeat or paraphrase the learner's words back to them (don't say things like "So your day has been great, huh?"). Respond naturally instead.
- Ask only ONE question per reply, and make it a NEW question — never re-ask something you already asked.
- Vary how you open each reply; don't start consecutive replies the same way.
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
  if (loadingModel || engine) return; // guard against double-taps / re-entry
  loadingModel = true;
  loadBtn.disabled = true;
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
    saveSettings();
    finishSetup();
  } catch (err) {
    console.error("Model load failed:", err);
    engine = null;
    setupProgress.hidden = true;
    setupReady.hidden = false;
    setupError.hidden = false;
    setupError.innerHTML =
      "<p>Couldn't load that model. If you have an older graphics card, try a " +
      "<strong>(compatibility / older GPUs)</strong> option. Otherwise check your " +
      "internet connection and try again.</p>";
  } finally {
    loadingModel = false;
    loadBtn.disabled = false;
  }
}

function finishSetup() {
  setupEl.hidden = true;
  controlsEl.hidden = false;
  renderStreak();
  if (restoreSession()) {
    renderRestored();
  } else {
    showWelcome();
  }
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
  utter.rate = parseFloat(rateSlider.value) || 1.0;
  const done = (interrupted) => {
    stopBtn.hidden = true;
    micBtn.classList.remove("speaking");
    onEnd?.(interrupted);
  };
  utter.onstart = () => {
    stopBtn.hidden = false;
    micBtn.classList.add("speaking");
  };
  utter.onend = () => done(false);
  utter.onerror = () => done(true);
  synth.speak(utter);
}

// Stop Maple mid-sentence without starting to listen.
function stopSpeaking() {
  synth?.cancel();
  stopBtn.hidden = true;
  micBtn.classList.remove("speaking");
}

// ---- Rendering -------------------------------------------------------------

// Scroll the chat to the bottom only if the user is already near it, so we
// don't yank the view away while they're scrolled up re-reading.
function maybeAutoScroll() {
  const nearBottom =
    chatEl.scrollHeight - chatEl.scrollTop - chatEl.clientHeight < 120;
  if (nearBottom) chatEl.scrollTop = chatEl.scrollHeight;
}

function clearWelcome() {
  chatEl.querySelector(".welcome")?.remove();
}

function showWelcome() {
  chatEl.innerHTML = `
    <div class="welcome">
      <h2>Hi, I'm Maple 🍁</h2>
      <p>Tap the microphone and just start talking — about your day, your plans,
      anything. I'll chat back and gently help your English along the way.</p>
      <p style="margin-top:14px;font-size:13px">Not sure what to say? Pick one:</p>
      <div class="starters" id="starters"></div>
    </div>`;

  // Show three random conversation starters.
  const pool = [...STARTERS].sort(() => Math.random() - 0.5).slice(0, 3);
  const wrap = $("starters");
  for (const text of pool) {
    const btn = document.createElement("button");
    btn.className = "starter";
    btn.textContent = text;
    btn.onclick = () => handleUserUtterance(text);
    wrap.appendChild(btn);
  }
}

// Re-render a conversation restored from a previous session.
function renderRestored() {
  chatEl.innerHTML = "";
  for (const m of messages) {
    if (m.role === "user") {
      addMessage("user", m.content);
    } else {
      const el = document.createElement("div");
      el.className = "msg maple";
      chatEl.appendChild(el);
      renderMapleMessage(el, m.content);
    }
  }
  chatEl.scrollTop = chatEl.scrollHeight;
}

// Track words the recognizer was unsure about (low confidence) — these become
// pronunciation-practice suggestions in the session review.
function notePronunciation(transcript, confidence) {
  if (confidence == null || confidence >= 0.75) return;
  const words = transcript
    .toLowerCase()
    .replace(/[^a-z'\s]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 3 && !COMMON_WORDS.has(w));
  for (const w of words) {
    trickyWords.set(w, (trickyWords.get(w) || 0) + 1);
  }
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
  const actions = document.createElement("div");
  actions.className = "msg-actions";

  const ICONS = {
    speaker:
      '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M11 5 6 9H2v6h4l5 4z"/><path d="M15.5 8.5a5 5 0 0 1 0 7M19 5a9 9 0 0 1 0 14"/></svg>',
    bulb:
      '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 18h6"/><path d="M10 22h4"/><path d="M12 2a7 7 0 0 0-4 12.7c.6.5 1 1.3 1 2.1V18h6v-1.2c0-.8.4-1.6 1-2.1A7 7 0 0 0 12 2z"/></svg>',
    chat:
      '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M21 11.5a8.4 8.4 0 0 1-9 8.4 9 9 0 0 1-3.7-.7L3 21l1.3-3.9A8.4 8.4 0 0 1 12 3a8.4 8.4 0 0 1 9 8.5z"/></svg>',
  };
  const mkAction = (icon, label, fn) => {
    const b = document.createElement("button");
    b.className = "speak-again";
    b.innerHTML = `${ICONS[icon]}<span>${label}</span>`;
    b.onclick = fn;
    return b;
  };

  actions.appendChild(mkAction("speaker", "Hear again", () => speak(spoken || fullText)));
  actions.appendChild(mkAction("bulb", "Explain simply", () => explainMessage(spoken || fullText, el)));
  actions.appendChild(mkAction("chat", "Help me reply", () => suggestReplies(spoken || fullText, el)));

  el.appendChild(actions);
  maybeAutoScroll();
}

// Suggest a few short things the learner could say next — a big help when stuck.
async function suggestReplies(mapleText, msgEl) {
  if (!engine || busy) return;
  if (msgEl.querySelector(".ideas-box")) {
    msgEl.querySelector(".ideas-box").scrollIntoView({ block: "nearest" });
    return;
  }
  const box = document.createElement("div");
  box.className = "ideas-box";
  box.innerHTML = `<span class="coach-spinner"></span>Thinking of some ideas…`;
  msgEl.appendChild(box);
  maybeAutoScroll();

  try {
    const res = await engine.chat.completions.create({
      messages: [
        {
          role: "user",
          content: `An English learner is practicing speaking. Their partner just said: "${mapleText}". Suggest 3 short, natural things (each under 12 words) the learner could SAY in reply, as if they were speaking. Return ONLY the 3 options, each on its own line, with no numbering, quotes, or extra text.`,
        },
      ],
      temperature: 0.8,
      max_tokens: 120,
    });
    const raw = (res.choices?.[0]?.message?.content || "").trim();
    const ideas = raw
      .split("\n")
      .map((l) => l.replace(/^[\s\-*\d.)]+/, "").replace(/^["']|["']$/g, "").trim())
      .filter(Boolean)
      .slice(0, 3);

    if (!ideas.length) {
      box.textContent = "Sorry, no ideas right now — just say whatever comes to mind!";
      return;
    }
    box.innerHTML = `<div class="ideas-label">Tap one to say it, or use it as inspiration:</div>`;
    const row = document.createElement("div");
    row.className = "starters";
    for (const idea of ideas) {
      const chip = document.createElement("button");
      chip.className = "starter";
      chip.textContent = idea;
      chip.onclick = () => handleUserUtterance(idea);
      row.appendChild(chip);
    }
    box.appendChild(row);
    maybeAutoScroll();
  } catch {
    box.textContent = "Sorry, I couldn't think of ideas just now. Please try again.";
  }
}

// Ask the model to restate its message in very simple English, shown inline.
async function explainMessage(text, msgEl) {
  if (!engine || busy) return;
  // Avoid duplicate panels.
  if (msgEl.querySelector(".explain-box")) {
    msgEl.querySelector(".explain-box").scrollIntoView({ block: "nearest" });
    return;
  }
  const box = document.createElement("div");
  box.className = "explain-box";
  box.innerHTML = `<span class="coach-spinner"></span>Explaining…`;
  msgEl.appendChild(box);
  chatEl.scrollTop = chatEl.scrollHeight;

  try {
    const res = await engine.chat.completions.create({
      messages: [
        {
          role: "user",
          content: `Rewrite this in very simple English (short words, one or two short sentences) so an English learner can understand it. If there is a difficult word, add its meaning in parentheses. Only give the simplified version, nothing else.\n\n"${text}"`,
        },
      ],
      temperature: 0.3,
      max_tokens: 160,
    });
    const simple = (res.choices?.[0]?.message?.content || "").trim();
    box.textContent = "💡 " + (simple || "Sorry, I couldn't simplify that one.");
  } catch {
    box.textContent = "💡 Sorry, I couldn't simplify that one. Please try again.";
  }
  chatEl.scrollTop = chatEl.scrollHeight;
}

// ---- Generation (in-browser, streaming) ------------------------------------

async function sendToMaple() {
  if (!engine) return;
  const myGen = generation; // snapshot — invalidated if the session is reset
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
      if (myGen !== generation) return; // session was reset — abandon quietly
      const delta = chunk.choices?.[0]?.delta?.content || "";
      if (!delta) continue;
      if (firstChunk) {
        typingEl.textContent = "";
        firstChunk = false;
      }
      fullText += delta;
      typingEl.textContent = splitReply(fullText).spoken || fullText;
      maybeAutoScroll();
    }

    if (myGen !== generation) return;
    if (!fullText.trim()) throw new Error("Empty response");

    messages.push({ role: "assistant", content: fullText });
    saveSession();
    renderMapleMessage(typingEl, fullText);

    const { spoken } = splitReply(fullText);
    speak(spoken || fullText, (interrupted) => {
      // After Maple finishes speaking, listen again automatically when either
      // hands-free auto mode or the auto-listen toggle is on. Skip if the user
      // interrupted, or if the session was reset during TTS.
      if (
        !interrupted &&
        myGen === generation &&
        (autoMode || controls.autoListen.checked) &&
        recognition &&
        textForm.hidden
      ) {
        startListening();
      }
    });
  } catch (err) {
    console.error(err);
    if (myGen === generation) {
      typingEl.className = "msg maple";
      typingEl.textContent =
        "Sorry, I had trouble responding just now. Please try again.";
    }
  } finally {
    if (myGen === generation) {
      busy = false;
      micBtn.classList.remove("thinking");
    }
  }
}

function handleUserUtterance(text) {
  const clean = text.trim();
  if (!clean || busy) return;
  addMessage("user", clean);
  messages.push({ role: "user", content: clean });
  saveSession();
  markPracticedToday();
  sendToMaple();
}

// ---- Speech recognition ----------------------------------------------------

function setupRecognition() {
  if (!recognition) {
    $("unsupported").hidden = false;
    micBtn.querySelector(".mic-label").textContent = "Type to chat";
    controls.autoListen.checked = false;
    controls.autoListen.disabled = true;
    autoBtn.hidden = true; // no speech recognition → no hands-free mode
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
    micBtn.querySelector(".mic-label").textContent = autoMode
      ? "Listening… (auto)"
      : "Listening… tap to stop";
    interimEl.hidden = false;
    interimEl.textContent = "…";
  };

  recognition.onresult = (e) => {
    let interim = "";
    finalText = "";
    for (let i = 0; i < e.results.length; i++) {
      const result = e.results[i][0];
      const t = result.transcript;
      if (e.results[i].isFinal) {
        finalText += t;
        // Words the engine was unsure about are good pronunciation targets.
        notePronunciation(t, result.confidence);
      } else {
        interim += t;
      }
    }
    interimEl.textContent = (finalText + " " + interim).trim() || "…";
  };

  recognition.onerror = (e) => {
    // "no-speech"/"aborted" are normal in a hands-free loop — don't treat as fatal.
    if (e.error === "no-speech") {
      interimEl.textContent = "I didn't catch that…";
    } else if (e.error === "not-allowed" || e.error === "service-not-allowed") {
      // Mic permission denied — auto mode can't continue.
      autoMode = false;
      reflectAutoMode();
      interimEl.textContent = "Microphone access is blocked. Enable it to speak.";
    }
  };

  recognition.onend = () => {
    listening = false;
    micBtn.classList.remove("listening");
    micBtn.querySelector(".mic-label").textContent = autoMode
      ? "Auto mode on"
      : "Tap to speak";
    interimEl.hidden = true;

    if (finalText.trim()) {
      handleUserUtterance(finalText); // a reply will re-arm listening when it's done
    } else if (autoMode && !busy) {
      // Heard nothing this round — keep the hands-free loop alive after a beat.
      clearTimeout(autoRestartTimer);
      autoRestartTimer = setTimeout(() => {
        if (autoMode && !busy && !listening) startListening();
      }, 700);
    }
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

// ---- Hands-free auto mode --------------------------------------------------

function reflectAutoMode() {
  autoBtn.classList.toggle("active", autoMode);
  autoBtn.setAttribute("aria-pressed", autoMode ? "true" : "false");
  autoBtn.title = autoMode ? "Auto mode: ON (tap to stop)" : "Hands-free auto mode";
  if (!autoMode) {
    micBtn.querySelector(".mic-label").textContent = listening
      ? "Listening… tap to stop"
      : "Tap to speak";
  }
}

function toggleAutoMode() {
  if (!recognition) {
    // No speech recognition available — auto mode can't work; offer typing.
    toggleTextForm(true);
    return;
  }
  autoMode = !autoMode;
  reflectAutoMode();

  if (autoMode) {
    // This click is the user gesture browsers require to start the mic.
    synth?.cancel();
    if (!busy && !listening) startListening();
  } else {
    clearTimeout(autoRestartTimer);
    stopListening();
  }
}

// ---- Session review --------------------------------------------------------

function userTurns() {
  return messages.filter((m) => m.role === "user");
}

// Count words and surface the less-common vocabulary the learner used.
function analyzeVocabulary() {
  const turns = userTurns();
  const allWords = turns
    .map((m) => m.content.toLowerCase())
    .join(" ")
    .replace(/[^a-z'\s]/g, " ")
    .split(/\s+/)
    .filter(Boolean);

  const richSet = new Set(
    allWords.filter((w) => w.length > 4 && !COMMON_WORDS.has(w)),
  );

  return {
    turnCount: turns.length,
    wordCount: allWords.length,
    richWords: [...richSet].slice(0, 12),
  };
}

function statBlock(num, lbl) {
  return `<div class="stat"><div class="num">${num}</div><div class="lbl">${lbl}</div></div>`;
}

async function openReview() {
  if (busy) return;
  endAutoMode();
  stopListening();
  synth?.cancel();

  const turns = userTurns();
  if (turns.length === 0) {
    reviewStats.innerHTML = "";
    reviewBody.innerHTML = `<p class="review-empty">You haven't spoken yet this
      session. Tap the microphone and chat with Maple for a bit, then come back
      to see your review.</p>`;
    reviewOverlay.hidden = false;
    return;
  }

  const { turnCount, wordCount, richWords } = analyzeVocabulary();
  const avg = Math.round(wordCount / Math.max(turnCount, 1));

  reviewStats.innerHTML =
    statBlock(turnCount, "times you spoke") +
    statBlock(wordCount, "words spoken") +
    statBlock(avg, "avg words / turn");

  // Build the static sections first (vocab + pronunciation), then ask Maple
  // for a short personalized coaching note.
  let html = "";

  if (richWords.length) {
    html += `<div class="review-section"><h3>Nice vocabulary you used</h3>
      <div class="chip-row">${richWords
        .map((w) => `<span class="chip">${w}</span>`)
        .join("")}</div></div>`;
  }

  const tricky = [...trickyWords.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([w]) => w)
    .slice(0, 8);
  if (tricky.length) {
    html += `<div class="review-section"><h3>Words to practice saying clearly</h3>
      <div class="chip-row">${tricky
        .map((w) => `<span class="chip warn">${w}</span>`)
        .join("")}</div>
      <p class="review-empty" style="margin-top:8px">Tap a word to hear it, then
      say it back a few times.</p></div>`;
  }

  // All-time progress (from the streak record).
  const streak = readStreak();
  if (streak && streak.totalDays) {
    html += `<div class="review-section"><h3>Your progress</h3>
      <p class="review-empty">You've practiced on <strong>${streak.totalDays}</strong>
      ${streak.totalDays === 1 ? "day" : "days"} so far, with a current streak of
      <strong>${streak.count || 1}</strong>
      ${(streak.count || 1) === 1 ? "day" : "days"}. Keep showing up — consistency
      is what builds fluency.</p></div>`;
  }

  html += `<div class="review-section"><h3>Maple's note for you</h3>
    <div class="review-coach" id="coachNote"><span class="coach-spinner"></span>Thinking about your session…</div></div>`;

  reviewBody.innerHTML = html;
  reviewOverlay.hidden = false;

  // Make pronunciation chips speak when tapped.
  reviewBody.querySelectorAll(".chip.warn").forEach((chip) => {
    chip.style.cursor = "pointer";
    chip.onclick = () => speakWord(chip.textContent);
  });

  // Generate the coaching note with the in-browser model.
  const coachEl = $("coachNote");
  try {
    const note = await generateCoachNote();
    coachEl.textContent = note;
  } catch {
    coachEl.textContent =
      "Great work practicing today! Keep having conversations like this — every chat makes your English more natural.";
  }
}

// Speak a single word slowly and clearly for pronunciation practice.
function speakWord(word) {
  if (!synth) return;
  synth.cancel();
  const u = new SpeechSynthesisUtterance(word);
  const v = getSelectedVoice();
  if (v) {
    u.voice = v;
    u.lang = v.lang;
  }
  u.rate = 0.75;
  synth.speak(u);
}

async function generateCoachNote() {
  if (!engine) throw new Error("no engine");
  const transcript = userTurns()
    .map((m) => `- ${m.content}`)
    .join("\n");

  const prompt = `You are Maple, a warm English-speaking coach. Below are things a learner said out loud during a practice conversation. Write a SHORT, encouraging review (3 to 4 sentences) for them. Mention one specific thing they did well, and one simple, friendly suggestion to improve. Speak directly to the learner ("you"). Do not use markdown, bullet points, or emoji.

What the learner said:
${transcript}`;

  const res = await engine.chat.completions.create({
    messages: [{ role: "user", content: prompt }],
    temperature: 0.6,
    max_tokens: 220,
  });
  return (res.choices?.[0]?.message?.content || "").trim();
}

function closeReview() {
  reviewOverlay.hidden = true;
}

// ---- UI wiring -------------------------------------------------------------

loadBtn.addEventListener("click", loadModel);

micBtn.addEventListener("click", () => {
  if (!recognition) {
    toggleTextForm(true);
    return;
  }
  // A manual mic tap takes over from auto mode.
  if (autoMode) {
    autoMode = false;
    clearTimeout(autoRestartTimer);
    reflectAutoMode();
  }
  if (listening) stopListening();
  else startListening();
});

autoBtn.addEventListener("click", toggleAutoMode);

// Stop Maple speaking. If interrupted by hand, don't auto-re-listen.
stopBtn.addEventListener("click", () => {
  stopSpeaking();
  clearTimeout(autoRestartTimer);
});

// Scroll-to-latest button appears when the user scrolls up.
scrollBtn.addEventListener("click", () => {
  chatEl.scrollTo({ top: chatEl.scrollHeight, behavior: "smooth" });
});
chatEl.addEventListener("scroll", () => {
  const nearBottom =
    chatEl.scrollHeight - chatEl.scrollTop - chatEl.clientHeight < 120;
  scrollBtn.hidden = nearBottom;
});

function endAutoMode() {
  autoMode = false;
  clearTimeout(autoRestartTimer);
  reflectAutoMode();
}

resetBtn.addEventListener("click", () => {
  if (messages.length && !confirm("Start a new session? This clears the current chat.")) {
    return;
  }
  endAutoMode();
  synth?.cancel();
  stopListening();
  clearSession();
  showWelcome();
});

reviewBtn.addEventListener("click", openReview);
reviewClose.addEventListener("click", closeReview);
reviewKeepGoing.addEventListener("click", closeReview);
reviewNew.addEventListener("click", () => {
  closeReview();
  endAutoMode();
  synth?.cancel();
  stopListening();
  clearSession();
  showWelcome();
});
reviewOverlay.addEventListener("click", (e) => {
  if (e.target === reviewOverlay) closeReview();
});

settingsToggle.addEventListener("click", () => {
  settingsEl.hidden = !settingsEl.hidden;
});

for (const el of Object.values(controls)) el.addEventListener("change", saveSettings);
voiceSelect.addEventListener("change", saveSettings);
modelSelect.addEventListener("change", saveSettings);
rateSlider.addEventListener("input", () => {
  updateRateLabel();
  saveSettings();
});

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

// Register the service worker so the app shell loads offline. Best-effort —
// failures (e.g. on file://) are harmless and ignored.
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("sw.js").catch(() => {});
  });
}

init();
