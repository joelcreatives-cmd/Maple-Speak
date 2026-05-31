// ---- Maple Speak: voice-based English practice ----------------------------

const $ = (id) => document.getElementById(id);

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

const controls = {
  level: $("level"),
  style: $("style"),
  corrections: $("corrections"),
  scenario: $("scenario"),
  autoSpeak: $("autoSpeak"),
  autoListen: $("autoListen"),
};

// ---- State -----------------------------------------------------------------

let messages = []; // { role: 'user' | 'assistant', content }
let listening = false;
let busy = false;
let voices = [];

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
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(data));
}

let pendingVoiceURI = null;

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
  // Prefer a saved choice, else a natural-sounding default.
  const preferred =
    pendingVoiceURI ||
    voices.find((v) => /natural|google|samantha|aria|jenny/i.test(v.name))
      ?.voiceURI ||
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

// The detailed-coaching feedback note shouldn't be read aloud. Split it out.
function splitReply(text) {
  const idx = text.search(/\n-{2,}\s*\n?\s*📝|\n📝\s*Feedback/i);
  if (idx === -1) return { spoken: text.trim(), feedback: null };
  const spoken = text.slice(0, idx).trim();
  let feedback = text
    .slice(idx)
    .replace(/^\s*\n?-{2,}\s*\n?/, "")
    .replace(/^📝\s*Feedback:?\s*/i, "")
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
  const w = chatEl.querySelector(".welcome");
  if (w) w.remove();
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

// ---- Talking to the server (streaming) -------------------------------------

async function sendToMaple() {
  busy = true;
  micBtn.classList.add("thinking");
  const typingEl = addTyping();

  const payload = {
    messages,
    level: controls.level.value,
    style: controls.style.value,
    corrections: controls.corrections.value,
    scenario: controls.scenario.value,
  };

  let fullText = "";
  let firstChunk = true;

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok || !res.body) throw new Error("Request failed");

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split("\n\n");
      buffer = parts.pop();
      for (const part of parts) {
        const line = part.match(/^event: (.+)\ndata: (.+)$/s);
        if (!line) continue;
        const [, event, dataStr] = line;
        const data = JSON.parse(dataStr);

        if (event === "delta") {
          if (firstChunk) {
            typingEl.textContent = "";
            firstChunk = false;
          }
          fullText += data.text;
          // Show only the spoken part live; feedback renders at the end.
          typingEl.textContent = splitReply(fullText).spoken || fullText;
          chatEl.scrollTop = chatEl.scrollHeight;
        } else if (event === "done") {
          fullText = data.text || fullText;
        } else if (event === "error") {
          throw new Error(data.message);
        }
      }
    }

    if (!fullText.trim()) throw new Error("Empty response");

    messages.push({ role: "assistant", content: fullText });
    renderMapleMessage(typingEl, fullText);

    const { spoken } = splitReply(fullText);
    speak(spoken || fullText, () => {
      if (controls.autoListen.checked && recognition && !textFormVisible()) {
        startListening();
      }
    });
  } catch (err) {
    console.error(err);
    typingEl.classList.remove("maple");
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
    if (e.error === "no-speech") {
      interimEl.textContent = "I didn't catch that — try again.";
    }
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
  synth?.cancel(); // stop Maple talking if user wants to jump in
  try {
    recognition.start();
  } catch {
    /* start() throws if already started; ignore */
  }
}

function stopListening() {
  if (recognition && listening) recognition.stop();
}

// ---- UI wiring -------------------------------------------------------------

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

for (const el of Object.values(controls)) {
  el.addEventListener("change", saveSettings);
}
voiceSelect.addEventListener("change", saveSettings);

// Text input fallback / alternative
function textFormVisible() {
  return !textForm.hidden;
}
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

loadSettings();
setupRecognition();
showWelcome();
