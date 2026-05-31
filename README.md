# 🍁 Maple Speak

A **free, private** web app for practicing **spoken English**. Tap the mic, talk
to **Maple** — an AI conversation partner — and it talks back out loud, gently
helping your English along the way.

**There's nothing to pay and no account to create.** The AI runs entirely
inside your browser, so your conversations never leave your computer.

## How it stays free

The usual cost in an app like this is calling a cloud AI. Maple avoids that
completely:

- 🧠 **The AI runs in your browser** via [WebLLM](https://github.com/mlc-ai/web-llm) +
  WebGPU. The model is downloaded once and cached, then runs on your own device
  — no API key, no server bill, and it even works offline afterward.
- 🎙️ **Speech-to-text** and 🔊 **text-to-speech** use your browser's built-in
  **Web Speech API** — also free.
- 🗄️ The included Node server is a tiny **zero-dependency** static file server.
  It only hands the files to your browser; it makes no AI calls.

## What it does

- 🎙️ **Speak naturally** — your voice is transcribed in the browser.
- 🔊 **Maple talks back** — replies are read aloud, so it feels like a real chat.
- 🔁 **Hands-free** — Maple auto-listens after it finishes speaking.
- ✍️ **Corrections, your way** — gentle conversational fixes, detailed coaching
  notes, or none.
- 🎚️ **Adapts to you** — level (beginner / intermediate / advanced) and partner style.
- 🎭 **Scenarios** — café, job interview, travel, shopping, doctor, or free chat.
- ⌨️ **Type fallback** if your browser has no speech support.

## Requirements

- A modern **Google Chrome** or **Microsoft Edge** (desktop/laptop, or Android)
  with **WebGPU** support — this is what runs the AI. Most up-to-date versions
  have it on by default.
- A computer with a reasonably capable GPU and enough memory. Smaller models
  (the default) work on more modest hardware.
- That's it — **no API key, no Node packages to install** for the AI.

## Run it

The simplest way is to serve the files with the included static server:

```bash
npm start          # → http://localhost:3000
```

(`npm start` just runs the built-in Node server — there are **no dependencies
to install**.) Any static file server works too, e.g. `python3 -m http.server`
from the `public/` folder.

Then open **http://localhost:3000**, choose a model, click **Load Maple**, and
start talking.

> The **first** time, your browser downloads the chosen AI model once (a few
> hundred MB to ~1 GB). It's cached after that and loads quickly — even offline.

## Choosing a model

On the start screen you can pick from a few small, free models:

| Model | Size | Notes |
| --- | --- | --- |
| **Llama 3.2 1B** (default) | ~0.7 GB | Small & fast — best for most people |
| **Qwen 2.5 1.5B** | ~1 GB | A little smarter |
| **Llama 3.2 3B** | ~1.8 GB | Bigger & smarter, but slower |
| *…1B / 1.5B (compatibility)* | larger | For older GPUs without `shader-f16` |

Bigger models give more natural conversation but need a stronger computer.

## A note on quality

These free, on-device models are much smaller than commercial cloud AIs, so
Maple's replies and corrections are simpler and occasionally imperfect. For
**everyday speaking practice** that's exactly what you want — a patient partner
that keeps you talking, at zero cost and with full privacy.

## Customizing Maple

Maple's personality is built in the browser. Open `public/app.js` and edit the
`LEVELS`, `STYLES`, `CORRECTIONS`, or `buildSystemPrompt` definitions, or the
`MODELS` list to offer different models.
