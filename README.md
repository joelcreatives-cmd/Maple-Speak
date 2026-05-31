# 🍁 Maple Speak

A voice-based web app for practicing **spoken English**. Tap the mic, talk to
**Maple** — a friendly AI conversation partner — and it talks back out loud,
gently helping your English along the way. It's built for real back-and-forth
conversation, not flashcards.

## What it does

- 🎙️ **Speak naturally** — your voice is transcribed in the browser (Web Speech API).
- 🔊 **Maple talks back** — replies are read aloud with text-to-speech, so it feels like a real chat.
- 🔁 **Hands-free mode** — Maple can auto-listen right after it finishes speaking, so you just keep talking.
- ✍️ **Gentle corrections** — choose how Maple handles your mistakes:
  - **Gentle, conversational** — slips in quick fixes without breaking the flow.
  - **Detailed coaching** — adds a short feedback note after each reply.
  - **None** — pure conversation, no corrections.
- 🎚️ **Adjust to you** — pick your level (beginner / intermediate / advanced) and a partner style.
- 🎭 **Scenarios** — practice a café order, a job interview, travel check-in, and more — or just free-chat.
- ⌨️ **Type fallback** — works without a mic too.

## Setup

You'll need [Node.js](https://nodejs.org/) 18+ and an Anthropic API key
([get one here](https://console.anthropic.com/)).

```bash
# 1. Install dependencies
npm install

# 2. Add your API key
cp .env.example .env
#    then edit .env and paste your key

# 3. Start the app
npm start
```

Then open **http://localhost:3000** and start talking.

> **Tip:** For the full speaking experience, use **Google Chrome** or
> **Microsoft Edge** (desktop or Android). These have the best Web Speech
> support. Other browsers can still type to chat. Microphone access requires
> `localhost` or HTTPS.

## How it works

- **`server.js`** — a small Express server. It keeps your API key safe on the
  server, builds Maple's personality from your settings, and streams Claude's
  replies back to the browser. Uses the `claude-opus-4-8` model, tuned for
  short, snappy, spoken-style replies.
- **`public/`** — the browser app. Speech recognition (speech → text) and
  speech synthesis (text → speech) run entirely in your browser; only the text
  of the conversation is sent to the server.

Your conversation is **not stored anywhere** — it lives only in the current
browser tab and is gone when you refresh or hit reset.

## Customizing Maple

Open `server.js` and edit the `LEVELS`, `STYLES`, `CORRECTIONS`, or
`buildSystemPrompt` definitions to change Maple's behavior, add new scenarios,
or adjust the teaching style.
