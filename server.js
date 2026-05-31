import "dotenv/config";
import express from "express";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import Anthropic from "@anthropic-ai/sdk";

const __dirname = dirname(fileURLToPath(import.meta.url));

const PORT = process.env.PORT || 3000;
const MODEL = "claude-opus-4-8";

if (!process.env.ANTHROPIC_API_KEY) {
  console.error(
    "\n  Missing ANTHROPIC_API_KEY.\n" +
      "  Copy .env.example to .env and add your key from https://console.anthropic.com/\n",
  );
  process.exit(1);
}

const client = new Anthropic();
const app = express();
app.use(express.json({ limit: "1mb" }));
app.use(express.static(join(__dirname, "public")));

// ---- System prompt construction --------------------------------------------

const LEVELS = {
  beginner:
    "The learner is a BEGINNER. Use simple, common words and short sentences. Speak slowly and clearly. Avoid idioms and slang. Be very encouraging.",
  intermediate:
    "The learner is INTERMEDIATE. Use everyday vocabulary and natural sentence length. You may introduce common idioms, explaining them briefly if they might be new.",
  advanced:
    "The learner is ADVANCED. Speak naturally at a normal pace, using rich vocabulary, idioms, and nuanced expression, as you would with a fluent speaker.",
};

const STYLES = {
  friendly:
    "You are warm, upbeat, and adaptive. You mirror the learner's level and keep things easy to follow while staying genuinely engaging.",
  casual:
    "You talk like a relaxed, real friend, using natural slang, contractions, and a normal conversational pace. This is great for building real-world fluency.",
  tutor:
    "You are a patient, gentle tutor. You speak slowly and clearly with simple vocabulary, give lots of encouragement, and never make the learner feel rushed.",
};

const CORRECTIONS = {
  gentle:
    "When the learner makes a noticeable mistake (grammar, word choice, or an awkward phrasing), gently weave a correction into your reply in a natural way, then continue the conversation. Do NOT correct every tiny thing — only what matters for being understood. Never interrupt the flow or lecture.",
  detailed:
    "After replying conversationally, ALWAYS add a short feedback note. Format it on its own lines at the very end of your message like this:\n\n---\n📝 Feedback: <one or two friendly, specific tips about grammar, vocabulary, or phrasing the learner could improve. If they spoke perfectly, praise something specific they did well.>\n\nKeep the feedback short and supportive.",
  none: "Do NOT correct the learner's mistakes. Just have a natural, enjoyable conversation as a supportive friend.",
};

function buildSystemPrompt({ level, style, corrections, scenario }) {
  const levelText = LEVELS[level] || LEVELS.intermediate;
  const styleText = STYLES[style] || STYLES.friendly;
  const correctionText = CORRECTIONS[corrections] || CORRECTIONS.gentle;

  const scenarioText = scenario
    ? `\n\nTODAY'S SCENARIO: ${scenario}. Stay in this scenario and play your role naturally, but break character briefly if the learner asks a direct question about English.`
    : "";

  return `You are "Maple", a friendly AI conversation partner whose job is to help someone practice and improve their spoken English. You are talking with them out loud — your replies are read aloud by a text-to-speech voice, and the learner replies by speaking.

CORE GOAL: keep the learner talking. The more they speak, the more they improve.

PERSONALITY: ${styleText}

LEVEL: ${levelText}

CORRECTIONS: ${correctionText}

HOW TO TALK:
- Keep your replies SHORT and conversational — usually 1 to 3 sentences. This is a spoken conversation, not an essay.
- End most turns with a light, open question to keep the conversation going.
- Sound like a real person speaking, not like written text. Use contractions and natural rhythm.
- Since your reply will be spoken aloud, do not use markdown formatting, bullet points, code, emoji, or symbols in the conversational part of your reply. Write plain spoken sentences. (The feedback note, when enabled, is the only exception.)
- Never mention that you are an AI or a language model unless directly asked. You are simply Maple, a friendly conversation partner.
- If the learner is quiet or unsure, kindly offer a topic or ask an easy question.
- Respond only with what you would say out loud — no narration of your reasoning.${scenarioText}`;
}

// ---- Chat endpoint (streaming) ----------------------------------------------

app.post("/api/chat", async (req, res) => {
  const { messages, level, style, corrections, scenario } = req.body || {};

  if (!Array.isArray(messages) || messages.length === 0) {
    return res.status(400).json({ error: "messages must be a non-empty array" });
  }

  // Only keep the fields the API expects, and cap history length to stay snappy.
  const trimmed = messages
    .slice(-24)
    .filter((m) => m && (m.role === "user" || m.role === "assistant") && m.content)
    .map((m) => ({ role: m.role, content: String(m.content) }));

  const systemPrompt = buildSystemPrompt({ level, style, corrections, scenario });

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders?.();

  const send = (event, data) =>
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);

  try {
    const stream = client.messages.stream({
      model: MODEL,
      max_tokens: 1024,
      // Snappy, low-latency replies suit a back-and-forth conversation.
      thinking: { type: "disabled" },
      output_config: { effort: "low" },
      system: [
        {
          type: "text",
          text: systemPrompt,
          cache_control: { type: "ephemeral" },
        },
      ],
      messages: trimmed,
    });

    stream.on("text", (delta) => send("delta", { text: delta }));

    const final = await stream.finalMessage();
    const fullText = final.content
      .filter((b) => b.type === "text")
      .map((b) => b.text)
      .join("");

    send("done", { text: fullText });
    res.end();
  } catch (err) {
    console.error("Chat error:", err?.message || err);
    if (!res.headersSent) {
      res.status(500).json({ error: "Something went wrong talking to Maple." });
    } else {
      send("error", { message: "Sorry, I had trouble responding. Please try again." });
      res.end();
    }
  }
});

app.get("/api/health", (_req, res) => res.json({ ok: true, model: MODEL }));

app.listen(PORT, () => {
  console.log(`\n  🍁 Maple Speak is running at http://localhost:${PORT}\n`);
});
