// Maple Speak service worker — keeps the app working offline while always
// preferring the freshest version when online. The AI model itself is cached
// separately by WebLLM in the browser's storage.

const CACHE = "maple-speak-v28";
const SHELL = [
  "./",
  "./index.html",
  "./style.css",
  "./app.js",
  "./manifest.webmanifest",
  "./icon.svg",
  "./logo.png",
  "./icon-192.png",
  "./icon-512.png",
  "./apple-touch-icon.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE).then((cache) => cache.addAll(SHELL)).then(() => self.skipWaiting()),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))),
      )
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;

  const url = new URL(request.url);
  // Only handle our own same-origin app-shell files. Let everything else
  // (fonts, the WebLLM CDN, model weights) go straight to the network so we
  // never interfere with how WebLLM caches large model files.
  if (url.origin !== self.location.origin) return;

  // Network-first: always try the live version, fall back to cache offline.
  // This guarantees users get updates immediately instead of being stuck on a
  // stale cached app, while still working with no connection.
  event.respondWith(
    fetch(request)
      .then((resp) => {
        const copy = resp.clone();
        caches.open(CACHE).then((cache) => cache.put(request, copy)).catch(() => {});
        return resp;
      })
      .catch(() => caches.match(request)),
  );
});

