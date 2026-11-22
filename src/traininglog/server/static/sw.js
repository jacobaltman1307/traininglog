const CACHE_NAME = "traininglog-v2";
const VENDOR_ASSETS = [
  "/static/vendor/uPlot.iife.min.js",
  "/static/vendor/uPlot.min.css",
  "/static/vendor/preact.module.js",
  "/static/vendor/hooks.module.js",
  "/static/vendor/htm.module.js",
  "/static/vendor/ui.mjs",
  "/icon.svg",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(VENDOR_ASSETS)).catch(() => {})
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (url.pathname.startsWith("/api/") || event.request.method !== "GET") {
    return;
  }
  if (url.pathname.startsWith("/static/vendor/")) {
    event.respondWith(
      caches.match(event.request).then((cached) => cached || fetch(event.request))
    );
    return;
  }
  event.respondWith(
    fetch(event.request).catch(() => caches.match(event.request))
  );
});
