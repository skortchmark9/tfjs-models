const VERSION = "v7";
const CACHE_NAME = `flexion-friend-${VERSION}`;

// const APP_STATIC_RESOURCES = [
//   "/",
//   "/index.html",
//   "/app.js",
//   "/cycletrack.json",
//   "/icons/wheel.svg",
// ];


// self.addEventListener("install", (event) => {
//   event.waitUntil(
//     (async () => {
//       const cache = await caches.open(CACHE_NAME);
//       cache.addAll(APP_STATIC_RESOURCES);
//     })()
//   );
// });


self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const names = await caches.keys();
      await Promise.all(
        names.map((name) => {
          if (name !== CACHE_NAME) {
            return caches.delete(name);
          }
        })
      );
      await clients.claim();
    })()
  );
});

/**
 * After the first load, all the assets should be cached.
 */
self.addEventListener("fetch", (event) => {
  event.respondWith((async () => {
    const cache = await caches.open(CACHE_NAME);
    const cachedResponse = await cache.match(event.request);
    if (cachedResponse) {
      // Return the cached response if it's available.
      return cachedResponse;
    } else {
      const response = await fetch(event.request);
      cache.put(event.request, response.clone());
      return response;
    }
  })());
});
