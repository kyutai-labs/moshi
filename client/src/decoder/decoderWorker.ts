export const DecoderWorker = new Worker(
  new URL("/assets/decoderWorker.min.js", import.meta.url),
);
