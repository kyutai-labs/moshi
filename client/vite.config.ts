import { ProxyOptions, defineConfig, loadEnv } from "vite";
import topLevelAwait from "vite-plugin-top-level-await";

export default defineConfig(({mode}) => {
  const env = loadEnv(mode, process.cwd());
  const proxyConf:Record<string, string | ProxyOptions> = env.VITE_QUEUE_API_URL ? {
    "/api": {
      target: env.VITE_QUEUE_API_URL,
      changeOrigin: true,
    },
  } : {};
  return {
    server: {
      host: "0.0.0.0",
      https: {
        cert: "./cert.pem",
        key: "./key.pem",
      },
      proxy:{
        ...proxyConf,
      }
    },
    plugins: [
      topLevelAwait({
        // The export name of top-level await promise for each chunk module
        promiseExportName: "__tla",
        // The function to generate import names of top-level await promise in each chunk module
        promiseImportName: i => `__tla_${i}`,
      }),
    ],
  };
});
