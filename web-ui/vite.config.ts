import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "node:path";

// In dev the FastAPI backend runs on :8000; Vite serves the SPA on :5173 and
// proxies /api so cookies/CORS stay simple. `npm run build` emits a static
// bundle into ../static/dist which the Python package serves directly.
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": resolve(__dirname, "src"),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
  build: {
    outDir: resolve(__dirname, "..", "static", "dist"),
    emptyOutDir: true,
    sourcemap: false,
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
  },
} as const);
