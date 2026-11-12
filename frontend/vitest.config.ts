import { lingui } from "@lingui/vite-plugin";
import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [lingui({ macroTransform: true })],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  test: {
    environment: "jsdom",
    // Node's own (experimental) localStorage would shadow the one of jsdom
    execArgv: ["--no-experimental-webstorage"],
    include: ["src/**/*.test.ts"],
    setupFiles: ["src/test/setupI18n.ts"],
  },
});
