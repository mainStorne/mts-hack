/// <reference types="vite/client" />

interface ImportMetaEnv {
  // Base URL of the support agent backend
  readonly VITE_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
