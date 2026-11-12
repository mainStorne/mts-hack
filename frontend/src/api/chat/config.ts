// Base URL of the backend; CORS is open, so the browser calls it directly
export const API_URL = (
  import.meta.env.VITE_API_URL || "http://localhost:8000"
).replace(/\/+$/, "");
