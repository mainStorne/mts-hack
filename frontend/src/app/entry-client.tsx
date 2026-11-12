// Must be first among app imports: `@/locales/lib` activates Lingui on eval
// so module-scope `i18n._(...)` in the App tree does not race.
import { initializeI18n } from "@/locales/lib";
import { createRoot } from "react-dom/client";
import { App } from "./app.tsx";
import { getRouter } from "./router.ts";
import "./styles.css";

// Init internationalization
initializeI18n();

// Create the router instance
const router = getRouter();

// Render the app
const container = document.getElementById("app") as HTMLDivElement;
const root = createRoot(container);
root.render(<App router={router} />);
