import "@/locales/lib";
import { queryClient } from "@/app/query-client.ts";
import { ToastContainer, ToastProvider } from "@/components/toast";
import { useTheme } from "@/lib/ui/use-theme.ts";
import { i18n } from "@lingui/core";
import { I18nProvider } from "@lingui/react";
import { QueryClientProvider } from "@tanstack/react-query";
import { Register, RouterProvider } from "@tanstack/react-router";
import { useEffect } from "react";

// Root app component
export function App({ router }: { router: Register["router"] }) {
  return (
    <I18nProvider i18n={i18n}>
      <QueryClientProvider client={queryClient}>
        <ToastProvider>
          <RouterProvider router={router} />
          <ToastContainer />
        </ToastProvider>

        <ThemeChanger />
      </QueryClientProvider>
    </I18nProvider>
  );
}

// Helper to change the theme class in <html> element
function ThemeChanger() {
  const { isDarkMode } = useTheme();
  useEffect(() => {
    document.documentElement.classList.toggle("dark", isDarkMode);
    document.documentElement.dataset.theme = isDarkMode ? "dark" : "light";
  }, [isDarkMode]);
  return null;
}
