import { useTheme } from "@/lib/ui/use-theme.ts";
import { useLingui } from "@lingui/react/macro";

export default function SwitchThemeButton() {
  const { setTernaryDarkMode } = useTheme();
  const { t } = useLingui();

  return (
    <button
      type="button"
      title={t`Switch theme`}
      className="hover:bg-base-300 flex items-center justify-center rounded-xl p-2"
      onClick={() =>
        setTernaryDarkMode((prev) => (prev === "dark" ? "light" : "dark"))
      }
    >
      <span className="icon-[material-symbols--light-mode-outline] flex text-2xl text-[#F0B132] dark:hidden" />
      <span className="icon-[material-symbols--dark-mode-outline] text-primary hidden text-2xl dark:flex" />
    </button>
  );
}
