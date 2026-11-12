import { LOCALE_LABELS, useAppLocale } from "@/locales/lib";
import { useLingui } from "@lingui/react/macro";

export function SwitchLocaleButton() {
  const { t } = useLingui();
  const [locale, setLocale] = useAppLocale();

  return (
    <button
      type="button"
      title={t`Switch language`}
      className="hover:bg-base-300 text-base-content/70 hover:text-base-content flex size-10 items-center justify-center rounded-xl text-sm font-semibold tracking-wide"
      onClick={() => setLocale(locale === "ru" ? "en" : "ru")}
    >
      {LOCALE_LABELS[locale]}
    </button>
  );
}
