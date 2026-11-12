import { cn } from "@/lib/ui/cn";
import { useLingui } from "@lingui/react/macro";
import { useCallback, useEffect, useState } from "react";
import { Toast, ToastType } from "./types.ts";

const toastBorders: Record<ToastType, string> = {
  success: "border-green-500",
  error: "border-red-500",
};

const toastColors: Record<ToastType, string> = {
  success: "text-green-500",
  error: "text-red-500",
};

const toastIcons: Record<ToastType, string> = {
  success: "icon-[material-symbols--check-circle-outline]",
  error: "icon-[material-symbols--error-outline]",
};

export function ToastItem({
  toast,
  onClose,
}: {
  toast: Toast;
  onClose: (id: string) => void;
}) {
  const { i18n, t } = useLingui();
  const [isEntering, setIsEntering] = useState(true);

  const handleClose = useCallback(() => onClose(toast.id), [onClose, toast.id]);

  // Start the appearing animation
  useEffect(() => {
    const timer = setTimeout(() => setIsEntering(false), 50);
    return () => clearTimeout(timer);
  }, []);

  // Close automatically when the duration is set
  useEffect(() => {
    if (toast.duration && toast.duration > 0) {
      const timer = setTimeout(handleClose, toast.duration);
      return () => clearTimeout(timer);
    }
  }, [toast.duration, handleClose]);

  return (
    <div
      role="status"
      className={cn(
        "alert transition-all duration-300 ease-in-out",
        "transform-gpu",
        (toast.isVisible === false || isEntering) &&
          "-translate-y-full opacity-0 sm:translate-x-full sm:translate-y-0",
        toastBorders[toast.type],
      )}
    >
      <span
        className={cn(
          "text-2xl",
          toastColors[toast.type],
          toastIcons[toast.type],
        )}
      />
      <div>
        <h4 className="font-bold">
          {typeof toast.title === "string" ? toast.title : i18n._(toast.title)}
        </h4>
        {toast.message && (
          <p className="text-xs">
            {typeof toast.message === "string"
              ? toast.message
              : i18n._(toast.message)}
          </p>
        )}
      </div>
      <button
        type="button"
        title={t`Close notification`}
        onClick={handleClose}
        className="btn btn-square btn-sm"
      >
        <span className="icon-[material-symbols--close] text-lg" />
      </button>
    </div>
  );
}
