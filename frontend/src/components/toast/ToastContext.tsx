import {
  createContext,
  PropsWithChildren,
  useCallback,
  useContext,
  useMemo,
  useState,
} from "react";
import { Toast, ToastContextType, ToastText } from "./types.ts";

const ToastContext = createContext<ToastContextType | undefined>(undefined);

const generateId = () => {
  return `toast-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
};

export function ToastProvider({ children }: PropsWithChildren) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const showToast = useCallback((toast: Omit<Toast, "id">) => {
    const newToast: Toast = {
      ...toast,
      id: generateId(),
      duration: toast.duration ?? 5000,
      isVisible: true,
    };

    setToasts((prev) => [...prev, newToast]);
  }, []);

  const hideToast = useCallback((id: string) => {
    // Mark the toast as leaving first and remove it once the animation is over
    setToasts((prev) =>
      prev.map((toast) =>
        toast.id === id ? { ...toast, isVisible: false } : toast,
      ),
    );

    setTimeout(() => {
      setToasts((prev) => prev.filter((toast) => toast.id !== id));
    }, 300);
  }, []);

  const showSuccess = useCallback(
    (title: ToastText, message?: ToastText) => {
      showToast({ type: "success", title, message });
    },
    [showToast],
  );

  const showError = useCallback(
    (title: ToastText, message?: ToastText) => {
      showToast({ type: "error", title, message });
    },
    [showToast],
  );

  const value = useMemo<ToastContextType>(
    () => ({ toasts, showToast, hideToast, showSuccess, showError }),
    [toasts, showToast, hideToast, showSuccess, showError],
  );

  return (
    <ToastContext.Provider value={value}>{children}</ToastContext.Provider>
  );
}

export function useToast() {
  const context = useContext(ToastContext);
  if (context === undefined) {
    throw new Error("useToast must be used within a ToastProvider");
  }
  return context;
}
