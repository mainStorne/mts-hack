import type { MessageDescriptor } from "@lingui/core";

export type ToastText = string | MessageDescriptor;

export type ToastType = "success" | "error";

export interface Toast {
  id: string;
  type: ToastType;
  title: ToastText;
  message?: ToastText;
  duration?: number;
  isVisible?: boolean;
}

export interface ToastContextType {
  toasts: Toast[];
  showToast: (toast: Omit<Toast, "id">) => void;
  hideToast: (id: string) => void;
  showSuccess: (title: ToastText, message?: ToastText) => void;
  showError: (title: ToastText, message?: ToastText) => void;
}
