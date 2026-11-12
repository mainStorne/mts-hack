import { i18n, type MessageDescriptor } from "@lingui/core";
import { msg } from "@lingui/core/macro";

// An error response of the backend: `{ detail: string }` with an HTTP status
export class ApiError extends Error {
  readonly status: number;
  readonly detail: string | null;

  constructor(status: number, detail: string | null) {
    super(detail ?? `Error ${status}`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }

  static async fromResponse(response: Response): Promise<ApiError> {
    let detail: string | null = null;
    try {
      const body: unknown = await response.json();
      const value = (body as { detail?: unknown } | null)?.detail;
      if (typeof value === "string") detail = value;
      else if (value != null) detail = JSON.stringify(value);
    } catch {
      // The body is not JSON
    }
    return new ApiError(response.status, detail);
  }
}

export function formatApiErrorMessage(error: unknown): string {
  const descriptor = formatApiErrorDescriptor(error);
  return typeof descriptor === "string" ? descriptor : i18n._(descriptor);
}

export function formatApiErrorDescriptor(
  error: unknown,
): string | MessageDescriptor {
  if (error instanceof ApiError) {
    // The detail of a 503 tells that the assistant is unavailable: it is shown as it is
    if (error.detail) return error.detail;
    const code = error.status.toString();
    return msg`Error ${code}`;
  }
  if (error instanceof TypeError) {
    const message = error.message;
    return msg`Network error: ${message}`;
  }
  if (error instanceof Error) return error.message;
  return error == null ? msg`Unknown error` : String(error);
}
