import { describe, expect, it } from "vitest";
import { ApiError, formatApiErrorMessage } from "./errors";

describe("formatApiErrorMessage", () => {
  it("shows the detail of the backend as it is", () => {
    expect(
      formatApiErrorMessage(new ApiError(503, "Assistant is unavailable")),
    ).toBe("Assistant is unavailable");
  });
  it("falls back to the status, the network error text and plain errors", () => {
    expect(formatApiErrorMessage(new ApiError(500, null))).toBe("Error 500");
    expect(formatApiErrorMessage(new TypeError("Failed to fetch"))).toBe(
      "Network error: Failed to fetch",
    );
    expect(formatApiErrorMessage(new Error("Boom"))).toBe("Boom");
    expect(formatApiErrorMessage(undefined)).toBe("Unknown error");
  });
});
