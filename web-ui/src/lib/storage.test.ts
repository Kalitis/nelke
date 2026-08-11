import { beforeEach, describe, expect, it } from "vitest";
import { loadString, removeString, saveString } from "./storage";

beforeEach(() => {
  window.localStorage.clear();
});

describe("storage helpers", () => {
  it("round-trips a value under the nelke: prefix", () => {
    saveString("profile", "dslab");
    expect(window.localStorage.getItem("nelke:profile")).toBe("dslab");
    expect(loadString("profile")).toBe("dslab");
  });

  it("returns null for a missing key", () => {
    expect(loadString("nope")).toBeNull();
  });

  it("removes a value", () => {
    saveString("profile", "dslab");
    removeString("profile");
    expect(loadString("profile")).toBeNull();
  });

  it("does not throw when localStorage is unavailable", () => {
    // Simulate a hardened/private mode where touching localStorage throws.
    const original = Object.getOwnPropertyDescriptor(window, "localStorage");
    Object.defineProperty(window, "localStorage", {
      configurable: true,
      get() {
        throw new Error("unavailable");
      },
    });
    try {
      expect(loadString("profile")).toBeNull();
      expect(() => saveString("profile", "x")).not.toThrow();
      expect(() => removeString("profile")).not.toThrow();
    } finally {
      // Restore the real jsdom localStorage.
      Object.defineProperty(window, "localStorage", original!);
    }
  });
});
