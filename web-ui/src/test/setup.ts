import "@testing-library/jest-dom/vitest";

// jsdom ships scrolling as a "not implemented" stub that logs a noisy error
// to stderr and throws. Our router scrolls to top on navigation; provide a
// real no-op so tests stay quiet and pass regardless of the catch block.
Object.defineProperty(window, "scrollTo", {
  writable: true,
  value: () => {},
});
