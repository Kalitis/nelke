import { beforeEach, describe, expect, it } from "vitest";
import { parsePath, useRouter } from "./router";

describe("parsePath", () => {
  it("routes / to chat", () => {
    expect(parsePath("/")).toEqual({ name: "chat" });
    expect(parsePath("")).toEqual({ name: "chat" });
  });

  it("routes /cycles to cycles list", () => {
    expect(parsePath("/cycles")).toEqual({ name: "cycles" });
  });

  it("routes /cycles/:id to cycle detail", () => {
    expect(parsePath("/cycles/abc123")).toEqual({ name: "cycle", id: "abc123" });
  });

  it("routes /memory to memory", () => {
    expect(parsePath("/memory")).toEqual({ name: "memory" });
  });

  it("routes /projects to projects list", () => {
    expect(parsePath("/projects")).toEqual({ name: "projects" });
  });

  it("routes /projects/:id to project detail", () => {
    expect(parsePath("/projects/abc-1")).toEqual({ name: "project", id: "abc-1" });
  });

  it("falls back to chat for unknown paths", () => {
    expect(parsePath("/whatever/else")).toEqual({ name: "chat" });
  });

  it("decodes the cycle id", () => {
    expect(parsePath("/cycles/20260811-7f96")).toEqual({ name: "cycle", id: "20260811-7f96" });
  });
});

describe("useRouter navigate", () => {
  beforeEach(() => {
    window.history.replaceState({}, "", "/");
    useRouter.setState({ route: { name: "chat" } });
  });

  it("pushes the path and updates the route", () => {
    useRouter.getState().navigate("/cycles");
    expect(window.location.pathname).toBe("/cycles");
    expect(useRouter.getState().route).toEqual({ name: "cycles" });
  });

  it("updates the cycle detail route with an id", () => {
    useRouter.getState().navigate("/cycles/abc-1");
    expect(useRouter.getState().route).toEqual({ name: "cycle", id: "abc-1" });
  });

  it("no-ops the history when navigating to the current path", () => {
    window.history.replaceState({}, "", "/memory");
    useRouter.getState().navigate("/memory");
    expect(window.location.pathname).toBe("/memory");
    expect(useRouter.getState().route).toEqual({ name: "memory" });
  });
});

describe("useRouter popstate", () => {
  beforeEach(() => {
    window.history.replaceState({}, "", "/");
    useRouter.setState({ route: { name: "chat" } });
  });

  it("syncs the route when the user hits back/forward", () => {
    const cleanup = useRouter.getState().init();
    try {
      window.history.pushState({}, "", "/memory");
      window.dispatchEvent(new PopStateEvent("popstate"));
      expect(useRouter.getState().route).toEqual({ name: "memory" });
    } finally {
      cleanup();
    }
  });
});
