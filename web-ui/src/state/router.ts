import { create } from "zustand";

// A tiny history-based client router. We intentionally avoid a dependency on
// react-router: the app has four screens (chat, cycles, cycle detail, memory)
// and this keeps the bundle small. The router parses window.location.pathname
// into a discriminated Route and re-renders on pushState/popstate.

export type Route =
  | { name: "chat" }
  | { name: "cycles" }
  | { name: "cycle"; id: string }
  | { name: "projects" }
  | { name: "project"; id: string }
  | { name: "memory" };

export function parsePath(pathname: string): Route {
  const parts = pathname.replace(/^\/+|\/+$/g, "").split("/");
  if (parts.length === 0 || parts[0] === "") return { name: "chat" };
  switch (parts[0]) {
    case "cycles":
      return parts[1] ? { name: "cycle", id: decodeURIComponent(parts[1]) } : { name: "cycles" };
    case "projects":
      return parts[1] ? { name: "project", id: decodeURIComponent(parts[1]) } : { name: "projects" };
    case "memory":
      return { name: "memory" };
    default:
      // Unknown path → fall back to the chat screen (the SPA is chat-centric).
      return { name: "chat" };
  }
}

interface RouterState {
  route: Route;
  /** Initialise the popstate listener. Safe to call once at app boot. */
  init: () => () => void;
  /** Push a new path onto the history stack and update the route. */
  navigate: (path: string) => void;
  /** Recompute the route from the current location (used after popstate). */
  sync: () => void;
}

export const useRouter = create<RouterState>((set, get) => ({
  route: typeof window !== "undefined" ? parsePath(window.location.pathname) : { name: "chat" },

  init: () => {
    const onPop = () => get().sync();
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  },

  navigate: (path) => {
    const target = path.startsWith("/") ? path : "/" + path;
    if (target === window.location.pathname) {
      get().sync();
      return;
    }
    window.history.pushState({}, "", target);
    set({ route: parsePath(target) });
    // Scroll to top on navigation — each screen starts at the top. Guard for
    // environments where scrollTo is unavailable (jsdom).
    try {
      window.scrollTo(0, 0);
    } catch {
      // no-op
    }
  },

  sync: () => set({ route: parsePath(window.location.pathname) }),
}));

/** Convenience hook returning the active route. */
export function useRoute(): Route {
  return useRouter((s) => s.route);
}
