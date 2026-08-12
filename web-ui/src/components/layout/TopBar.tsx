import { useChatStore } from "@/state/chatStore";
import { useRouter, useRoute } from "@/state/router";
import type { Route } from "@/state/router";

const NAV_ITEMS: { label: string; path: string; route: Route["name"] }[] = [
  { label: "Chat", path: "/", route: "chat" },
  { label: "Cycles", path: "/cycles", route: "cycles" },
  { label: "Memory", path: "/memory", route: "memory" },
];

export function TopBar({ onOpenSidebar }: { onOpenSidebar: () => void }) {
  const chat = useChatStore((s) => s.chat);
  const profiles = useChatStore((s) => s.profiles);
  const profile = useChatStore((s) => s.profile);
  const setProfile = useChatStore((s) => s.setProfile);
  const streaming = useChatStore((s) => s.streaming);
  const navigate = useRouter((s) => s.navigate);
  const route = useRoute();

  return (
    <header className="flex items-center gap-3 border-b border-edge bg-panel/70 px-4 py-2.5 backdrop-blur">
      <button
        type="button"
        onClick={onOpenSidebar}
        className="rounded-lg p-2 text-zinc-400 hover:bg-panel2 lg:hidden"
        aria-label="Open sidebar"
      >
        ☰
      </button>
      <nav className="flex items-center gap-1" aria-label="Primary">
        {NAV_ITEMS.map((item) => {
          // "cycle" detail screen is part of the cycles section.
          const active = route.name === item.route || (item.route === "cycles" && route.name === "cycle");
          return (
            <button
              key={item.path}
              type="button"
              onClick={() => navigate(item.path)}
              className={
                "rounded-lg px-2.5 py-1 text-sm transition-colors " +
                (active ? "bg-panel2 text-accent" : "text-zinc-400 hover:bg-panel2 hover:text-zinc-100")
              }
              aria-current={active ? "page" : undefined}
            >
              {item.label}
            </button>
          );
        })}
      </nav>
      <h1 className="flex-1 truncate text-sm font-medium text-zinc-200">
        {route.name === "chat" ? (chat?.title || "Nelke") : ""}
      </h1>
      <select
        value={profile ?? undefined}
        onChange={(e) => setProfile(e.target.value)}
        className="rounded-lg border border-edge bg-panel2 px-2 py-1 text-xs text-zinc-300 focus:border-accent focus:outline-none"
        aria-label="Provider profile"
      >
        {profiles.map((p) => (
          <option key={p.name} value={p.name}>
            {p.name} · {p.model}
          </option>
        ))}
      </select>
      {streaming && (
        <span className="flex items-center gap-1.5 text-xs text-accent">
          <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-accent" />
          streaming
        </span>
      )}
    </header>
  );
}
