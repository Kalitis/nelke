import { useChatStore } from "@/state/chatStore";

export function TopBar({ onOpenSidebar }: { onOpenSidebar: () => void }) {
  const chat = useChatStore((s) => s.chat);
  const profiles = useChatStore((s) => s.profiles);
  const profile = useChatStore((s) => s.profile);
  const setProfile = useChatStore((s) => s.setProfile);
  const streaming = useChatStore((s) => s.streaming);

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
      <h1 className="flex-1 truncate text-sm font-medium text-zinc-200">
        {chat?.title || "Nelke"}
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
