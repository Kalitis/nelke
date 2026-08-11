import { useEffect } from "react";
import { useMemoryStore } from "@/state/memoryStore";
import { Markdown } from "@/components/chat/Markdown";
import { Spinner } from "@/components/ui/Spinner";

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function fileName(path: string): string {
  const parts = path.split("/");
  return parts[parts.length - 1] || path;
}

export function MemoryView() {
  const files = useMemoryStore((s) => s.files);
  const selectedName = useMemoryStore((s) => s.selectedName);
  const content = useMemoryStore((s) => s.content);
  const loading = useMemoryStore((s) => s.loading);
  const error = useMemoryStore((s) => s.error);
  const load = useMemoryStore((s) => s.load);
  const select = useMemoryStore((s) => s.select);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="flex flex-1 min-h-0">
      {/* File list */}
      <aside className="w-72 shrink-0 overflow-y-auto border-r border-edge bg-panel">
        <div className="px-4 py-3">
          <h1 className="text-sm font-medium text-zinc-100">Memory</h1>
          <p className="text-[11px] text-zinc-500">{files.length} file{files.length === 1 ? "" : "s"}</p>
        </div>
        <nav className="px-2 pb-3">
          {files.length === 0 ? (
            <p className="px-2 py-6 text-center text-sm text-zinc-600">No memory files</p>
          ) : (
            <ul className="space-y-0.5">
              {files.map((f) => {
                const active = f.name === selectedName;
                return (
                  <li key={f.name}>
                    <button
                      type="button"
                      onClick={() => void select(f.name)}
                      className={
                        "block w-full rounded-lg px-2.5 py-2 text-left text-sm transition-colors " +
                        (active ? "bg-panel2 text-zinc-100" : "text-zinc-300 hover:bg-panel/60")
                      }
                      title={f.name}
                    >
                      <div className="truncate">{fileName(f.name)}</div>
                      <div className="truncate text-[11px] text-zinc-500">
                        {f.name} · {formatSize(f.size)}
                      </div>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </nav>
      </aside>

      {/* Content viewer */}
      <div className="flex min-w-0 flex-1 flex-col">
        {error ? (
          <div className="flex-1 px-4 py-6">
            <p className="text-sm text-red-400">{error}</p>
          </div>
        ) : !selectedName ? (
          <div className="flex flex-1 items-center justify-center px-4 text-center text-sm text-zinc-600">
            Select a memory file to view its contents.
          </div>
        ) : loading ? (
          <div className="flex flex-1 items-center justify-center">
            <Spinner className="h-6 w-6" />
          </div>
        ) : (
          <div className="flex-1 overflow-y-auto">
            <div className="mx-auto max-w-3xl px-4 py-6">
              <p className="mb-3 text-[11px] text-zinc-500">{selectedName}</p>
              <Markdown content={content} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
