import { useEffect, useState } from "react";
import { useProjectsStore } from "@/state/projectsStore";
import { useRouter } from "@/state/router";
import { Spinner } from "@/components/ui/Spinner";
import { Markdown } from "@/components/chat/Markdown";
import { api } from "@/api/client";

function formatTime(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "—" : d.toLocaleString([], {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

function shortId(id: string): string {
  return id.length > 10 ? id.slice(-8) : id;
}

function frontendLabel(frontend: string): string {
  return ({ telegram: "tg", tui: "tui", web: "web" }[frontend] ?? frontend) || "?";
}

const MEMORY_PREFIX = (projectId: string) => `projects/${projectId}/`;

export function ProjectDetailView({ projectId }: { projectId: string }) {
  const detail = useProjectsStore((s) => s.detail);
  const loading = useProjectsStore((s) => s.loading);
  const error = useProjectsStore((s) => s.error);
  const storeError = error;
  const loadDetail = useProjectsStore((s) => s.loadDetail);
  const clearDetail = useProjectsStore((s) => s.clearDetail);
  const updateProject = useProjectsStore((s) => s.updateProject);
  const deleteProject = useProjectsStore((s) => s.deleteProject);
  const navigate = useRouter((s) => s.navigate);

  // Inline stage edit state.
  const [editingStage, setEditingStage] = useState(false);
  const [stageDraft, setStageDraft] = useState("");
  // Selected memory file (relative name within the project, e.g. "notes.md").
  const [selectedMemory, setSelectedMemory] = useState<string | null>(null);
  const [memoryContent, setMemoryContent] = useState<string>("");
  const [memoryLoading, setMemoryLoading] = useState(false);

  useEffect(() => {
    void loadDetail(projectId);
    return () => clearDetail();
  }, [projectId, loadDetail, clearDetail]);

  // Load the first memory file by default when the project loads.
  useEffect(() => {
    if (detail && detail.id === projectId && !selectedMemory) {
      const first = detail.memory_files[0];
      if (first) setSelectedMemory(first.name);
    }
  }, [detail, projectId, selectedMemory]);

  // Fetch the selected memory note's content (via the existing /api/memory
  // endpoint, which serves nested paths under the global memory root).
  useEffect(() => {
    if (!detail || !selectedMemory) {
      setMemoryContent("");
      return;
    }
    let cancelled = false;
    setMemoryLoading(true);
    api
      .memoryFile(MEMORY_PREFIX(detail.id) + selectedMemory)
      .then((r) => {
        if (!cancelled) setMemoryContent(r.content);
      })
      .catch(() => {
        if (!cancelled) setMemoryContent("(could not load note)");
      })
      .finally(() => {
        if (!cancelled) setMemoryLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [detail, selectedMemory]);

  if (loading && !detail) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <Spinner className="h-6 w-6" />
      </div>
    );
  }
  if (storeError && !detail) {
    return (
      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl px-4 py-6">
          <button type="button" onClick={() => navigate("/projects")} className="mb-3 text-sm text-accent hover:underline">← All projects</button>
          <p className="text-sm text-red-400">{storeError}</p>
        </div>
      </div>
    );
  }
  if (!detail) return null;

  const saveStage = async () => {
    const ok = await updateProject(detail.id, { stage: stageDraft.trim() });
    if (ok) setEditingStage(false);
  };

  const onDelete = async () => {
    if (!window.confirm(`Delete project "${detail.name}"? Its chats are kept but unlinked.`)) return;
    const ok = await deleteProject(detail.id);
    if (ok) navigate("/projects");
  };

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-3xl px-4 py-6">
        <button type="button" onClick={() => navigate("/projects")} className="mb-3 text-sm text-accent hover:underline">← All projects</button>

        {/* Project card header */}
        <div className="mb-1 flex items-center gap-2">
          <h1 className="text-lg font-medium text-zinc-100">{detail.name}</h1>
          {editingStage ? (
            <span className="flex items-center gap-1">
              <input
                type="text"
                value={stageDraft}
                onChange={(e) => setStageDraft(e.target.value)}
                placeholder="idea"
                className="w-24 rounded-md border border-edge bg-canvas px-2 py-0.5 text-[12px] text-zinc-100 focus:border-accent focus:outline-none"
                autoFocus
                onKeyDown={(e) => {
                  if (e.key === "Enter") void saveStage();
                  if (e.key === "Escape") setEditingStage(false);
                }}
              />
              <button type="button" onClick={() => void saveStage()} className="text-[11px] text-accent hover:underline">save</button>
              <button type="button" onClick={() => setEditingStage(false)} className="text-[11px] text-zinc-500 hover:underline">cancel</button>
            </span>
          ) : (
            <button
              type="button"
              title="Edit stage"
              onClick={() => {
                setStageDraft(detail.stage);
                setEditingStage(true);
              }}
              className={`rounded-md px-2 py-0.5 text-[11px] uppercase ${detail.stage ? "bg-accent/20 text-accent" : "bg-panel2 text-zinc-500"} hover:opacity-80`}
            >
              {detail.stage || "set stage"}
            </button>
          )}
        </div>
        <p className="mb-3 text-[11px] text-zinc-500">
          id {detail.id} · created {formatTime(detail.created_at)} · updated {formatTime(detail.updated_at)}
        </p>

        {/* Description */}
        <div className="mb-6 rounded-lg border border-edge bg-panel px-4 py-3">
          <p className="mb-1 text-[11px] uppercase text-zinc-500">Description</p>
          {detail.description ? (
            <div className="text-sm text-zinc-200">
              <Markdown content={detail.description} />
            </div>
          ) : (
            <p className="text-sm italic text-zinc-600">(no description)</p>
          )}
        </div>

        {/* Memory + Chats split */}
        <div className="grid gap-4 md:grid-cols-[16rem_1fr]">
          {/* Memory column */}
          <section>
            <h2 className="mb-2 text-sm font-medium text-zinc-300">
              Memory ({detail.memory_files.length})
            </h2>
            {detail.memory_files.length === 0 ? (
              <p className="text-sm text-zinc-600">No memory notes yet.</p>
            ) : (
              <ul className="space-y-1">
                {detail.memory_files.map((m) => (
                  <li key={m.name}>
                    <button
                      type="button"
                      onClick={() => setSelectedMemory(m.name)}
                      className={
                        "block w-full rounded-lg px-2.5 py-2 text-left text-sm transition-colors " +
                        (selectedMemory === m.name
                          ? "bg-panel2 text-zinc-100"
                          : "text-zinc-300 hover:bg-panel/60")
                      }
                    >
                      <span className="block truncate">{m.name}</span>
                      <span className="text-[11px] text-zinc-500">{m.size} B</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </section>

          {/* Detail column: memory content + chats */}
          <section className="min-w-0 space-y-6">
            <div>
              <h2 className="mb-2 text-sm font-medium text-zinc-300">
                {selectedMemory ? `Note: ${selectedMemory}` : "Memory note"}
              </h2>
              <div className="rounded-lg border border-edge bg-panel px-4 py-3">
                {memoryLoading ? (
                  <Spinner className="h-5 w-5" />
                ) : selectedMemory ? (
                  <div className="prose prose-invert max-w-none text-sm text-zinc-200">
                    <Markdown content={memoryContent} />
                  </div>
                ) : (
                  <p className="text-sm italic text-zinc-600">Select a memory note on the left.</p>
                )}
              </div>
            </div>

            <div>
              <h2 className="mb-2 text-sm font-medium text-zinc-300">
                Chats ({detail.chats.length})
              </h2>
              {detail.chats.length === 0 ? (
                <p className="text-sm text-zinc-600">No chats linked to this project.</p>
              ) : (
                <ul className="space-y-1">
                  {detail.chats.map((c) => (
                    <li
                      key={c.id}
                      className="flex items-center gap-2 rounded-md border border-edge bg-panel px-3 py-2 text-sm"
                    >
                      <span className="rounded bg-panel2 px-1.5 py-0.5 text-[10px] uppercase text-zinc-400">
                        {frontendLabel(c.frontend)}
                      </span>
                      <span className="font-mono text-[12px] text-zinc-300">{shortId(c.id)}</span>
                      <span className="text-[11px] text-zinc-500">
                        · {c.message_count ?? 0} msgs · {formatTime(c.started_at)}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </section>
        </div>

        <div className="mt-8 border-t border-edge pt-4">
          <button
            type="button"
            onClick={() => void onDelete()}
            className="text-[12px] text-red-400 hover:underline"
          >
            Delete project
          </button>
        </div>
      </div>
    </div>
  );
}
