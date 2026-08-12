import { useEffect } from "react";
import { useCyclesStore, type SubagentCard } from "@/state/cyclesStore";
import { ToolCallBlock, type ToolEntry } from "@/components/chat/ToolCallBlock";
import { Spinner } from "@/components/ui/Spinner";
import type { CycleWorker } from "@/state/types";

const STATUS_STYLES: Record<string, string> = {
  pending: "bg-panel2 text-zinc-400",
  running: "bg-accent/20 text-accent",
  done: "bg-emerald-500/20 text-emerald-300",
  error: "bg-red-500/20 text-red-300",
};

function statusClass(status: string): string {
  return STATUS_STYLES[status] ?? STATUS_STYLES.pending;
}

/**
 * One card per parallel worker. The card merges the persisted `worker` row
 * (static title/status from the DB) with its live state from the SSE stream
 * (`card` accumulates tokens/tools as the worker runs).
 */
function WorkerCard({ worker, card }: { worker: CycleWorker; card?: SubagentCard }) {
  const status = card?.status ?? worker.status;
  const title = card?.title ?? worker.title;
  const content = card?.content ?? "";
  const tools: ToolEntry[] = (card?.tools ?? []).map((t) => ({
    name: t.name,
    args: t.args,
    snippet: t.snippet,
  }));
  const isRunning = status === "running";

  return (
    <div className="flex min-h-[8rem] flex-col rounded-lg border border-edge bg-panel px-3 py-2.5">
      <div className="mb-2 flex items-center gap-2">
        <span className="shrink-0 rounded-md bg-panel2 px-1.5 py-0.5 text-[10px] font-mono text-zinc-500">
          #{worker.worker_index}
        </span>
        <span className="min-w-0 flex-1 truncate text-sm font-medium text-zinc-100" title={title}>
          {title}
        </span>
        <span className={`shrink-0 rounded-md px-1.5 py-0.5 text-[10px] uppercase ${statusClass(status)}`}>
          {status}
        </span>
      </div>

      {content && (
        <pre className="mb-2 max-h-40 overflow-auto whitespace-pre-wrap break-words rounded border border-edge bg-canvas/60 px-2 py-1.5 font-mono text-[11px] text-zinc-300">
          {content}
        </pre>
      )}

      {tools.length > 0 && (
        <div className="mb-1">
          {tools.map((t, i) => (
            <ToolCallBlock key={i} tool={t} />
          ))}
        </div>
      )}

      <div className="mt-auto flex items-center gap-2 text-[11px] text-zinc-600">
        {isRunning ? (
          <>
            <Spinner className="h-3 w-3" />
            {card?.lastEventKind ? <span className="truncate">{card.lastEventKind}…</span> : <span>working…</span>}
          </>
        ) : card?.lastEvent ? (
          <span className="truncate">{card.lastEvent}</span>
        ) : worker.status === "pending" ? (
          <span>queued</span>
        ) : (
          <span>{worker.status}</span>
        )}
      </div>
    </div>
  );
}

/**
 * Renders a grid of worker cards for a cycle. When the cycle is running it
 * subscribes to the live SSE stream so each card fills in as its worker runs;
 * on completion the canonical `detail.workers` drives the final view.
 */
export function WorkerGrid({ cycleId }: { cycleId: string }) {
  const detail = useCyclesStore((s) => s.detail);
  const liveWorkers = useCyclesStore((s) => s.liveWorkers);
  const startLive = useCyclesStore((s) => s.startLive);
  const stopLive = useCyclesStore((s) => s.stopLive);
  const isRunning = detail?.status === "running";

  useEffect(() => {
    if (!isRunning) return;
    startLive(cycleId);
    return () => {
      // Keep the subscription across re-renders; only stop when the grid
      // unmounts or the cycle finishes (the cycle_result event refreshes
      // detail so the next render drops `isRunning`).
      stopLive();
    };
  }, [isRunning, cycleId, startLive, stopLive]);

  if (!detail || detail.workers.length === 0) return null;

  return (
    <div className="mb-6">
      <h2 className="mb-2 text-sm font-medium text-zinc-300">Workers</h2>
      <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
        {detail.workers.map((w) => (
          <WorkerCard key={w.id} worker={w} card={liveWorkers[w.id]} />
        ))}
      </div>
    </div>
  );
}
