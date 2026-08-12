import { useEffect } from "react";
import { useCyclesStore } from "@/state/cyclesStore";
import { useRouter } from "@/state/router";
import { Spinner } from "@/components/ui/Spinner";
import { Markdown } from "@/components/chat/Markdown";
import { WorkerGrid } from "./WorkerGrid";
import type { CycleDetail, CycleEvent } from "@/state/types";

function formatTime(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "—" : d.toLocaleString([], {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

function payloadSummary(ev: CycleEvent): string {
  const p = ev.payload || {};
  if (typeof p.summary === "string") return p.summary;
  if (typeof p.commit_sha === "string") return `commit ${p.commit_sha.slice(0, 8)}`;
  if (typeof p.step === "number") return `step ${p.step}`;
  return "";
}

/**
 * One folded Timeline row. Consecutive `agent_token` events are coalesced into
 * a single `agent_text` row (grouped by worker), so streamed tokens read as
 * continuous prose instead of one row per token.
 */
interface TimelineRow {
  kind: string;
  seq: number;
  workerId: string | null;
  text: string;
  /** Present for pass-through rows; absent for folded `agent_text` rows. */
  ev?: CycleEvent;
}

function workerIdOf(ev: CycleEvent): string | null {
  const w = ev.payload?.worker_id;
  return typeof w === "string" && w.length > 0 ? w : null;
}

/**
 * Fold consecutive `agent_token` and `agent_text` events into one `agent_text`
 * row, grouped by worker. Newer cycles persist one `agent_text` per turn
 * (text in `payload.text`); legacy cycles stored one `agent_token` per token
 * (text in `payload.token`). Both are coalesced into continuous prose. Other
 * kinds pass through unchanged.
 */
function foldTimeline(events: CycleEvent[]): TimelineRow[] {
  const rows: TimelineRow[] = [];
  let acc: { workerId: string | null; text: string; seq: number } | null = null;
  const flush = () => {
    if (acc && acc.text.length > 0) {
      rows.push({ kind: "agent_text", seq: acc.seq, workerId: acc.workerId, text: acc.text });
    }
    acc = null;
  };
  for (const ev of events) {
    if (ev.kind === "agent_token" || ev.kind === "agent_text") {
      const chunk = ev.kind === "agent_text"
        ? ((ev.payload?.text as string | undefined) ?? ev.message ?? "")
        : ((ev.payload?.token as string | undefined) ?? ev.message ?? "");
      if (chunk.length === 0) continue;
      const wid = workerIdOf(ev);
      if (!acc || acc.workerId !== wid) {
        flush();
        acc = { workerId: wid, text: "", seq: ev.seq };
      }
      acc.text += chunk;
    } else {
      flush();
      rows.push({ kind: ev.kind, seq: ev.seq, workerId: workerIdOf(ev), text: ev.message, ev });
    }
  }
  flush();
  return rows;
}

function TimelineRowView({ row, workerLabel }: { row: TimelineRow; workerLabel: string | null }) {
  if (row.kind === "agent_text") {
    return (
      <li className="py-1.5 text-sm">
        <details className="group rounded-md border border-edge bg-panel px-3 py-2">
          <summary className="flex cursor-pointer items-center gap-2 text-[12px] text-zinc-300 hover:text-zinc-100">
            <span className="text-[11px] text-zinc-600">#{row.seq}</span>
            <span className="text-[11px] text-zinc-500">agent_text</span>
            {workerLabel && <span className="rounded bg-panel2 px-1.5 py-0.5 text-[10px] text-zinc-400">{workerLabel}</span>}
            <span className="ml-auto text-[11px] text-zinc-600">{row.text.length} chars</span>
          </summary>
          <div className="mt-2 border-t border-edge pt-2 text-sm text-zinc-200">
            <Markdown content={row.text} />
          </div>
        </details>
      </li>
    );
  }
  // Non-token rows reuse the compact event layout.
  const step = (row.ev?.payload?.step as number | undefined) ?? row.seq;
  const summary = row.ev ? payloadSummary(row.ev) : "";
  const isTool = row.kind === "agent_tool" || row.kind === "agent_tool_result";
  return (
    <li className="flex gap-3 py-1.5 text-sm">
      <span className="w-16 shrink-0 text-[11px] text-zinc-600">#{step}</span>
      <span className="w-28 shrink-0 text-[11px] text-zinc-500">{row.kind}</span>
      <span className={`min-w-0 flex-1 ${isTool ? "font-mono text-[12px] text-zinc-400" : "text-zinc-200"}`}>
        {row.text}
        {workerLabel && row.workerId && <span className="ml-2 text-zinc-600">· {workerLabel}</span>}
        {summary && <span className="ml-2 text-zinc-600">· {summary}</span>}
      </span>
    </li>
  );
}

export function CycleDetailView({ cycleId }: { cycleId: string }) {
  const detail = useCyclesStore((s) => s.detail);
  const loading = useCyclesStore((s) => s.loading);
  const error = useCyclesStore((s) => s.error);
  const loadDetail = useCyclesStore((s) => s.loadDetail);
  const clearDetail = useCyclesStore((s) => s.clearDetail);
  const navigate = useRouter((s) => s.navigate);

  useEffect(() => {
    void loadDetail(cycleId);
    return () => clearDetail();
  }, [cycleId, loadDetail, clearDetail]);

  // Poll while running.
  useEffect(() => {
    if (detail?.status !== "running") return;
    const id = window.setInterval(() => void loadDetail(cycleId), 5000);
    return () => window.clearInterval(id);
  }, [detail?.status, cycleId, loadDetail]);

  if (loading && !detail) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <Spinner className="h-6 w-6" />
      </div>
    );
  }
  if (error && !detail) {
    return (
      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl px-4 py-6">
          <button type="button" onClick={() => navigate("/cycles")} className="mb-3 text-sm text-accent hover:underline">← All cycles</button>
          <p className="text-sm text-red-400">{error}</p>
        </div>
      </div>
    );
  }
  if (!detail) return null;

  const d: CycleDetail = detail;
  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-3xl px-4 py-6">
        <button type="button" onClick={() => navigate("/cycles")} className="mb-3 text-sm text-accent hover:underline">← All cycles</button>
        <div className="mb-1 flex items-center gap-2">
          <h1 className="text-lg font-medium text-zinc-100">Cycle {d.id}</h1>
          <span className="rounded-md bg-panel2 px-2 py-0.5 text-[11px] uppercase text-zinc-400">{d.status}</span>
        </div>
        <p className="mb-3 text-[11px] text-zinc-500">
          branch {d.branch || "—"} · started {formatTime(d.started_at)} · ended {formatTime(d.ended_at)}
          {d.ai_verdict && <> · AI: {d.ai_verdict}</>}
          {d.human_verdict && <> · human: {d.human_verdict}</>}
        </p>
        <div className="mb-6 rounded-lg border border-edge bg-panel px-4 py-3">
          <p className="mb-1 text-[11px] uppercase text-zinc-500">Objective</p>
          <div className="text-sm text-zinc-200">
            <Markdown content={d.objective || "(no objective)"} />
          </div>
        </div>

        {/* Parallel worker cards (only populated by parallel-mode cycles). */}
        <WorkerGrid cycleId={d.id} />

        <h2 className="mb-2 text-sm font-medium text-zinc-300">Steps</h2>
        {d.steps.length === 0 ? (
          <p className="mb-6 text-sm text-zinc-600">No steps recorded yet.</p>
        ) : (
          <ul className="mb-6 space-y-1">
            {d.steps.map((s) => (
              <li key={s.step} className="flex gap-3 rounded-md border border-edge bg-panel px-3 py-2 text-sm">
                <span className="w-10 shrink-0 text-[11px] text-zinc-600">#{s.step}</span>
                <span className="w-20 shrink-0 text-[11px] text-zinc-500">{s.status}</span>
                <span className="min-w-0 flex-1 text-zinc-200">{s.summary || (s.commit_sha ? `commit ${s.commit_sha.slice(0, 8)}` : "—")}</span>
              </li>
            ))}
          </ul>
        )}

        <h2 className="mb-2 text-sm font-medium text-zinc-300">Timeline</h2>
        {d.events.length === 0 ? (
          <p className="text-sm text-zinc-600">No events yet.</p>
        ) : (
          <ul className="divide-y divide-edge">
            {(() => {
              const workerLabelById = new Map<string, string>();
              for (const w of d.workers) {
                workerLabelById.set(w.id, w.title || `worker ${w.worker_index}`);
              }
              return foldTimeline(d.events).map((row, i) => (
                <TimelineRowView
                  key={`${row.seq}-${i}`}
                  row={row}
                  workerLabel={row.workerId ? workerLabelById.get(row.workerId) ?? null : null}
                />
              ));
            })()}
          </ul>
        )}
      </div>
    </div>
  );
}
