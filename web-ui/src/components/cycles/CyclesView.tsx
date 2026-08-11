import { useEffect } from "react";
import { useCyclesStore } from "@/state/cyclesStore";
import { useRouter } from "@/state/router";
import { Spinner } from "@/components/ui/Spinner";
import type { CycleStatus, CycleSummary } from "@/state/types";

function formatTime(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "—" : d.toLocaleString([], {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

const STATUS_STYLES: Record<string, string> = {
  running: "bg-accent/20 text-accent",
  merged: "bg-emerald-500/20 text-emerald-300",
  rejected: "bg-red-500/20 text-red-300",
  stuck: "bg-zinc-500/20 text-zinc-400",
};

function statusBadgeClass(status: CycleStatus): string {
  return STATUS_STYLES[status] ?? "bg-panel2 text-zinc-400";
}

function CycleCard({ cycle }: { cycle: CycleSummary }) {
  const navigate = useRouter((s) => s.navigate);
  const stepCount = cycle.steps.length;
  const lastStep = cycle.steps[stepCount - 1]?.step ?? 0;

  return (
    <button
      type="button"
      onClick={() => navigate(`/cycles/${cycle.id}`)}
      className="w-full rounded-lg border border-edge bg-panel px-4 py-3 text-left transition-colors hover:border-accent/60 hover:bg-panel2"
    >
      <div className="flex items-start gap-3">
        <span className={`mt-0.5 shrink-0 rounded-md px-2 py-0.5 text-[11px] font-medium uppercase ${statusBadgeClass(cycle.status)}`}>
          {cycle.status}
        </span>
        <div className="min-w-0 flex-1">
          <p className="line-clamp-2 text-sm text-zinc-100">{cycle.objective || "(no objective)"}</p>
          <p className="mt-1 truncate text-[11px] text-zinc-500">
            {cycle.id} · branch {cycle.branch || "—"} · {stepCount} step{stepCount === 1 ? "" : "s"} (last #{lastStep}) · {formatTime(cycle.started_at)}
          </p>
          {(cycle.ai_verdict || cycle.human_verdict) && (
            <p className="mt-1 text-[11px] text-zinc-500">
              {cycle.ai_verdict && <span className="mr-2">AI: {cycle.ai_verdict}</span>}
              {cycle.human_verdict && <span>human: {cycle.human_verdict}</span>}
            </p>
          )}
        </div>
      </div>
    </button>
  );
}

export function CyclesView() {
  const cycles = useCyclesStore((s) => s.cycles);
  const loadCycles = useCyclesStore((s) => s.loadCycles);
  const error = useCyclesStore((s) => s.error);

  useEffect(() => {
    void loadCycles();
  }, [loadCycles]);

  // Poll for progress while any cycle is running.
  useEffect(() => {
    if (!cycles.some((c) => c.status === "running")) return;
    const id = window.setInterval(() => void loadCycles(), 5000);
    return () => window.clearInterval(id);
  }, [loadCycles, cycles]);

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-3xl px-4 py-6">
        <h1 className="mb-4 text-lg font-medium text-zinc-100">Self-improvement cycles</h1>
        {error && <p className="mb-3 text-sm text-red-400">{error}</p>}
        {cycles.length === 0 ? (
          <p className="py-10 text-center text-sm text-zinc-600">
            No cycles yet. Run one from the chat or <code className="text-zinc-400">nelke improve</code>.
          </p>
        ) : (
          <div className="space-y-2">
            {cycles.map((c) => <CycleCard key={c.id} cycle={c} />)}
          </div>
        )}
      </div>
    </div>
  );
}

export function CyclesLoading() {
  return (
    <div className="flex flex-1 items-center justify-center">
      <Spinner className="h-6 w-6" />
    </div>
  );
}
