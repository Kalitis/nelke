import { useEffect, useState } from "react";
import { Dialog, DialogPanel } from "@headlessui/react";
import { useProjectsStore } from "@/state/projectsStore";
import { useRouter } from "@/state/router";
import type { ProjectSummary } from "@/state/types";

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

/** A short badge for the project's free-form stage label. */
function StageBadge({ stage }: { stage: string }) {
  if (!stage) {
    return <span className="mt-0.5 shrink-0 rounded-md bg-panel2 px-2 py-0.5 text-[11px] uppercase text-zinc-500">no stage</span>;
  }
  return (
    <span className="mt-0.5 shrink-0 rounded-md bg-accent/20 px-2 py-0.5 text-[11px] font-medium uppercase text-accent">
      {stage}
    </span>
  );
}

function ProjectCard({ project }: { project: ProjectSummary }) {
  const navigate = useRouter((s) => s.navigate);
  return (
    <button
      type="button"
      onClick={() => navigate(`/projects/${project.id}`)}
      className="w-full rounded-lg border border-edge bg-panel px-4 py-3 text-left transition-colors hover:border-accent/60 hover:bg-panel2"
    >
      <div className="flex items-start gap-3">
        <StageBadge stage={project.stage} />
        <div className="min-w-0 flex-1">
          <p className="line-clamp-1 text-sm font-medium text-zinc-100">{project.name}</p>
          {project.description ? (
            <p className="mt-0.5 line-clamp-2 text-[12px] text-zinc-400">{project.description}</p>
          ) : (
            <p className="mt-0.5 line-clamp-1 text-[12px] italic text-zinc-600">(no description)</p>
          )}
          <p className="mt-1 truncate text-[11px] text-zinc-500">
            {project.chat_count} chat{project.chat_count === 1 ? "" : "s"} · updated {formatTime(project.updated_at)} · id {shortId(project.id)}
          </p>
        </div>
      </div>
    </button>
  );
}

export function ProjectsView() {
  const projects = useProjectsStore((s) => s.projects);
  const loadProjects = useProjectsStore((s) => s.loadProjects);
  const error = useProjectsStore((s) => s.error);
  const [modalOpen, setModalOpen] = useState(false);

  useEffect(() => {
    void loadProjects();
  }, [loadProjects]);

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-3xl px-4 py-6">
        <div className="mb-4 flex items-center justify-between gap-3">
          <h1 className="text-lg font-medium text-zinc-100">Projects</h1>
          <button
            type="button"
            onClick={() => setModalOpen(true)}
            className="rounded-lg border border-edge bg-panel2 px-3 py-1.5 text-sm font-medium text-zinc-100 transition-colors hover:bg-edge"
          >
            + New project
          </button>
        </div>
        {error && <p className="mb-3 text-sm text-red-400">{error}</p>}
        {projects.length === 0 ? (
          <p className="py-10 text-center text-sm text-zinc-600">
            No projects yet. Group related chats and memory under a project with “New project”.
          </p>
        ) : (
          <div className="space-y-2">
            {projects.map((p) => <ProjectCard key={p.id} project={p} />)}
          </div>
        )}
      </div>
      <NewProjectModal open={modalOpen} onClose={() => setModalOpen(false)} />
    </div>
  );
}

function NewProjectModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const createProject = useProjectsStore((s) => s.createProject);
  const navigate = useRouter((s) => s.navigate);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [stage, setStage] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const reset = () => {
    setName("");
    setDescription("");
    setStage("");
    setError(null);
    setSubmitting(false);
  };

  const handleSubmit = async () => {
    const trimmed = name.trim();
    if (!trimmed || submitting) return;
    setSubmitting(true);
    setError(null);
    const id = await createProject(trimmed, {
      description: description.trim(),
      stage: stage.trim(),
    });
    setSubmitting(false);
    if (id) {
      reset();
      onClose();
      navigate(`/projects/${id}`);
    } else {
      setError(useProjectsStore.getState().error ?? "Failed to create project");
    }
  };

  return (
    <Dialog
      open={open}
      onClose={() => {
        if (!submitting) {
          reset();
          onClose();
        }
      }}
      className="relative z-50"
    >
      <div className="fixed inset-0 bg-black/60" />
      <div className="fixed inset-0 flex items-center justify-center p-4">
        <DialogPanel className="w-full max-w-lg rounded-xl border border-edge bg-panel p-5 shadow-xl">
          <h2 className="mb-3 text-base font-medium text-zinc-100">New project</h2>
          <label className="mb-1 block text-[11px] uppercase text-zinc-500">Name</label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. Nelke web UI"
            className="mb-3 w-full rounded-lg border border-edge bg-canvas px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-600 focus:border-accent focus:outline-none"
            autoFocus
          />
          <label className="mb-1 block text-[11px] uppercase text-zinc-500">Description</label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="One-line summary of what this project is about"
            rows={3}
            className="mb-3 w-full resize-none rounded-lg border border-edge bg-canvas px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-600 focus:border-accent focus:outline-none"
          />
          <label className="mb-3 block text-[11px] uppercase text-zinc-500">
            Stage <span className="normal-case text-zinc-600">(free-form: idea / active / done…)</span>
          </label>
          <input
            type="text"
            value={stage}
            onChange={(e) => setStage(e.target.value)}
            placeholder="idea"
            className="mb-3 w-full rounded-lg border border-edge bg-canvas px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-600 focus:border-accent focus:outline-none"
          />
          {error && <p className="mb-3 text-sm text-red-400">{error}</p>}
          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={() => {
                reset();
                onClose();
              }}
              disabled={submitting}
              className="rounded-lg border border-edge bg-panel2 px-3 py-1.5 text-sm text-zinc-300 hover:bg-edge disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={handleSubmit}
              disabled={submitting || !name.trim()}
              className="rounded-lg border border-accent/40 bg-accent/20 px-3 py-1.5 text-sm font-medium text-accent hover:bg-accent/30 disabled:opacity-50"
            >
              {submitting ? "Creating…" : "Create project"}
            </button>
          </div>
        </DialogPanel>
      </div>
    </Dialog>
  );
}
