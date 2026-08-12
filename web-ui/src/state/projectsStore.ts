import { create } from "zustand";
import { api } from "@/api/client";
import type { ProjectDetail, ProjectSummary } from "@/state/types";

interface ProjectsState {
  projects: ProjectSummary[];
  detail: ProjectDetail | null;
  loading: boolean;
  error: string | null;
  // id of the most recently created project, so the UI can navigate to its
  // detail page right after `createProject` returns.
  lastCreatedId: string | null;

  loadProjects: () => Promise<void>;
  loadDetail: (id: string) => Promise<void>;
  clearDetail: () => void;
  createProject: (
    name: string,
    opts?: { description?: string; stage?: string },
  ) => Promise<string | null>;
  updateProject: (
    id: string,
    fields: { name?: string; description?: string; stage?: string },
  ) => Promise<boolean>;
  deleteProject: (id: string) => Promise<boolean>;
}

export const useProjectsStore = create<ProjectsState>((set, get) => ({
  projects: [],
  detail: null,
  loading: false,
  error: null,
  lastCreatedId: null,

  loadProjects: async () => {
    try {
      const projects = await api.projectsList();
      set({ projects, error: null });
    } catch (err) {
      set({ error: String(err) });
    }
  },

  loadDetail: async (id) => {
    set({ loading: true, error: null });
    try {
      const detail = await api.projectDetail(id);
      set({ detail, loading: false });
    } catch (err) {
      set({ loading: false, error: String(err), detail: null });
    }
  },

  clearDetail: () => set({ detail: null, loading: false }),

  createProject: async (name, opts) => {
    try {
      const result = await api.createProject(name, opts);
      set({ lastCreatedId: result.id });
      await get().loadProjects();
      return result.id;
    } catch (err) {
      set({ error: String(err) });
      return null;
    }
  },

  updateProject: async (id, fields) => {
    try {
      await api.updateProject(id, fields);
      await get().loadProjects();
      // Refresh the open detail if it is the one we just edited.
      if (get().detail?.id === id) await get().loadDetail(id);
      return true;
    } catch (err) {
      set({ error: String(err) });
      return false;
    }
  },

  deleteProject: async (id) => {
    try {
      await api.deleteProject(id);
      await get().loadProjects();
      if (get().detail?.id === id) set({ detail: null });
      return true;
    } catch (err) {
      set({ error: String(err) });
      return false;
    }
  },
}));
