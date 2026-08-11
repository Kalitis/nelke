import { create } from "zustand";
import { api } from "@/api/client";
import type { CycleDetail, CycleSummary } from "@/state/types";

interface CyclesState {
  cycles: CycleSummary[];
  detail: CycleDetail | null;
  loading: boolean;
  error: string | null;

  loadCycles: () => Promise<void>;
  loadDetail: (id: string) => Promise<void>;
  clearDetail: () => void;
}

export const useCyclesStore = create<CyclesState>((set) => ({
  cycles: [],
  detail: null,
  loading: false,
  error: null,

  loadCycles: async () => {
    try {
      const cycles = await api.cyclesList();
      set({ cycles, error: null });
    } catch (err) {
      set({ error: String(err) });
    }
  },

  loadDetail: async (id) => {
    set({ loading: true, error: null });
    try {
      const detail = await api.cycleDetail(id);
      set({ detail, loading: false });
    } catch (err) {
      set({ loading: false, error: String(err), detail: null });
    }
  },

  clearDetail: () => set({ detail: null, loading: false }),
}));
