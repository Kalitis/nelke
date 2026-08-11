import { create } from "zustand";
import { api } from "@/api/client";
import type { MemoryFile } from "@/state/types";

interface MemoryState {
  files: MemoryFile[];
  selectedName: string | null;
  content: string;
  loading: boolean;
  error: string | null;

  load: () => Promise<void>;
  select: (name: string) => Promise<void>;
}

export const useMemoryStore = create<MemoryState>((set, get) => ({
  files: [],
  selectedName: null,
  content: "",
  loading: false,
  error: null,

  load: async () => {
    try {
      const files = await api.memoryFiles();
      set({ files, error: null });
      // Auto-select the first file when nothing is chosen yet.
      if (!get().selectedName && files.length > 0) {
        await get().select(files[0].name);
      }
    } catch (err) {
      set({ error: String(err) });
    }
  },

  select: async (name) => {
    set({ selectedName: name, loading: true, error: null });
    try {
      const { content } = await api.memoryFile(name);
      set({ content, loading: false });
    } catch (err) {
      set({ loading: false, error: String(err), content: "" });
    }
  },
}));
