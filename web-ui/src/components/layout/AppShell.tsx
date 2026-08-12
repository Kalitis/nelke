import { useEffect, useState } from "react";
import { Dialog, DialogPanel } from "@headlessui/react";
import { Sidebar } from "./Sidebar";
import { TopBar } from "./TopBar";
import { Transcript } from "@/components/chat/Transcript";
import { Composer } from "@/components/chat/Composer";
import { CyclesView } from "@/components/cycles/CyclesView";
import { CycleDetailView } from "@/components/cycles/CycleDetailView";
import { ProjectsView } from "@/components/projects/ProjectsView";
import { ProjectDetailView } from "@/components/projects/ProjectDetailView";
import { MemoryView } from "@/components/memory/MemoryView";
import { useChatStore } from "@/state/chatStore";
import { useRouter, useRoute } from "@/state/router";

export function AppShell() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const loadProfiles = useChatStore((s) => s.loadProfiles);
  const loadChats = useChatStore((s) => s.loadChats);
  const chats = useChatStore((s) => s.chats);
  const route = useRoute();
  const initRouter = useRouter((s) => s.init);

  // Boot: pull profiles + chat list once; subscribe to history navigation.
  useEffect(() => {
    void loadProfiles();
    void loadChats();
  }, [loadProfiles, loadChats]);

  useEffect(() => {
    const cleanup = initRouter();
    return cleanup;
  }, [initRouter]);

  // Keep the chat list fresh (titles/counts change as turns complete).
  useEffect(() => {
    if (chats.length === 0) return;
    const id = window.setInterval(() => {
      void loadChats();
    }, 15000);
    return () => window.clearInterval(id);
  }, [loadChats, chats.length]);

  const onChatScreen = route.name === "chat";

  return (
    <div className="flex h-screen overflow-hidden bg-canvas">
      {/* Desktop sidebar — only on the chat screen (cycles/memory have their own panels). */}
      {onChatScreen && (
        <div className="hidden lg:block">
          <Sidebar />
        </div>
      )}

      {/* Mobile drawer */}
      {onChatScreen && (
        <Dialog open={sidebarOpen} onClose={setSidebarOpen} className="relative z-40 lg:hidden">
          <div className="fixed inset-0 bg-black/60" />
          <div className="fixed inset-y-0 left-0">
            <DialogPanel className="h-full w-72">
              <Sidebar onClose={() => setSidebarOpen(false)} />
            </DialogPanel>
          </div>
        </Dialog>
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar onOpenSidebar={() => setSidebarOpen(true)} />
        {route.name === "chat" ? (
          <>
            <Transcript />
            <Composer />
          </>
        ) : route.name === "cycles" ? (
          <CyclesView />
        ) : route.name === "cycle" ? (
          <CycleDetailView cycleId={route.id} />
        ) : route.name === "projects" ? (
          <ProjectsView />
        ) : route.name === "project" ? (
          <ProjectDetailView projectId={route.id} />
        ) : route.name === "memory" ? (
          <MemoryView />
        ) : null}
      </div>
    </div>
  );
}
