import { useEffect, useState } from "react";
import { Dialog, DialogPanel } from "@headlessui/react";
import { Sidebar } from "./Sidebar";
import { TopBar } from "./TopBar";
import { Transcript } from "@/components/chat/Transcript";
import { Composer } from "@/components/chat/Composer";
import { useChatStore } from "@/state/chatStore";

export function AppShell() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const loadProfiles = useChatStore((s) => s.loadProfiles);
  const loadChats = useChatStore((s) => s.loadChats);
  const chats = useChatStore((s) => s.chats);

  // Boot: pull profiles + chat list once.
  useEffect(() => {
    void loadProfiles();
    void loadChats();
  }, [loadProfiles, loadChats]);

  // Keep the chat list fresh (titles/counts change as turns complete).
  useEffect(() => {
    if (chats.length === 0) return;
    const id = window.setInterval(() => {
      void loadChats();
    }, 15000);
    return () => window.clearInterval(id);
  }, [loadChats, chats.length]);

  return (
    <div className="flex h-screen overflow-hidden bg-canvas">
      {/* Desktop sidebar */}
      <div className="hidden lg:block">
        <Sidebar />
      </div>

      {/* Mobile drawer */}
      <Dialog open={sidebarOpen} onClose={setSidebarOpen} className="relative z-40 lg:hidden">
        <div className="fixed inset-0 bg-black/60" />
        <div className="fixed inset-y-0 left-0">
          <DialogPanel className="h-full w-72">
            <Sidebar onClose={() => setSidebarOpen(false)} />
          </DialogPanel>
        </div>
      </Dialog>

      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar onOpenSidebar={() => setSidebarOpen(true)} />
        <Transcript />
        <Composer />
      </div>
    </div>
  );
}
