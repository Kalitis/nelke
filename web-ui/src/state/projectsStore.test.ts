import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useProjectsStore } from "./projectsStore";
import type { ProjectDetail, ProjectSummary } from "@/state/types";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function urlFetchMock(handler: (url: string, init?: RequestInit) => Response | Promise<Response>) {
  const mock = vi.fn<typeof globalThis.fetch>(async (input, init) => handler(String(input), init));
  globalThis.fetch = mock as never;
  return mock;
}

const SAMPLE: ProjectSummary = {
  id: "p1",
  name: "Nelke",
  description: "agent",
  stage: "idea",
  meta: {},
  created_at: null,
  updated_at: null,
  chat_count: 0,
};

describe("projectsStore", () => {
  let originalFetch: typeof globalThis.fetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
    useProjectsStore.setState({
      projects: [],
      detail: null,
      loading: false,
      error: null,
      lastCreatedId: null,
    });
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("loads the project list", async () => {
    urlFetchMock(async (url) => {
      expect(url).toBe("/api/projects");
      return jsonResponse([SAMPLE]);
    });

    await useProjectsStore.getState().loadProjects();
    const { projects } = useProjectsStore.getState();
    expect(projects).toHaveLength(1);
    expect(projects[0].id).toBe("p1");
  });

  it("loads a project detail", async () => {
    const detail: ProjectDetail = { ...SAMPLE, chats: [], memory_files: [{ name: "INDEX.md", size: 5 }] };
    urlFetchMock(async (_url, init) => {
      expect(init).toBeUndefined();
      return jsonResponse(detail);
    });

    await useProjectsStore.getState().loadDetail("p1");
    const { detail: loaded } = useProjectsStore.getState();
    expect(loaded?.id).toBe("p1");
    expect(loaded?.memory_files[0].name).toBe("INDEX.md");
    expect(useProjectsStore.getState().loading).toBe(false);
  });

  it("createProject posts and records lastCreatedId", async () => {
    const fetchMock = urlFetchMock(async (url, init) => {
      if (url === "/api/projects" && init?.method === "POST") {
        return jsonResponse({ id: "p-new", name: "New" });
      }
      if (url === "/api/projects") {
        return jsonResponse([{ ...SAMPLE, id: "p-new", name: "New" }]);
      }
      return jsonResponse([]);
    });

    const id = await useProjectsStore.getState().createProject("New", { stage: "active" });
    expect(id).toBe("p-new");
    expect(useProjectsStore.getState().lastCreatedId).toBe("p-new");
    expect(useProjectsStore.getState().projects[0].id).toBe("p-new");

    const postCall = fetchMock.mock.calls.find((c) => String(c[0]) === "/api/projects");
    const init = postCall![1] as RequestInit;
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body as string)).toEqual({
      name: "New",
      description: "",
      stage: "active",
    });
  });

  it("updateProject patches and refreshes the open detail", async () => {
    const detail: ProjectDetail = { ...SAMPLE, stage: "idea", chats: [], memory_files: [] };
    urlFetchMock(async (url, init) => {
      if (url === `/api/projects/p1` && init?.method === "PATCH") {
        return jsonResponse({ ok: true });
      }
      if (url === "/api/projects") {
        return jsonResponse([{ ...SAMPLE, stage: "active" }]);
      }
      if (url === `/api/projects/p1`) {
        return jsonResponse({ ...detail, stage: "active" });
      }
      return jsonResponse({});
    });

    useProjectsStore.setState({ detail });
    const ok = await useProjectsStore.getState().updateProject("p1", { stage: "active" });
    expect(ok).toBe(true);
    expect(useProjectsStore.getState().detail?.stage).toBe("active");
  });

  it("deleteProject removes the project and clears the open detail", async () => {
    const detail: ProjectDetail = { ...SAMPLE, chats: [], memory_files: [] };
    const fetchMock = urlFetchMock(async (url, init) => {
      if (url === `/api/projects/p1` && init?.method === "DELETE") {
        return jsonResponse({ ok: true });
      }
      if (url === "/api/projects") {
        return jsonResponse([]);
      }
      return jsonResponse({});
    });

    useProjectsStore.setState({ detail });
    const ok = await useProjectsStore.getState().deleteProject("p1");
    expect(ok).toBe(true);
    expect(useProjectsStore.getState().detail).toBeNull();
    expect(useProjectsStore.getState().projects).toEqual([]);
    expect(fetchMock.mock.calls.some((c) => String(c[0]) === "/api/projects/p1" && (c[1] as RequestInit).method === "DELETE")).toBe(true);
  });

  it("records an error when a request fails", async () => {
    urlFetchMock(async () => {
      throw new Error("network down");
    });

    await useProjectsStore.getState().loadProjects();
    expect(useProjectsStore.getState().error).toBe("Error: network down");
  });
});
