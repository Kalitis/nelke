# Self-improvement cycles

Each project has its **own** self-improvement cycle. Cycles are not global and
are not shared between projects: every project drives an independent improve →
verify → commit loop that only touches files inside that project.

## Nelke's example cycle

The Nelke example is the reference self-improvement cycle. It lives at the
**repo root**, and its localization files live there as well (in the same root
directory, alongside the cycle config).

Reference configuration:

- `examples/nelke_cycle.yaml` — the Nelke example cycle
- `examples/project_cycle.yaml` — a sample cycle for a user project

## Per-project cycles

Because each project has its own cycle, the loop for one project never affects
another project's files, commits, or localization. To give a project a cycle:

1. Create a cycle config rooted at that project's root (see the samples under
   `examples/`).
2. Place the project's localization files in that project's root.
3. Run the cycle for that project; it will only touch that project's scope.

See `examples/nelke_cycle.yaml` and `examples/project_cycle.yaml` for concrete
shapes.
