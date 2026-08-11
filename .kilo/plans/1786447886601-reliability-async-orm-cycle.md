# Nelke: Reliability & Observability + Self-Improvement Cycle

## Scope
**In:** (1) structured logging; (2) SQLAlchemy 2.0 async ORM + Alembic + aiosqlite rewrite of the persistence layer; (3) merge-conflict recovery (auto-rebase + 1 retry); (4) proactive `improve`-offer on task degradation across all 4 frontends; (5) auto-cleanup of stuck cycles in `doctor`.
**Out of scope:** `plan_first` (do NOT touch); memory intelligence (auto-tag/cross-link/dedup); frontend parity beyond the improve-offer; security/sandbox; versioning hygiene; CI; vestigial-code removal; Postgres (DB stays SQLite, local-first).

## Resolved decisions
- **D1 Logging:** stdlib `logging`, single logger `"nelke"`, text default + `NELKE_LOG_FORMAT=json` + `NELKE_LOG_FILE` + `NELKE_LOG_LEVEL` (default INFO). No new deps.
- **D2 Persistence:** full rewrite to **SQLAlchemy 2.0 async ORM + Alembic + aiosqlite**. New deps `sqlalchemy[asyncio]>=2.0`, `aiosqlite>=0.20`, `alembic>=1.13`. Models in `core/models.py`; `db.py` becomes an async repository (`AsyncSession`). Services still return `dict` to contain frontend blast radius (frontends only add `await` + improve-offer). CLI commands wrapped in `asyncio.run`. Tests rewritten async. `boot_check()` must stay side-effect-free & network-free.
- **D3 Merge-conflict:** on `GitError` from `merge_no_ff` → `git rebase main` on the branch → retry merge once; on second failure, current `merge-conflict` dead-end + clear event with reason. Shared helper used by both the cycle engine and `services.resolve_review`.
- **D4 plan_first:** OUT of scope.
- **D5 Improve-offer:** all 4 frontends, non-blocking. Add `degradation: DegradationReport | None` to `AgentResult`; launching reuses existing `run_cycle`.
- **D6 Auto-cleanup:** `doctor` calls `reconcile_stale_cycles`.

## Phases (each keeps `pytest`/`ruff`/`mypy` green so it can land as one cycle/review unit)

### Phase 1 — Structured logging (additive, low risk)
1. New `src/nelke/core/logging.py`: `setup_logging()` reads `NELKE_LOG_LEVEL`/`NELKE_LOG_FORMAT`/`NELKE_LOG_FILE`; StreamHandler(stderr) + optional FileHandler; JSON formatter when `format=json`. Idempotent (don't double-add handlers).
2. Call `setup_logging()` from `main.py` app callback (before dispatch) and each frontend `launch()` (`web.py:537`, `tui.launch`, `telegram_bot.launch`, cli entry).
3. Classify the ~30 silent `except Exception: pass`/`# noqa: BLE001` sites and replace:
   - Persistence-best-effort (`cycle.py:303,325,336`; `services.py:97,102`) → `logger.debug("...", exc_info=True)`.
   - UI/render best-effort (`cli.py:113,226,388,397`; `tui.py:301,476,514,560`; `telegram_bot.py:261,406,415`) → `logger.debug(..., exc_info=True)`.
   - Real-error swallowing (`web.py:385` runner; `cycle.py:268` top-level) → `logger.exception(...)`.
   - Tool/LLM that already returns failure to caller (`agent.py:170,177`; `llm.py:243,290`) → `logger.debug(..., exc_info=True)` (keep returning failure).
4. Add `logger.info` parallel to key cycle transitions already emitting `CycleEvent` (`cycle_start`, `commit`, `boot_check_failed`, `ai_review`, `awaiting_human`, `merged`, `rejected`) so logs narrate a cycle even without the DB.
5. New `tests/test_logging.py`: text/json formats, file output, level filtering, idempotent re-setup.

### Phase 2 — SQLAlchemy async ORM + Alembic (big; land as ONE focused unit)
1. Add deps to `pyproject.toml`: `sqlalchemy[asyncio]>=2.0`, `aiosqlite>=0.20`, `alembic>=1.13`; `uv lock`.
2. New `src/nelke/core/models.py`: declarative `Base` + ORM models for all 9 tables (`sessions, messages, cycles, cycle_steps, review_requests, tasks, usage_events, cycle_events`). Match current schema exactly incl. `cache_read_tokens`, `cache_read_pct`.
3. Alembic: `alembic.ini` (`script_location = migrations`), `migrations/env.py` configured for async aiosqlite + `nelke.core.models` metadata; `migrations/versions/0001_initial.py` capturing the current schema. Initial migration must be idempotent (`CREATE TABLE IF NOT EXISTS`) so existing DBs `stamp` at baseline and fresh DBs build. Add `core/db.py::run_migrations()` that programmatically runs `alembic upgrade head` (async) — replaces `Database.migrate()`.
4. Rewrite `src/nelke/core/db.py`:
   - `async_engine(db_path)` + `async_sessionmaker`, cached per process keyed by db_path; `async def close()` for engine disposal.
   - `class Database` → async repository: `async def create_session(...)`, `async def add_message(...)`, etc. Keep public method names (call sites only add `await`). Use ORM `select`/`insert`.
   - `run_migrations()` called where `migrate()` was.
5. Convert `services.py`: all DB-accessing sync fns → `async def` (`list_chats`, `get_chat`, `get_chat_messages`, `create_chat`, `rename_chat`, `delete_chat`, `list_open_reviews`, `get_review`, `resolve_review`, `list_cycles`, `get_cycle_detail`, `_cycle_summary`, `build_message_tree`, `edit_message`, `delete_message`, `set_active_message`, `reconcile_stale_cycles`, `_persist_new_messages`, `_persist_usage`, `_title_from_first_user`). Keep returning `dict`.
6. Remove duplicated `find_repo`/`open_db`/`open_memory`/`get_llm` in `cli.py:50-86` — use `services.*`.
7. `cycle.py`: `self.db.*` → `await self.db.*` (~15 sites); `emit()` DB writes keep try/except + add `await` + `logger.debug`.
8. Frontends:
   - `web.py`: add `await` to all `services.*`/`db.*`; add FastAPI lifespan engine cleanup.
   - `telegram_bot.py`: add `await`.
   - `tui.py`: handlers calling services at `275,287,300,315,356,368,447,452,484,500,521,537,551` → `async` + `await` (Textual supports async action handlers).
   - `cli.py`: each DB-touching handler (`review_*`, `memory_*`, `db_*`, `config_*`, `doctor`, `improve`, `chat`, `task`) → async inner fn via `asyncio.run`. Engine per-command (create+close within the command); engine per-process in web/TG.
9. `__init__.py` `boot_check()`: verify `models` import is side-effect-free; boot_check stays network-free.
10. Tests → async: add async engine fixture in `conftest.py` (function-scoped tmp db + migrate). Convert `test_db.py`, `test_services.py`, `test_cycle.py`, `test_web.py`, `test_tui.py`, `test_telegram.py` (verify each; `test_session_analyzer.py`/`test_agent.py` likely db-free).
11. Existing-DB upgrade regression: copy an old-shape `~/.nelke/nelke.db`, run `alembic upgrade head`, assert row counts preserved (no data loss).

### Phase 3 — Merge-conflict recovery (small)
1. Add `GitRepo.rebase(base: str) -> GitResult` (`gitops.py`).
2. New helper `merge_with_retry(repo, branch, cycle_id)` in `cycle.py` near `merge_cycle_branch` (cycle.py:115): on `GitError` → `checkout branch` → `rebase main` → `checkout main` → retry `merge_no_ff` once; on second failure raise `GitError` with reason.
3. Use it in `cycle.py:489` and `services.resolve_review` (services.py:383). DB: stay `merged` on success; on exhaustion set `merge-conflict` + emit `merge_conflict` event with reason; emit `merge_retry` on the attempt.
4. Tests: `test_cycle.py` + `test_services.py` — synthesize a conflicting commit on `main`, assert one rebase+retry, final status.

### Phase 4 — Proactive `improve`-offer on degradation (additive, 4 frontends)
1. `AgentResult`: add `degradation: DegradationReport | None = None` (`agent.py`); populate in `_maybe_degrade` (agent.py:143) — keep the side-effect `on_degraded` callback too.
2. CLI (`cli.py`): after a run, if `result.degradation.degraded`, prompt `Nelke struggled: {reasons}. Run \`nelke improve "{suggested_objective}"\`? [y/N]`; on `y` call `cli.improve(suggested_objective, ...)`.
3. Web (`web.py` + `static/app.js`): SSE event `degraded` `{reasons, suggested_objective}` in chat stream; non-blocking banner + button → POST `/api/improve`.
4. TUI (`tui.py`): notification/banner; on confirm call `_run_improve(suggested_objective)`.
5. Telegram (`telegram_bot.py`): message + inline keyboard offering to start the cycle (callback dispatches to the existing improve handler).
6. Tests: `test_agent.py` (degradation populated), `test_cli_streams.py`/`test_web.py`/`test_tui.py`/`test_telegram.py` assert the offer surfaces non-blocking.

### Phase 5 — Auto-cleanup of stuck cycles in `doctor` (trivial)
1. `cli.doctor()` (`cli.py`): call `await services.reconcile_stale_cycles(...)`; print marked `{id, branch, reason}`.
2. Test: assert doctor reports stuck cycles.

## Risks
- **Phase 2 is large & atomic**: intermediate half-converted states fail the gate. Do Phase 2 as one focused cycle/unit, not spread across many. Phase 1 first (logging aids debugging Phase 2).
- **Existing `~/.nelke/nelke.db` upgrade**: initial Alembic migration must not drop/lose data — stamp-and-upgrade + regression test on an old-shape DB.
- **`boot_check()` weight**: keep model imports side-effect-free; boot_check stays fast & network-free.
- **CLI engine lifecycle**: `asyncio.run` per command creates fresh loops — create+close the engine within each command; module-level engine only in web/TG.
- **Textual async handlers**: confirm `async def action_*`/`on_mount` work (Textual 8.x supports it).
- **Dirty working tree NOW**: 24 modified files + untracked `scripts/`, `web-ui/`. The cycle engine refuses a dirty tree (cycle.py:254). Commit/stash before any `nelke improve` for these phases.

## Validation (per phase)
- Gate (mandatory after each phase): `uv run pytest -q`, `uv run ruff check .`, `uv run mypy src/nelke`.
- Phase 2 extra: `uv run alembic upgrade head` on (a) fresh tmp DB and (b) a copy of an old-shape `~/.nelke/nelke.db` → assert no data loss (row counts preserved). `uv run nelke doctor` clean. `nelke boot_check` passes.
- Phase 5: `uv run nelke doctor` reports stuck cycles.

## Open questions for implementer
- `NELKE_LOG_LEVEL` default INFO (recommended) — confirm noise acceptable; if chatty, default WARNING and gate-keep cycle transitions at INFO.
- Logging vs `cycle_events` table: keep separate — events are persisted structured trace; logging is stdout/file narration. Do not duplicate into a new table.
