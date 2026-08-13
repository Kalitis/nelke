# ML Sandbox (WSL)

## Расположение
- WSL дистрибутив: **Ubuntu** (WSL 2), пользователь **n4** (вход без пароля через `wsl -d Ubuntu`).
- Рабочая папка sandbox: `/home/n4/workspace/ml`
- Rust: установлен через rustup в `~/.cargo` + `~/.rustup` (rustc/cargo 1.97.1). Не в PATH — подключать `source ~/.cargo/env`.
- Python venv (ML-среда): `/home/n4/workspace/ml/venv` (Python 3.14.4).
- Ещё одна папка проекта: `/home/n4/workspace/contai`.

## ML-пакеты в venv
- torch 2.11.0(+cpu), numpy 2.5.1, sympy 1.14, triton 3.6, gymnasium 1.3.0, box2d-py, pygame.
- Лежит wheel `torch-2.11.0+cu128` в `~/workspace/ml/` (CUDA-сборка ещё не установлена).
- CUDA-библиотеки (nvidia-*) установлены в venv (cublas, cudnn, nccl и т.д., CUDA 12.8), но torch стоит CPU-only + активировать CUDA.

Эта среда понадобится вскоре (задача Игоря).

Related: [Nelke Development Vector (2026-08-11)](chats/20260811054228-2ada9b87/development/roadmap.md)

Related: [О пользователе](chats/20260811070422-6ed4ba1d/about-user.md)

Related: [Проект Nelke](chats/20260811070422-6ed4ba1d/projects/nelke.md)

Related: [Циклы самоулучшения](chats/20260811105752-37bb9979/cycles/notes.md)

Related: [Наблюдения](chats/20260811105752-37bb9979/notes/workspace.md)

Related: [Цикл улучшения 20260811121107-6f0a032f](chats/20260811121107-6f0a032f/cycles/notes.md)

Related: [Рабочий каталог 20260811121107-6f0a032f](chats/20260811121107-6f0a032f/notes/workspace.md)

Related: [Зачем существует N4 — Нова и REBIRTH](chats/unknown/rebirth-nova.md)

Related: [N4 / REBIRTH — рабочая конституция строителя](chats/unknown/roadmap.md)

Related: [LLM providers](facts/llms.md)

Related: [Lessons](lessons.md)

Related: [Skills](skills.md)

Related: [Оценка двух подходов к долговременной памяти (steering vs merging) — 2026-08-12](chats/20260812093926-ff525969/cycles/notes.md)

Related: [N4 / Nachtschatten — долговременная память: прототипы steering / merging / associative](chats/20260812093926-ff525969/cycles/nachtschatten-memory-prototypes.md)
