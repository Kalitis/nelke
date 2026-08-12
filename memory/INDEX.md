# Nelke Memory Index

## .
- [Lessons](lessons.md) — - A self-improvement commit that cannot import/boot must be reverted — the boot check is the rollback gate. #lessons governance
- [Skills](skills.md) — - Use the tool-calling loop: read/glob/grep before editing; write minimal diffs. #skills agent tools

## chats
- [Nelke Development Vector (2026-08-11)](chats/20260811054228-2ada9b87/development/roadmap.md) — Tags: roadmap, development, self-improvement, vector #roadmap development self-improvement vector
- [О пользователе](chats/20260811070422-6ed4ba1d/about-user.md) — - **Имя:** Игорь Калитис
- [Проект Nelke](chats/20260811070422-6ed4ba1d/projects/nelke.md) — - Входит в серию N4: Nelke, Nagelkraut, Nachtschatten, Nieswurz.
- [Циклы самоулучшения](chats/20260811105752-37bb9979/cycles/notes.md) — Начало нового цикла. Задача: определить недостатки в возможностях (агента Nelke) и предложить/реализовать улучшения.
- [Наблюдения](chats/20260811105752-37bb9979/notes/workspace.md) — - bash-инструмент работает через cmd (Windows). Доступны: dir, type, и т.п. (не ls/pwd/head).
- [Цикл улучшения 20260811121107-6f0a032f](chats/20260811121107-6f0a032f/cycles/notes.md) — Статус: идёт. Начало — разминка выполнена (рекурсивный скрипт + скачивание файла).
- [Рабочий каталог 20260811121107-6f0a032f](chats/20260811121107-6f0a032f/notes/workspace.md) — - Задача: определить недостатки возможностей Nelke и реализовать улучшения.
- [N4 / Nachtschatten — долговременная память: прототипы steering / merging / associative](chats/20260812093926-ff525969/cycles/nachtschatten-memory-prototypes.md) — Игорь изобрёл два подхода к долговременной памяти: **Steering** и **Merging**. Nelke добавила третий (associative library) и сравнила все три на игрушечном трансформере. Это вероятная основа будущего **Nachtschatten** (персистентная память локальных агентов).
- [Оценка двух подходов к долговременной памяти (steering vs merging) — 2026-08-12](chats/20260812093926-ff525969/cycles/notes.md) — Игорь показал два изобретённых подхода к памяти: **Steering** (гиперсеть генерирует LoRA-матрицы из контекстного вектора, динамически, отключается на None) и **Merging** (гиперсеть генерирует LoRA из внутреннего скрытого состояния и разово вшивает в linear.weight, выключая гиперчик через флаг is_merged). Оба работают на игрушечных примерах (угадывают Key→Val и сохраняют язык [1,2,3,4]→[2,3,4,5]).
- [Зачем существует N4 — Нова и REBIRTH](chats/unknown/rebirth-nova.md) — Дополнение к «рабочей конституции строителя» (chats/unknown/roadmap.md). Записано из прямого разговора с Игорем.
- [N4 / REBIRTH — рабочая конституция строителя](chats/unknown/roadmap.md) — Записано из прямого разговора с Игорем (пользователь, совладелец направления).

## facts
- [LLM providers](facts/llms.md) — - Nelke talks to any OpenAI-compatible endpoint via the `openai` library. #facts llm providers
- [ML Sandbox (WSL)](facts/ml-sandbox-wsl.md) — - WSL дистрибутив: **Ubuntu** (WSL 2), пользователь **n4** (вход без пароля через `wsl -d Ubuntu`).
