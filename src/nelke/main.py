"""Nelke CLI entry point: a single Typer app dispatching to all frontends.

CLI is the phase-2 frontend; web/tui/bot launchers delegate to their adapters
(reporting not-implemented until their phases land).
"""

from __future__ import annotations

import typer

from nelke import __version__
from nelke.frontends import cli

app = typer.Typer(
    name="nelke",
    help="Nelke — self-improving general-purpose agent.",
    no_args_is_help=True,
    add_completion=False,
)

review_app = typer.Typer(help="Human review gate for self-improvement cycles.", no_args_is_help=True)
memory_app = typer.Typer(help="Manage the markdown memory store.", no_args_is_help=True)
config_app = typer.Typer(help="Configuration and provider profiles.", no_args_is_help=True)
db_app = typer.Typer(help="Inspect the Nelke SQLite database.", no_args_is_help=True)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"Nelke {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(None, "--version", "-V", callback=_version_callback, is_eager=True),
) -> None:
    pass


# --------------------------------------------------------------------------- #
# chat / task / improve
# --------------------------------------------------------------------------- #
@app.command()
def chat(
    text: str | None = typer.Argument(None, help="Single message; omit for an interactive session."),
    profile: str | None = typer.Option(None, "--profile", "-p", help="Provider profile name."),
) -> None:
    """Chat with Nelke (interactive if no text is given)."""
    _check_llm(profile)
    if text:
        cli.run_task(text, profile=profile, interactive=False)
    else:
        _interactive_chat(profile)


@app.command("task")
def task_command(
    text: str = typer.Argument(..., help="The task for Nelke to complete."),
    profile: str | None = typer.Option(None, "--profile", "-p"),
) -> None:
    """Run a one-shot task (non-interactive)."""
    _check_llm(profile)
    cli.run_task(text, profile=profile, interactive=True)


@app.command()
def improve(
    objective: str = typer.Argument(..., help="What to improve in the repo."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Auto-approve the human review gate."),
    profile: str | None = typer.Option(None, "--profile", "-p"),
) -> None:
    """Run a self-improvement cycle on this repository."""
    _check_llm(profile)
    cli.improve(objective, yes=yes, profile=profile)


# --------------------------------------------------------------------------- #
# review
# --------------------------------------------------------------------------- #
@review_app.command("list")
def review_list() -> None:
    cli.review_list()


@review_app.command("approve")
def review_approve(request_id: str = typer.Argument(...)) -> None:
    cli.review_approve(request_id)


@review_app.command("reject")
def review_reject(request_id: str = typer.Argument(...)) -> None:
    cli.review_reject(request_id)


app.add_typer(review_app, name="review")


# --------------------------------------------------------------------------- #
# memory
# --------------------------------------------------------------------------- #
@memory_app.command("list")
def memory_list() -> None:
    cli.memory_list()


@memory_app.command("show")
def memory_show(name: str = typer.Argument(...)) -> None:
    cli.memory_show(name)


@memory_app.command("edit")
def memory_edit(name: str = typer.Argument(...)) -> None:
    cli.memory_edit(name)


@memory_app.command("recall")
def memory_recall(
    query: str = typer.Argument(...),
    top_k: int = typer.Option(8, "--top-k"),
) -> None:
    cli.memory_recall(query, top_k)


app.add_typer(memory_app, name="memory")


# --------------------------------------------------------------------------- #
# config / db / doctor
# --------------------------------------------------------------------------- #
@config_app.command("show")
def config_show() -> None:
    cli.config_show()


@config_app.command("init")
def config_init() -> None:
    cli.config_init()


app.add_typer(config_app, name="config")


@db_app.command("status")
def db_status() -> None:
    """Show database table counts."""
    cli.db_status()


@db_app.command("cleanup")
def db_cleanup() -> None:
    """Mark abandoned running cycles as stuck/failed."""
    cli.db_cleanup()


app.add_typer(db_app, name="db")


@app.command()
def doctor() -> None:
    """Check environment: tools, profiles, DB, boot_check."""
    cli.doctor()


# --------------------------------------------------------------------------- #
# launchers for the other frontends
# --------------------------------------------------------------------------- #
@app.command()
def web(
    host: str | None = typer.Option(None, "--host", "-h", help="Bind host (default 127.0.0.1)."),
    port: int | None = typer.Option(None, "--port", "-p", help="Bind port (default 8000)."),
) -> None:
    """Launch the web frontend (FastAPI + SSE)."""
    from nelke.frontends import web as web_frontend
    from nelke.frontends.telegram_bot import start_companion as telegram_companion

    telegram_companion()
    web_frontend.launch(host=host, port=port)


@app.command()
def tui() -> None:
    """Launch the TUI frontend (Textual)."""
    from nelke.frontends import tui as tui_frontend
    from nelke.frontends.telegram_bot import start_companion as telegram_companion

    telegram_companion()
    tui_frontend.launch()


@app.command()
def bot() -> None:
    """Launch the Telegram bot frontend (aiogram)."""
    from nelke.frontends import telegram_bot as telegram_frontend

    telegram_frontend.launch()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _check_llm(profile: str | None) -> None:
    from nelke.config import ProfileError, get_profile, load_profiles

    try:
        get_profile(profile, load_profiles())
    except ProfileError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc


def _interactive_chat(profile: str | None) -> None:
    from rich.prompt import Prompt

    from nelke.frontends.telegram_bot import start_companion as telegram_companion

    telegram_companion()
    _intro()
    while True:
        try:
            text = Prompt.ask("\n[bold cyan]you[/]")
        except (EOFError, KeyboardInterrupt):
            typer.echo("\nbye")
            raise typer.Exit(0) from None
        if not text.strip():
            continue
        if text.strip().lower() in {"/exit", "/quit", "/q"}:
            raise typer.Exit(0)
        if text.strip().lower() in {"/help", "/h"}:
            typer.echo("commands: /exit  /help")
            continue
        cli.run_task(text, profile=profile, interactive=True)


def _intro() -> None:
    typer.echo(typer.style(f"Nelke {__version__} — type your message (/exit to quit)", fg="green"))


if __name__ == "__main__":
    app()
