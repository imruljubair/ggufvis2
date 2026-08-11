"""Command-line and interactive navigation for final v1."""

from __future__ import annotations

import argparse
import shutil
import sys
import termios
# import time
import tty
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .diagram import Diagram, build_diagram
from .explainer import (
    BlockLoopDetail,
    DEFAULT_SAMPLE_LIMIT,
    MINIMUM_SAMPLE_LIMIT,
    detail_for_step,
    render_matrix_detail,
    selection_shape,
)
from .gguf import GGUFError, read_gguf, read_gguf_url, resolve_ollama_model
from .model import config_from_gguf
from .renderer import render_diagram
from .terminal import read_key


# Optional Model View selection delay, currently disabled:
# LAYER_SELECTION_DELAY_SECONDS = 0.035

VIEW_LABEL_COLOR = (125, 125, 125)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Render a standalone metadata-driven matrix diagram for dense "
            "Llama, original Gemma, Qwen2/Qwen2.5, Qwen3, or DeepSeek-R1 "
            "Distill GGUF models."
        )
    )
    result.add_argument(
        "gguf",
        help=(
            "local GGUF path, Ollama model name with --ollama, or direct "
            "HTTP(S) GGUF link with --url"
        ),
    )
    source_mode = result.add_mutually_exclusive_group()
    source_mode.add_argument(
        "--ollama",
        action="store_true",
        help="resolve the positional argument as an Ollama model name",
    )
    source_mode.add_argument(
        "--url",
        action="store_true",
        help=(
            "parse a remote GGUF through in-memory HTTP range requests "
            "without saving the model"
        ),
    )
    result.add_argument(
        "--ollama-models-dir",
        metavar="PATH",
        help=(
            "Ollama models directory; defaults to OLLAMA_MODELS or "
            "~/.ollama/models"
        ),
    )
    result.add_argument("--no-color", action="store_true")
    result.add_argument(
        "--static",
        action="store_true",
        help="print one operation without entering interactive mode",
    )
    result.add_argument(
        "--step",
        type=int,
        default=1,
        help="1-based operation used by --static (default: 1)",
    )
    return result


def _human_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.1f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024
    return f"{value} B"


def _source_header(
    value: str, *, is_ollama: bool = False, is_remote: bool = False
) -> str:
    """Build the single persistent source line shown above the diagram."""
    if is_ollama:
        return f"Ollama: {value}"
    if is_remote:
        filename = unquote(Path(urlsplit(value).path).name)
        return f"Remote: {filename or value}"
    return f"GGUF: {Path(value).expanduser().name}"


def _fit_header(header: str, width: int) -> str:
    """Middle-truncate a source line while preserving its useful suffix."""
    if len(header) <= width:
        return header
    if width <= 1:
        return "…"[:width]
    available = width - 1
    left = (available + 1) // 2
    right = available - left
    return f"{header[:left]}…{header[-right:] if right else ''}"


def _view_header(
    source_header: str,
    view: str,
    width: int,
    use_color: bool,
) -> str:
    """Append a persistent, low-contrast view label to the source header."""
    suffix = f" [{view}]"
    if width <= len(suffix):
        visible = _fit_header(suffix.strip(), width)
        prefix = ""
        label = visible
    else:
        prefix = _fit_header(source_header, width - len(suffix))
        label = suffix
        visible = f"{prefix}{label}"
    if not use_color:
        return visible
    red, green, blue = VIEW_LABEL_COLOR
    return (
        f"{prefix}\033[38;2;{red};{green};{blue}m"
        f"{label}\033[0m"
    )


def _viewport(
    diagram: Diagram,
    step: int,
    use_color: bool,
    requested_start: int | None,
    source_header: str,
) -> tuple[str, int]:
    terminal = shutil.get_terminal_size(
        fallback=(120, diagram.panels.canvas_height + 1)
    )
    canvas = render_diagram(diagram, step, annotation_position="right")
    stacked = False
    if terminal.columns < canvas.width:
        # The annotation is wider than the model but still fits comfortably
        # below it in ordinary 80-column terminals.
        canvas = render_diagram(diagram, step, annotation_position="below")
        stacked = True
    if terminal.columns < canvas.width:
        header = _view_header(
            source_header, "model view", terminal.columns, use_color
        )
        return (
            f"{header}\n"
            f"Terminal is {terminal.columns} columns wide; this diagram needs "
            f"at least {canvas.width}. Widen the window, then press an arrow key.",
            0,
        )

    header = _view_header(
        source_header, "model view", terminal.columns, use_color
    )
    lines = canvas.render(use_color=use_color).splitlines()
    if stacked:
        # Keep the below-diagram card visible while the matrix viewport moves.
        # Keep the trailing spacer and annotation card together.
        model_lines = lines[: diagram.panels.canvas_height]
        card_lines = lines[diagram.panels.canvas_height :]
        viewport_height = min(
            len(model_lines),
            # Reserve rows for the header, status, and narrow-layout tip.
            max(4, terminal.lines - len(card_lines) - 3),
        )
        last_start = max(0, len(model_lines) - viewport_height)
    else:
        model_lines = lines
        card_lines = []
        # Reserve one row each for the persistent header and status.
        viewport_height = min(len(lines), max(4, terminal.lines - 2))
        last_start = max(0, len(lines) - viewport_height)
    if requested_start is None:
        centered = diagram.operations[step].row - viewport_height // 2
        start = max(0, min(centered, last_start))
    else:
        start = max(0, min(requested_start, last_start))
    end = start + viewport_height
    visible_lines = model_lines[start:end]
    if card_lines:
        visible_lines.extend(card_lines)
    visible = "\n".join(visible_lines)
    detail_hint = (
        "  Right: Explainer"
        if detail_for_step(diagram, step) is not None
        else ""
    )
    status = (
        f"Rows {start + 1}-{end}/{len(model_lines)}  "
        f"Up/Down: step  PgUp/PgDn: view{detail_hint}  q: quit"
    )
    if stacked:
        tip = (
            "Tip: widen the terminal to place annotations beside the diagram."
        )
        return f"{header}\n{visible}\n{tip}\n{status}", start
    return f"{header}\n{visible}\n{status}", start


def _detail_viewport(
    diagram: Diagram,
    step: int,
    row: int,
    column: int,
    use_color: bool,
    source_header: str,
    animation_stage: int = 2,
) -> tuple[str, int, int, int, int]:
    """Render a complete, visual-only matrix grid sized to the terminal."""
    terminal = shutil.get_terminal_size(fallback=(80, 24))
    header = _view_header(
        source_header, "explainer view", terminal.columns, use_color
    )
    for sample_limit in range(
        DEFAULT_SAMPLE_LIMIT,
        MINIMUM_SAMPLE_LIMIT - 1,
        -1,
    ):
        row_count, column_count = selection_shape(
            diagram, step, sample_limit
        )
        selected_row = max(0, min(row, row_count - 1))
        selected_column = max(0, min(column, column_count - 1))
        canvas = render_matrix_detail(
            diagram,
            step,
            selected_row,
            selected_column,
            sample_limit=sample_limit,
            animation_stage=animation_stage,
        )
        if (
            canvas.width <= terminal.columns
            and canvas.height + 3 <= terminal.lines
        ):
            controls = (
                "Esc: back to Model View  q: quit"
                if isinstance(detail_for_step(diagram, step), BlockLoopDetail)
                else "Arrows: select  Esc: back to Model View  q: quit"
            )
            return (
                f"{header}\n{canvas.render(use_color=use_color)}\n\n{controls}",
                selected_row,
                selected_column,
                row_count,
                column_count,
            )
    return (
        f"{header}\nTerminal is too small for the matrix grid.",
        0,
        0,
        1,
        1,
    )


def _run_interactive(
    diagram: Diagram, use_color: bool, source_header: str
) -> None:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        terminal_width = shutil.get_terminal_size(fallback=(120, 24)).columns
        print(
            _view_header(
                source_header, "model view", terminal_width, use_color
            )
        )
        print(render_diagram(diagram, 0).render(use_color=use_color))
        print("Interactive mode requires a terminal. Use --static for one frame.")
        return

    descriptor = sys.stdin.fileno()
    previous = termios.tcgetattr(descriptor)
    step = 0
    viewport_start: int | None = None
    detail_mode = False
    detail_row = 0
    detail_column = 0
    try:
        tty.setraw(descriptor)
        sys.stdout.write("\033[?1049h\033[?25l")
        while True:
            if detail_mode:
                (
                    frame,
                    detail_row,
                    detail_column,
                    detail_row_count,
                    detail_column_count,
                ) = _detail_viewport(
                    diagram,
                    step,
                    detail_row,
                    detail_column,
                    use_color,
                    source_header,
                )
                actual_start = 0
            else:
                frame, actual_start = _viewport(
                    diagram, step, use_color, viewport_start, source_header
                )
            sys.stdout.write(
                f"\033[2J\033[H{frame.replace(chr(10), chr(13) + chr(10))}"
            )
            sys.stdout.flush()
            key = read_key()
            if detail_mode:
                if key == "up":
                    detail_row = max(0, detail_row - 1)
                elif key == "down":
                    detail_row = min(detail_row_count - 1, detail_row + 1)
                elif key == "left":
                    detail_column = max(0, detail_column - 1)
                elif key == "right":
                    detail_column = min(
                        detail_column_count - 1,
                        detail_column + 1,
                    )
                elif key == "escape":
                    detail_mode = False
                    viewport_start = None
                elif key.lower() == "q" or key == "\x03":
                    break
                continue
            if key == "down":
                next_step = min(step + 1, len(diagram.operations) - 1)
                if next_step != step:
                    step = next_step
                    viewport_start = None
                    # time.sleep(LAYER_SELECTION_DELAY_SECONDS)
            elif key == "up":
                next_step = max(step - 1, 0)
                if next_step != step:
                    step = next_step
                    viewport_start = None
                    # time.sleep(LAYER_SELECTION_DELAY_SECONDS)
            elif key == "page_up":
                viewport_start = actual_start - 10
            elif key == "page_down":
                viewport_start = actual_start + 10
            elif key == "right" and detail_for_step(diagram, step) is not None:
                detail_mode = True
                detail_row = 0
                detail_column = 0
            elif key.lower() == "q" or key == "\x03":
                break
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, previous)
        sys.stdout.write("\033[?25h\033[?1049l")
        sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.ollama_models_dir and not args.ollama:
            raise GGUFError("--ollama-models-dir requires --ollama")
        if args.url:
            model = read_gguf_url(args.gguf)
        else:
            path = (
                resolve_ollama_model(args.gguf, args.ollama_models_dir)
                if args.ollama
                else Path(args.gguf).expanduser()
            )
            model = read_gguf(path)
        if model.source_url:
            print(
                "remote: parsed metadata and "
                f"{len(model.tensors)} tensor descriptors; transferred "
                f"{_human_bytes(model.remote_bytes_transferred)} of "
                f"{_human_bytes(model.remote_file_size)}; no local model "
                "file created",
                file=sys.stderr,
            )
        config = config_from_gguf(model)
        diagram = build_diagram(config)
        source_header = _source_header(
            args.gguf, is_ollama=args.ollama, is_remote=args.url
        )
        if not 1 <= args.step <= len(diagram.operations):
            raise GGUFError(
                f"--step must be between 1 and {len(diagram.operations)}"
            )
        for warning in config.warnings:
            print(f"warning: {warning}", file=sys.stderr)

        use_color = not args.no_color
        if args.static:
            terminal_width = shutil.get_terminal_size(
                fallback=(120, diagram.panels.canvas_height + 2)
            ).columns
            print(
                _view_header(
                    source_header, "model view", terminal_width, use_color
                )
            )
            print(
                render_diagram(diagram, args.step - 1).render(
                    use_color=use_color
                )
            )
        else:
            _run_interactive(diagram, use_color, source_header)
    except GGUFError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0
