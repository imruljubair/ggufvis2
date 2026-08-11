"""Focused tests for the standalone final-v1 implementation."""

from __future__ import annotations

import ast
import http.server
import io
import os
import re
import struct
import threading
import unicodedata
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from ggufvis.cli import (
    _detail_viewport,
    _fit_header,
    _source_header,
    _viewport,
)
from ggufvis.annotations import card_lines, maximum_card_width
from ggufvis.diagram import (
    LEARNED_WEIGHTS,
    LEVEL_ONE_WIDTH,
    LEVEL_ONE_X,
    LEVEL_TWO_WIDTH,
    LEVEL_TWO_X,
    PAIR_GAP,
    build_diagram,
)
from ggufvis.explainer import (
    ACTIVATION_CELL,
    ANNOTATOR_COLOR,
    COLUMN_HIGHLIGHT,
    DIMENSION_COLOR,
    ENDPOINT_CELL_COUNT,
    ENDPOINT_COLUMN_SPACING,
    GRID_BACKGROUND,
    GRID_COLOR,
    INACTIVE_CELL,
    MASKED_CELL,
    NORMALIZED_HIGHLIGHT,
    RESULT_HIGHLIGHT,
    RESULT_CELL,
    RMS_CELL,
    RMS_HIGHLIGHT,
    ROPE_SECOND_HIGHLIGHT,
    ROW_HIGHLIGHT,
    WEIGHT_CELL,
    detail_for_step,
    render_matrix_detail,
    selection_shape,
)
from ggufvis.gguf import GGUFError, GGUFModel, TensorInfo, read_gguf_url
from ggufvis.model import config_from_gguf
from ggufvis.renderer import (
    ACTIVE_BORDER,
    ACTIVE_LEARNED_BORDER_MARKER,
    COMPACT_CARD_TEXT,
    INACTIVE_BORDER,
    INACTIVE_LEARNED_BORDER_MARKER,
    render_diagram,
)
from ggufvis.terminal import read_key


def _tensor(name: str, *shape: int) -> TensorInfo:
    return TensorInfo(name, shape, 0, 0)


def _qwen3_model(*, tied: bool = False) -> GGUFModel:
    hidden, head_dim, ffn, vocab = 4096, 128, 12288, 151936
    metadata = {
        "general.architecture": "qwen3",
        "general.name": "Synthetic Qwen3 8B",
        "qwen3.block_count": 36,
        "qwen3.context_length": 40960,
        "qwen3.embedding_length": hidden,
        "qwen3.feed_forward_length": ffn,
        "qwen3.attention.head_count": 32,
        "qwen3.attention.head_count_kv": 8,
        "qwen3.attention.key_length": head_dim,
        "qwen3.attention.value_length": head_dim,
        "qwen3.rope.freq_base": 1_000_000.0,
        "qwen3.vocab_size": vocab,
    }
    tensors = [
        _tensor("token_embd.weight", hidden, vocab),
        _tensor("blk.0.attn_q.weight", hidden, 32 * head_dim),
        _tensor("blk.0.attn_k.weight", hidden, 8 * head_dim),
        _tensor("blk.0.attn_v.weight", hidden, 8 * head_dim),
        _tensor("blk.0.attn_output.weight", hidden, 32 * head_dim),
        _tensor("blk.0.attn_q_norm.weight", head_dim),
        _tensor("blk.0.attn_k_norm.weight", head_dim),
        _tensor("blk.0.ffn_gate.weight", hidden, ffn),
        _tensor("blk.0.ffn_up.weight", hidden, ffn),
        _tensor("blk.0.ffn_down.weight", ffn, hidden),
        _tensor("blk.35.attn_norm.weight", hidden),
    ]
    if not tied:
        tensors.append(_tensor("output.weight", hidden, vocab))
    return GGUFModel(
        Path("synthetic-qwen3.gguf"), 3, metadata, tuple(tensors)
    )


def _deepseek_qwen3_model() -> GGUFModel:
    base = _qwen3_model()
    metadata = dict(base.metadata)
    # The current Ollama 8B tag uses this name and does not contain the
    # literal word "Distill"; its qwen3 architecture still identifies the
    # dense backbone on which the DeepSeek-R1 behavior was trained.
    metadata["general.name"] = "DeepSeek-R1-0528-Qwen3-8B"
    return GGUFModel(base.path, base.version, metadata, base.tensors)


def _deepseek_llama_model() -> GGUFModel:
    base = _qwen3_model()
    metadata = {
        key.replace("qwen3.", "llama."): value
        for key, value in base.metadata.items()
    }
    metadata["general.architecture"] = "llama"
    metadata["general.name"] = "DeepSeek-R1-Distill-Llama-70B"
    tensors = tuple(
        tensor
        for tensor in base.tensors
        if tensor.name
        not in {
            "blk.0.attn_q_norm.weight",
            "blk.0.attn_k_norm.weight",
        }
    )
    return GGUFModel(base.path, base.version, metadata, tensors)


def _gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _remote_fixture() -> bytes:
    """Create a tiny valid header followed by a large fake tensor payload."""
    header = bytearray(b"GGUF")
    header.extend(struct.pack("<IQQ", 3, 1, 2))
    for key, value in (
        ("general.architecture", "llama"),
        ("general.name", "Remote Test"),
    ):
        header.extend(_gguf_string(key))
        header.extend(struct.pack("<I", 8))  # GGUF string value type
        header.extend(_gguf_string(value))
    header.extend(_gguf_string("token_embd.weight"))
    header.extend(struct.pack("<IQQIQ", 2, 4, 8, 0, 0))
    return bytes(header) + b"\0" * (2 * 1024 * 1024)


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    data = _remote_fixture()
    requested_ranges: list[tuple[int, int]] = []

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
        if match is None:
            self.send_error(400, "Range required")
            return
        start, requested_end = map(int, match.groups())
        if start >= len(self.data):
            self.send_response(416)
            self.end_headers()
            return
        end = min(requested_end, len(self.data) - 1)
        self.requested_ranges.append((start, end))
        payload = self.data[start : end + 1]
        self.send_response(206)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header(
            "Content-Range", f"bytes {start}-{end}/{len(self.data)}"
        )
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args) -> None:
        pass


class FinalV1Tests(unittest.TestCase):
    def test_matrix_product_detail_is_visual_and_navigable(self) -> None:
        diagram = build_diagram(config_from_gguf(_qwen3_model()))
        detail = detail_for_step(diagram, 0)
        canvas = render_matrix_detail(diagram, 0, 3, 4)
        plain = canvas.render(use_color=False)

        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertEqual(detail.left.key, "token_embedding")
        self.assertEqual(detail.right.key, "token_one_hot")
        self.assertEqual(detail.result.key, "embedding_x")
        self.assertIn("151936", plain)
        self.assertIn("4096", plain)
        self.assertIn("seq", plain)
        self.assertEqual(plain.count("151936"), 2)
        self.assertEqual(plain.count("4096"), 2)
        self.assertEqual(plain.count("seq"), 2)
        self.assertNotIn("E [", plain)
        self.assertNotIn("T [", plain)
        self.assertNotIn("X [", plain)
        self.assertIn("E", plain)
        self.assertIn("T", plain)
        self.assertIn("X", plain)
        self.assertIn("EMBEDDING", plain)
        self.assertRegex(plain, r"━+seq━+")
        self.assertRegex(plain, r"━+151936━+")
        self.assertNotIn("├", plain)
        self.assertNotIn("┼", plain)
        border_runs = {len(run) for run in re.findall(r"━+", plain)}
        self.assertGreater(max(border_runs), min(border_runs))
        self.assertNotIn(" × ", plain)
        self.assertEqual(plain.count(" = "), 1)
        self.assertGreater(
            sum(
                color == GRID_BACKGROUND
                for colors in canvas.background_colors
                for color in colors
            ),
            0,
        )
        self.assertGreaterEqual(
            sum(
                color == ROW_HIGHLIGHT
                for colors in canvas.colors
                for color in colors
            ),
            8,
        )
        self.assertGreaterEqual(
            sum(
                color == COLUMN_HIGHLIGHT
                for colors in canvas.colors
                for color in colors
            ),
            8,
        )
        column_rows = sorted(
            {
                row_index
                for row_index, colors in enumerate(canvas.colors)
                if COLUMN_HIGHLIGHT in colors
            }
        )
        self.assertTrue(
            any(
                second - first == 1
                for first, second in zip(column_rows, column_rows[1:])
            )
        )
        self.assertGreater(
            sum(
                color == RESULT_HIGHLIGHT
                for colors in canvas.colors
                for color in colors
            ),
            0,
        )
        self.assertEqual(
            {
                color
                for pixels_row, colors_row in zip(
                    canvas.pixels, canvas.colors
                )
                for pixel, color in zip(pixels_row, colors_row)
                if pixel in "0123456789"
            },
            {DIMENSION_COLOR},
        )
        self.assertEqual(
            {
                color
                for pixels_row, colors_row in zip(
                    canvas.pixels, canvas.colors
                )
                for pixel, color in zip(pixels_row, colors_row)
                if pixels_row[:9] == list("EMBEDDING")
                and pixel in "EMBEDDING"
            },
            {ANNOTATOR_COLOR},
        )
        pixels = "".join("".join(row) for row in canvas.pixels)
        self.assertGreaterEqual(pixels.count(INACTIVE_CELL), 60)
        self.assertEqual(
            {
                color
                for pixels_row, colors_row in zip(
                    canvas.pixels, canvas.colors
                )
                for pixel, color in zip(pixels_row, colors_row)
                if pixel == INACTIVE_CELL
            },
            {GRID_COLOR},
        )
        self.assertGreaterEqual(pixels.count(WEIGHT_CELL), 8)
        self.assertGreaterEqual(pixels.count(ACTIVATION_CELL), 8)
        self.assertEqual(pixels.count(RESULT_CELL), 2)
        self.assertGreaterEqual(pixels.count("·"), 4)
        self.assertEqual(pixels.count("→"), 1)
        self.assertEqual(pixels.count("↓"), 1)
        self.assertIn("· ·", plain)
        self.assertTrue(plain.startswith("EMBEDDING • Token embedding\n"))
        self.assertIn("X(out): ■ =", plain)
        embedding_lines = plain.splitlines()
        formula_y = next(
            index
            for index, line in enumerate(embedding_lines)
            if "X(out): ■ =" in line
        )
        self.assertEqual(embedding_lines[formula_y - 1], "")
        self.assertIn("■ = (●×●) + (●×●) + ... + (●×●)", plain)
        self.assertIn("[■]", plain)
        self.assertIn("E(learned)", plain)
        self.assertIn("T(in)", plain)
        self.assertIn("X(out)", plain)
        for qualifier in ("(in)", "(learned)", "(out)"):
            qualifier_colors = next(
                colors[start : start + len(qualifier)]
                for pixels, colors in zip(canvas.pixels, canvas.colors)
                if (start := "".join(pixels).find(qualifier)) >= 0
            )
            self.assertTrue(
                all(color == DIMENSION_COLOR for color in qualifier_colors)
            )

        above_omission = render_matrix_detail(diagram, 0, 2, 3)
        below_omission = render_matrix_detail(diagram, 0, 3, 3)

        def result_cell_y(detail_canvas) -> int:
            return next(
                y
                for y, (pixels_row, backgrounds_row) in enumerate(
                    zip(
                        detail_canvas.pixels,
                        detail_canvas.background_colors,
                    )
                )
                for pixel, background in zip(
                    pixels_row,
                    backgrounds_row,
                )
                if pixel == RESULT_CELL and background == GRID_BACKGROUND
            )

        self.assertGreater(
            result_cell_y(below_omission) - result_cell_y(above_omission),
            1,
        )

        moved = render_matrix_detail(diagram, 0, 0, 0)
        for character in ("E", "T", "X", "→", "↓"):
            original_positions = {
                (x, y)
                for y, pixels_row in enumerate(canvas.pixels)
                for x, pixel in enumerate(pixels_row)
                if pixel == character
            }
            moved_positions = {
                (x, y)
                for y, pixels_row in enumerate(moved.pixels)
                for x, pixel in enumerate(pixels_row)
                if pixel == character
            }
            self.assertEqual(original_positions, moved_positions)

    def test_left_matrix_edges_use_one_consistent_pair_gap(self) -> None:
        diagram = build_diagram(config_from_gguf(_qwen3_model()))
        right_x = diagram.matrix("x").x
        aligned_left_keys = (
            "token_embedding",
            "gamma",
            "wk",
            "gamma_k",
            "wv",
            "wq",
            "gamma_q",
            "kt_rope",
            "v_attention",
            "wo",
            "x_residual",
            "gamma_post",
            "wg",
            "wu",
            "wd",
            "r_final",
            "gamma_final",
            "wlm",
        )

        for key in aligned_left_keys:
            with self.subTest(matrix=key):
                matrix = diagram.matrix(key)
                right_edge = matrix.x + diagram.width_for(matrix.columns)
                self.assertEqual(right_x - right_edge, PAIR_GAP)

    def test_nested_panels_expand_around_matrices_without_overlap(self) -> None:
        diagram = build_diagram(config_from_gguf(_qwen3_model()))
        shift = diagram.horizontal_shift
        panel_groups = (
            (
                LEVEL_ONE_X,
                LEVEL_ONE_WIDTH + 2 * shift,
                (
                    "wk", "k", "gamma_k", "k_norm", "k_rope",
                    "wv", "v_projection", "wq", "q", "gamma_q",
                    "q_norm", "q_rope", "kt_rope", "a",
                    "a_softmax", "v_attention", "h",
                ),
            ),
            (
                LEVEL_TWO_X,
                LEVEL_TWO_WIDTH + 2 * shift,
                (
                    "wq", "q", "gamma_q", "q_norm", "q_rope",
                    "kt_rope", "a", "a_softmax", "v_attention", "h",
                ),
            ),
            (
                LEVEL_ONE_X,
                LEVEL_ONE_WIDTH + 2 * shift,
                ("wg", "g", "wu", "u", "p", "wd", "m"),
            ),
        )

        for panel_x, panel_width, matrix_keys in panel_groups:
            panel_right = panel_x + panel_width - 1
            for key in matrix_keys:
                with self.subTest(panel_x=panel_x, matrix=key):
                    matrix = diagram.matrix(key)
                    matrix_right = (
                        matrix.x + diagram.width_for(matrix.columns) - 1
                    )
                    self.assertGreater(matrix.x, panel_x)
                    self.assertLess(matrix_right, panel_right)

    def test_model_view_marks_learned_matrices_on_bottom_border(self) -> None:
        diagram = build_diagram(config_from_gguf(_qwen3_model()))
        canvas = render_diagram(diagram, 0)

        for matrix in diagram.matrices:
            width = diagram.width_for(matrix.columns)
            height = diagram.height_for(matrix.rows)
            marker = canvas.pixels[matrix.y + height - 1][
                matrix.x + width - 2
            ]
            marker_color = canvas.colors[matrix.y + height - 1][
                matrix.x + width - 2
            ]
            with self.subTest(matrix=matrix.key):
                if matrix.key in LEARNED_WEIGHTS:
                    expected_marker = (
                        ACTIVE_LEARNED_BORDER_MARKER
                        if matrix.key in diagram.operations[0].matrices
                        else INACTIVE_LEARNED_BORDER_MARKER
                    )
                    self.assertEqual(marker, expected_marker)
                    expected_color = (
                        ACTIVE_BORDER
                        if matrix.key
                        in diagram.operations[0].matrices
                        else INACTIVE_BORDER
                    )
                    self.assertEqual(marker_color, expected_color)
                else:
                    self.assertNotIn(
                        marker,
                        (
                            ACTIVE_LEARNED_BORDER_MARKER,
                            INACTIVE_LEARNED_BORDER_MARKER,
                        ),
                    )

    def test_rmsnorm_step_has_visual_and_navigable_detail(self) -> None:
        diagram = build_diagram(config_from_gguf(_qwen3_model()))
        norm_step = next(
            index
            for index, operation in enumerate(diagram.operations)
            if operation.result == "x_norm"
        )

        detail = detail_for_step(diagram, norm_step)
        self.assertIsNotNone(detail)
        canvas = render_matrix_detail(diagram, norm_step, 2, 3)
        plain = canvas.render(use_color=False)

        self.assertIn("ATTENTION", plain)
        self.assertTrue(plain.startswith("ATTENTION • RMSNorm\n"))
        self.assertIn("Xᵦ", plain)
        self.assertIn("γ", plain)
        self.assertIn("Xᵦ′", plain)
        self.assertIn("● = √((●² + ... + ●²) / 4096 + ε)", plain)
        self.assertIn("● = ● / ●", plain)
        self.assertIn("■ = ● × ●", plain)
        self.assertIn("[■]", plain)
        self.assertIn("Xᵦ(in)", plain)
        self.assertIn("γ(learned)", plain)
        self.assertIn("Xᵦ′(out)", plain)
        self.assertIn("rms", plain)
        self.assertIn("X̂ᵦ", plain)
        self.assertEqual(plain.count("→"), 1)
        self.assertEqual(plain.count("↓"), 2)
        self.assertEqual(plain.count("←"), 1)
        formula_lines = [
            line
            for line in plain.splitlines()
            if any(prefix in line for prefix in ("rms:", "X̂ᵦ:", "Xᵦ′(out):"))
        ]
        self.assertEqual(len(formula_lines), 3)
        first_formula_y = plain.splitlines().index(formula_lines[0])
        self.assertEqual(plain.splitlines()[first_formula_y - 1], "")
        self.assertTrue(all(line.startswith("    ") for line in formula_lines))
        def display_position(line: str, index: int) -> int:
            return sum(
                not unicodedata.combining(character)
                for character in line[:index]
            )

        self.assertEqual(
            len(
                {
                    display_position(line, line.index(":"))
                    for line in formula_lines
                }
            ),
            1,
        )
        self.assertEqual(
            len(
                {
                    display_position(
                        line,
                        line.index(marker, line.index(":")),
                    )
                    for line, marker in zip(
                        formula_lines,
                        (RMS_CELL, ACTIVATION_CELL, RESULT_CELL),
                    )
                }
            ),
            1,
        )
        self.assertGreaterEqual(plain.count(ACTIVATION_CELL), 9)
        colored_circles = {
            color: sum(
                pixel == ACTIVATION_CELL and pixel_color == color
                for pixels, colors in zip(canvas.pixels, canvas.colors)
                for pixel, pixel_color in zip(pixels, colors)
            )
            for color in (
                ROW_HIGHLIGHT,
                RMS_HIGHLIGHT,
                NORMALIZED_HIGHLIGHT,
            )
        }
        self.assertEqual(colored_circles[ROW_HIGHLIGHT], 2)
        self.assertEqual(colored_circles[RMS_HIGHLIGHT], 3)
        self.assertGreaterEqual(colored_circles[NORMALIZED_HIGHLIGHT], 3)
        self.assertEqual(plain.count(RESULT_CELL), 2)
        self.assertGreaterEqual(plain.count(INACTIVE_CELL), 80)
        self.assertEqual(selection_shape(diagram, norm_step, 6), (6, 6))

    def test_attention_score_formula_uses_two_activation_symbols(self) -> None:
        diagram = build_diagram(config_from_gguf(_qwen3_model()))
        score_step = next(
            index
            for index, operation in enumerate(diagram.operations)
            if operation.result == "a"
        )

        canvas = render_matrix_detail(
            diagram,
            score_step,
            0,
            0,
        )
        plain = canvas.render(use_color=False)

        self.assertIn("■ = (●×●) + (●×●) + ... + (●×●)", plain)
        self.assertIn("S(out): ■ =", plain)
        self.assertIn("[■]", plain)
        score_formula = next(
            line for line in plain.splitlines() if "■ = (●×●)" in line
        )
        self.assertTrue(score_formula.startswith("    "))
        self.assertNotIn("■×●", plain)
        detail = detail_for_step(diagram, score_step)
        assert detail is not None
        self.assertIn(f"{detail.left.label}(in)", plain)
        self.assertIn(f"{detail.right.label}(in)", plain)
        self.assertEqual(ENDPOINT_CELL_COUNT, 3)
        self.assertEqual(ENDPOINT_COLUMN_SPACING, 3)
        weight_positions = [
            x
            for pixels, colors, backgrounds in zip(
                canvas.pixels,
                canvas.colors,
                canvas.background_colors,
            )
            for x, (pixel, color, background) in enumerate(
                zip(pixels, colors, backgrounds)
            )
            if pixel == ACTIVATION_CELL
            and color == ROW_HIGHLIGHT
            and background == GRID_BACKGROUND
        ]
        self.assertEqual(len(weight_positions), 6)
        self.assertEqual(
            [
                weight_positions[index + 1] - weight_positions[index]
                for index in (0, 1, 3, 4)
            ],
            [3, 3, 3, 3],
        )

    def test_causal_softmax_scales_masks_and_normalizes_columns(self) -> None:
        diagram = build_diagram(config_from_gguf(_qwen3_model()))
        softmax_step = next(
            index
            for index, operation in enumerate(diagram.operations)
            if operation.result == "a_softmax"
        )

        allowed = render_matrix_detail(diagram, softmax_step, 0, 5)
        masked = render_matrix_detail(diagram, softmax_step, 5, 0)
        plain = allowed.render(use_color=False)

        self.assertTrue(plain.startswith("ATTENTION • Causal softmax\n"))
        self.assertIn("S(in)", plain)
        self.assertIn("A(out)", plain)
        self.assertNotIn("Prev:", plain)
        self.assertNotIn("Next:", plain)
        self.assertNotIn("Ŝ", plain)
        self.assertIn("● = ● / √128", plain)
        self.assertNotIn("÷√dₖ→", plain)
        self.assertNotIn("+Mᶜ→", plain)
        self.assertNotIn("p_q", plain)
        self.assertIn("~ = −∞", plain)
        self.assertIn("● = e(●) + e(●) + ... + e(●)", plain)
        self.assertIn("■ = e(●) / ●", plain)
        self.assertIn("× = 0", plain)
        causal_formula_lines = [
            line
            for line in plain.splitlines()
            if any(prefix in line for prefix in ("L:", "p:", "A(out):"))
        ]
        self.assertEqual(len(causal_formula_lines), 5)
        first_formula_y = plain.splitlines().index(causal_formula_lines[0])
        self.assertEqual(plain.splitlines()[first_formula_y - 1], "")
        self.assertTrue(
            all(line.startswith("    ") for line in causal_formula_lines)
        )
        self.assertEqual(
            len({line.index(":") for line in causal_formula_lines}),
            1,
        )
        self.assertNotIn("e^", plain)
        self.assertNotIn("e⁻∞", plain)
        self.assertNotIn("softmax↓", plain)
        self.assertEqual(plain.count("[●]"), 2)
        self.assertEqual(plain.count("[■]"), 1)
        masked_plain = masked.render(use_color=False)
        self.assertEqual(masked_plain.count("[~]"), 1)
        self.assertEqual(masked_plain.count("[×]"), 1)
        self.assertTrue(
            any("↓" in line and line.rstrip().endswith("p") for line in plain.splitlines())
        )
        self.assertTrue(
            any(
                "↓" in line and line.rstrip().endswith("A(out)")
                for line in plain.splitlines()
            )
        )
        self.assertEqual(selection_shape(diagram, softmax_step, 6), (6, 6))
        inactive_colors = {
            color
            for pixels, colors in zip(allowed.pixels, allowed.colors)
            for pixel, color in zip(pixels, colors)
            if pixel == INACTIVE_CELL
        }
        self.assertEqual(inactive_colors, {GRID_COLOR})
        highlighted_column_cells = sum(
            pixel == INACTIVE_CELL and color == COLUMN_HIGHLIGHT
            for pixels, colors, backgrounds in zip(
                allowed.pixels,
                allowed.colors,
                allowed.background_colors,
            )
            for pixel, color, background in zip(
                pixels,
                colors,
                backgrounds,
            )
            if background == GRID_BACKGROUND
        )
        self.assertEqual(highlighted_column_cells, 0)
        inactive_mask_colors = {
            color
            for pixels, colors, backgrounds in zip(
                allowed.pixels,
                allowed.colors,
                allowed.background_colors,
            )
            for pixel, color, background in zip(
                pixels,
                colors,
                backgrounds,
            )
            if pixel == MASKED_CELL and background == GRID_BACKGROUND
        }
        self.assertEqual(
            inactive_mask_colors,
            {ROW_HIGHLIGHT},
        )
        selected_mask_colors = {
            color
            for pixels, colors, backgrounds in zip(
                masked.pixels,
                masked.colors,
                masked.background_colors,
            )
            for pixel, color, background in zip(
                pixels,
                colors,
                backgrounds,
            )
            if pixel == MASKED_CELL
            and background == GRID_BACKGROUND
            and color != GRID_COLOR
        }
        self.assertEqual(
            selected_mask_colors,
            {ROW_HIGHLIGHT},
        )
        matrix_mask_count = sum(
            pixel == MASKED_CELL and background == GRID_BACKGROUND
            for pixels, backgrounds in zip(
                allowed.pixels,
                allowed.background_colors,
            )
            for pixel, background in zip(pixels, backgrounds)
        )
        self.assertEqual(matrix_mask_count, 15)
        matrix_results = sum(
            pixel == RESULT_CELL and background == GRID_BACKGROUND
            for pixels, backgrounds in zip(
                allowed.pixels,
                allowed.background_colors,
            )
            for pixel, background in zip(pixels, backgrounds)
        )
        self.assertEqual(matrix_results, 1)
        self.assertEqual(
            sum(
                pixel == RESULT_CELL and background == GRID_BACKGROUND
                for pixels, backgrounds in zip(
                    masked.pixels,
                    masked.background_colors,
                )
                for pixel, background in zip(pixels, backgrounds)
            ),
            0,
        )
        self.assertLessEqual(allowed.width, 80)

    def test_gated_activation_shows_activation_and_pointwise_product(self) -> None:
        base = config_from_gguf(_qwen3_model())
        for activation in ("SiLU", "GELU"):
            with self.subTest(activation=activation):
                diagram = build_diagram(replace(base, activation=activation))
                gated_step = next(
                    index
                    for index, operation in enumerate(diagram.operations)
                    if operation.result == "p"
                )
                initial = render_matrix_detail(diagram, gated_step, 0, 0)
                moved = render_matrix_detail(diagram, gated_step, 5, 5)
                plain = initial.render(use_color=False)

                self.assertTrue(plain.startswith("MLP • Gated activation\n"))
                self.assertIn("G(in)", plain)
                self.assertIn("U(in)", plain)
                self.assertIn(f"{activation}(G)", plain)
                self.assertIn("P(out)", plain)
                self.assertEqual(plain.count("↓"), 1)
                self.assertEqual(plain.count("⊙"), 1)
                self.assertIn("P(out): ■ = ● ⊙ ●", plain)
                if activation == "SiLU":
                    self.assertIn(
                        "SiLU(G): ● = ● / (1 + e(−●))",
                        plain,
                    )
                    self.assertNotIn("σ(G)", plain)
                else:
                    self.assertIn(
                        "GELU(G): ● = ½● × (1 + erf(● / √2))",
                        plain,
                    )
                self.assertEqual(plain.count("[■]"), 1)
                self.assertEqual(moved.render(False).count("[■]"), 1)
                self.assertEqual(
                    selection_shape(diagram, gated_step, 6),
                    (6, 6),
                )
                self.assertLessEqual(initial.width, 80)
                self.assertLessEqual(initial.height + 3, 32)

    def test_residual_additions_show_two_inputs_and_elementwise_sum(self) -> None:
        diagram = build_diagram(config_from_gguf(_qwen3_model()))
        expectations = {
            "r": (
                "ATTENTION • Attention residual",
                "Xᵦ(in)",
                "Y(in)",
                "R(out)",
            ),
            "x_out": (
                "MLP • MLP residual",
                "R(in)",
                "M(in)",
                "Xᵦ₊₁(out)",
            ),
        }
        for result, labels in expectations.items():
            with self.subTest(result=result):
                step = next(
                    index
                    for index, operation in enumerate(diagram.operations)
                    if operation.result == result
                )
                initial = render_matrix_detail(diagram, step, 0, 0)
                moved = render_matrix_detail(diagram, step, 5, 5)
                plain = initial.render(use_color=False)

                for label in labels:
                    self.assertIn(label, plain)
                self.assertEqual(plain.count("→"), 1)
                self.assertEqual(plain.count("↓"), 1)
                self.assertEqual(plain.count("[■]"), 1)
                self.assertIn(f"{labels[3]}: ■ = ● + ●", plain)
                self.assertEqual(moved.render(False).count("[■]"), 1)
                self.assertEqual(selection_shape(diagram, step, 6), (6, 6))
                self.assertLessEqual(initial.width, 80)
                self.assertLessEqual(initial.height + 3, 32)

    def test_block_routing_explainer_is_intentionally_disabled(self) -> None:
        diagram = build_diagram(config_from_gguf(_qwen3_model()))
        step = next(
            index
            for index, operation in enumerate(diagram.operations)
            if operation.result == "x"
        )
        self.assertIsNone(detail_for_step(diagram, step))

    def test_rope_detail_aligns_rotation_rows_with_input_and_output(self) -> None:
        diagram = build_diagram(config_from_gguf(_qwen3_model()))
        rope_step = next(
            index
            for index, operation in enumerate(diagram.operations)
            if operation.result == "k_rope"
        )

        detail = detail_for_step(diagram, rope_step)
        self.assertIsNotNone(detail)
        first = render_matrix_detail(diagram, rope_step, 0, 0)
        moved = render_matrix_detail(diagram, rope_step, 5, 5)
        plain = first.render(use_color=False)

        self.assertTrue(
            plain.startswith("ATTENTION • K rotary position encoding\n")
        )
        self.assertIn("K′", plain)
        self.assertNotIn("pair", plain)
        self.assertIn("Kᵣ", plain)
        self.assertNotIn("R(m", plain)
        self.assertIn("R", plain)
        self.assertEqual(plain.count("[0]"), 2)
        self.assertEqual(plain.count("[64]"), 2)
        self.assertIn("┃[0]", plain)
        self.assertIn("┃[64]", plain)
        self.assertIn("[0]─┐", plain)
        self.assertIn("[64]┘", plain)
        self.assertEqual(plain.count("d/2"), 2)
        self.assertNotIn("128/2", plain)
        self.assertIn("K′(in)", plain)
        self.assertIn("R(in)", plain)
        self.assertIn("Kᵣ(out)", plain)
        self.assertEqual(plain.count("┏"), 3)
        self.assertEqual(plain.count("⋮"), 1)
        self.assertEqual(ROPE_SECOND_HIGHLIGHT, (90, 180, 225))
        self.assertIn("○  ○  ○  ○  ○  ○", plain)
        self.assertNotIn("○  ○  ○   ○  ○  ○", plain)
        self.assertIn("■ = ● × cos(m×θᵢ) − ● × sin(m×θᵢ)", plain)
        self.assertIn("■ = ● × sin(m×θᵢ) + ● × cos(m×θᵢ)", plain)
        self.assertEqual(plain.count("Kᵣ(out): ■ ="), 2)
        self.assertIn("R(in): θᵢ = 1000000^(−2i/128)", plain)
        first_rope_formula = next(
            index
            for index, line in enumerate(plain.splitlines())
            if "Kᵣ(out): ■ =" in line
        )
        self.assertEqual(plain.splitlines()[first_rope_formula - 1], "")
        self.assertIn("θᵢ = 1000000^(−2i/128)", plain)
        rope_formula_lines = [
            line
            for line in plain.splitlines()
            if "cos(m×θᵢ)" in line
            or line.lstrip().startswith("θᵢ =")
        ]
        self.assertTrue(
            all(line.startswith("    ") for line in rope_formula_lines)
        )
        self.assertEqual(plain.count("↓"), 1)
        self.assertEqual(plain.count("→"), 2)
        self.assertEqual(plain.count(RESULT_CELL), 4)
        self.assertEqual(plain.count("·"), 48)
        self.assertEqual(selection_shape(diagram, rope_step, 6), (4, 6))
        self.assertGreaterEqual(plain.count(INACTIVE_CELL), 80)

        red_rotation_y = next(
            y
            for y, (pixels, backgrounds) in enumerate(
                zip(first.pixels, first.background_colors)
            )
            if "cos(" in (line := "".join(pixels))
            and "sin(" in line
            and line.index("cos(") < line.index("sin(")
            and GRID_BACKGROUND in backgrounds
        )
        blue_rotation_y = next(
            y
            for y, (pixels, backgrounds) in enumerate(
                zip(first.pixels, first.background_colors)
            )
            if "cos(" in (line := "".join(pixels))
            and "sin(" in line
            and line.index("sin(") < line.index("cos(")
            and GRID_BACKGROUND in backgrounds
        )
        result_rows = [
            y
            for y, (pixels, backgrounds) in enumerate(
                zip(first.pixels, first.background_colors)
            )
            if any(
                pixel == RESULT_CELL and background == GRID_BACKGROUND
                for pixel, background in zip(pixels, backgrounds)
            )
        ]
        self.assertEqual(result_rows, [red_rotation_y, blue_rotation_y])

        def selected_positions(canvas) -> set[tuple[int, int]]:
            return {
                (x, y)
                for y, (pixels, backgrounds) in enumerate(
                    zip(canvas.pixels, canvas.background_colors)
                )
                for x, (pixel, background) in enumerate(
                    zip(pixels, backgrounds)
                )
                if pixel == ACTIVATION_CELL
                and background == GRID_BACKGROUND
            }

        self.assertNotEqual(
            selected_positions(first),
            selected_positions(moved),
        )
        next_pair = render_matrix_detail(diagram, rope_step, 1, 0)
        self.assertNotEqual(
            selected_positions(first),
            selected_positions(next_pair),
        )
        next_pair_plain = next_pair.render(use_color=False)
        self.assertIn("[1]", next_pair_plain)
        self.assertIn("[65]", next_pair_plain)
        first_split_y = next(
            y
            for y, pixels in enumerate(first.pixels)
            if "d/2" in "".join(pixels)
        )
        next_split_y = next(
            y
            for y, pixels in enumerate(next_pair.pixels)
            if "d/2" in "".join(pixels)
        )
        self.assertEqual(next_split_y, first_split_y + 1)
        numeric_rotation = render_matrix_detail(diagram, rope_step, 1, 1)
        numeric_plain = numeric_rotation.render(use_color=False)
        self.assertIn("cos(1×0.806)", numeric_plain)
        self.assertIn("sin(1×0.806)", numeric_plain)
        for middle_column in (2, 3):
            middle_plain = render_matrix_detail(
                diagram,
                rope_step,
                1,
                middle_column,
            ).render(use_color=False)
            for collision in ("○●", "●○", "○■", "■○"):
                self.assertNotIn(collision, middle_plain)

        def matrix_corners(canvas) -> set[tuple[int, int]]:
            return {
                (x, y)
                for y, pixels in enumerate(canvas.pixels)
                if "seq" in "".join(pixels)
                for x, pixel in enumerate(pixels)
                if pixel == "┏" and x >= canvas.width // 3
            }

        self.assertEqual(
            matrix_corners(first),
            matrix_corners(next_pair),
        )

        left_edge = render_matrix_detail(diagram, rope_step, 0, 0)
        right_edge = render_matrix_detail(diagram, rope_step, 0, 5)

        def active_x(canvas, glyph: str) -> set[int]:
            return {
                x
                for pixels, backgrounds in zip(
                    canvas.pixels, canvas.background_colors
                )
                for x, (pixel, background) in enumerate(
                    zip(pixels, backgrounds)
                )
                if pixel == glyph and background == GRID_BACKGROUND
            }

        activation_travel = max(active_x(right_edge, ACTIVATION_CELL)) - max(
            active_x(left_edge, ACTIVATION_CELL)
        )
        result_travel = max(active_x(right_edge, RESULT_CELL)) - max(
            active_x(left_edge, RESULT_CELL)
        )
        self.assertEqual(activation_travel, result_travel)

    def test_head_concatenation_shows_sampled_heads_from_three_kv_groups(self) -> None:
        diagram = build_diagram(config_from_gguf(_qwen3_model()))
        concat_step = next(
            index
            for index, operation in enumerate(diagram.operations)
            if operation.result == "h_merge"
        )

        initial = render_matrix_detail(diagram, concat_step, 0, 0)
        moved = render_matrix_detail(diagram, concat_step, 3, 5)
        plain = initial.render(use_color=False)
        moved_plain = moved.render(use_color=False)

        self.assertTrue(plain.startswith("ATTENTION • Head concatenation\n"))
        self.assertIn("H(out)", plain)
        self.assertIn("KV₀", plain)
        self.assertIn("KV₃", plain)
        self.assertIn("KV₇", plain)
        self.assertIn("H₀,₀(in)", plain)
        self.assertIn("H₀,₁(in)", plain)
        self.assertIn("H₃,₀(in)", plain)
        self.assertIn("H₃,₁(in)", plain)
        self.assertIn("H₇,₂(in)", plain)
        self.assertIn("H₇,₃(in)", plain)
        self.assertIn("KV₀ ─ Q₀ → H₀,₀(in) → 128", plain)
        self.assertNotIn("KV₀ →", plain)
        self.assertEqual(plain.count("128 ·"), 6)
        self.assertEqual(plain.count("→"), 12)
        self.assertEqual(plain.count("⋮"), 4)
        self.assertEqual(plain.count("[■]"), 1)
        self.assertIn("128×8×4=4096", plain)
        self.assertEqual(selection_shape(diagram, concat_step, 6), (6, 6))
        self.assertEqual((initial.width, initial.height), (57, 31))
        self.assertEqual(
            (initial.width, initial.height),
            (moved.width, moved.height),
        )
        self.assertIn("KV₃", moved_plain)
        self.assertNotIn("KV₅", moved_plain)

        def colors_for(canvas, text: str) -> list[object]:
            return next(
                colors[start : start + len(text)]
                for pixels, colors in zip(canvas.pixels, canvas.colors)
                if (start := "".join(pixels).find(text)) >= 0
            )

        self.assertTrue(
            all(color == COLUMN_HIGHLIGHT for color in colors_for(initial, "KV₀"))
        )
        self.assertTrue(
            all(color == ROW_HIGHLIGHT for color in colors_for(initial, "Q₀"))
        )
        self.assertIn("KV₀ ─ Q₁ → H₀,₁(in) → 128", plain)
        h00_y = next(
            y
            for y, pixels in enumerate(initial.pixels)
            if "H₀,₀(in)" in "".join(pixels)
        )
        self.assertEqual("".join(initial.pixels[h00_y]).count("·"), 6)
        self.assertGreaterEqual(
            "".join(initial.pixels[h00_y - 1]).count(INACTIVE_CELL),
            5,
        )
        self.assertEqual(
            "".join(initial.pixels[h00_y + 1]).count(INACTIVE_CELL),
            6,
        )
        with patch(
            "ggufvis.cli.shutil.get_terminal_size",
            return_value=os.terminal_size((80, 36)),
        ):
            frame, _, _, rows, columns = _detail_viewport(
                diagram,
                concat_step,
                0,
                0,
                False,
                "Ollama: qwen3",
            )
        self.assertNotIn("Terminal is too small", frame)
        self.assertEqual((rows, columns), (6, 6))
        self.assertLessEqual(len(frame.splitlines()), 36)

    def test_matrix_detail_adapts_without_clipping_labels(self) -> None:
        diagram = build_diagram(config_from_gguf(_qwen3_model()))
        with patch(
            "ggufvis.cli.shutil.get_terminal_size",
            return_value=os.terminal_size((80, 32)),
        ):
            frame, _, _, rows, columns = _detail_viewport(
                diagram,
                0,
                0,
                0,
                False,
                "Ollama: qwen3",
            )

        self.assertEqual((rows, columns), (6, 6))
        self.assertTrue(frame.startswith("Ollama: qwen3 [explainer view]\n"))
        self.assertIn("151936", frame)
        self.assertIn("4096", frame)
        self.assertIn("seq", frame)
        self.assertLessEqual(len(frame.splitlines()), 32)
        self.assertLessEqual(max(map(len, frame.splitlines())), 80)
        self.assertIn(
            "(●×●) + ... + (●×●)\n\nArrows:",
            frame,
        )
        self.assertIn("Esc: back to Model View", frame)

    def test_matrix_detail_keeps_separate_glyphs_with_heading_block(self) -> None:
        diagram = build_diagram(config_from_gguf(_qwen3_model()))
        with patch(
            "ggufvis.cli.shutil.get_terminal_size",
            return_value=os.terminal_size((80, 27)),
        ):
            frame, _, _, rows, columns = _detail_viewport(
                diagram,
                0,
                0,
                0,
                False,
                "Ollama: qwen3",
            )

        self.assertEqual((rows, columns), (3, 3))
        self.assertGreaterEqual(frame.count(ACTIVATION_CELL), 6)
        self.assertNotIn("Terminal is too small", frame)
        self.assertLessEqual(len(frame.splitlines()), 27)
        self.assertIn("EMBEDDING • Token embedding\n", frame)

    def test_repository_launcher_is_named_ggufvis2(self) -> None:
        repository = Path(__file__).parents[1]

        self.assertTrue((repository / "ggufvis2.py").is_file())
        self.assertFalse((repository / "ggufvis.py").exists())

    def test_terminal_reader_recognizes_horizontal_arrows(self) -> None:
        with patch("ggufvis.terminal.sys.stdin", io.StringIO("\x1b[C")):
            self.assertEqual(read_key(), "right")
        with patch("ggufvis.terminal.sys.stdin", io.StringIO("\x1b[D")):
            self.assertEqual(read_key(), "left")

    def test_terminal_reader_uses_unbuffered_descriptor_input(self) -> None:
        read_descriptor, write_descriptor = os.pipe()
        os.write(write_descriptor, b"\x1b[B")
        os.close(write_descriptor)
        with os.fdopen(read_descriptor, encoding="latin-1") as input_stream:
            with patch("ggufvis.terminal.sys.stdin", input_stream):
                self.assertEqual(read_key(), "down")

    def test_rope_navigation_has_no_multiframe_delay(self) -> None:
        cli_source = (Path(__file__).parents[1] / "ggufvis" / "cli.py").read_text()

        self.assertNotIn("isinstance(detail, RoPEDetail)", cli_source)
        detail_keys = cli_source.split(
            '            if detail_mode:\n                if key == "up":',
            1,
        )[1].split("                continue", 1)[0]
        self.assertNotIn("time.sleep", detail_keys)
        cli_tree = ast.parse(cli_source)
        self.assertFalse(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "sleep"
                for node in ast.walk(cli_tree)
            )
        )

    def test_deepseek_distill_uses_qwen_or_llama_backbone(self) -> None:
        qwen = config_from_gguf(_deepseek_qwen3_model())
        llama = config_from_gguf(_deepseek_llama_model())
        qwen_diagram = build_diagram(qwen)
        qwen_card = card_lines(qwen_diagram, 0)
        qwen_render = render_diagram(qwen_diagram, 0).render(use_color=False)

        self.assertTrue(qwen.is_deepseek_r1_distill)
        self.assertEqual(qwen.architecture, "qwen3")
        self.assertEqual(qwen.backbone_name, "Qwen3")
        self.assertTrue(qwen.qk_norm)
        self.assertIn(
            "DeepSeek-R1 Distill (Qwen3) Block b ×36", qwen_render
        )
        self.assertEqual(qwen_card[0], "[01/24] EMBEDDING")

        self.assertTrue(llama.is_deepseek_r1_distill)
        self.assertEqual(llama.architecture, "llama")
        self.assertEqual(llama.backbone_name, "Llama")
        self.assertFalse(llama.qk_norm)
        self.assertEqual(len(build_diagram(llama).operations), 22)

    def test_annotation_titles_do_not_repeat_model_identity(self) -> None:
        base = config_from_gguf(_qwen3_model())
        for architecture in ("llama", "gemma", "qwen2", "qwen3"):
            with self.subTest(architecture=architecture):
                config = replace(
                    base,
                    architecture=architecture,
                    is_deepseek_r1_distill=False,
                )
                title = card_lines(build_diagram(config), 0)[0]
                self.assertEqual(title, "[01/24] EMBEDDING")

    def test_native_deepseek_architecture_is_not_guessed(self) -> None:
        base = _qwen3_model()
        metadata = dict(base.metadata)
        metadata["general.architecture"] = "deepseek2"
        native = GGUFModel(base.path, base.version, metadata, base.tensors)

        with self.assertRaisesRegex(
            GGUFError, "native DeepSeek architecture"
        ):
            config_from_gguf(native)

    def test_qwen_dimensions_norms_and_annotations_are_integrated(self) -> None:
        config = config_from_gguf(_qwen3_model())
        diagram = build_diagram(config)
        result_keys = [operation.result for operation in diagram.operations]
        step = result_keys.index("k_norm")
        lines = card_lines(diagram, step)
        rendered = render_diagram(diagram, step).render(use_color=False)

        self.assertEqual(config.attention_kind, "GQA")
        self.assertEqual(len(diagram.operations), 24)
        self.assertEqual(diagram.height_for("seq"), 7)
        self.assertEqual(diagram.height_for("151936"), 7)
        self.assertEqual(lines[0], "[05/24] ATTENTION")
        self.assertEqual(lines[2], "Equation: K′ = RMSNorm(K; γk)")
        self.assertEqual(
            lines[3],
            "Source: blk.b.attn_k_norm.weight",
        )
        score_step = result_keys.index("a")
        attention_step = result_keys.index("a_softmax")
        head_step = result_keys.index("h")
        self.assertIn(
            "S = Kᵣᵀ × Qᵣ",
            card_lines(diagram, score_step)[2],
        )
        self.assertIn(
            "A = softmax(S/√128 + Mᶜ)",
            card_lines(diagram, attention_step)[2],
        )
        self.assertIn(
            "Hᵢⱼ = V × A",
            card_lines(diagram, head_step)[2],
        )
        self.assertIn(
            "X = E × T",
            card_lines(diagram, 0)[2],
        )
        self.assertIn(
            "          vocab = 151936",
            card_lines(diagram, 0)[2],
        )
        block_input_step = result_keys.index("x")
        block_input_card = card_lines(diagram, block_input_step)
        self.assertEqual(block_input_card[0], "[02/24] BLOCK INPUT")
        self.assertEqual(
            block_input_card[1],
            "Operation: Route hidden state into block",
        )
        self.assertIn(
            "Xᵦ = X when b=0;\n"
            "          otherwise output of block b−1",
            block_input_card[2],
        )
        self.assertEqual(
            block_input_card[3],
            "Source: none (runtime activation routing)",
        )
        self.assertEqual(
            diagram.operations[block_input_step].matrices,
            frozenset({"embedding_x", "x"}),
        )
        norm_step = result_keys.index("x_norm")
        self.assertIn(
            "Xᵦ′ = RMSNorm(Xᵦ; γ)",
            card_lines(diagram, norm_step)[2],
        )
        output_step = result_keys.index("x_out")
        self.assertIn(
            "Xᵦ₊₁ = R + M",
            card_lines(diagram, output_step)[2],
        )
        self.assertIn(" Qwen3 Block b ×36 ", rendered)
        self.assertIn(" Head end ┘", rendered)
        self.assertIn(" GQA end ┘", rendered)
        self.assertLessEqual(maximum_card_width(diagram), 61)

    def test_tied_output_uses_embedding_source(self) -> None:
        diagram = build_diagram(config_from_gguf(_qwen3_model(tied=True)))
        lines = card_lines(diagram, len(diagram.operations) - 1)

        self.assertEqual(diagram.matrix("wlm").label, "Eᵀ")
        self.assertEqual(
            lines[3],
            "Source: token_embd.weight (tied)",
        )

    def test_remote_reader_uses_ranges_without_saving_the_model(self) -> None:
        _RangeHandler.requested_ranges = []
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        url = (
            f"http://127.0.0.1:{server.server_address[1]}/remote-test.gguf"
        )

        model = read_gguf_url(url)

        self.assertEqual(model.metadata["general.architecture"], "llama")
        self.assertEqual(model.tensor("token_embd.weight").shape, (4, 8))
        self.assertEqual(model.remote_file_size, len(_RangeHandler.data))
        self.assertLess(model.remote_bytes_transferred, len(_RangeHandler.data))
        self.assertTrue(_RangeHandler.requested_ranges)
        self.assertFalse(model.path.exists())

    def test_annotation_can_stack_below_for_narrow_terminals(self) -> None:
        diagram = build_diagram(config_from_gguf(_qwen3_model()))
        beside = render_diagram(diagram, 0, annotation_position="right")
        below = render_diagram(diagram, 0, annotation_position="below")
        below_text = below.render(use_color=False)

        self.assertLess(below.width, beside.width)
        self.assertGreater(below.height, beside.height)
        self.assertIn("[01/24] EMBEDDING", below_text)
        self.assertIn("Source: token_embd.weight", below_text)

    def test_annotation_height_grows_with_wrapped_text(self) -> None:
        diagram = build_diagram(config_from_gguf(_qwen3_model()))
        result_keys = [operation.result for operation in diagram.operations]
        single_line_step = result_keys.index("x_norm")
        two_line_step = result_keys.index("x")

        single_line = render_diagram(
            diagram, single_line_step, annotation_position="below"
        )
        two_lines = render_diagram(
            diagram, two_line_step, annotation_position="below"
        )

        self.assertEqual(two_lines.height, single_line.height + 1)
        card_start = diagram.panels.canvas_height + 1
        single_card = single_line.render(use_color=False).splitlines()[card_start:]
        two_line_card = two_lines.render(use_color=False).splitlines()[card_start:]
        self.assertEqual(len(single_card), 5)
        self.assertEqual(len(two_line_card), 6)
        self.assertIn("Source:", single_card[-2])
        self.assertIn("Source:", two_line_card[-2])
        self.assertTrue(single_card[-1].startswith("└"))
        self.assertTrue(two_line_card[-1].startswith("└"))
        self.assertIn(
            "otherwise output of block b−1",
            two_lines.render(use_color=False),
        )

    def test_unselected_operations_have_low_contrast_compact_annotations(self) -> None:
        diagram = build_diagram(config_from_gguf(_qwen3_model()))
        beside = render_diagram(diagram, 0, annotation_position="right")
        below = render_diagram(diagram, 0, annotation_position="below")
        operation = "[06/24] K rotary position encoding"

        def colors_for(text: str) -> list[object]:
            return next(
                colors[start : start + len(text)]
                for pixels, colors in zip(beside.pixels, beside.colors)
                if (start := "".join(pixels).find(text)) >= 0
            )

        self.assertTrue(
            all(color == COMPACT_CARD_TEXT for color in colors_for(operation))
        )
        beside_text = beside.render(use_color=False)
        self.assertNotIn("[06/24] ATTENTION", beside_text)
        self.assertIn(operation, beside_text)
        self.assertNotIn(operation, below.render(use_color=False))

    def test_narrow_viewport_keeps_annotation_visible(self) -> None:
        diagram = build_diagram(config_from_gguf(_qwen3_model()))
        with patch(
            "ggufvis.cli.shutil.get_terminal_size",
            return_value=os.terminal_size((81, 30)),
        ):
            frame, _ = _viewport(
                diagram, 0, False, None, "GGUF: synthetic-qwen3.gguf"
            )

        self.assertNotIn("Widen the window", frame)
        self.assertIn("[01/24] EMBEDDING", frame)
        self.assertIn("Source: token_embd.weight", frame)
        self.assertIn(
            "Tip: widen the terminal to place annotations beside the diagram.",
            frame,
        )
        self.assertTrue(
            frame.startswith("GGUF: synthetic-qwen3.gguf [model view]\n")
        )
        self.assertIn("Right: Explainer", frame)

    def test_source_header_is_adaptive_and_source_aware(self) -> None:
        local = _source_header("/models/llama.gguf")
        ollama = _source_header("deepseek-r1:8b", is_ollama=True)
        remote = _source_header(
            "https://huggingface.co/a/b/resolve/main/"
            "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
            is_remote=True,
        )

        self.assertEqual(local, "GGUF: llama.gguf")
        self.assertEqual(ollama, "Ollama: deepseek-r1:8b")
        self.assertEqual(
            remote,
            "Remote: DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
        )
        fitted = _fit_header(remote, 30)
        self.assertEqual(len(fitted), 30)
        self.assertIn("…", fitted)
        self.assertTrue(fitted.endswith("Q4_K_M.gguf"))

    def test_mha_and_mqa_have_clean_non_nested_labels(self) -> None:
        base = config_from_gguf(_qwen3_model())
        mha = replace(
            base,
            qk_norm=False,
            query_heads=16,
            kv_heads=16,
            key_head_dim=256,
            value_head_dim=256,
        )
        mqa = replace(
            base,
            qk_norm=False,
            query_heads=16,
            kv_heads=1,
            key_head_dim=256,
            value_head_dim=256,
        )
        mha_text = render_diagram(build_diagram(mha), 0).render(False)
        mqa_text = render_diagram(build_diagram(mqa), 0).render(False)

        self.assertIn(" MHA end ┘", mha_text)
        self.assertNotIn(" Head end ┘", mha_text)
        self.assertIn(" Shared K/V end ┘", mqa_text)
        self.assertIn(" Query end ┘", mqa_text)

        for config, expected_label in ((mha, "H₀(in)"), (mqa, "H₀(in)")):
            with self.subTest(explainer=config.attention_kind):
                diagram = build_diagram(config)
                concat_step = next(
                    index
                    for index, operation in enumerate(diagram.operations)
                    if operation.result == "h_merge"
                )
                detail = render_matrix_detail(
                    diagram,
                    concat_step,
                    0,
                    0,
                ).render(False)
                self.assertIn(expected_label, detail)
                self.assertIn("[■]", detail)

    def test_final_folder_has_no_historical_visualizer_imports(self) -> None:
        folder = Path(__file__).parents[1] / "ggufvis"
        historical = {
            f"ggufvis{version}" for version in range(2, 9)
        } | {"tensor_diagram_interactive_full_block_external_annotations"}
        imported: set[str] = set()
        for path in folder.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module)

        self.assertTrue(historical.isdisjoint(imported))


if __name__ == "__main__":
    unittest.main()
