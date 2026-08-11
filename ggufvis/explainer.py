"""Visual-only matrix multiplication details for interactive navigation."""

from __future__ import annotations

from dataclasses import dataclass
import unicodedata

from .annotations import OPERATION_BY_RESULT, SECTION_BY_RESULT
from .diagram import LEARNED_WEIGHTS, Diagram, Matrix
from .terminal import Canvas, put


DEFAULT_SAMPLE_LIMIT = 6
MINIMUM_SAMPLE_LIMIT = 3
ENDPOINT_CELL_COUNT = 3
ENDPOINT_COLUMN_SPACING = 3
WEIGHT_CELL = "●"
ACTIVATION_CELL = "●"
RESULT_CELL = "■"
RMS_CELL = "●"
INACTIVE_CELL = "○"
MASKED_CELL = "~"
ROW_HIGHLIGHT = (218, 168, 82)
COLUMN_HIGHLIGHT = (112, 178, 118)
RESULT_HIGHLIGHT = (211, 70, 86)
RMS_HIGHLIGHT = (92, 150, 205)
NORMALIZED_HIGHLIGHT = (190, 110, 195)
ROPE_FIRST_HIGHLIGHT = (211, 70, 86)
ROPE_SECOND_HIGHLIGHT = (90, 180, 225)
GRID_BACKGROUND = (28, 28, 28)
GRID_COLOR = (125, 125, 125)
DIMENSION_COLOR = (145, 145, 145)
ANNOTATOR_COLOR = (235, 70, 205)
OPERATION_COLOR = (190, 155, 205)
VARIABLE_COLOR = (235, 195, 95)
LABEL_COLOR = "bright_white"
HEADING_HEIGHT = 2
FORMULA_INDENT = 4
CAUSAL_CELL_SPACING = 3


# Only true matrix products use the row × column detail. Element-wise,
# normalization, routing, and parameter-free steps remain in the main view.
PRODUCT_MATRICES: dict[str, tuple[str, str, str]] = {
    "embedding_x": ("token_embedding", "token_one_hot", "embedding_x"),
    "k": ("wk", "x_norm", "k"),
    "v_projection": ("wv", "x_norm", "v_projection"),
    "q": ("wq", "x_norm", "q"),
    "a": ("kt_rope", "q_rope", "a"),
    "h": ("v_attention", "a_softmax", "h"),
    "y": ("wo", "h_merge", "y"),
    "g": ("wg", "r_norm", "g"),
    "u": ("wu", "r_norm", "u"),
    "m": ("wd", "p", "m"),
    "logits": ("wlm", "z", "logits"),
}

RMSNORM_MATRICES: dict[str, tuple[str, str, str]] = {
    "x_norm": ("x", "gamma", "x_norm"),
    "k_norm": ("k", "gamma_k", "k_norm"),
    "q_norm": ("q", "gamma_q", "q_norm"),
    "r_norm": ("r", "gamma_post", "r_norm"),
    "z": ("x_out", "gamma_final", "z"),
}


@dataclass(frozen=True)
class MatrixDetail:
    """The two operands and result shown in the visual matrix view."""

    left: Matrix
    right: Matrix
    result: Matrix


@dataclass(frozen=True)
class RMSNormDetail:
    """Input, learned scale, and result for one RMSNorm operation."""

    input: Matrix
    scale: Matrix
    result: Matrix


@dataclass(frozen=True)
class RoPEDetail:
    """Input and output matrices for one rotary-position operation."""

    input: Matrix
    result: Matrix


@dataclass(frozen=True)
class CausalSoftmaxDetail:
    """Raw scores and normalized causal attention weights."""

    scores: Matrix
    result: Matrix


@dataclass(frozen=True)
class HeadConcatDetail:
    """Per-query attention heads and their row-wise concatenation."""

    input: Matrix
    result: Matrix


@dataclass(frozen=True)
class GatedActivationDetail:
    """Gate, up-projection, and pointwise gated output matrices."""

    gate: Matrix
    up: Matrix
    result: Matrix


@dataclass(frozen=True)
class ResidualDetail:
    """Residual stream, branch contribution, and elementwise sum."""

    residual: Matrix
    branch: Matrix
    result: Matrix


@dataclass(frozen=True)
class BlockLoopDetail:
    """Initial hidden state and one selected transformer-block iteration."""

    initial: Matrix
    result: Matrix


@dataclass(frozen=True)
class BoxLayout:
    """Dimension-scaled geometry and sampled navigation bounds."""

    matrix: Matrix
    width: int
    height: int
    row_count: int
    column_count: int

    @property
    def box_offset(self) -> int:
        # The row dimension overlaps the left border and extends outward.
        return max(0, len(self.matrix.rows) - 1)

    @property
    def total_width(self) -> int:
        return self.box_offset + self.width


def detail_for_step(
    diagram: Diagram, step: int
) -> (
    MatrixDetail
    | RMSNormDetail
    | RoPEDetail
    | CausalSoftmaxDetail
    | HeadConcatDetail
    | GatedActivationDetail
    | ResidualDetail
    | BlockLoopDetail
    | None
):
    """Return an interactive detail supported by the selected operation."""
    result = diagram.operations[step].result
    # Block routing is already explained by the Model View annotation. Keep
    # the static BlockLoopDetail renderer available in case we want to restore
    # it later, but do not register an Operation Explainer for `x` now.
    # if result == "x":
    #     return BlockLoopDetail(
    #         diagram.matrix("embedding_x"),
    #         diagram.matrix("x_out"),
    #     )
    keys = PRODUCT_MATRICES.get(result)
    if keys is not None:
        return MatrixDetail(*(diagram.matrix(key) for key in keys))
    keys = RMSNORM_MATRICES.get(result)
    if keys is not None:
        return RMSNormDetail(*(diagram.matrix(key) for key in keys))
    if result == "k_rope":
        input_key = "k_norm" if diagram.config.qk_norm else "k"
        return RoPEDetail(diagram.matrix(input_key), diagram.matrix(result))
    if result == "q_rope":
        input_key = "q_norm" if diagram.config.qk_norm else "q"
        return RoPEDetail(diagram.matrix(input_key), diagram.matrix(result))
    if result == "a_softmax":
        return CausalSoftmaxDetail(
            diagram.matrix("a"),
            diagram.matrix(result),
        )
    if result == "h_merge":
        return HeadConcatDetail(
            diagram.matrix("h"),
            diagram.matrix(result),
        )
    if result == "p":
        return GatedActivationDetail(
            diagram.matrix("g"),
            diagram.matrix("u"),
            diagram.matrix(result),
        )
    if result == "r":
        return ResidualDetail(
            diagram.matrix("x_residual"),
            diagram.matrix("y"),
            diagram.matrix(result),
        )
    if result == "x_out":
        return ResidualDetail(
            diagram.matrix("r_final"),
            diagram.matrix("m"),
            diagram.matrix(result),
        )
    return None


def _sample_count(dimension: str, limit: int) -> int:
    maximum = max(MINIMUM_SAMPLE_LIMIT, limit)
    if dimension == "seq":
        return maximum
    return min(int(dimension), maximum)


def _horizontal_extent(diagram: Diagram, dimension: str) -> int:
    """Encode column dimensions as visibly narrow or wide matrix boxes."""
    if dimension == "seq":
        return 20
    rank = diagram.height_for(dimension) - 3
    return max(13, min(37, 17 + 4 * rank))


def _vertical_extent(diagram: Diagram, dimension: str) -> int:
    """Encode row dimensions compactly for terminal aspect ratios."""
    if dimension == "seq":
        return 7
    rank = diagram.height_for(dimension) - 3
    return max(5, min(11, 5 + rank))


def _layout(
    diagram: Diagram,
    matrix: Matrix,
    sample_limit: int,
    *,
    minimum_width: int = 0,
    minimum_height: int = 0,
) -> BoxLayout:
    return BoxLayout(
        matrix=matrix,
        width=max(
            minimum_width,
            _horizontal_extent(diagram, matrix.columns),
        ),
        height=max(
            minimum_height,
            _vertical_extent(diagram, matrix.rows),
        ),
        row_count=_sample_count(matrix.rows, sample_limit),
        column_count=_sample_count(matrix.columns, sample_limit),
    )


def _with_collapsed_rows(layout: BoxLayout) -> BoxLayout:
    if layout.matrix.rows == "1" or layout.row_count < 2:
        return layout
    return BoxLayout(
        layout.matrix,
        layout.width,
        max(layout.height, layout.row_count + 4),
        layout.row_count,
        layout.column_count,
    )


def selection_shape(
    diagram: Diagram,
    step: int,
    sample_limit: int,
) -> tuple[int, int]:
    """Return the sampled result navigation bounds."""
    detail = detail_for_step(diagram, step)
    if detail is None:
        raise ValueError("selected operation has no matrix-product detail")
    rope_pairs = (
        max(2, min(4, sample_limit - 2))
        if isinstance(detail, RoPEDetail)
        else 0
    )
    layout = _layout(
        diagram,
        detail.result,
        sample_limit,
        minimum_height=2 * rope_pairs + 6 if rope_pairs else 0,
    )
    if isinstance(detail, RoPEDetail):
        return rope_pairs, layout.column_count
    if isinstance(detail, CausalSoftmaxDetail):
        endpoint_count = _causal_endpoint_count(sample_limit)
        visible_count = 2 * endpoint_count
        return visible_count, visible_count
    if isinstance(detail, HeadConcatDetail):
        return len(_head_concat_sampled_heads(diagram)), layout.column_count
    if isinstance(detail, GatedActivationDetail):
        return layout.row_count, layout.column_count
    if isinstance(detail, ResidualDetail):
        return layout.row_count, layout.column_count
    if isinstance(detail, BlockLoopDetail):
        return 1, 1
    return layout.row_count, layout.column_count


def _scaled_position(index: int, count: int, start: int, end: int) -> int:
    if count <= 1 or end <= start:
        return start
    return start + round(index * (end - start) / (count - 1))


def _cell_marker(matrix: Matrix) -> str:
    return WEIGHT_CELL if matrix.key in LEARNED_WEIGHTS else ACTIVATION_CELL


def _operand_role(matrix: Matrix) -> str:
    return "learned" if matrix.key in LEARNED_WEIGHTS else "input"


def _role_qualifier(role: str | None) -> str:
    return {
        "input": "(in)",
        "learned": "(learned)",
        "output": "(out)",
    }.get(role, "")


def _role_overhang(role: str | None) -> int:
    qualifier = _role_qualifier(role)
    return len(qualifier) - len(qualifier) // 2


def _matrix_name_span(
    box_x: int,
    width: int,
    name: str,
    role: str | None,
) -> tuple[int, int]:
    """Return the half-overhanging label span used above a matrix."""
    qualifier = _role_qualifier(role)
    start = max(
        0,
        box_x + max(0, width - len(name)) - len(qualifier) // 2,
    )
    return start, start + len(name) + len(qualifier)


def _draw_matrix_name(
    canvas: Canvas,
    box_x: int,
    y: int,
    width: int,
    name: str,
    role: str | None,
) -> None:
    """Draw a yellow variable and a gray role around its upper-right corner."""
    qualifier = _role_qualifier(role)
    name_x, _ = _matrix_name_span(box_x, width, name, role)
    put(canvas, name_x, y - 1, name, VARIABLE_COLOR, None, bold=True)
    if qualifier:
        put(
            canvas,
            name_x + len(name),
            y - 1,
            qualifier,
            DIMENSION_COLOR,
            None,
            bold=True,
        )


def _detail_operation_name(result: str) -> str:
    if result in RMSNORM_MATRICES:
        return "RMSNorm"
    if result == "x":
        return "Block recurrence"
    return OPERATION_BY_RESULT[result]


def _detail_context_labels(diagram: Diagram, step: int) -> tuple[str, str]:
    """Name the operations immediately before and after an explainer."""
    previous = (
        OPERATION_BY_RESULT[diagram.operations[step - 1].result]
        if step > 0
        else "Model input"
    )
    following = (
        OPERATION_BY_RESULT[diagram.operations[step + 1].result]
        if step + 1 < len(diagram.operations)
        else "Model output"
    )
    return f"Prev: {previous}", f"Next: {following}"


def _detail_context_width(diagram: Diagram, step: int) -> int:
    return 2 * max(map(len, _detail_context_labels(diagram, step))) + 2


def _draw_detail_context(
    canvas: Canvas,
    diagram: Diagram,
    step: int,
    input_center_x: int,
    output_center_x: int,
    output_bottom_y: int,
) -> None:
    """Draw incoming and outgoing operation-context arrows."""
    previous, following = _detail_context_labels(diagram, step)
    previous_x = max(
        0,
        min(
            input_center_x - len(previous) // 2,
            canvas.width - len(previous),
        ),
    )
    put(
        canvas,
        previous_x,
        HEADING_HEIGHT - 2,
        previous,
        OPERATION_COLOR,
        None,
        bold=True,
    )
    put(
        canvas,
        input_center_x,
        HEADING_HEIGHT - 1,
        "↓",
        OPERATION_COLOR,
        None,
        bold=True,
    )

    following_x = output_center_x - len(following) - 2
    if following_x < 0:
        following_x = output_center_x + 2
    put(
        canvas,
        following_x,
        output_bottom_y,
        following,
        OPERATION_COLOR,
        None,
        bold=True,
    )
    put(
        canvas,
        output_center_x,
        output_bottom_y,
        "↓",
        OPERATION_COLOR,
        None,
        bold=True,
    )


def _draw_detail_heading(canvas: Canvas, result: str) -> None:
    section = SECTION_BY_RESULT[result]
    operation = _detail_operation_name(result)
    put(
        canvas,
        0,
        0,
        section,
        ANNOTATOR_COLOR,
        None,
        bold=True,
    )
    put(
        canvas,
        len(section) + 1,
        0,
        "•",
        DIMENSION_COLOR,
        None,
        bold=True,
    )
    put(
        canvas,
        len(section) + 3,
        0,
        operation,
        OPERATION_COLOR,
        None,
        bold=True,
    )


def _detail_heading_width(result: str) -> int:
    return (
        len(SECTION_BY_RESULT[result])
        + 3
        + len(_detail_operation_name(result))
    )


def _endpoint_blocks(
    canvas: Canvas,
    x: int,
    y: int,
    width: int,
    color: tuple[int, int, int],
    marker: str,
) -> None:
    """Draw small boxes at both row ends with a dotted collapsed middle."""
    for index in range(ENDPOINT_CELL_COUNT):
        put(
            canvas,
            x + 2 + index * ENDPOINT_COLUMN_SPACING,
            y,
            marker,
            color,
            GRID_BACKGROUND,
            bold=True,
        )
        put(
            canvas,
            x + width - 3 - index * ENDPOINT_COLUMN_SPACING,
            y,
            marker,
            color,
            GRID_BACKGROUND,
            bold=True,
        )
    dots = "· ·"
    put(
        canvas,
        x + max(1, (width - len(dots)) // 2),
        y,
        dots,
        GRID_COLOR,
        GRID_BACKGROUND,
    )


def _vertical_endpoint_blocks(
    canvas: Canvas,
    x: int,
    y: int,
    height: int,
    color: tuple[int, int, int],
    marker: str,
) -> None:
    """Draw small boxes at both column ends with a dotted collapsed middle."""
    for index in range(ENDPOINT_CELL_COUNT):
        put(
            canvas,
            x,
            y + 1 + index,
            marker,
            color,
            GRID_BACKGROUND,
            bold=True,
        )
        put(
            canvas,
            x,
            y + height - 2 - index,
            marker,
            color,
            GRID_BACKGROUND,
            bold=True,
        )
    for dot_y in (y + height // 2 - 1, y + height // 2):
        put(
            canvas,
            x,
            dot_y,
            "·",
            GRID_COLOR,
            GRID_BACKGROUND,
        )


def _inactive_axis_positions(
    start: int,
    end: int,
    count: int,
) -> tuple[int, ...]:
    if count <= 1:
        return ((start + end + 1) // 2,)
    return tuple(
        _scaled_position(index, count, start, end)
        for index in range(count)
    )


def _inactive_endpoint_positions(
    near: int,
    far: int,
    step: int,
) -> tuple[int, ...]:
    positions = {
        near + index * step for index in range(ENDPOINT_CELL_COUNT)
    }
    positions.update(
        far - index * step for index in range(ENDPOINT_CELL_COUNT)
    )
    return tuple(sorted(position for position in positions if near <= position <= far))


def _collapsed_row_positions(
    y: int,
    height: int,
    count: int,
) -> tuple[int, ...]:
    upper_count = (count + 1) // 2
    lower_count = count - upper_count
    return (
        *(y + 1 + index for index in range(upper_count)),
        *(
            y + height - 1 - lower_count + index
            for index in range(lower_count)
        ),
    )


def _collapsed_dot_rows(y: int, height: int) -> tuple[int, int]:
    return y + height // 2 - 1, y + height // 2


def _draw_inactive_cells(
    canvas: Canvas,
    box_x: int,
    y: int,
    layout: BoxLayout,
    *,
    endpoint_columns: bool,
    endpoint_rows: bool,
    collapsed_rows: bool,
) -> None:
    if endpoint_columns:
        x_positions = _inactive_endpoint_positions(
            box_x + 2,
            box_x + layout.width - 3,
            ENDPOINT_COLUMN_SPACING,
        )
    else:
        x_positions = _inactive_axis_positions(
            box_x + 2,
            box_x + layout.width - 3,
            layout.column_count,
        )
    if collapsed_rows:
        y_positions = _collapsed_row_positions(
            y,
            layout.height,
            layout.row_count,
        )
    elif endpoint_rows:
        y_positions = _inactive_endpoint_positions(
            y + 1,
            y + layout.height - 2,
            1,
        )
    else:
        y_positions = _inactive_axis_positions(
            y + 1,
            y + layout.height - 2,
            layout.row_count,
        )
    for marker_y in y_positions:
        for marker_x in x_positions:
            put(
                canvas,
                marker_x,
                marker_y,
                INACTIVE_CELL,
                GRID_COLOR,
                GRID_BACKGROUND,
            )
    if collapsed_rows:
        for dot_y in _collapsed_dot_rows(y, layout.height):
            for marker_x in x_positions:
                put(
                    canvas,
                    marker_x,
                    dot_y,
                    "·",
                    GRID_COLOR,
                    GRID_BACKGROUND,
                )


def _draw_box(
    canvas: Canvas,
    origin_x: int,
    y: int,
    layout: BoxLayout,
    *,
    name_side: str | None,
    show_dimensions: bool = True,
    active_row: int | None = None,
    active_column: int | None = None,
    active_cell: tuple[int, int] | None = None,
    active_cell_marker: str = RESULT_CELL,
    active_cell_color: tuple[int, int, int] = RESULT_HIGHLIGHT,
    inactive_cells: bool = False,
    inactive_endpoint_columns: bool = False,
    inactive_endpoint_rows: bool = False,
    collapsed_rows: bool = False,
    io_role: str | None = None,
) -> None:
    box_x = origin_x + layout.box_offset
    canvas.fill_rect(
        box_x,
        y,
        layout.width,
        layout.height,
        GRID_BACKGROUND,
    )
    canvas.fancy_box(
        box_x,
        y,
        layout.width,
        layout.height,
        "heavy",
        GRID_COLOR,
    )

    if show_dimensions:
        columns = layout.matrix.columns
        put(
            canvas,
            box_x + max(1, (layout.width - len(columns)) // 2),
            y,
            columns,
            DIMENSION_COLOR,
            GRID_BACKGROUND,
            bold=True,
        )
        rows = layout.matrix.rows
        put(
            canvas,
            box_x - len(rows) + 1,
            y + (1 if rows == "1" else 2),
            rows,
            DIMENSION_COLOR,
            None,
            bold=True,
        )

    if inactive_cells:
        _draw_inactive_cells(
            canvas,
            box_x,
            y,
            layout,
            endpoint_columns=inactive_endpoint_columns,
            endpoint_rows=inactive_endpoint_rows,
            collapsed_rows=collapsed_rows,
        )

    if active_row is not None:
        marker_y = (
            _collapsed_row_positions(
                y,
                layout.height,
                layout.row_count,
            )[active_row]
            if collapsed_rows
            else _scaled_position(
                active_row,
                layout.row_count,
                y + 1,
                y + layout.height - 2,
            )
        )
        _endpoint_blocks(
            canvas,
            box_x,
            marker_y,
            layout.width,
            ROW_HIGHLIGHT,
            _cell_marker(layout.matrix),
        )
    if active_column is not None:
        marker_x = _scaled_position(
            active_column,
            layout.column_count,
            box_x + 2,
            box_x + layout.width - 3,
        )
        if collapsed_rows:
            for marker_y in _collapsed_row_positions(
                y,
                layout.height,
                layout.row_count,
            ):
                put(
                    canvas,
                    marker_x,
                    marker_y,
                    _cell_marker(layout.matrix),
                    COLUMN_HIGHLIGHT,
                    GRID_BACKGROUND,
                    bold=True,
                )
            for dot_y in _collapsed_dot_rows(y, layout.height):
                put(
                    canvas,
                    marker_x,
                    dot_y,
                    "·",
                    GRID_COLOR,
                    GRID_BACKGROUND,
                )
        else:
            _vertical_endpoint_blocks(
                canvas,
                marker_x,
                y,
                layout.height,
                COLUMN_HIGHLIGHT,
                _cell_marker(layout.matrix),
            )
    if active_cell is not None:
        active_row_index, active_column_index = active_cell
        marker_x = _scaled_position(
            active_column_index,
            layout.column_count,
            box_x + 2,
            box_x + layout.width - 3,
        )
        marker_y = (
            _collapsed_row_positions(
                y,
                layout.height,
                layout.row_count,
            )[active_row_index]
            if collapsed_rows
            else _scaled_position(
                active_row_index,
                layout.row_count,
                y + 1,
                y + layout.height - 2,
            )
        )
        marker_text = (
            f"[{active_cell_marker}]"
            if active_cell_marker == RESULT_CELL
            else active_cell_marker
        )
        put(
            canvas,
            marker_x - (1 if len(marker_text) == 3 else 0),
            marker_y,
            marker_text,
            active_cell_color,
            GRID_BACKGROUND,
            bold=True,
        )
    # Matrix names sit just outside their assigned vertical border without
    # replacing or touching it.
    if name_side is not None:
        _draw_matrix_name(
            canvas,
            box_x,
            y,
            layout.width,
            layout.matrix.label,
            io_role,
        )


def _formula_text(left_marker: str, right_marker: str) -> str:
    term = f"({left_marker}×{right_marker})"
    return f"{RESULT_CELL} = {term} + {term} + ... + {term}"


def _formula_prefix_width(prefixes: tuple[str, ...]) -> int:
    return max((_terminal_width(prefix) for prefix in prefixes), default=0)


def _terminal_width(text: str) -> int:
    """Return the rendered width for labels containing combining marks."""
    return sum(not unicodedata.combining(character) for character in text)


def _formula_body_x(prefixes: tuple[str, ...]) -> int:
    return FORMULA_INDENT + _formula_prefix_width(prefixes) + 2


def _draw_formula_prefix(
    canvas: Canvas,
    y: int,
    prefix: str,
    prefix_width: int,
) -> int:
    """Right-align a colored variable prefix before a shared equation column."""
    prefix_x = FORMULA_INDENT + prefix_width - _terminal_width(prefix)
    qualifier_start = prefix.find("(")
    if qualifier_start < 0:
        put(canvas, prefix_x, y, prefix, VARIABLE_COLOR, None, bold=True)
    else:
        put(
            canvas,
            prefix_x,
            y,
            prefix[:qualifier_start],
            VARIABLE_COLOR,
            None,
            bold=True,
        )
        put(
            canvas,
            prefix_x + qualifier_start,
            y,
            prefix[qualifier_start:],
            DIMENSION_COLOR,
            None,
            bold=True,
        )
    colon_x = prefix_x + len(prefix)
    put(
        canvas,
        colon_x,
        y,
        ":",
        LABEL_COLOR,
        None,
    )
    return colon_x + 2


def _draw_formula(
    canvas: Canvas,
    y: int,
    left_marker: str,
    right_marker: str,
    result_name: str,
) -> None:
    prefixes = (f'{result_name}{_role_qualifier("output")}',)
    prefix_width = _formula_prefix_width(prefixes)
    start_x = _draw_formula_prefix(
        canvas,
        y,
        prefixes[0],
        prefix_width,
    )
    term_tokens = (
        ("(", LABEL_COLOR, False),
        (left_marker, ROW_HIGHLIGHT, True),
        ("×", LABEL_COLOR, False),
        (right_marker, COLUMN_HIGHLIGHT, True),
        (")", LABEL_COLOR, False),
    )
    tokens = (
        (RESULT_CELL, RESULT_HIGHLIGHT, True),
        (" = ", LABEL_COLOR, False),
        *term_tokens,
        (" + ", LABEL_COLOR, False),
        *term_tokens,
        (" + ... + ", LABEL_COLOR, False),
        *term_tokens,
    )
    cursor = start_x
    for text, color, bold in tokens:
        put(canvas, cursor, y, text, color, None, bold=bold)
        cursor += len(text)


def _rms_formula_lines(
    dimension: str,
) -> tuple[tuple[tuple[str, object, bool], ...], ...]:
    """Build the three explicit element-level RMSNorm stages."""
    return (
        (
            (RMS_CELL, RMS_HIGHLIGHT, True),
            (" = √((", LABEL_COLOR, False),
            (ACTIVATION_CELL, COLUMN_HIGHLIGHT, True),
            ("² + ... + ", LABEL_COLOR, False),
            (ACTIVATION_CELL, COLUMN_HIGHLIGHT, True),
            (f"²) / {dimension} + ε)", LABEL_COLOR, False),
        ),
        (
            (ACTIVATION_CELL, NORMALIZED_HIGHLIGHT, True),
            (" = ", LABEL_COLOR, False),
            (ACTIVATION_CELL, COLUMN_HIGHLIGHT, True),
            (" / ", LABEL_COLOR, False),
            (RMS_CELL, RMS_HIGHLIGHT, True),
        ),
        (
            (RESULT_CELL, RESULT_HIGHLIGHT, True),
            (" = ", LABEL_COLOR, False),
            (ACTIVATION_CELL, NORMALIZED_HIGHLIGHT, True),
            (" × ", LABEL_COLOR, False),
            (WEIGHT_CELL, ROW_HIGHLIGHT, True),
        ),
    )


def _normalized_name(name: str) -> str:
    return f"{name[0]}\N{COMBINING CIRCUMFLEX ACCENT}{name[1:]}"


def _rms_formula_prefixes(
    input_name: str,
    result_name: str,
) -> tuple[str, str, str]:
    return (
        "rms",
        _normalized_name(input_name),
        f'{result_name}{_role_qualifier("output")}',
    )


def _rms_formula_width(
    dimension: str,
    input_name: str,
    result_name: str,
) -> int:
    prefixes = _rms_formula_prefixes(input_name, result_name)
    body_width = max(
        sum(len(text) for text, _, _ in line)
        for line in _rms_formula_lines(dimension)
    )
    return _formula_body_x(prefixes) + body_width


def _draw_rms_formula(
    canvas: Canvas,
    y: int,
    dimension: str,
    input_name: str,
    result_name: str,
) -> None:
    prefixes = _rms_formula_prefixes(input_name, result_name)
    prefix_width = _formula_prefix_width(prefixes)
    for line_index, (prefix, tokens) in enumerate(
        zip(prefixes, _rms_formula_lines(dimension))
    ):
        cursor = _draw_formula_prefix(
            canvas,
            y + line_index,
            prefix,
            prefix_width,
        )
        for text, color, bold in tokens:
            put(canvas, cursor, y + line_index, text, color, None, bold=bold)
            cursor += len(text)


def _render_rmsnorm_detail(
    diagram: Diagram,
    step: int,
    detail: RMSNormDetail,
    row: int,
    column: int,
    sample_limit: int,
) -> Canvas:
    """Render RMSNorm with the matrix placement used by the visual draft."""
    result = _with_collapsed_rows(
        _layout(diagram, detail.result, sample_limit)
    )
    input_layout = _with_collapsed_rows(
        _layout(
            diagram,
            detail.input,
            sample_limit,
        )
    )
    rms_matrix = Matrix(
        "rms_intermediate",
        "rms",
        0,
        0,
        detail.input.columns,
        "1",
        detail.input.panel_background,
    )
    rms = BoxLayout(
        matrix=rms_matrix,
        width=input_layout.width,
        height=3,
        row_count=1,
        column_count=result.column_count,
    )
    normalized_matrix = Matrix(
        "normalized_intermediate",
        _normalized_name(detail.input.label),
        0,
        0,
        detail.input.columns,
        detail.input.rows,
        detail.input.panel_background,
    )
    normalized = _with_collapsed_rows(
        _layout(diagram, normalized_matrix, sample_limit)
    )
    # All matrices with the same row dimension share one slightly taller
    # geometry. The extra row exposes one additional endpoint marker.
    vector_height = max(
        input_layout.height,
        normalized.height,
        result.height,
    ) + 1
    input_layout = BoxLayout(
        input_layout.matrix,
        input_layout.width,
        vector_height,
        input_layout.row_count,
        input_layout.column_count,
    )
    normalized = BoxLayout(
        normalized.matrix,
        normalized.width,
        vector_height,
        normalized.row_count,
        normalized.column_count,
    )
    result = BoxLayout(
        result.matrix,
        result.width,
        vector_height,
        result.row_count,
        result.column_count,
    )
    scale = BoxLayout(
        matrix=detail.scale,
        width=7,
        height=vector_height,
        row_count=result.row_count,
        column_count=1,
    )
    selected_row = max(0, min(row, result.row_count - 1))
    selected_column = max(0, min(column, result.column_count - 1))

    gap = 1
    normalized_x = 0
    normalized_box_x = normalized_x + normalized.box_offset
    result_box_x = normalized_box_x + normalized.width + result.box_offset + gap
    result_x = result_box_x - result.box_offset
    # The scale vector is the rightmost operand. Reserve the result name and
    # its own exterior row dimension before positioning the narrow box.
    scale_box_x = (
        result_box_x
        + result.width
        + scale.box_offset
        + gap
    )
    scale_x = scale_box_x - scale.box_offset
    _, result_label_end = _matrix_name_span(
        result_box_x,
        result.width,
        detail.result.label,
        "output",
    )
    scale_label_start, _ = _matrix_name_span(
        scale_box_x,
        scale.width,
        detail.scale.label,
        "learned",
    )
    label_clearance = max(0, result_label_end + 1 - scale_label_start)
    scale_box_x += label_clearance
    scale_x += label_clearance
    input_x = normalized_box_x - input_layout.box_offset
    rms_x = normalized_box_x - rms.box_offset
    rms_box_x = normalized_box_x

    shift = max(
        0,
        -min(rms_x, normalized_x, result_x, input_x, scale_x),
    )
    rms_x += shift
    normalized_x += shift
    result_x += shift
    input_x += shift
    scale_x += shift
    rms_box_x += shift
    normalized_box_x += shift
    result_box_x += shift
    scale_box_x += shift

    group_width = max(
        rms_x + rms.total_width,
        normalized_x + normalized.total_width,
        scale_x + scale.total_width,
        input_x + input_layout.total_width,
        result_x + result.total_width,
        input_layout.width + normalized_box_x + _role_overhang("input"),
        result_box_x + result.width + _role_overhang("output"),
        scale_box_x + scale.width + _role_overhang("learned"),
    )
    formula_width = _rms_formula_width(
        detail.input.rows,
        detail.input.label,
        detail.result.label,
    )
    result_key = diagram.operations[step].result
    canvas_width = max(
        group_width,
        formula_width,
        # _detail_context_width(diagram, step),
        _detail_heading_width(result_key),
    )
    center_shift = max(0, (canvas_width - group_width) // 2)
    rms_x += center_shift
    normalized_x += center_shift
    result_x += center_shift
    input_x += center_shift
    scale_x += center_shift
    rms_box_x += center_shift
    normalized_box_x += center_shift
    result_box_x += center_shift
    scale_box_x += center_shift

    top_y = HEADING_HEIGHT
    rms_y = top_y + input_layout.height + 1
    bottom_y = rms_y + rms.height + 1
    matrices_height = bottom_y + max(scale.height, normalized.height, result.height)
    formula_y = matrices_height + 1
    canvas = Canvas(canvas_width, formula_y + 3)

    _draw_box(
        canvas,
        input_x,
        top_y,
        input_layout,
        name_side="right",
        active_column=selected_column,
        inactive_cells=True,
        collapsed_rows=True,
        io_role="input",
    )
    _draw_box(
        canvas,
        rms_x,
        rms_y,
        rms,
        name_side="left",
        active_cell=(0, selected_column),
        active_cell_marker=RMS_CELL,
        active_cell_color=RMS_HIGHLIGHT,
        inactive_cells=True,
    )
    _draw_box(
        canvas,
        scale_x,
        bottom_y,
        scale,
        name_side="right",
        inactive_cells=True,
        collapsed_rows=True,
        io_role="learned",
    )
    _draw_box(
        canvas,
        normalized_x,
        bottom_y,
        normalized,
        name_side="left",
        active_cell=(selected_row, selected_column),
        active_cell_marker=ACTIVATION_CELL,
        active_cell_color=NORMALIZED_HIGHLIGHT,
        inactive_cells=True,
        collapsed_rows=True,
    )
    _draw_box(
        canvas,
        result_x,
        bottom_y,
        result,
        name_side="right",
        active_cell=(selected_row, selected_column),
        inactive_cells=True,
        collapsed_rows=True,
        io_role="output",
    )

    scale_marker_y = _collapsed_row_positions(
        bottom_y,
        scale.height,
        scale.row_count,
    )[selected_row]
    put(
        canvas,
        scale_box_x + scale.width // 2,
        scale_marker_y,
        WEIGHT_CELL,
        ROW_HIGHLIGHT,
        GRID_BACKGROUND,
        bold=True,
    )
    put(
        canvas,
        rms_box_x + rms.width // 2,
        rms_y - 1,
        "↓",
        COLUMN_HIGHLIGHT,
        None,
        bold=True,
    )
    put(
        canvas,
        rms_box_x + rms.width // 2,
        rms_y + rms.height,
        "↓",
        RMS_HIGHLIGHT,
        None,
        bold=True,
    )
    put(
        canvas,
        normalized_box_x + normalized.width,
        bottom_y + normalized.height // 2,
        "→",
        NORMALIZED_HIGHLIGHT,
        None,
        bold=True,
    )
    put(
        canvas,
        scale_box_x - scale.box_offset - 1,
        bottom_y + scale.height // 2,
        "←",
        ROW_HIGHLIGHT,
        None,
        bold=True,
    )
    # Optional operation context, retained for possible re-enabling:
    # _draw_detail_context(
    #     canvas,
    #     diagram,
    #     step,
    #     normalized_box_x + input_layout.width // 2,
    #     result_box_x + result.width // 2,
    #     bottom_y + result.height,
    # )
    _draw_rms_formula(
        canvas,
        formula_y,
        detail.input.rows,
        detail.input.label,
        detail.result.label,
    )
    _draw_detail_heading(canvas, result_key)
    return canvas


def _rope_formula_lines() -> tuple[tuple[tuple[str, object, bool], ...], ...]:
    return (
        (
            (RESULT_CELL, ROPE_FIRST_HIGHLIGHT, True),
            (" = ", LABEL_COLOR, False),
            (ACTIVATION_CELL, ROPE_FIRST_HIGHLIGHT, True),
            (" × cos(m×θᵢ) − ", LABEL_COLOR, False),
            (ACTIVATION_CELL, ROPE_SECOND_HIGHLIGHT, True),
            (" × sin(m×θᵢ)", LABEL_COLOR, False),
        ),
        (
            (RESULT_CELL, ROPE_SECOND_HIGHLIGHT, True),
            (" = ", LABEL_COLOR, False),
            (ACTIVATION_CELL, ROPE_FIRST_HIGHLIGHT, True),
            (" × sin(m×θᵢ) + ", LABEL_COLOR, False),
            (ACTIVATION_CELL, ROPE_SECOND_HIGHLIGHT, True),
            (" × cos(m×θᵢ)", LABEL_COLOR, False),
        ),
    )


def _draw_rope_formulas(
    canvas: Canvas,
    y: int,
    theta_formula: str,
    result_name: str,
) -> None:
    prefixes = (
        f'{result_name}{_role_qualifier("output")}',
        f'{result_name}{_role_qualifier("output")}',
        f'R{_role_qualifier("input")}',
    )
    prefix_width = _formula_prefix_width(prefixes)
    for line_index, tokens in enumerate(_rope_formula_lines()):
        cursor = _draw_formula_prefix(
            canvas,
            y + line_index,
            prefixes[line_index],
            prefix_width,
        )
        for text, color, bold in tokens:
            put(canvas, cursor, y + line_index, text, color, None, bold=bold)
            cursor += len(text)
    theta_y = y + len(_rope_formula_lines())
    theta_x = _draw_formula_prefix(
        canvas,
        theta_y,
        prefixes[-1],
        prefix_width,
    )
    put(canvas, theta_x, theta_y, theta_formula, LABEL_COLOR, None)


def _format_rope_frequency(value: float) -> str:
    if value >= 0.001:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{value:.2e}"


def _draw_rope_pair(
    canvas: Canvas,
    box_x: int,
    y: int,
    layout: BoxLayout,
    pair: int,
    token: int,
    *,
    output: bool = False,
    active: bool = True,
) -> None:
    column_positions = tuple(
        _scaled_position(
            index,
            layout.column_count,
            box_x + 2,
            box_x + layout.width - 3,
        )
        for index in range(layout.column_count)
    )
    marker_x = _scaled_position(
        token,
        layout.column_count,
        box_x + 2,
        box_x + layout.width - 3,
    )
    pair_count = layout.row_count
    pair_positions = [
        (y + 1 + index, y + pair_count + 3 + index)
        for index in range(pair_count)
    ]
    for dot_y in range(y + pair_count + 1, y + pair_count + 3):
        for column_x in column_positions:
            put(
                canvas,
                column_x,
                dot_y,
                "·",
                GRID_COLOR,
                GRID_BACKGROUND,
            )
    trailing_dots = range(
        y + 2 * pair_count + 3,
        y + 2 * pair_count + 5,
    )
    for dot_y in trailing_dots:
        for column_x in column_positions:
            put(
                canvas,
                column_x,
                dot_y,
                "·",
                GRID_COLOR,
                GRID_BACKGROUND,
            )
    for first_position, second_position in pair_positions:
        for column_x in column_positions:
            put(
                canvas,
                column_x,
                first_position,
                INACTIVE_CELL,
                GRID_COLOR,
                GRID_BACKGROUND,
            )
            put(
                canvas,
                column_x,
                second_position,
                INACTIVE_CELL,
                GRID_COLOR,
                GRID_BACKGROUND,
            )
    if not active:
        return
    first_y, second_y = pair_positions[pair]
    marker = RESULT_CELL if output else ACTIVATION_CELL
    marker_text = f"[{marker}]" if output else marker
    marker_offset = 1 if output else 0
    put(
        canvas,
        marker_x - marker_offset,
        first_y,
        marker_text,
        ROPE_FIRST_HIGHLIGHT,
        GRID_BACKGROUND,
        bold=True,
    )
    put(
        canvas,
        marker_x - marker_offset,
        second_y,
        marker_text,
        ROPE_SECOND_HIGHLIGHT,
        GRID_BACKGROUND,
        bold=True,
    )


def _draw_rope_split_indicator(
    canvas: Canvas,
    index_x: int,
    x: int,
    first_y: int,
    second_y: int,
    first_index: int,
    second_index: int,
    label: str,
) -> None:
    """Join the selected half-indices and label their d/2 separation."""
    for index, line_y in (
        (first_index, first_y),
        (second_index, second_y),
    ):
        connector_start = index_x + len(f"[{index}]")
        for connector_x in range(connector_start, x):
            put(canvas, connector_x, line_y, "─", DIMENSION_COLOR, None)
    put(canvas, x, first_y, "┐", DIMENSION_COLOR, None)
    put(canvas, x, second_y, "┘", DIMENSION_COLOR, None)
    for line_y in range(first_y + 1, second_y):
        put(canvas, x, line_y, "│", DIMENSION_COLOR, None)
    put(
        canvas,
        x,
        (first_y + second_y) // 2,
        label,
        DIMENSION_COLOR,
        None,
    )


def _render_rope_detail(
    diagram: Diagram,
    step: int,
    detail: RoPEDetail,
    row: int,
    column: int,
    sample_limit: int,
    animation_stage: int,
) -> Canvas:
    """Render RoPE as an input, one selected rotation, and an output."""
    visible_pairs = max(2, min(4, sample_limit - 2))
    base_layout = _layout(
        diagram,
        detail.input,
        sample_limit,
        minimum_height=2 * visible_pairs + 6,
    )
    base = BoxLayout(
        base_layout.matrix,
        base_layout.width,
        base_layout.height,
        visible_pairs,
        base_layout.column_count,
    )
    input_layout = base
    result_layout = BoxLayout(
        detail.result,
        base.width,
        base.height,
        base.row_count,
        base.column_count,
    )
    selected_pair = max(0, min(row, base.row_count - 1))
    selected_token = max(0, min(column, base.column_count - 1))

    feature_dimension = int(detail.input.rows)

    def rotation_lines(pair_index: int, token_index: int) -> tuple[str, str]:
        frequency = diagram.config.rope_frequency_base ** (
            -2 * pair_index / feature_dimension
        )
        angle = f"{token_index}×{_format_rope_frequency(frequency)}"
        return (
            f"cos({angle})  −sin({angle})",
            f"sin({angle})   cos({angle})",
        )

    first_rotation, second_rotation = rotation_lines(
        selected_pair,
        selected_token,
    )
    rotation_slot_width = max(
        max(len(first), len(second)) + 2
        for pair_index in range(base.row_count)
        for token_index in range(base.column_count)
        for first, second in (rotation_lines(pair_index, token_index),)
    )
    rotation_box_width = rotation_slot_width
    rotation_box_x = 0
    result_box_x = rotation_slot_width + result_layout.box_offset + 2
    result_x = result_box_x - result_layout.box_offset
    input_box_x = result_box_x
    input_x = input_box_x - input_layout.box_offset
    index_width = max(
        len(f"[{base.row_count - 1}]"),
        len(f"[{feature_dimension // 2 + base.row_count - 1}]"),
    )
    split_label = "d/2"
    split_indicator_width = len(split_label)
    group_width = (
        result_box_x
        + result_layout.width
        + index_width
        + split_indicator_width
    )
    group_width = max(
        group_width,
        input_box_x + input_layout.width + _role_overhang("input"),
        result_box_x + result_layout.width + _role_overhang("output"),
    )
    base_value = diagram.config.rope_frequency_base
    base_text = (
        str(int(base_value))
        if base_value.is_integer()
        else f"{base_value:g}"
    )
    theta_formula = f"θᵢ = {base_text}^(−2i/{feature_dimension})"
    formula_width = max(
        max(
            sum(len(text) for text, _, _ in line)
            for line in _rope_formula_lines()
        ),
        len(theta_formula),
    )
    rope_formula_prefixes = (
        f'{detail.result.label}{_role_qualifier("output")}',
        f'R{_role_qualifier("input")}',
    )
    result_key = diagram.operations[step].result
    canvas_width = max(
        group_width,
        _formula_body_x(rope_formula_prefixes) + formula_width,
        # _detail_context_width(diagram, step),
        _detail_heading_width(result_key),
    )
    center_shift = max(0, (canvas_width - group_width) // 2)
    rotation_box_x += center_shift
    input_x += center_shift
    input_box_x += center_shift
    result_x += center_shift
    result_box_x += center_shift

    top_y = HEADING_HEIGHT
    result_y = top_y + input_layout.height + 1
    formula_y = result_y + result_layout.height + 1
    canvas = Canvas(canvas_width, formula_y + 3)

    _draw_box(
        canvas,
        input_x,
        top_y,
        input_layout,
        name_side="right",
        io_role="input",
    )
    _draw_box(
        canvas,
        result_x,
        result_y,
        result_layout,
        name_side="right",
        io_role="output",
    )
    _draw_rope_pair(
        canvas,
        input_box_x,
        top_y,
        input_layout,
        selected_pair,
        selected_token,
        active=animation_stage >= 1,
    )
    _draw_rope_pair(
        canvas,
        result_box_x,
        result_y,
        result_layout,
        selected_pair,
        selected_token,
        output=True,
        active=animation_stage >= 2,
    )

    input_first_y = top_y + 1 + selected_pair
    input_second_y = top_y + visible_pairs + 3 + selected_pair
    result_first_y = result_y + 1 + selected_pair
    result_second_y = result_y + visible_pairs + 3 + selected_pair
    rotation_box_y = result_first_y - 1
    rotation_box_height = result_second_y - result_first_y + 3
    canvas.fill_rect(
        rotation_box_x,
        rotation_box_y,
        rotation_box_width,
        rotation_box_height,
        GRID_BACKGROUND,
    )
    canvas.fancy_box(
        rotation_box_x,
        rotation_box_y,
        rotation_box_width,
        rotation_box_height,
        "heavy",
        GRID_COLOR,
    )
    _draw_matrix_name(
        canvas,
        rotation_box_x - 2,
        rotation_box_y,
        rotation_box_width,
        "R",
        "input",
    )
    first_rotation_x = (
        rotation_box_x + rotation_box_width - len(first_rotation) - 1
    )
    second_rotation_x = (
        rotation_box_x + rotation_box_width - len(second_rotation) - 1
    )
    put(
        canvas,
        first_rotation_x,
        result_first_y,
        first_rotation,
        ROPE_FIRST_HIGHLIGHT,
        GRID_BACKGROUND,
    )
    put(
        canvas,
        second_rotation_x,
        result_second_y,
        second_rotation,
        ROPE_SECOND_HIGHLIGHT,
        GRID_BACKGROUND,
    )
    put(
        canvas,
        rotation_box_x + rotation_box_width,
        result_first_y,
        "→",
        ROPE_FIRST_HIGHLIGHT,
        None,
        bold=True,
    )
    put(
        canvas,
        rotation_box_x + rotation_box_width,
        result_second_y,
        "→",
        ROPE_SECOND_HIGHLIGHT,
        None,
        bold=True,
    )
    put(
        canvas,
        rotation_box_x + rotation_box_width // 2,
        (result_first_y + result_second_y) // 2,
        "⋮",
        GRID_COLOR,
        GRID_BACKGROUND,
    )
    put(
        canvas,
        input_box_x + input_layout.width // 2,
        top_y + base.height,
        "↓",
        ROPE_FIRST_HIGHLIGHT,
        None,
        bold=True,
    )
    first_index = selected_pair
    second_index = selected_pair + feature_dimension // 2
    index_x = input_box_x + input_layout.width
    if animation_stage >= 1:
        put(
            canvas,
            index_x,
            input_first_y,
            f"[{first_index}]",
            ROPE_FIRST_HIGHLIGHT,
            None,
        )
        put(
            canvas,
            index_x,
            input_second_y,
            f"[{second_index}]",
            ROPE_SECOND_HIGHLIGHT,
            None,
        )
        _draw_rope_split_indicator(
            canvas,
            index_x,
            index_x + index_width,
            input_first_y,
            input_second_y,
            first_index,
            second_index,
            split_label,
        )
    if animation_stage >= 2:
        put(
            canvas,
            index_x,
            result_first_y,
            f"[{first_index}]",
            ROPE_FIRST_HIGHLIGHT,
            None,
        )
        put(
            canvas,
            index_x,
            result_second_y,
            f"[{second_index}]",
            ROPE_SECOND_HIGHLIGHT,
            None,
        )
        _draw_rope_split_indicator(
            canvas,
            index_x,
            index_x + index_width,
            result_first_y,
            result_second_y,
            first_index,
            second_index,
            split_label,
        )
    # Optional operation context, retained for possible re-enabling:
    # _draw_detail_context(
    #     canvas,
    #     diagram,
    #     step,
    #     input_box_x + input_layout.width // 2,
    #     result_box_x + result_layout.width // 2,
    #     result_y + result_layout.height,
    # )
    _draw_rope_formulas(
        canvas,
        formula_y,
        theta_formula,
        detail.result.label,
    )
    _draw_detail_heading(canvas, result_key)
    return canvas


def _causal_slot(index: int, endpoint_count: int) -> int:
    """Insert one visual omission slot between sampled sequence endpoints."""
    return index if index < endpoint_count else index + 1


def _causal_endpoint_count(sample_limit: int) -> int:
    """Show three sequence endpoints on each side at full detail."""
    return max(1, min(3, sample_limit // 2))


def _draw_causal_grid(
    canvas: Canvas,
    box_x: int,
    y: int,
    width: int,
    endpoint_count: int,
    selected_key: int,
    selected_query: int,
    *,
    name: str,
    stage: str,
    io_role: str | None = None,
) -> None:
    slot_count = 2 * endpoint_count + 1
    height = slot_count + 2
    canvas.fill_rect(box_x, y, width, height, GRID_BACKGROUND)
    canvas.fancy_box(box_x, y, width, height, "heavy", GRID_COLOR)
    put(
        canvas,
        box_x + max(1, (width - len("seq")) // 2),
        y,
        "seq",
        DIMENSION_COLOR,
        GRID_BACKGROUND,
        bold=True,
    )
    put(
        canvas,
        box_x - 2,
        y + 2,
        "seq",
        DIMENSION_COLOR,
        None,
        bold=True,
    )
    _draw_matrix_name(canvas, box_x, y, width, name, io_role)

    selected_key_slot = _causal_slot(selected_key, endpoint_count)
    selected_query_slot = _causal_slot(selected_query, endpoint_count)
    for row_slot in range(slot_count):
        for column_slot in range(slot_count):
            marker_x = box_x + 2 + CAUSAL_CELL_SPACING * column_slot
            marker_y = y + 1 + row_slot
            if row_slot == endpoint_count or column_slot == endpoint_count:
                put(
                    canvas,
                    marker_x,
                    marker_y,
                    "·",
                    GRID_COLOR,
                    GRID_BACKGROUND,
                )
                continue

            key_index = (
                row_slot if row_slot < endpoint_count else row_slot - 1
            )
            query_index = (
                column_slot
                if column_slot < endpoint_count
                else column_slot - 1
            )
            is_selected = (
                row_slot == selected_key_slot
                and column_slot == selected_query_slot
            )
            is_visible = key_index <= query_index
            if stage == "scores":
                column_is_selected = column_slot == selected_query_slot
                marker = ACTIVATION_CELL if column_is_selected else INACTIVE_CELL
                color = COLUMN_HIGHLIGHT if column_is_selected else GRID_COLOR
            elif stage == "masked":
                column_is_selected = column_slot == selected_query_slot
                marker = (
                    ACTIVATION_CELL
                    if is_visible and column_is_selected
                    else INACTIVE_CELL if is_visible else MASKED_CELL
                )
                color = (
                    ROW_HIGHLIGHT
                    if column_is_selected or not is_visible
                    else GRID_COLOR
                )
            else:
                marker = INACTIVE_CELL if is_visible else "×"
                color = GRID_COLOR if is_visible else RESULT_HIGHLIGHT
                if is_selected:
                    marker = RESULT_CELL if is_visible else "×"
                    color = RESULT_HIGHLIGHT
            if is_selected:
                put(
                    canvas,
                    marker_x - 1,
                    marker_y,
                    f"[{marker}]",
                    color,
                    GRID_BACKGROUND,
                    bold=True,
                )
            else:
                put(
                    canvas,
                    marker_x,
                    marker_y,
                    marker,
                    color,
                    GRID_BACKGROUND,
                    bold=is_selected,
                )


def _draw_causal_vector(
    canvas: Canvas,
    box_x: int,
    y: int,
    width: int,
    endpoint_count: int,
    selected_query: int,
) -> None:
    """Draw one softmax partition sum p for each sampled query column."""
    slot_count = 2 * endpoint_count + 1
    canvas.fill_rect(box_x, y, width, 3, GRID_BACKGROUND)
    canvas.fancy_box(box_x, y, width, 3, "heavy", GRID_COLOR)
    put(
        canvas,
        box_x + width - len("p"),
        y - 1,
        "p",
        VARIABLE_COLOR,
        None,
        bold=True,
    )
    selected_query_slot = _causal_slot(selected_query, endpoint_count)
    for slot in range(slot_count):
        marker_x = box_x + 2 + CAUSAL_CELL_SPACING * slot
        if slot == endpoint_count:
            put(canvas, marker_x, y + 1, "·", GRID_COLOR, GRID_BACKGROUND)
            continue
        is_selected = slot == selected_query_slot
        marker = ACTIVATION_CELL if is_selected else INACTIVE_CELL
        put(
            canvas,
            marker_x,
            y + 1,
            marker,
            NORMALIZED_HIGHLIGHT if is_selected else GRID_COLOR,
            GRID_BACKGROUND,
            bold=is_selected,
        )


def _causal_formula_lines(
    head_dimension: int,
) -> tuple[tuple[tuple[str, object, bool], ...], ...]:
    """Build the colored symbolic scale, mask, sum, and softmax equations."""
    return (
        (
            (ACTIVATION_CELL, ROW_HIGHLIGHT, True),
            (" = ", LABEL_COLOR, False),
            (ACTIVATION_CELL, COLUMN_HIGHLIGHT, True),
            (f" / √{head_dimension}", LABEL_COLOR, False),
        ),
        (
            (MASKED_CELL, ROW_HIGHLIGHT, True),
            (" = −∞", LABEL_COLOR, False),
        ),
        (
            (ACTIVATION_CELL, NORMALIZED_HIGHLIGHT, True),
            (" = e(", LABEL_COLOR, False),
            (ACTIVATION_CELL, ROW_HIGHLIGHT, True),
            (") + e(", LABEL_COLOR, False),
            (ACTIVATION_CELL, ROW_HIGHLIGHT, True),
            (") + ... + e(", LABEL_COLOR, False),
            (ACTIVATION_CELL, ROW_HIGHLIGHT, True),
            (")", LABEL_COLOR, False),
        ),
        (
            (RESULT_CELL, RESULT_HIGHLIGHT, True),
            (" = e(", LABEL_COLOR, False),
            (ACTIVATION_CELL, ROW_HIGHLIGHT, True),
            (") / ", LABEL_COLOR, False),
            (ACTIVATION_CELL, NORMALIZED_HIGHLIGHT, True),
        ),
        (
            ("×", RESULT_HIGHLIGHT, True),
            (" = 0", LABEL_COLOR, False),
        ),
    )


def _draw_causal_formulas(
    canvas: Canvas,
    y: int,
    head_dimension: int,
    masked_name: str,
    probability_name: str,
    result_name: str,
) -> None:
    prefixes = (
        masked_name,
        masked_name,
        probability_name,
        f'{result_name}{_role_qualifier("output")}',
        f'{result_name}{_role_qualifier("output")}',
    )
    prefix_width = _formula_prefix_width(prefixes)
    for line_index, (prefix, tokens) in enumerate(
        zip(prefixes, _causal_formula_lines(head_dimension))
    ):
        cursor = _draw_formula_prefix(
            canvas,
            y + line_index,
            prefix,
            prefix_width,
        )
        for text, color, bold in tokens:
            put(canvas, cursor, y + line_index, text, color, None, bold=bold)
            cursor += len(text)


def _render_causal_softmax_detail(
    diagram: Diagram,
    step: int,
    detail: CausalSoftmaxDetail,
    row: int,
    column: int,
    sample_limit: int,
) -> Canvas:
    """Render scaling, causal masking, and column-wise normalization."""
    endpoint_count = _causal_endpoint_count(sample_limit)
    visible_count = 2 * endpoint_count
    selected_key = max(0, min(row, visible_count - 1))
    selected_query = max(0, min(column, visible_count - 1))
    slot_count = visible_count + 1
    grid_width = CAUSAL_CELL_SPACING * (slot_count - 1) + 5
    grid_height = slot_count + 2
    left_box_x = 3
    masked_box_x = left_box_x + grid_width + 2
    vector_height = 3
    vector_x = masked_box_x
    vector_y = HEADING_HEIGHT + grid_height + 1
    result_y = vector_y + vector_height + 1

    head_dimension = diagram.config.key_head_dim
    formula_lines = _causal_formula_lines(head_dimension)
    formula_prefixes = (
        "L",
        "L",
        "p",
        f'{detail.result.label}{_role_qualifier("output")}',
        f'{detail.result.label}{_role_qualifier("output")}',
    )
    top_width = max(
        left_box_x + grid_width + _role_overhang("input"),
        masked_box_x + grid_width + _role_overhang("output"),
    )
    canvas_width = max(
        top_width,
        _detail_heading_width(diagram.operations[step].result),
        # _detail_context_width(diagram, step),
        *(
            _formula_body_x(formula_prefixes)
            + sum(len(text) for text, _, _ in line)
            for line in formula_lines
        ),
    )
    y = HEADING_HEIGHT
    formula_y = result_y + grid_height + 1
    canvas = Canvas(canvas_width, formula_y + len(formula_lines))

    _draw_causal_grid(
        canvas,
        left_box_x,
        y,
        grid_width,
        endpoint_count,
        selected_key,
        selected_query,
        name=detail.scores.label,
        stage="scores",
        io_role="input",
    )
    _draw_causal_grid(
        canvas,
        masked_box_x,
        y,
        grid_width,
        endpoint_count,
        selected_key,
        selected_query,
        name="L",
        stage="masked",
    )
    arrow_y = y + grid_height // 2
    put(
        canvas,
        left_box_x + grid_width,
        arrow_y,
        "→",
        OPERATION_COLOR,
        None,
        bold=True,
    )
    put(
        canvas,
        masked_box_x + grid_width // 2,
        y + grid_height,
        "↓",
        OPERATION_COLOR,
        None,
        bold=True,
    )
    _draw_causal_vector(
        canvas,
        vector_x,
        vector_y,
        grid_width,
        endpoint_count,
        selected_query,
    )
    put(
        canvas,
        vector_x + grid_width // 2,
        vector_y + vector_height,
        "↓",
        OPERATION_COLOR,
        None,
        bold=True,
    )
    _draw_causal_grid(
        canvas,
        masked_box_x,
        result_y,
        grid_width,
        endpoint_count,
        selected_key,
        selected_query,
        name=detail.result.label,
        stage="result",
        io_role="output",
    )
    # Optional operation context, retained for possible re-enabling:
    # _draw_detail_context(
    #     canvas,
    #     diagram,
    #     step,
    #     left_box_x + grid_width // 2,
    #     masked_box_x + grid_width // 2,
    #     result_y + grid_height,
    # )
    _draw_causal_formulas(
        canvas,
        formula_y,
        head_dimension,
        "L",
        "p",
        detail.result.label,
    )
    _draw_detail_heading(canvas, diagram.operations[step].result)
    return canvas


def _draw_elementwise_matrix(
    canvas: Canvas,
    box_x: int,
    y: int,
    width: int,
    endpoint_count: int,
    selected_row: int,
    selected_column: int,
    rows: str,
    columns: str,
    name: str,
    role: str | None,
    marker: str,
    color: tuple[int, int, int],
    *,
    bracketed: bool = False,
    show_active_cell: bool = True,
    border_color: object = GRID_COLOR,
    show_dimensions: bool = True,
) -> None:
    """Draw one sampled matrix in the elementwise gated-activation flow."""
    slot_count = 2 * endpoint_count + 1
    height = slot_count + 2
    canvas.fill_rect(box_x, y, width, height, GRID_BACKGROUND)
    canvas.fancy_box(box_x, y, width, height, "heavy", border_color)
    dimension_color = DIMENSION_COLOR
    if show_dimensions:
        put(
            canvas,
            box_x + max(1, (width - len(columns)) // 2),
            y,
            columns,
            dimension_color,
            GRID_BACKGROUND,
            bold=True,
        )
        put(
            canvas,
            box_x - len(rows) + 1,
            y + (1 if rows == "1" else 2),
            rows,
            dimension_color,
            None,
            bold=True,
        )
    if name:
        _draw_matrix_name(canvas, box_x, y, width, name, role)

    selected_row_slot = _causal_slot(selected_row, endpoint_count)
    selected_column_slot = _causal_slot(selected_column, endpoint_count)
    for row_slot in range(slot_count):
        for column_slot in range(slot_count):
            marker_x = box_x + 2 + CAUSAL_CELL_SPACING * column_slot
            marker_y = y + 1 + row_slot
            if row_slot == endpoint_count or column_slot == endpoint_count:
                put(
                    canvas,
                    marker_x,
                    marker_y,
                    "·",
                    GRID_COLOR,
                    GRID_BACKGROUND,
                )
                continue
            if show_active_cell and (
                row_slot == selected_row_slot
                and column_slot == selected_column_slot
            ):
                active_text = f"[{marker}]" if bracketed else marker
                put(
                    canvas,
                    marker_x - (1 if bracketed else 0),
                    marker_y,
                    active_text,
                    color,
                    GRID_BACKGROUND,
                    bold=True,
                )
            else:
                put(
                    canvas,
                    marker_x,
                    marker_y,
                    INACTIVE_CELL,
                    GRID_COLOR,
                    GRID_BACKGROUND,
                )


def _gated_formula_lines(
    activation: str,
) -> tuple[tuple[tuple[str, object, bool], ...], ...]:
    """Build activation and pointwise-product equations for one element."""
    if activation == "SiLU":
        return (
            (
                (ACTIVATION_CELL, NORMALIZED_HIGHLIGHT, True),
                (" = ", LABEL_COLOR, False),
                (ACTIVATION_CELL, COLUMN_HIGHLIGHT, True),
                (" / (1 + e(−", LABEL_COLOR, False),
                (ACTIVATION_CELL, COLUMN_HIGHLIGHT, True),
                ("))", LABEL_COLOR, False),
            ),
            (
                (RESULT_CELL, RESULT_HIGHLIGHT, True),
                (" = ", LABEL_COLOR, False),
                (ACTIVATION_CELL, NORMALIZED_HIGHLIGHT, True),
                (" ⊙ ", LABEL_COLOR, False),
                (ACTIVATION_CELL, ROW_HIGHLIGHT, True),
            ),
        )
    return (
        (
            (ACTIVATION_CELL, NORMALIZED_HIGHLIGHT, True),
            (" = ½", LABEL_COLOR, False),
            (ACTIVATION_CELL, COLUMN_HIGHLIGHT, True),
            (" × (1 + erf(", LABEL_COLOR, False),
            (ACTIVATION_CELL, COLUMN_HIGHLIGHT, True),
            (" / √2))", LABEL_COLOR, False),
        ),
        (
            (RESULT_CELL, RESULT_HIGHLIGHT, True),
            (" = ", LABEL_COLOR, False),
            (ACTIVATION_CELL, NORMALIZED_HIGHLIGHT, True),
            (" ⊙ ", LABEL_COLOR, False),
            (ACTIVATION_CELL, ROW_HIGHLIGHT, True),
        ),
    )


def _gated_formula_prefixes(
    activation: str,
    result_name: str,
) -> tuple[str, ...]:
    if activation == "SiLU":
        return (
            "SiLU(G)",
            f'{result_name}{_role_qualifier("output")}',
        )
    return (
        "GELU(G)",
        f'{result_name}{_role_qualifier("output")}',
    )


def _draw_gated_formulas(
    canvas: Canvas,
    y: int,
    activation: str,
    result_name: str,
) -> None:
    prefixes = _gated_formula_prefixes(activation, result_name)
    prefix_width = _formula_prefix_width(prefixes)
    for line_index, (prefix, tokens) in enumerate(
        zip(prefixes, _gated_formula_lines(activation))
    ):
        cursor = _draw_formula_prefix(
            canvas,
            y + line_index,
            prefix,
            prefix_width,
        )
        for text, color, bold in tokens:
            put(canvas, cursor, y + line_index, text, color, None, bold=bold)
            cursor += len(text)


def _render_gated_activation_detail(
    diagram: Diagram,
    step: int,
    detail: GatedActivationDetail,
    row: int,
    column: int,
    sample_limit: int,
) -> Canvas:
    """Render activation followed by an elementwise gate/value product."""
    endpoint_count = _causal_endpoint_count(sample_limit)
    visible_count = 2 * endpoint_count
    selected_row = max(0, min(row, visible_count - 1))
    selected_column = max(0, min(column, visible_count - 1))
    slot_count = 2 * endpoint_count + 1
    box_width = CAUSAL_CELL_SPACING * (slot_count - 1) + 5
    box_height = slot_count + 2
    left_x = len(detail.result.rows) + 1
    horizontal_gap = 5
    right_x = left_x + box_width + horizontal_gap
    top_y = HEADING_HEIGHT + 1
    vertical_gap = 3
    bottom_y = top_y + box_height + vertical_gap
    formula_y = bottom_y + box_height + 2
    activation = diagram.config.activation
    formula_prefixes = _gated_formula_prefixes(
        activation,
        detail.result.label,
    )
    formula_lines = _gated_formula_lines(activation)
    formula_width = _formula_body_x(formula_prefixes) + max(
        sum(len(text) for text, _, _ in line)
        for line in formula_lines
    )
    canvas_width = max(
        right_x + box_width + _role_overhang("output"),
        formula_width,
        _detail_heading_width(diagram.operations[step].result),
    )
    canvas = Canvas(canvas_width, formula_y + len(formula_lines))

    _draw_elementwise_matrix(
        canvas,
        left_x,
        top_y,
        box_width,
        endpoint_count,
        selected_row,
        selected_column,
        detail.gate.rows,
        detail.gate.columns,
        detail.gate.label,
        "input",
        ACTIVATION_CELL,
        COLUMN_HIGHLIGHT,
    )
    _draw_elementwise_matrix(
        canvas,
        right_x,
        top_y,
        box_width,
        endpoint_count,
        selected_row,
        selected_column,
        detail.gate.rows,
        detail.gate.columns,
        f"{activation}(G)",
        None,
        ACTIVATION_CELL,
        NORMALIZED_HIGHLIGHT,
    )
    _draw_elementwise_matrix(
        canvas,
        left_x,
        bottom_y,
        box_width,
        endpoint_count,
        selected_row,
        selected_column,
        detail.up.rows,
        detail.up.columns,
        detail.up.label,
        "input",
        ACTIVATION_CELL,
        ROW_HIGHLIGHT,
    )
    _draw_elementwise_matrix(
        canvas,
        right_x,
        bottom_y,
        box_width,
        endpoint_count,
        selected_row,
        selected_column,
        detail.result.rows,
        detail.result.columns,
        detail.result.label,
        "output",
        RESULT_CELL,
        RESULT_HIGHLIGHT,
        bracketed=True,
    )

    arrow_x = left_x + box_width + horizontal_gap // 2
    put(canvas, arrow_x, top_y + box_height // 2, "→", OPERATION_COLOR, None, bold=True)
    put(canvas, arrow_x, bottom_y + box_height // 2, "→", OPERATION_COLOR, None, bold=True)
    vertical_x = right_x + box_width // 2
    put(
        canvas,
        vertical_x,
        top_y + box_height + 1,
        "↓",
        OPERATION_COLOR,
        None,
        bold=True,
    )

    _draw_gated_formulas(
        canvas,
        formula_y,
        activation,
        detail.result.label,
    )
    _draw_detail_heading(canvas, diagram.operations[step].result)
    return canvas


def _draw_residual_formula(
    canvas: Canvas,
    y: int,
    result_name: str,
) -> None:
    prefix = f'{result_name}{_role_qualifier("output")}'
    cursor = _draw_formula_prefix(
        canvas,
        y,
        prefix,
        _formula_prefix_width((prefix,)),
    )
    for text, color, bold in (
        (RESULT_CELL, RESULT_HIGHLIGHT, True),
        (" = ", LABEL_COLOR, False),
        (ACTIVATION_CELL, COLUMN_HIGHLIGHT, True),
        (" + ", LABEL_COLOR, False),
        (ACTIVATION_CELL, ROW_HIGHLIGHT, True),
    ):
        put(canvas, cursor, y, text, color, None, bold=bold)
        cursor += len(text)


def _render_residual_detail(
    diagram: Diagram,
    step: int,
    detail: ResidualDetail,
    row: int,
    column: int,
    sample_limit: int,
) -> Canvas:
    """Render two same-shaped inputs joining in an elementwise residual sum."""
    endpoint_count = _causal_endpoint_count(sample_limit)
    visible_count = 2 * endpoint_count
    selected_row = max(0, min(row, visible_count - 1))
    selected_column = max(0, min(column, visible_count - 1))
    slot_count = 2 * endpoint_count + 1
    box_width = CAUSAL_CELL_SPACING * (slot_count - 1) + 5
    box_height = slot_count + 2
    left_x = len(detail.result.rows) + 1
    horizontal_gap = 5
    right_x = left_x + box_width + horizontal_gap
    top_y = HEADING_HEIGHT + 1
    vertical_gap = 3
    bottom_y = top_y + box_height + vertical_gap
    formula_y = bottom_y + box_height + 2
    formula_prefix = f'{detail.result.label}{_role_qualifier("output")}'
    formula_width = (
        _formula_body_x((formula_prefix,))
        + len(f"{RESULT_CELL} = {ACTIVATION_CELL} + {ACTIVATION_CELL}")
    )
    canvas_width = max(
        right_x + box_width + _role_overhang("output"),
        formula_width,
        _detail_heading_width(diagram.operations[step].result),
    )
    canvas = Canvas(canvas_width, formula_y + 1)

    _draw_elementwise_matrix(
        canvas,
        right_x,
        top_y,
        box_width,
        endpoint_count,
        selected_row,
        selected_column,
        detail.branch.rows,
        detail.branch.columns,
        detail.branch.label,
        "input",
        ACTIVATION_CELL,
        ROW_HIGHLIGHT,
    )
    _draw_elementwise_matrix(
        canvas,
        left_x,
        bottom_y,
        box_width,
        endpoint_count,
        selected_row,
        selected_column,
        detail.residual.rows,
        detail.residual.columns,
        detail.residual.label,
        "input",
        ACTIVATION_CELL,
        COLUMN_HIGHLIGHT,
    )
    _draw_elementwise_matrix(
        canvas,
        right_x,
        bottom_y,
        box_width,
        endpoint_count,
        selected_row,
        selected_column,
        detail.result.rows,
        detail.result.columns,
        detail.result.label,
        "output",
        RESULT_CELL,
        RESULT_HIGHLIGHT,
        bracketed=True,
    )

    put(
        canvas,
        left_x + box_width + horizontal_gap // 2,
        bottom_y + box_height // 2,
        "→",
        OPERATION_COLOR,
        None,
        bold=True,
    )
    put(
        canvas,
        right_x + box_width // 2,
        top_y + box_height + 1,
        "↓",
        OPERATION_COLOR,
        None,
        bold=True,
    )
    _draw_residual_formula(
        canvas,
        formula_y,
        detail.result.label,
    )
    _draw_detail_heading(canvas, diagram.operations[step].result)
    return canvas


def _render_block_loop_detail(
    diagram: Diagram,
    step: int,
    detail: BlockLoopDetail,
    row: int,
    sample_limit: int,
) -> Canvas:
    """Render the two static sources of a generic block's input state."""
    endpoint_count = _causal_endpoint_count(sample_limit)
    slot_count = 2 * endpoint_count + 1
    box_width = CAUSAL_CELL_SPACING * (slot_count - 1) + 5
    box_height = slot_count + 2
    box_x = 19
    top_y = HEADING_HEIGHT + 1
    center_y = top_y + box_height // 2
    left_arrow_x = box_x - 2
    left_label_x = left_arrow_x - len("X(in)") - 1
    right_arrow_x = box_x + box_width + 1
    right_label_x = right_arrow_x + 2
    canvas_width = max(
        right_label_x + len("from Block b−1"),
        _detail_heading_width(diagram.operations[step].result),
    )
    canvas = Canvas(canvas_width, top_y + box_height)

    _draw_elementwise_matrix(
        canvas,
        box_x,
        top_y,
        box_width,
        endpoint_count,
        0,
        0,
        detail.result.rows,
        detail.result.columns,
        "",
        None,
        ACTIVATION_CELL,
        OPERATION_COLOR,
        show_active_cell=False,
        border_color=OPERATION_COLOR,
        show_dimensions=True,
    )
    block_title = "Block b"
    put(
        canvas,
        box_x + (box_width - len(block_title)) // 2,
        top_y - 1,
        block_title,
        VARIABLE_COLOR,
        None,
        bold=True,
    )

    put(canvas, left_label_x, center_y, "X", VARIABLE_COLOR, None, bold=True)
    put(
        canvas,
        left_label_x + 1,
        center_y,
        "(in)",
        DIMENSION_COLOR,
        None,
        bold=True,
    )
    put(canvas, left_label_x, center_y + 2, "if b=0", DIMENSION_COLOR, None)
    put(
        canvas,
        left_arrow_x,
        center_y,
        "→",
        COLUMN_HIGHLIGHT,
        None,
        bold=True,
    )

    put(canvas, right_label_x, center_y, "Xᵦ", VARIABLE_COLOR, None, bold=True)
    put(
        canvas,
        right_label_x + 2,
        center_y,
        "(in)",
        DIMENSION_COLOR,
        None,
        bold=True,
    )
    put(
        canvas,
        right_label_x,
        center_y + 2,
        "from Block b−1",
        DIMENSION_COLOR,
        None,
    )
    put(
        canvas,
        right_label_x,
        center_y + 3,
        "if b>0",
        DIMENSION_COLOR,
        None,
    )
    put(
        canvas,
        right_arrow_x,
        center_y,
        "←",
        OPERATION_COLOR,
        None,
        bold=True,
    )
    _draw_detail_heading(canvas, diagram.operations[step].result)
    return canvas


def _subscript(value: int) -> str:
    return str(value).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))


def _head_concat_sampled_heads(diagram: Diagram) -> tuple[int, ...]:
    """Return heads from only KV 0, the fixed midpoint, and the final KV."""
    config = diagram.config
    queries_per_kv = config.queries_per_kv
    last_group = config.kv_heads - 1
    middle_group = last_group // 2

    def pair_for(group: int, preferred: int) -> tuple[int, ...]:
        if queries_per_kv == 1:
            return (group * queries_per_kv,)
        first_local = max(0, min(preferred, queries_per_kv - 2))
        return tuple(
            group * queries_per_kv + local
            for local in (first_local, first_local + 1)
        )

    heads: set[int] = set(pair_for(0, 0))
    heads.update(pair_for(middle_group, 0))

    # Finish with the actual last two query heads in the final KV group.
    last_default = max(0, queries_per_kv - 2)
    heads.update(pair_for(last_group, last_default))
    return tuple(sorted(heads))


def _head_concat_visible_heads(
    diagram: Diagram,
) -> tuple[int | None, ...]:
    """Insert omissions between the three fixed sampled KV regions."""
    ordered_heads = _head_concat_sampled_heads(diagram)

    visible: list[int | None] = []
    for head in ordered_heads:
        if visible and isinstance(visible[-1], int) and head > visible[-1] + 1:
            visible.append(None)
        visible.append(head)
    return tuple(visible)


def _head_concat_labels(
    diagram: Diagram,
    head: int,
) -> tuple[str, str, str]:
    config = diagram.config
    group, local = divmod(head, config.queries_per_kv)
    if config.attention_kind == "GQA":
        suffix = f"{_subscript(group)},{_subscript(local)}"
        return (
            f"KV{_subscript(group)}",
            f"Q{_subscript(local)}",
            f"H{suffix}",
        )
    if config.attention_kind == "MQA":
        suffix = _subscript(local)
        return "KV₀", f"Q{suffix}", f"H{suffix}"
    suffix = _subscript(group)
    return f"KV{suffix}", f"Q{suffix}", f"H{suffix}"


def _draw_head_concat_separator(
    canvas: Canvas,
    x: int,
    y: int,
    width: int,
) -> None:
    canvas.line(x + 1, y, x + width - 2, y, "━", GRID_COLOR)
    canvas.point(x, y, "┣", GRID_COLOR)
    canvas.point(x + width - 1, y, "┫", GRID_COLOR)


def _render_head_concat_detail(
    diagram: Diagram,
    step: int,
    detail: HeadConcatDetail,
    row: int,
    column: int,
    sample_limit: int,
) -> Canvas:
    """Render connector trees feeding three-row concatenated head boxes."""
    sampled_heads = _head_concat_sampled_heads(diagram)
    selected_index = max(0, min(row, len(sampled_heads) - 1))
    selected_head = sampled_heads[selected_index]
    column_count = _sample_count(detail.result.columns, sample_limit)
    selected_column = max(0, min(column, column_count - 1))
    visible = _head_concat_visible_heads(diagram)
    head_positions: dict[int, tuple[int, int]] = {}
    omission_positions: list[int] = []
    top_y = HEADING_HEIGHT
    cursor_y = top_y
    for item in visible:
        if item is None:
            omission_positions.append(cursor_y + 1)
            cursor_y += 2
            continue
        head_positions[item] = (cursor_y, cursor_y + 2)
        cursor_y += 4
    box_height = cursor_y - top_y + 1
    bottom_y = top_y + box_height - 1
    separators = sorted(
        {
            boundary
            for box_y, _ in head_positions.values()
            for boundary in (box_y, box_y + 4)
            if top_y < boundary < bottom_y
        }
    )

    visible_heads = tuple(head_positions)
    labels_by_head = {
        head: _head_concat_labels(diagram, head) for head in visible_heads
    }
    kv_width = max(len(labels[0]) for labels in labels_by_head.values())
    query_width = max(len(labels[1]) for labels in labels_by_head.values())
    head_width = max(
        len(labels[2]) + len(_role_qualifier("input"))
        for labels in labels_by_head.values()
    )
    kv_x = 0
    branch_x = kv_width + 2
    query_x = branch_x + 1
    query_arrow_x = query_x + query_width + 1
    head_x = query_arrow_x + 2
    head_arrow_x = head_x + head_width + 1
    result_width = _horizontal_extent(diagram, detail.result.columns)
    head_dimension = str(diagram.config.value_head_dim)
    result_dimension = (
        f"{head_dimension}×{diagram.config.kv_heads}×"
        f"{diagram.config.queries_per_kv}={detail.result.rows}"
    )
    result_x = head_arrow_x + len(head_dimension) + 1
    bracket_x = result_x + result_width + 1
    canvas_width = max(
        bracket_x + len(result_dimension),
        _detail_heading_width(diagram.operations[step].result),
    )
    canvas = Canvas(canvas_width, top_y + box_height)

    canvas.fill_rect(result_x, top_y, result_width, box_height, GRID_BACKGROUND)
    canvas.fancy_box(
        result_x,
        top_y,
        result_width,
        box_height,
        "heavy",
        GRID_COLOR,
    )
    for separator_y in separators:
        _draw_head_concat_separator(
            canvas,
            result_x,
            separator_y,
            result_width,
        )

    _draw_matrix_name(canvas, result_x, top_y, result_width, "H", "output")
    dimension_x = result_x + max(1, (result_width - len("seq")) // 2)
    put(
        canvas,
        dimension_x,
        top_y,
        "seq",
        DIMENSION_COLOR,
        GRID_BACKGROUND,
        bold=True,
    )
    result_columns = tuple(
        _scaled_position(
            index,
            column_count,
            result_x + 2,
            result_x + result_width - 3,
        )
        for index in range(column_count)
    )

    for head in visible_heads:
        center_y = head_positions[head][1]
        kv_label, query_label, head_label = labels_by_head[head]
        is_selected = head == selected_head
        put(
            canvas,
            kv_x,
            center_y,
            kv_label,
            COLUMN_HIGHLIGHT if is_selected else DIMENSION_COLOR,
            None,
            bold=is_selected,
        )
        for connector_x in range(len(kv_label) + 1, query_x - 1):
            put(
                canvas,
                connector_x,
                center_y,
                "─",
                OPERATION_COLOR,
                None,
                bold=True,
            )
        put(
            canvas,
            query_x,
            center_y,
            query_label,
            ROW_HIGHLIGHT if is_selected else DIMENSION_COLOR,
            None,
            bold=is_selected,
        )
        put(
            canvas,
            query_arrow_x,
            center_y,
            "→",
            OPERATION_COLOR,
            None,
            bold=True,
        )
        put(
            canvas,
            head_x,
            center_y,
            head_label,
            VARIABLE_COLOR if is_selected else DIMENSION_COLOR,
            None,
            bold=is_selected,
        )
        qualifier = _role_qualifier("input")
        put(
            canvas,
            head_x + len(head_label),
            center_y,
            qualifier,
            DIMENSION_COLOR,
            None,
            bold=is_selected,
        )
        put(
            canvas,
            head_arrow_x,
            center_y,
            "→",
            OPERATION_COLOR,
            None,
            bold=True,
        )

    for omission_y in omission_positions:
        put(
            canvas,
            branch_x,
            omission_y,
            "⋮",
            GRID_COLOR,
            None,
            bold=True,
        )
        put(
            canvas,
            result_x + result_width // 2,
            omission_y,
            "⋮",
            GRID_COLOR,
            GRID_BACKGROUND,
            bold=True,
        )

    for item in visible_heads:
        box_y, center_y = head_positions[item]
        row_positions = (box_y + 1, box_y + 2, box_y + 3)
        for marker_y in (row_positions[0], row_positions[2]):
            for marker_x in result_columns:
                put(
                    canvas,
                    marker_x,
                    marker_y,
                    INACTIVE_CELL,
                    GRID_COLOR,
                    GRID_BACKGROUND,
                )
        for marker_x in result_columns:
            put(
                canvas,
                marker_x,
                row_positions[1],
                "·",
                GRID_COLOR,
                GRID_BACKGROUND,
            )
        put(
            canvas,
            result_x - len(head_dimension) + 1,
            center_y,
            head_dimension,
            DIMENSION_COLOR,
            GRID_BACKGROUND,
            bold=True,
        )
        if item == selected_head:
            put(
                canvas,
                result_columns[selected_column] - 1,
                row_positions[0],
                f"[{RESULT_CELL}]",
                RESULT_HIGHLIGHT,
                GRID_BACKGROUND,
                bold=True,
            )

    canvas.point(bracket_x, top_y, "┐", DIMENSION_COLOR)
    canvas.point(bracket_x, bottom_y, "┘", DIMENSION_COLOR)
    for bracket_y in range(top_y + 1, bottom_y):
        canvas.point(bracket_x, bracket_y, "│", DIMENSION_COLOR)
    put(
        canvas,
        bracket_x,
        top_y + box_height // 2,
        result_dimension,
        DIMENSION_COLOR,
        None,
        bold=True,
    )
    _draw_detail_heading(canvas, diagram.operations[step].result)
    return canvas


def render_matrix_detail(
    diagram: Diagram,
    step: int,
    row: int,
    column: int,
    *,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
    animation_stage: int = 2,
) -> Canvas:
    """Render dimension-scaled boxes with compact contribution markers."""
    detail = detail_for_step(diagram, step)
    if detail is None:
        raise ValueError("selected operation has no matrix-product detail")
    limit = max(MINIMUM_SAMPLE_LIMIT, sample_limit)
    if isinstance(detail, CausalSoftmaxDetail):
        return _render_causal_softmax_detail(
            diagram,
            step,
            detail,
            row,
            column,
            limit,
        )
    if isinstance(detail, RMSNormDetail):
        return _render_rmsnorm_detail(
            diagram,
            step,
            detail,
            row,
            column,
            limit,
        )
    if isinstance(detail, RoPEDetail):
        return _render_rope_detail(
            diagram,
            step,
            detail,
            row,
            column,
            limit,
            animation_stage,
        )
    if isinstance(detail, HeadConcatDetail):
        return _render_head_concat_detail(
            diagram,
            step,
            detail,
            row,
            column,
            limit,
        )
    if isinstance(detail, GatedActivationDetail):
        return _render_gated_activation_detail(
            diagram,
            step,
            detail,
            row,
            column,
            limit,
        )
    if isinstance(detail, ResidualDetail):
        return _render_residual_detail(
            diagram,
            step,
            detail,
            row,
            column,
            limit,
        )
    if isinstance(detail, BlockLoopDetail):
        return _render_block_loop_detail(
            diagram,
            step,
            detail,
            row,
            limit,
        )
    left = _with_collapsed_rows(
        _layout(
            diagram,
            detail.left,
            limit,
            minimum_width=25,
        )
    )
    right = _with_collapsed_rows(
        _layout(
            diagram,
            detail.right,
            limit,
            minimum_height=12,
        )
    )
    result = _with_collapsed_rows(
        _layout(diagram, detail.result, limit)
    )
    selected_row = max(0, min(row, result.row_count - 1))
    selected_column = max(0, min(column, result.column_count - 1))
    left_role = _operand_role(detail.left)
    right_role = _operand_role(detail.right)

    gap = 1
    left_x = 0
    result_x = left.total_width + gap
    right_x = result_x + result.box_offset - right.box_offset
    shift = max(0, -min(left_x, right_x, result_x))
    left_x += shift
    right_x += shift
    result_x += shift

    result_key = diagram.operations[step].result
    top_y = HEADING_HEIGHT
    bottom_y = top_y + right.height + 1
    canvas_width = max(
        left_x + left.total_width,
        right_x + right.total_width,
        result_x + result.total_width,
        left_x + left.total_width + _role_overhang(left_role),
        right_x + right.total_width + _role_overhang(right_role),
        result_x + result.total_width + _role_overhang("output"),
        _detail_heading_width(result_key),
        # _detail_context_width(diagram, step),
        _formula_body_x(
            (f'{detail.result.label}{_role_qualifier("output")}',)
        )
        + len(_formula_text(_cell_marker(detail.left), _cell_marker(detail.right))),
    )
    matrices_height = bottom_y + max(left.height, result.height)
    formula_y = matrices_height + 1
    canvas_height = formula_y + 1
    canvas = Canvas(canvas_width, canvas_height)

    _draw_box(
        canvas,
        right_x,
        top_y,
        right,
        name_side="right",
        active_column=selected_column,
        inactive_cells=True,
        collapsed_rows=True,
        io_role=right_role,
    )
    _draw_box(
        canvas,
        left_x,
        bottom_y,
        left,
        name_side="left",
        active_row=selected_row,
        inactive_cells=True,
        inactive_endpoint_columns=True,
        collapsed_rows=True,
        io_role=left_role,
    )
    _draw_box(
        canvas,
        result_x,
        bottom_y,
        result,
        name_side="right",
        show_dimensions=True,
        active_cell=(selected_row, selected_column),
        inactive_cells=True,
        collapsed_rows=True,
        io_role="output",
    )
    left_box_x = left_x + left.box_offset
    horizontal_arrow_y = bottom_y + left.height // 2
    put(
        canvas,
        left_box_x + left.width,
        horizontal_arrow_y,
        "→",
        ROW_HIGHLIGHT,
        None,
        bold=True,
    )
    right_box_x = right_x + right.box_offset
    vertical_arrow_x = right_box_x + right.width // 2
    put(
        canvas,
        vertical_arrow_x,
        top_y + right.height,
        "↓",
        COLUMN_HIGHLIGHT,
        None,
        bold=True,
    )
    # Optional operation context, retained for possible re-enabling:
    # _draw_detail_context(
    #     canvas,
    #     diagram,
    #     step,
    #     right_box_x + right.width // 2,
    #     result_x + result.box_offset + result.width // 2,
    #     bottom_y + result.height,
    # )
    _draw_formula(
        canvas,
        formula_y,
        _cell_marker(left.matrix),
        _cell_marker(right.matrix),
        detail.result.label,
    )
    _draw_detail_heading(canvas, result_key)
    return canvas
