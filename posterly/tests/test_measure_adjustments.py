"""Unit tests for ``compute_adjustment_hints`` -- the per-column
adjustment hints surfaced when ``measure`` fails its gap/spread gate.

The hint computation is a pure function (no Playwright, no DOM): given
each column's last-card-bottom and the footer-strip top, it yields the
target bottom plus a "keep / grow / trim" hint per column. Testing it as
a pure function pins the contract without booting a browser.

The hint exists because the alternative -- a reader doing
``strip_top - (min_gap + max_gap) / 2 - col_bottom`` mentally for every
column on every failed run -- reliably costs an extra rebuild per loop.
"""
from __future__ import annotations

from _posterly.measure import compute_adjustment_hints


def _hints(adjustments: list[tuple[str, float, str]]) -> dict[str, str]:
    """Pull just the hint string keyed by column name."""
    return {name: hint for name, _b, hint in adjustments}


def test_target_is_centre_of_gap_band() -> None:
    """``target_bottom = strip_top - (min_gap + max_gap) / 2``. Aiming
    for the centre keeps a tiny post-edit drift inside the band."""
    target_gap, target_bottom, _ = compute_adjustment_hints(
        bottoms=[("col0", 3000.0)],
        strip_top=3120.0,
        min_gap=30.0,
        max_gap=50.0,
    )
    assert target_gap == 40.0
    assert target_bottom == 3080.0


def test_keep_grow_trim_classification() -> None:
    """Within ``keep_tol_px`` of the target -> 'keep'; positive delta
    (column too short) -> 'grow ~N px'; negative delta (too tall)
    -> 'trim ~N px'. Boundary at exactly ``keep_tol_px`` is inclusive."""
    bottoms = [
        ("col0", 3080.0),  # delta 0
        ("col1", 3083.0),  # delta -3 (within 5)
        ("col2", 3075.0),  # delta +5 -> still 'keep' (boundary)
        ("col3", 3050.0),  # delta +30 -> grow
        ("col4", 3120.0),  # delta -40 -> trim
    ]
    _, _, adj = compute_adjustment_hints(
        bottoms=bottoms,
        strip_top=3120.0,
        min_gap=30.0,
        max_gap=50.0,
    )
    h = _hints(adj)
    assert h["col0"] == "keep"
    assert h["col1"] == "keep"
    assert h["col2"] == "keep"
    assert h["col3"] == "grow ~30 px"
    assert h["col4"] == "trim ~40 px"


def test_keep_tol_is_configurable() -> None:
    """Tighter tolerance promotes near-target columns out of 'keep'."""
    _, _, adj = compute_adjustment_hints(
        bottoms=[("col0", 3083.0)],  # delta -3
        strip_top=3120.0,
        min_gap=30.0,
        max_gap=50.0,
        keep_tol_px=1.0,
    )
    assert _hints(adj)["col0"] == "trim ~3 px"


def test_hints_round_to_whole_pixels() -> None:
    """Sub-pixel deltas round to int. Half-pixel goes to even via
    Python's banker's rounding -- both directions tested so the suite
    fails fast if someone swaps to int-truncation."""
    _, _, adj = compute_adjustment_hints(
        bottoms=[
            ("col0", 3050.6),  # delta +29.4 -> 29
            ("col1", 3049.5),  # delta +30.5 -> 30 (banker's)
            ("col2", 3110.4),  # delta -30.4 -> 30
        ],
        strip_top=3120.0,
        min_gap=30.0,
        max_gap=50.0,
    )
    h = _hints(adj)
    assert h["col0"] == "grow ~29 px"
    assert h["col1"] == "grow ~30 px"
    assert h["col2"] == "trim ~30 px"


def test_preserves_input_order_and_names() -> None:
    """Hints come back in the same order the caller supplied. Hero rows
    keep their hero name -- the cmd_measure caller mixes 'col0', 'col1',
    'hero' in one list and the print loop relies on stable order."""
    _, _, adj = compute_adjustment_hints(
        bottoms=[("col1", 3000.0), ("hero", 3060.0), ("col0", 3120.0)],
        strip_top=3120.0,
        min_gap=30.0,
        max_gap=50.0,
    )
    assert [name for name, _b, _h in adj] == ["col1", "hero", "col0"]


def test_handles_off_canvas_strip_top() -> None:
    """If the footer-strip itself rendered past the canvas (a real
    failure mode caught by the structure gate), the hints still compute
    -- the caller is the right place to decide whether to print them."""
    _, target_bottom, adj = compute_adjustment_hints(
        bottoms=[("col0", 3000.0), ("col1", 3050.0)],
        strip_top=3500.0,  # off-canvas
        min_gap=30.0,
        max_gap=50.0,
    )
    assert target_bottom == 3460.0
    h = _hints(adj)
    # Both columns are far below the (impossible) target -> grow.
    assert h["col0"].startswith("grow ")
    assert h["col1"].startswith("grow ")
