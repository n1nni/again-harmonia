"""Alignment from OSMD SVG space to scan pixel space.

For each (system, staff) we compute a list of measure pairs:
``(svg_measure_box, scan_measure_box)``. Mapping a note position is then a
plain affine into the matching scan box — chosen by which SVG measure box
the note falls into.

This makes the dot positions and the regenerated SVG overlay share one
transform: a dot at OSMD ``(cx, cy)`` and the rendered glyph at the same
``(cx, cy)`` always land on the same scan pixel by construction.

The IoU/Hungarian machinery is kept, but only as a *diagnostic* — it
populates ``confidence`` and never moves ``scan_x`` / ``scan_y``. The only
post-mapping adjustment is an optional y-only snap to the pitch grid (line
or space, including ledgers), capped at half a line-spacing so the snap can
never alter pitch perception.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .iou_optimizer import get_binary_scan, notehead_iou
from .notehead_detector import (
    assign_notes_to_candidates,
    detect_noteheads_in_band,
    filter_candidates_to_pitch_grid,
)
from .staff_detector import Staff, System, detect_barlines
from .svg_parser import SVGMeasure, SVGSystem


AnchorList = List[Tuple[float, float]]
Box = Tuple[float, float, float, float]  # (left_x, right_x, top_y, bot_y)
MeasurePair = Tuple[Box, Box]


def align(
    scan_path: str,
    svg_systems: Sequence[SVGSystem],
    scan_systems: Sequence[System],
) -> List[dict]:
    """Run the full alignment pipeline. Returns aligned note dicts."""
    binary = get_binary_scan(scan_path)
    results: List[dict] = []

    pair_count = min(len(svg_systems), len(scan_systems))
    for sys_idx in range(pair_count):
        svg_sys = svg_systems[sys_idx]
        scan_sys = scan_systems[sys_idx]

        # Barlines once per system.
        try:
            scan_barline_xs = detect_barlines(scan_path, scan_sys)
        except Exception:
            scan_barline_xs = []

        # Fallback anchor list — used when a note doesn't fall inside any
        # SVG measure box (clefs, key signatures, hanging articulations).
        anchors = _build_barline_anchors(
            svg_sys.barline_xs,
            scan_barline_xs,
            (svg_sys.left_x, svg_sys.right_x),
            (scan_sys.left_x, scan_sys.right_x),
        )

        # Measure-affine pairs, indexed by staff_idx within this system.
        measure_pairs_by_staff = _build_measure_pairs(svg_sys, scan_sys, scan_barline_xs)

        staff_count = min(len(svg_sys.staves), len(scan_sys.staves))
        for staff_idx in range(staff_count):
            svg_staff = svg_sys.staves[staff_idx]
            scan_staff = scan_sys.staves[staff_idx]
            system_notes = [n for n in svg_sys.notes if n.staff_idx == staff_idx]
            measure_pairs = measure_pairs_by_staff.get(staff_idx, [])

            scan_line_ys = [float(ln.y) for ln in scan_staff.lines]

            # Scale ellipse radii from SVG space into scan-pixel space.
            scan_staff_height = max(1.0, scan_staff.bot_y - scan_staff.top_y)
            svg_staff_height = max(1.0, svg_staff.bot_y - svg_staff.top_y)
            y_scale = scan_staff_height / svg_staff_height

            # Detect actual notehead candidates inside this scan staff band
            # (with pad for ledger lines). Used only for the diagnostic
            # IoU/Hungarian score now — never to move dots.
            candidates = detect_noteheads_in_band(
                binary,
                scan_staff.top_y,
                scan_staff.bot_y,
                scan_staff.left_x,
                scan_staff.right_x,
                scan_staff.line_spacing,
                pad_factor=4.0,
            )
            candidates = filter_candidates_to_pitch_grid(
                candidates,
                line_ys=scan_line_ys,
                line_spacing=scan_staff.line_spacing,
                ledger_count=5,
                tol_factor=0.40,
            )

            snap_radius_x = max(8.0, scan_staff.line_spacing * 3.5)
            snap_radius_y = max(6.0, scan_staff.line_spacing * 2.5)
            half_spacing = max(1.0, scan_staff.line_spacing * 0.5)

            mapped_positions: List[Tuple[float, float]] = []
            note_geom: List[Tuple[float, float]] = []
            for note in system_notes:
                mapped = _map_point_through_measure(note.cx, note.cy, measure_pairs)
                if mapped is None:
                    mx = _map_x_piecewise(note.cx, anchors)
                    my = _map_y_affine(
                        note.cy,
                        svg_staff.top_y, svg_staff.bot_y,
                        scan_staff.top_y, scan_staff.bot_y,
                    )
                else:
                    mx, my = mapped

                # y-only pitch-grid snap (≤ half a line-spacing).
                my_snapped = _snap_y_to_pitch_grid(
                    my, scan_line_ys, scan_staff.line_spacing, half_spacing,
                )
                mapped_positions.append((mx, my_snapped))
                note_geom.append((max(2.0, note.rx * y_scale), max(2.0, note.ry * y_scale)))

            # Hungarian assignment kept for diagnostic confidence only —
            # we record the assignment score but do not consume best_x/best_y.
            assigned = assign_notes_to_candidates(
                mapped_positions, candidates, snap_radius_x, snap_radius_y,
            )

            for idx, note in enumerate(system_notes):
                scan_x, scan_y = mapped_positions[idx]
                rx_scan, ry_scan = note_geom[idx]

                snap = assigned.get(idx)
                if snap is not None:
                    _bx, _by, score = snap
                    confidence = max(0.0, 1.0 - min(1.0, score))
                    snapped_diag = True
                else:
                    # No matching candidate — fall back to a single IoU read
                    # at the mapped position. Cheap, no grid search.
                    confidence = float(
                        notehead_iou(binary, scan_x, scan_y, rx_scan, ry_scan)
                    )
                    snapped_diag = False

                results.append({
                    "note_id": note.note_id,
                    "scan_x": int(round(scan_x)),
                    "scan_y": int(round(scan_y)),
                    "coarse_x": float(scan_x),
                    "coarse_y": float(scan_y),
                    "pitch": note.pitch,
                    "duration": note.duration,
                    "system_idx": sys_idx,
                    "staff_idx": staff_idx,
                    "measure_idx": note.measure_idx,
                    "confidence": float(confidence),
                    "rx": rx_scan,
                    "ry": ry_scan,
                    "snapped": snapped_diag,
                })

    return results


# --- measure-affine ---------------------------------------------------------


def _build_measure_pairs(
    svg_sys: SVGSystem,
    scan_sys: System,
    scan_barline_xs: Sequence[float],
) -> Dict[int, List[MeasurePair]]:
    """Pair SVG measure boxes with scan measure boxes, keyed by staff_idx.

    SVG side: ``svg_sys.measures`` (already grouped per staff).
    Scan side: split each scan staff into N+1 sequential boxes using the N
    detected barlines as interior x-boundaries; y comes from the staff's
    top_y / bot_y. Pair Nth-by-position.
    """
    pairs_by_staff: Dict[int, List[MeasurePair]] = {}
    if not svg_sys.measures:
        return pairs_by_staff

    # Group SVG measures by staff_idx, sorted by left_x for sequential pairing.
    by_staff: Dict[int, List[SVGMeasure]] = {}
    for m in svg_sys.measures:
        by_staff.setdefault(m.staff_idx, []).append(m)
    for lst in by_staff.values():
        lst.sort(key=lambda m: m.left_x)

    sorted_barlines = sorted(float(x) for x in scan_barline_xs)

    staff_count = len(scan_sys.staves)
    for staff_idx, svg_measures in by_staff.items():
        if staff_idx < 0 or staff_idx >= staff_count:
            continue
        scan_staff = scan_sys.staves[staff_idx]

        # Build scan-side x-boundaries: [left, b1, ..., bN, right], filtered
        # to barlines that actually fall within the staff's x-extent.
        interior = [
            x for x in sorted_barlines
            if scan_staff.left_x < x < scan_staff.right_x
        ]
        boundaries = [float(scan_staff.left_x)] + interior + [float(scan_staff.right_x)]
        scan_boxes: List[Box] = []
        for i in range(len(boundaries) - 1):
            lx = boundaries[i]
            rx = boundaries[i + 1]
            if rx - lx < 2.0:
                continue
            scan_boxes.append(
                (lx, rx, float(scan_staff.top_y), float(scan_staff.bot_y))
            )

        n = min(len(svg_measures), len(scan_boxes))
        if n == 0:
            continue
        pairs: List[MeasurePair] = []
        for i in range(n):
            sm = svg_measures[i]
            svg_box: Box = (
                float(sm.left_x), float(sm.right_x),
                float(sm.top_y), float(sm.bot_y),
            )
            pairs.append((svg_box, scan_boxes[i]))
        pairs_by_staff[staff_idx] = pairs

    return pairs_by_staff


def _map_point_through_measure(
    svg_x: float,
    svg_y: float,
    measure_pairs: Sequence[MeasurePair],
) -> Optional[Tuple[float, float]]:
    """Find the SVG box containing svg_x and affine-map (svg_x, svg_y) into
    its paired scan box. Returns None if no SVG box contains svg_x."""
    if not measure_pairs:
        return None
    for svg_box, scan_box in measure_pairs:
        slx, srx, sty, sby = svg_box
        if slx <= svg_x <= srx:
            clx, crx, cty, cby = scan_box
            mapped_x = _map_x_affine(svg_x, slx, srx, clx, crx)
            mapped_y = _map_x_affine(svg_y, sty, sby, cty, cby)
            return mapped_x, mapped_y
    return None


def _snap_y_to_pitch_grid(
    y: float,
    line_ys: Sequence[float],
    line_spacing: float,
    max_dist: float,
) -> float:
    """Snap y to the nearest pitch position (line or space, including ledgers).

    The pitch grid steps by half a line-spacing. Snap is bounded by
    ``max_dist`` so a wildly off mapping is left alone rather than yanked
    onto the wrong line.
    """
    if not line_ys or line_spacing <= 0:
        return y
    step = line_spacing / 2.0
    top = float(line_ys[0])
    k = round((y - top) / step)
    snapped = top + k * step
    if abs(snapped - y) <= max_dist:
        return snapped
    return y


# --- coarse mapping fallback ------------------------------------------------


def _map_x_affine(x: float, a0: float, a1: float, b0: float, b1: float) -> float:
    span = a1 - a0
    if abs(span) < 1e-6:
        return b0
    t = (x - a0) / span
    return b0 + t * (b1 - b0)


def _map_y_affine(y: float, a0: float, a1: float, b0: float, b1: float) -> float:
    return _map_x_affine(y, a0, a1, b0, b1)


def _build_barline_anchors(
    svg_barline_xs: Sequence[float],
    scan_barline_xs: Sequence[float],
    svg_xs: Tuple[float, float],
    scan_xs: Tuple[float, float],
    tolerance: float = 60.0,
) -> AnchorList:
    """Greedy pairing of SVG barlines to scan barlines via predicted position."""
    anchors: AnchorList = [(float(svg_xs[0]), float(scan_xs[0]))]

    svg_sorted = sorted(svg_barline_xs)
    scan_sorted = sorted(scan_barline_xs)
    used: set = set()
    svg_span = max(1e-6, svg_xs[1] - svg_xs[0])

    for svx in svg_sorted:
        t = (svx - svg_xs[0]) / svg_span
        predicted = scan_xs[0] + t * (scan_xs[1] - scan_xs[0])
        best_idx = -1
        best_dist = float("inf")
        for i, scx in enumerate(scan_sorted):
            if i in used:
                continue
            dist = abs(scx - predicted)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx >= 0 and best_dist <= tolerance:
            anchors.append((float(svx), float(scan_sorted[best_idx])))
            used.add(best_idx)

    anchors.append((float(svg_xs[1]), float(scan_xs[1])))
    anchors.sort(key=lambda a: a[0])
    deduped: AnchorList = []
    for svx, scx in anchors:
        if deduped and abs(svx - deduped[-1][0]) < 1e-3:
            continue
        deduped.append((svx, scx))

    for i in range(1, len(deduped)):
        if deduped[i][1] < deduped[i - 1][1]:
            deduped[i] = (deduped[i][0], deduped[i - 1][1])
    return deduped


def _map_x_piecewise(svg_x: float, anchors: AnchorList) -> float:
    if not anchors:
        return svg_x
    xs_svg = [a[0] for a in anchors]
    xs_scan = [a[1] for a in anchors]
    return float(np.interp(svg_x, xs_svg, xs_scan))
