"""Parse an OSMD-rendered SVG string into structured systems / staves / notes.

OSMD's SVG markup varies across minor versions, so this parser keeps things
geometric: it groups horizontal ``<line>`` elements into 5-line staves, then
clusters staves into systems by vertical gap. Barlines are detected by
vertical ``<line>`` elements that span most of a system's height. Note
positions come from the frontend's GraphicSheet walk (passed in via
``osmd_notes``) because OSMD's SVG does not reliably carry notehead centers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from lxml import etree


# MusicXML scaling — used by callers that need tenths → pixel conversion.
MM_PER_TENTH = 6.1807 / 40.0


def tenths_to_px(tenths: float, dpi: float = 96.0) -> float:
    """Convert MusicXML <tenths> values to pixels at a given DPI."""
    return tenths * MM_PER_TENTH * dpi / 25.4


@dataclass
class SVGNote:
    note_id: str
    cx: float
    cy: float
    rx: float
    ry: float
    staff_idx: int
    system_idx: int
    measure_idx: int
    pitch: str
    duration: str
    bbox: Tuple[float, float, float, float]


@dataclass
class SVGStaff:
    staff_idx: int
    top_y: float
    bot_y: float
    left_x: float
    right_x: float
    line_ys: List[float] = field(default_factory=list)
    line_spacing: float = 0.0


@dataclass
class SVGMeasure:
    """Bbox of a single measure on a single staff in OSMD pixel space."""
    staff_idx: int
    measure_idx: int
    left_x: float
    right_x: float
    top_y: float
    bot_y: float


@dataclass
class SVGSystem:
    system_idx: int
    top_y: float
    bot_y: float
    left_x: float
    right_x: float
    staves: List[SVGStaff] = field(default_factory=list)
    notes: List[SVGNote] = field(default_factory=list)
    barline_xs: List[float] = field(default_factory=list)
    measures: List[SVGMeasure] = field(default_factory=list)


_NS = {"svg": "http://www.w3.org/2000/svg"}


# --- helpers ---------------------------------------------------------------


def _strip_namespace(xml: str) -> str:
    """lxml chokes on default namespaces for xpath — strip the attr."""
    return re.sub(r'\sxmlns="[^"]+"', "", xml, count=1)


def _float(attr: Optional[str], default: float = 0.0) -> float:
    if attr is None:
        return default
    try:
        return float(attr)
    except ValueError:
        return default


def _parse_transform(transform: Optional[str]) -> Tuple[float, float]:
    """Return (tx, ty) for a simple translate() transform; zero otherwise."""
    if not transform:
        return 0.0, 0.0
    m = re.search(r"translate\(\s*([-\d.]+)[ ,]+([-\d.]+)\s*\)", transform)
    if not m:
        m = re.search(r"translate\(\s*([-\d.]+)\s*\)", transform)
        if m:
            return float(m.group(1)), 0.0
        return 0.0, 0.0
    return float(m.group(1)), float(m.group(2))


def _accumulate_transform(elem) -> Tuple[float, float]:
    """Walk parents and sum translate offsets."""
    tx = ty = 0.0
    node = elem
    while node is not None:
        dx, dy = _parse_transform(node.get("transform"))
        tx += dx
        ty += dy
        node = node.getparent()
    return tx, ty


def _collect_lines(root) -> List[Dict[str, float]]:
    """All <line> elements with absolute coords (transforms flattened)."""
    out: List[Dict[str, float]] = []
    for line in root.iter("line"):
        x1 = _float(line.get("x1"))
        y1 = _float(line.get("y1"))
        x2 = _float(line.get("x2"))
        y2 = _float(line.get("y2"))
        tx, ty = _accumulate_transform(line)
        out.append({
            "x1": x1 + tx, "y1": y1 + ty,
            "x2": x2 + tx, "y2": y2 + ty,
        })
    return out


def _cluster_values(vals: List[float], max_gap: float) -> List[List[int]]:
    """Return clusters of indices whose values are within max_gap of each other."""
    if not vals:
        return []
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    clusters: List[List[int]] = [[order[0]]]
    for i in order[1:]:
        if vals[i] - vals[clusters[-1][-1]] <= max_gap:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    return clusters


def _group_staves_from_ys(
    line_ys: List[float], spacing_tol: float = 0.3
) -> List[List[int]]:
    """Group ascending y positions into windows of 5 with near-equal gaps."""
    if len(line_ys) < 5:
        return []
    order = sorted(range(len(line_ys)), key=lambda i: line_ys[i])
    sorted_ys = [line_ys[i] for i in order]
    staves: List[List[int]] = []
    i = 0
    while i <= len(sorted_ys) - 5:
        window = sorted_ys[i : i + 5]
        gaps = [window[k + 1] - window[k] for k in range(4)]
        mean_gap = sum(gaps) / 4.0
        if mean_gap <= 0:
            i += 1
            continue
        if all(abs(g - mean_gap) <= max(1.0, mean_gap * spacing_tol) for g in gaps):
            staves.append(order[i : i + 5])
            i += 5
        else:
            i += 1
    return staves


# --- public API ------------------------------------------------------------


def parse_osmd_svg(
    svg_string: str,
    osmd_notes: Optional[Iterable[dict]] = None,
    osmd_systems: Optional[Iterable[dict]] = None,
    osmd_measures: Optional[Iterable[dict]] = None,
) -> List[SVGSystem]:
    """Parse an OSMD SVG string into SVGSystem structures.

    Two sources of structural info, in priority order:
      1. ``osmd_systems`` — system/staff bounding boxes extracted directly
         from OSMD's GraphicSheet on the client. Most reliable.
      2. SVG geometry fallback — clusters horizontal ``<line>`` elements
         into staves/systems. Depends on OSMD emitting ``<line>`` for staff
         lines, which varies by version.

    ``osmd_notes`` supplies the notehead centers either way.
    ``osmd_measures`` supplies per-measure bboxes (from OSMD's GraphicSheet)
    used by the per-measure affine in ``aligner.py``.
    """
    # Preferred path: use OSMD's own structural data from the client.
    if osmd_systems:
        return _build_from_osmd_systems(
            list(osmd_systems),
            osmd_notes,
            svg_string if svg_string and svg_string.strip() else None,
            osmd_measures,
        )

    if not svg_string or not svg_string.strip():
        return []

    xml = _strip_namespace(svg_string)
    root = etree.fromstring(xml.encode("utf-8"))

    lines = _collect_lines(root)

    horizontals: List[Dict[str, float]] = []
    verticals: List[Dict[str, float]] = []
    for ln in lines:
        dx = abs(ln["x2"] - ln["x1"])
        dy = abs(ln["y2"] - ln["y1"])
        if dx >= max(dy * 3.0, 8.0):
            horizontals.append(ln)
        elif dy >= max(dx * 3.0, 8.0):
            verticals.append(ln)

    if not horizontals:
        return []

    # Merge horizontal strokes into candidate "staff line rows" by y.
    h_ys = [(ln["y1"] + ln["y2"]) / 2.0 for ln in horizontals]
    # Collapse horizontals on roughly the same row.
    row_clusters = _cluster_values(h_ys, max_gap=2.0)
    row_ys: List[float] = []
    row_x_ranges: List[Tuple[float, float]] = []
    for cluster in row_clusters:
        ys = [h_ys[i] for i in cluster]
        x_mins = [min(horizontals[i]["x1"], horizontals[i]["x2"]) for i in cluster]
        x_maxs = [max(horizontals[i]["x1"], horizontals[i]["x2"]) for i in cluster]
        row_ys.append(sum(ys) / len(ys))
        row_x_ranges.append((min(x_mins), max(x_maxs)))

    # Staves: 5 consecutive rows with near-equal gaps.
    stave_groups = _group_staves_from_ys(row_ys)
    if not stave_groups:
        return []

    # Compute each staff's bbox & spacing.
    staves: List[SVGStaff] = []
    for group_idx, group in enumerate(stave_groups):
        group_ys = sorted(row_ys[i] for i in group)
        lefts = [row_x_ranges[i][0] for i in group]
        rights = [row_x_ranges[i][1] for i in group]
        gaps = [group_ys[k + 1] - group_ys[k] for k in range(4)]
        staves.append(
            SVGStaff(
                staff_idx=-1,  # assigned after grouping into systems
                top_y=group_ys[0],
                bot_y=group_ys[-1],
                left_x=min(lefts),
                right_x=max(rights),
                line_ys=list(group_ys),
                line_spacing=float(sum(gaps) / 4.0),
            )
        )

    staves.sort(key=lambda s: s.top_y)

    # Group staves into systems: large vertical gap → new system.
    mean_spacing = sum(s.line_spacing for s in staves) / max(1, len(staves))
    systems: List[SVGSystem] = []
    current: List[SVGStaff] = [staves[0]]
    for st in staves[1:]:
        gap = st.top_y - current[-1].bot_y
        if gap > mean_spacing * 4.0:
            systems.append(_finalize_system(len(systems), current))
            current = [st]
        else:
            current.append(st)
    systems.append(_finalize_system(len(systems), current))

    # Barlines: vertical lines that span at least 70% of a system's height.
    for sys in systems:
        sys_top = sys.top_y
        sys_bot = sys.bot_y
        sys_height = max(1.0, sys_bot - sys_top)
        xs: List[float] = []
        for v in verticals:
            vtop = min(v["y1"], v["y2"])
            vbot = max(v["y1"], v["y2"])
            # Vertical must mostly lie within the system band.
            if vbot < sys_top - sys_height * 0.1:
                continue
            if vtop > sys_bot + sys_height * 0.1:
                continue
            overlap = min(vbot, sys_bot) - max(vtop, sys_top)
            if overlap < sys_height * 0.7:
                continue
            xs.append((v["x1"] + v["x2"]) / 2.0)
        # Merge close barline xs (within 2 px of each other).
        xs.sort()
        merged: List[float] = []
        for x in xs:
            if merged and abs(x - merged[-1]) < 3.0:
                merged[-1] = (merged[-1] + x) / 2.0
            else:
                merged.append(x)
        sys.barline_xs = merged

    # Map note entries from the OSMD GraphicSheet into the systems.
    if osmd_notes:
        for raw in osmd_notes:
            try:
                sys_idx = int(raw.get("systemIdx", 0))
                staff_idx = int(raw.get("staffIdx", 0))
                cx = float(raw.get("svgX", 0.0))
                cy = float(raw.get("svgY", 0.0))
                rx = float(raw.get("svgRx", 5.0))
                ry = float(raw.get("svgRy", 4.0))
            except (TypeError, ValueError):
                continue
            if sys_idx < 0 or sys_idx >= len(systems):
                continue
            sys = systems[sys_idx]
            if staff_idx < 0 or staff_idx >= len(sys.staves):
                staff_idx = max(0, min(staff_idx, len(sys.staves) - 1))
            # If the note's y is outside the reported system band, skip —
            # probably a disagreement between SVG geometry and the graphic
            # model (e.g. grace notes positioned above the staff).
            if cy < sys.top_y - rx * 10 or cy > sys.bot_y + rx * 10:
                # Keep anyway but at its reported y — the aligner will
                # later warp it into the scan's staff band.
                pass
            note = SVGNote(
                note_id=str(raw.get("noteId", f"n_{sys_idx}_{staff_idx}_{cx:.1f}")),
                cx=cx,
                cy=cy,
                rx=max(1.0, rx),
                ry=max(1.0, ry),
                staff_idx=staff_idx,
                system_idx=sys_idx,
                measure_idx=int(raw.get("measureIdx", 0)),
                pitch=str(raw.get("pitch", "?")),
                duration=str(raw.get("duration", "quarter")),
                bbox=(cx - rx, cy - ry, cx + rx, cy + ry),
            )
            sys.notes.append(note)

    return systems


def _finalize_system(system_idx: int, staves: List[SVGStaff]) -> SVGSystem:
    for i, st in enumerate(staves):
        st.staff_idx = i
    return SVGSystem(
        system_idx=system_idx,
        top_y=min(s.top_y for s in staves),
        bot_y=max(s.bot_y for s in staves),
        left_x=min(s.left_x for s in staves),
        right_x=max(s.right_x for s in staves),
        staves=list(staves),
        notes=[],
        barline_xs=[],
    )


def _build_from_osmd_systems(
    osmd_systems: List[dict],
    osmd_notes: Optional[Iterable[dict]],
    svg_string: Optional[str],
    osmd_measures: Optional[Iterable[dict]] = None,
) -> List[SVGSystem]:
    """Build SVGSystem structures from OSMD-extracted data."""
    systems: List[SVGSystem] = []
    for sys_idx, raw_sys in enumerate(osmd_systems):
        raw_staves = raw_sys.get("staves") or []
        staves: List[SVGStaff] = []
        for j, rs in enumerate(raw_staves):
            top_y = float(rs.get("top_y", 0.0))
            bot_y = float(rs.get("bot_y", top_y + 40.0))
            left_x = float(rs.get("left_x", 0.0))
            right_x = float(rs.get("right_x", left_x + 1.0))
            line_spacing = (bot_y - top_y) / 4.0 if bot_y > top_y else 10.0
            # Evenly space 5 lines across top→bot.
            line_ys = [top_y + k * line_spacing for k in range(5)]
            staves.append(
                SVGStaff(
                    staff_idx=j,
                    top_y=top_y,
                    bot_y=bot_y,
                    left_x=left_x,
                    right_x=right_x,
                    line_ys=line_ys,
                    line_spacing=float(line_spacing),
                )
            )
        if not staves:
            continue
        sys = SVGSystem(
            system_idx=sys_idx,
            top_y=float(raw_sys.get("top_y", min(s.top_y for s in staves))),
            bot_y=float(raw_sys.get("bot_y", max(s.bot_y for s in staves))),
            left_x=float(raw_sys.get("left_x", min(s.left_x for s in staves))),
            right_x=float(raw_sys.get("right_x", max(s.right_x for s in staves))),
            staves=staves,
            notes=[],
            barline_xs=[],
        )
        systems.append(sys)

    # Attach notes.
    if osmd_notes:
        for raw in osmd_notes:
            try:
                sys_idx = int(raw.get("systemIdx", 0))
                staff_idx = int(raw.get("staffIdx", 0))
                cx = float(raw.get("svgX", 0.0))
                cy = float(raw.get("svgY", 0.0))
                rx = float(raw.get("svgRx", 5.0))
                ry = float(raw.get("svgRy", 4.0))
            except (TypeError, ValueError):
                continue
            if sys_idx < 0 or sys_idx >= len(systems):
                continue
            sys = systems[sys_idx]
            if staff_idx < 0 or staff_idx >= len(sys.staves):
                staff_idx = max(0, min(staff_idx, len(sys.staves) - 1))
            sys.notes.append(
                SVGNote(
                    note_id=str(raw.get("noteId", f"n_{sys_idx}_{staff_idx}_{cx:.1f}")),
                    cx=cx,
                    cy=cy,
                    rx=max(1.0, rx),
                    ry=max(1.0, ry),
                    staff_idx=staff_idx,
                    system_idx=sys_idx,
                    measure_idx=int(raw.get("measureIdx", 0)),
                    pitch=str(raw.get("pitch", "?")),
                    duration=str(raw.get("duration", "quarter")),
                    bbox=(cx - rx, cy - ry, cx + rx, cy + ry),
                )
            )

    # Attach per-measure bboxes to their owning systems.
    if osmd_measures:
        for raw in osmd_measures:
            try:
                sys_idx = int(raw.get("system_idx", 0))
                staff_idx = int(raw.get("staff_idx", 0))
                measure_idx = int(raw.get("measure_idx", 0))
                left_x = float(raw.get("left_x", 0.0))
                right_x = float(raw.get("right_x", left_x + 1.0))
                top_y = float(raw.get("top_y", 0.0))
                bot_y = float(raw.get("bot_y", top_y + 1.0))
            except (TypeError, ValueError):
                continue
            if sys_idx < 0 or sys_idx >= len(systems):
                continue
            sys = systems[sys_idx]
            if right_x <= left_x or bot_y <= top_y:
                continue
            sys.measures.append(
                SVGMeasure(
                    staff_idx=staff_idx,
                    measure_idx=measure_idx,
                    left_x=left_x,
                    right_x=right_x,
                    top_y=top_y,
                    bot_y=bot_y,
                )
            )
        # Sort each system's measures by (staff, measure) for stable iteration.
        for sys in systems:
            sys.measures.sort(key=lambda m: (m.staff_idx, m.measure_idx, m.left_x))

    # Best-effort barline extraction if the SVG came along. Non-fatal.
    if svg_string:
        try:
            _attach_barlines_from_svg(svg_string, systems)
        except Exception:
            pass
    return systems


def _attach_barlines_from_svg(svg_string: str, systems: List[SVGSystem]) -> None:
    xml = _strip_namespace(svg_string)
    root = etree.fromstring(xml.encode("utf-8"))
    lines = _collect_lines(root)
    verticals = []
    for ln in lines:
        dx = abs(ln["x2"] - ln["x1"])
        dy = abs(ln["y2"] - ln["y1"])
        if dy >= max(dx * 3.0, 8.0):
            verticals.append(ln)
    for sys in systems:
        if not verticals:
            sys.barline_xs = []
            continue
        sys_top = sys.top_y
        sys_bot = sys.bot_y
        sys_height = max(1.0, sys_bot - sys_top)
        xs: List[float] = []
        for v in verticals:
            vtop = min(v["y1"], v["y2"])
            vbot = max(v["y1"], v["y2"])
            if vbot < sys_top - sys_height * 0.1:
                continue
            if vtop > sys_bot + sys_height * 0.1:
                continue
            overlap = min(vbot, sys_bot) - max(vtop, sys_top)
            if overlap < sys_height * 0.7:
                continue
            xs.append((v["x1"] + v["x2"]) / 2.0)
        xs.sort()
        merged: List[float] = []
        for x in xs:
            if merged and abs(x - merged[-1]) < 3.0:
                merged[-1] = (merged[-1] + x) / 2.0
            else:
                merged.append(x)
        sys.barline_xs = merged
