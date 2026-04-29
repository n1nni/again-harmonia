"""Three-stage score alignment pipeline (new — does NOT use IoU).

Stage 1 (this commit): deskew the scan, detect staff lines and the system's
left/right barlines, compute the expected staff-line layout from MusicXML's
``<defaults>``, and report a residual after fitting a 4-DOF similarity per
system. Stage 1 emits a debug PNG that overlays detected lines (green),
detected barlines (cyan), and the XML-derived lines transformed into scan
space (red); they should sit on top of each other when Stage 1 is healthy.

Stage 2 (similarity transform per system) and Stage 3 (notehead matching
+ per-note dx) live in this module too but are not implemented yet — only
the function stubs are listed in __all__ so the API surface is visible.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from lxml import etree
from scipy.signal import find_peaks


__all__ = [
    "Line",
    "StaffLine",
    "SystemBand",
    "deskew",
    "detect_systems",
    "detect_staff_lines",
    "detect_barlines",
    "compute_svg_staff_lines",
    "fit_system_transform",
    "main",
]


# --- data types -------------------------------------------------------------


@dataclass
class Line:
    """A horizontal line segment with two endpoints."""
    x_left: float
    x_right: float
    y: float


@dataclass
class StaffLine(Line):
    system_idx: int
    staff_idx: int  # 0..n_staves-1 within its system
    line_idx: int   # 0..4 within its staff (0 = topmost)


@dataclass
class SystemBand:
    system_idx: int
    top_y: int
    bot_y: int
    staff_lines: List[StaffLine]
    left_barline_x: float
    right_barline_x: float
    # System-level ink extent: the leftmost / rightmost column of the
    # horizontal-stroke image that has *any* ink across this system's
    # y-band. More robust than a single barline (which can be missing or
    # internal) and more robust than per-line extent (one bad row
    # contaminates a min/max). This is what we anchor the similarity fit on.
    left_ink_x: float = 0.0
    right_ink_x: float = 0.0
    # Median line-to-line spacing within a single staff of this system,
    # in scan pixels. Used to size notehead templates in Stage 3a.
    line_spacing_px: float = 0.0


@dataclass
class NoteHead:
    """A notehead expected by the MusicXML, in MusicXML tenths from page TL."""
    system_idx: int
    staff_idx: int
    measure_idx: int
    x_xml: float
    y_xml: float
    pitch: str  # e.g. "D4"
    duration: str  # 'whole'|'half'|'quarter'|'eighth'|'16th'|'32nd'
    # Filled by visualizers / stage 2:
    x_scan: float = 0.0
    y_scan: float = 0.0


@dataclass
class NoteheadDetection:
    """A notehead actually detected on the (deskewed) scan."""
    x: float
    y: float
    note_type: str  # 'filled' or 'open'
    score: float
    system_idx: int


@dataclass
class LyricAnchor:
    """An XML <lyric> anchor: aligned x with parent note, y with the lyric's
    own default-y. MusicXML uses + up; we keep the original tenths here and
    let callers convert via the system's similarity transform."""
    system_idx: int
    staff_idx: int
    x_xml: float
    y_xml: float
    text: str


# --- Stage 1.1 — deskew -----------------------------------------------------


def deskew(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """Rotate ``image`` so the staff lines run perfectly horizontal.

    Detects long horizontal strokes (which are mostly staff lines), fits a
    line to each contour with ``cv2.fitLine``, takes the median angle, and
    rotates around the image center via ``cv2.warpAffine``.

    Returns ``(deskewed_image, angle_deg)``. Positive ``angle_deg`` means
    the original was tilted clockwise; we rotate by the same angle to undo.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = binary.shape
    kernel_w = max(50, w // 20)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    angles: List[float] = []
    for cnt in contours:
        if len(cnt) < 30:
            continue
        line_fit = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        vx, vy = float(line_fit[0]), float(line_fit[1])
        ang = float(np.degrees(np.arctan2(vy, vx)))
        # Stay well within near-horizontal — anything else is not a staff line.
        if abs(ang) > 8.0:
            continue
        angles.append(ang)

    if not angles:
        return image, 0.0

    angle_deg = float(np.median(angles))
    if abs(angle_deg) < 0.05:
        return image, angle_deg

    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    border = (255, 255, 255) if image.ndim == 3 else 255
    deskewed = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )
    return deskewed, angle_deg


# --- Stage 1.2 — staff-line and barline detection ---------------------------


def _binarize(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _horizontal_strokes(binary: np.ndarray) -> np.ndarray:
    """Long horizontal strokes only — staff lines, not stems or noteheads."""
    w = binary.shape[1]
    kernel_w = max(50, w // 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def _cluster_close_peaks(peaks: List[int], max_gap: int = 2) -> List[int]:
    if not peaks:
        return []
    peaks = sorted(peaks)
    clusters: List[List[int]] = [[peaks[0]]]
    for p in peaks[1:]:
        if p - clusters[-1][-1] <= max_gap:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [int(round(sum(c) / len(c))) for c in clusters]


def _group_5_lines_per_staff(
    line_ys: List[int], spacing_tol: float = 0.30,
) -> List[List[int]]:
    """Pack consecutive detected line-rows into 5-line staves."""
    if len(line_ys) < 5:
        return []

    line_ys = sorted(line_ys)
    gaps = [line_ys[i + 1] - line_ys[i] for i in range(len(line_ys) - 1)]
    if not gaps:
        return []
    median_gap = float(np.median(gaps))
    small = [g for g in gaps if g <= median_gap * 1.5]
    target = float(np.median(small)) if small else median_gap
    tol = max(1.5, target * spacing_tol)

    staves: List[List[int]] = []
    used: set = set()
    for start in range(len(line_ys)):
        if start in used:
            continue
        anchor = line_ys[start]
        predicted = [anchor + k * target for k in range(5)]
        matched_idx: List[int] = []
        rows: List[int] = []
        for k, pred in enumerate(predicted):
            best_j = -1
            best_d = tol
            for j, y in enumerate(line_ys):
                if j in used or j in matched_idx:
                    continue
                d = abs(y - pred)
                if d < best_d:
                    best_d = d
                    best_j = j
            if best_j >= 0:
                matched_idx.append(best_j)
                rows.append(line_ys[best_j])
            else:
                rows.append(int(round(pred)))
        if len(matched_idx) >= 3:  # at least 3 of 5 lines really detected
            rows.sort()
            staves.append(rows)
            used.update(matched_idx)

    staves.sort(key=lambda s: s[0])
    return staves


def _split_staves_into_systems(
    staves: List[List[int]], line_spacings: List[float],
) -> List[List[int]]:
    """Split staves into systems by the largest jump in inter-staff gaps."""
    if not staves:
        return []
    if len(staves) == 1:
        return [[0]]

    gaps = [staves[i][0] - staves[i - 1][-1] for i in range(1, len(staves))]
    mean_spacing = float(np.mean(line_spacings)) if line_spacings else 5.0
    floor = max(1.0, mean_spacing * 3.0)
    ceiling_guess = mean_spacing * 6.0

    if max(gaps) <= ceiling_guess:
        threshold = max(gaps) + 1
    else:
        sorted_gaps = sorted(gaps)
        best_jump = 0.0
        split_at = float(sorted_gaps[-1])
        for i in range(1, len(sorted_gaps)):
            jump = sorted_gaps[i] - sorted_gaps[i - 1]
            if jump > best_jump:
                best_jump = jump
                split_at = (sorted_gaps[i] + sorted_gaps[i - 1]) / 2.0
        threshold = max(split_at, floor)

    systems: List[List[int]] = [[0]]
    for idx, gap in enumerate(gaps, start=1):
        if gap <= threshold:
            systems[-1].append(idx)
        else:
            systems.append([idx])
    return systems


def _line_x_extent(horizontal: np.ndarray, y: int) -> Tuple[int, int]:
    """Active-pixel x-extent on the given row of the horizontal-stroke image."""
    h, w = horizontal.shape
    y = max(0, min(h - 1, int(y)))
    row = horizontal[y, :]
    if row.max() == 0:
        return 0, w - 1
    active = np.where(row > 0)[0]
    if active.size == 0:
        return 0, w - 1
    return int(active[0]), int(active[-1])


def detect_systems(image: np.ndarray) -> List[SystemBand]:
    """Detect every system, staff, staff-line, and per-system barlines.

    The deskew step should run first so staff lines are horizontal.
    """
    binary = _binarize(image)
    horizontal = _horizontal_strokes(binary)

    row_sum = horizontal.sum(axis=1)
    if row_sum.max() == 0:
        return []
    threshold = row_sum.max() * 0.30
    peaks, _ = find_peaks(row_sum, height=threshold, distance=2)
    line_ys = _cluster_close_peaks(peaks.tolist(), max_gap=2)
    if len(line_ys) < 5:
        return []

    staves_lines = _group_5_lines_per_staff(line_ys)
    if not staves_lines:
        return []

    line_spacings = [
        float(np.mean([s[k + 1] - s[k] for k in range(4)])) for s in staves_lines
    ]
    sys_groups = _split_staves_into_systems(staves_lines, line_spacings)

    out: List[SystemBand] = []
    for sys_idx, group in enumerate(sys_groups):
        staff_lines: List[StaffLine] = []
        for staff_idx, gi in enumerate(group):
            for line_idx, y in enumerate(staves_lines[gi]):
                xl, xr = _line_x_extent(horizontal, y)
                staff_lines.append(StaffLine(
                    x_left=float(xl),
                    x_right=float(xr),
                    y=float(y),
                    system_idx=sys_idx,
                    staff_idx=staff_idx,
                    line_idx=line_idx,
                ))
        if not staff_lines:
            continue
        top_y = int(min(sl.y for sl in staff_lines))
        bot_y = int(max(sl.y for sl in staff_lines))
        left_x, right_x = detect_barlines(binary, (top_y, bot_y))

        # System-level ink extent from the horizontal-stroke band. Any
        # column with non-zero ink anywhere in the band counts.
        band = horizontal[top_y:bot_y + 1, :]
        col_any = band.any(axis=0)
        active_cols = np.where(col_any)[0]
        if active_cols.size > 0:
            left_ink = float(active_cols[0])
            right_ink = float(active_cols[-1])
        else:
            left_ink = float(min(sl.x_left for sl in staff_lines))
            right_ink = float(max(sl.x_right for sl in staff_lines))

        # Median line-to-line spacing inside a staff (4 gaps × n_staves).
        spacings: List[float] = []
        from collections import defaultdict
        by_staff: dict = defaultdict(list)
        for sl in staff_lines:
            by_staff[sl.staff_idx].append(sl.y)
        for ys in by_staff.values():
            ys_sorted = sorted(ys)
            for k in range(len(ys_sorted) - 1):
                spacings.append(ys_sorted[k + 1] - ys_sorted[k])
        line_spacing_px = float(np.median(spacings)) if spacings else 0.0

        out.append(SystemBand(
            system_idx=sys_idx,
            top_y=top_y,
            bot_y=bot_y,
            staff_lines=staff_lines,
            left_barline_x=left_x,
            right_barline_x=right_x,
            left_ink_x=left_ink,
            right_ink_x=right_ink,
            line_spacing_px=line_spacing_px,
        ))
    return out


def detect_staff_lines(image: np.ndarray, system_idx: int) -> List[StaffLine]:
    """Spec-shaped wrapper: run full system detection and return one system's
    staff lines. Calling repeatedly for different ``system_idx`` re-detects
    the whole page each time; ``detect_systems`` is the efficient path."""
    bands = detect_systems(image)
    if system_idx < 0 or system_idx >= len(bands):
        return []
    return bands[system_idx].staff_lines


def detect_barlines(
    binary: np.ndarray, system_band: Tuple[int, int],
) -> Tuple[float, float]:
    """Leftmost and rightmost barline x-positions inside a system band.

    ``system_band`` is ``(top_y, bot_y)``; ``binary`` has staff ink == 255.
    """
    top_y, bot_y = system_band
    band = binary[top_y:bot_y + 1, :]
    if band.size == 0:
        return 0.0, float(binary.shape[1] - 1)

    band_height = bot_y - top_y + 1
    kernel_h = max(5, int(band_height * 0.7))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_h))
    vertical = cv2.morphologyEx(band, cv2.MORPH_OPEN, kernel)
    col_sum = vertical.sum(axis=0)
    if col_sum.max() == 0:
        return 0.0, float(binary.shape[1] - 1)
    threshold = col_sum.max() * 0.50
    active = np.where(col_sum > threshold)[0]
    if active.size == 0:
        return 0.0, float(binary.shape[1] - 1)
    return float(active[0]), float(active[-1])


# --- Stage 1.3 — XML-derived expected staff lines ---------------------------


def _read_float(parent, *path: str, default: float) -> float:
    node = parent
    for p in path:
        if node is None:
            return default
        node = node.find(p)
    if node is None or node.text is None:
        return default
    try:
        return float(node.text)
    except (TypeError, ValueError):
        return default


def compute_svg_staff_lines(
    musicxml_path: str, n_systems: Optional[int] = None,
) -> List[List[StaffLine]]:
    """Expected staff-line positions per system, in MusicXML tenths.

    Coordinate system: x increases right, y increases down (print space).
    The origin is the top-left of the page. Returned lines are the *ideal*
    layout the engraver targeted; Stage 2 will fit a similarity from this
    coord system to the deskewed scan.

    Number of staves per system is inferred from the count of ``<part>``
    elements (one staff per part). System count comes from ``n_systems``
    when given; otherwise it is derived from ``<print new-system="yes"/>``
    markers in part 0.
    """
    tree = etree.parse(musicxml_path)
    root = tree.getroot()

    # Page + system + staff layout numbers (tenths).
    defaults = root.find("defaults")
    page_layout = defaults.find("page-layout") if defaults is not None else None
    margins = page_layout.find("page-margins") if page_layout is not None else None
    sys_layout = defaults.find("system-layout") if defaults is not None else None
    staff_layout = defaults.find("staff-layout") if defaults is not None else None

    page_height = _read_float(page_layout, "page-height", default=1921.0)
    page_width = _read_float(page_layout, "page-width", default=1358.0)
    top_margin = _read_float(margins, "top-margin", default=97.0)
    left_margin = _read_float(margins, "left-margin", default=130.0)
    right_margin = _read_float(margins, "right-margin", default=130.0)
    system_distance = _read_float(sys_layout, "system-distance", default=165.0)
    top_system_distance = _read_float(sys_layout, "top-system-distance", default=165.0)
    staff_distance = _read_float(staff_layout, "staff-distance", default=82.0)

    # Per-measure overrides for top-system-distance (Finale puts this on the
    # first <print> of each system so the first system can sit lower than the
    # default rule). We honor it for the first system only.
    parts = root.findall("score-partwise/part") if root.tag == "score-partwise" else root.findall("part")
    if not parts:
        parts = root.xpath("//part")
    n_staves = len(parts) if parts else 3
    if n_staves <= 0:
        n_staves = 3

    # Inferred system count from new-system markers in part 0.
    if n_systems is None:
        if not parts:
            n_systems = 1
        else:
            measures = parts[0].findall("measure")
            breaks = 1  # first system always exists
            for m in measures:
                p = m.find("print")
                if p is not None and p.get("new-system") == "yes":
                    breaks += 1
            n_systems = breaks

    # Per-system overrides from each <print> that starts a system. None means
    # "no explicit override" — for the first system we then fall back to
    # ``top-system-distance`` from <defaults>; for later systems we fall
    # back to ``system-distance`` measured from the previous system's bottom.
    explicit_top_per_system: List[Optional[float]] = []
    explicit_left_margin_per_system: List[Optional[float]] = []
    explicit_right_margin_per_system: List[Optional[float]] = []
    if parts:
        first_print_seen = False
        for m in parts[0].findall("measure"):
            p = m.find("print")
            if p is None:
                continue
            new_system = (p.get("new-system") == "yes")
            if (not first_print_seen) or new_system:
                tsd_node = p.find("system-layout/top-system-distance")
                if tsd_node is not None and tsd_node.text is not None:
                    try:
                        explicit_top_per_system.append(float(tsd_node.text))
                    except ValueError:
                        explicit_top_per_system.append(None)
                else:
                    explicit_top_per_system.append(None)

                left_node = p.find("system-layout/system-margins/left-margin")
                right_node = p.find("system-layout/system-margins/right-margin")
                explicit_left_margin_per_system.append(
                    float(left_node.text)
                    if left_node is not None and left_node.text is not None
                    else None
                )
                explicit_right_margin_per_system.append(
                    float(right_node.text)
                    if right_node is not None and right_node.text is not None
                    else None
                )
                first_print_seen = True
        while len(explicit_top_per_system) < n_systems:
            explicit_top_per_system.append(None)
        while len(explicit_left_margin_per_system) < n_systems:
            explicit_left_margin_per_system.append(None)
        while len(explicit_right_margin_per_system) < n_systems:
            explicit_right_margin_per_system.append(None)

    # Geometry per the MusicXML spec:
    #   top-system-distance: page top → top staff line of system's first staff.
    #   staff-distance:      bottom line of one staff → top line of next.
    #   system-distance:     bottom line of last staff in system N →
    #                        top line of first staff in system N+1.
    staff_height_tenths = 40.0  # 4 inter-line gaps × 10 tenths

    out: List[List[StaffLine]] = []
    # First system y: explicit top-system-distance if present, otherwise
    # the page-default top-system-distance.
    first_tsd = (
        explicit_top_per_system[0]
        if explicit_top_per_system and explicit_top_per_system[0] is not None
        else top_system_distance
    )
    cur_sys_top: float = top_margin + first_tsd

    for sys_idx in range(n_systems):
        # Per-system left/right system-margins offsets (additive to page margins).
        sys_left_off = (
            explicit_left_margin_per_system[sys_idx]
            if sys_idx < len(explicit_left_margin_per_system)
            and explicit_left_margin_per_system[sys_idx] is not None
            else 0.0
        )
        sys_right_off = (
            explicit_right_margin_per_system[sys_idx]
            if sys_idx < len(explicit_right_margin_per_system)
            and explicit_right_margin_per_system[sys_idx] is not None
            else 0.0
        )
        x_left = left_margin + sys_left_off
        x_right = (page_width - right_margin) - sys_right_off

        lines: List[StaffLine] = []
        staff_top = cur_sys_top
        for staff_idx in range(n_staves):
            for line_idx in range(5):
                y = staff_top + line_idx * 10.0
                lines.append(StaffLine(
                    x_left=x_left,
                    x_right=x_right,
                    y=y,
                    system_idx=sys_idx,
                    staff_idx=staff_idx,
                    line_idx=line_idx,
                ))
            staff_top += staff_height_tenths + staff_distance
        out.append(lines)
        # Bottom staff line of this system = staff_top - staff_distance.
        last_line_y = staff_top - staff_distance
        # Next system's top line: explicit top-system-distance wins, else
        # default system-distance from this system's bottom.
        next_idx = sys_idx + 1
        if (
            next_idx < len(explicit_top_per_system)
            and explicit_top_per_system[next_idx] is not None
        ):
            cur_sys_top = top_margin + explicit_top_per_system[next_idx]
        else:
            cur_sys_top = last_line_y + system_distance

    return out


# --- Stage 1.4 — sanity-check fit (peeks into Stage 2 for residual report) --


def fit_system_transform(
    svg_lines: List[StaffLine],
    scan_band: SystemBand,
) -> Tuple[Optional[np.ndarray], float, float, int]:
    """Fit a 4-DOF similarity (RANSAC) from XML coords → scan coords.

    Returns ``(M_2x3, mean_residual_px, max_residual_px, n_pairs)`` or
    ``(None, inf, inf, 0)`` if there aren't enough valid pairs.

    Point pairs come from staff-line endpoints (where the horizontal ink
    actually starts and ends on the scan). We use per-line extents — not
    the system's bounding barlines — because a faint or absent closing
    barline silently mis-anchors the fit and forces a non-uniform scale.
    """
    by_key = {(sl.staff_idx, sl.line_idx): sl for sl in scan_band.staff_lines}

    # Anchor at the system's horizontal-stroke ink extent (computed once at
    # detection time, see ``SystemBand.left_ink_x`` / ``right_ink_x``).
    sys_left_x = scan_band.left_ink_x
    sys_right_x = scan_band.right_ink_x

    src_pts: List[List[float]] = []
    dst_pts: List[List[float]] = []
    for svg_sl in svg_lines:
        scan_sl = by_key.get((svg_sl.staff_idx, svg_sl.line_idx))
        if scan_sl is None:
            continue
        src_pts.append([svg_sl.x_left, svg_sl.y])
        src_pts.append([svg_sl.x_right, svg_sl.y])
        dst_pts.append([sys_left_x, scan_sl.y])
        dst_pts.append([sys_right_x, scan_sl.y])

    n = len(src_pts)
    if n < 4:
        return None, float("inf"), float("inf"), n

    src = np.asarray(src_pts, dtype=np.float32).reshape(-1, 1, 2)
    dst = np.asarray(dst_pts, dtype=np.float32).reshape(-1, 1, 2)
    M, _inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0,
    )
    if M is None:
        return None, float("inf"), float("inf"), n

    src_h = np.hstack([
        np.asarray(src_pts, dtype=np.float64),
        np.ones((n, 1), dtype=np.float64),
    ])
    mapped = src_h @ M.T
    residuals = np.linalg.norm(mapped - np.asarray(dst_pts, dtype=np.float64), axis=1)
    return M, float(residuals.mean()), float(residuals.max()), n


# --- XML notehead extraction (driven by default-x + pitch) ------------------


_DIATONIC: dict = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6}


def _diatonic(step: str, octave: int) -> int:
    """Diatonic position with C0 = 0, ascending one per scale step."""
    return _DIATONIC.get(step.upper(), 0) + octave * 7


def _pitch_to_y_offset(step: str, octave: int, alter: int, clef_sign: str) -> float:
    """Y offset (MusicXML tenths) of a notehead from the staff's TOP staff line.

    Treble (G clef): top line = F5, bottom line = E4.
    Bass (F clef):   top line = A3, bottom line = G2.
    Each diatonic step downward = +5 tenths (half a line spacing); ``alter``
    only changes the accidental, not the y position.
    """
    note = _diatonic(step, octave)
    if clef_sign.upper() == "F":
        top_line = _diatonic("A", 3)
    else:
        top_line = _diatonic("F", 5)
    return (top_line - note) * 5.0


def _detect_clefs(parts) -> List[str]:
    """Return clef sign ('G' or 'F') for each part, taking the first <clef>."""
    out: List[str] = []
    for part in parts:
        sign = "G"
        clef_node = part.find("measure/attributes/clef/sign")
        if clef_node is not None and clef_node.text:
            sign = clef_node.text.strip().upper()
        out.append(sign if sign in ("G", "F", "C") else "G")
    return out


def extract_xml_noteheads(
    musicxml_path: str,
    svg_systems_layout: List[List[StaffLine]],
) -> List[NoteHead]:
    """Walk the MusicXML and emit one NoteHead per pitched note.

    x = page-margin-x + cumulative measure widths in this system + default-x
    y = staff_top + pitch-derived offset
    """
    tree = etree.parse(musicxml_path)
    root = tree.getroot()
    parts = root.findall("score-partwise/part") if root.tag == "score-partwise" else root.findall("part")
    if not parts:
        parts = root.xpath("//part")
    if not parts:
        return []
    clef_signs = _detect_clefs(parts)

    # Index staff_top y per (system_idx, staff_idx) by pulling the
    # topmost line (line_idx == 0) out of svg_systems_layout.
    staff_tops: dict = {}
    sys_x_left: dict = {}
    for sys_idx, lines in enumerate(svg_systems_layout):
        for sl in lines:
            if sl.line_idx == 0:
                staff_tops[(sys_idx, sl.staff_idx)] = sl.y
            sys_x_left.setdefault(sys_idx, sl.x_left)
            if sl.x_left < sys_x_left[sys_idx]:
                sys_x_left[sys_idx] = sl.x_left

    out: List[NoteHead] = []
    durations_map = {
        "whole": "whole", "half": "half", "quarter": "quarter",
        "eighth": "eighth", "16th": "16th", "32nd": "32nd",
        "64th": "64th",
    }

    for part_idx, part in enumerate(parts):
        clef = clef_signs[part_idx] if part_idx < len(clef_signs) else "G"
        cur_sys = 0
        cumul_x = 0.0  # tenths consumed by previous measures in this system
        first_print_seen = False

        for m_node in part.findall("measure"):
            print_el = m_node.find("print")
            if print_el is not None:
                new_sys = print_el.get("new-system") == "yes"
                if first_print_seen and new_sys:
                    cur_sys += 1
                    cumul_x = 0.0
                first_print_seen = True

            sys_left_tenths = sys_x_left.get(cur_sys, 0.0)
            measure_left_x = sys_left_tenths + cumul_x
            staff_top = staff_tops.get((cur_sys, part_idx), 0.0)

            try:
                measure_idx = int(m_node.get("number", "0")) - 1
            except (TypeError, ValueError):
                measure_idx = 0

            for note in m_node.findall("note"):
                if note.find("rest") is not None:
                    continue
                dx_attr = note.get("default-x")
                if dx_attr is None:
                    continue
                try:
                    dx = float(dx_attr)
                except ValueError:
                    continue
                pitch_el = note.find("pitch")
                if pitch_el is None:
                    continue
                step_el = pitch_el.find("step")
                octave_el = pitch_el.find("octave")
                alter_el = pitch_el.find("alter")
                if step_el is None or octave_el is None:
                    continue
                step = (step_el.text or "C").strip()
                try:
                    octave = int(octave_el.text)
                except (TypeError, ValueError):
                    continue
                try:
                    alter = int(alter_el.text) if alter_el is not None and alter_el.text else 0
                except ValueError:
                    alter = 0

                y_off = _pitch_to_y_offset(step, octave, alter, clef)
                x_xml = measure_left_x + dx
                y_xml = staff_top + y_off

                type_el = note.find("type")
                duration = durations_map.get(
                    (type_el.text.strip().lower() if type_el is not None and type_el.text else ""),
                    "quarter",
                )
                acc = "#" if alter == 1 else "b" if alter == -1 else ""
                out.append(NoteHead(
                    system_idx=cur_sys,
                    staff_idx=part_idx,
                    measure_idx=measure_idx,
                    x_xml=x_xml,
                    y_xml=y_xml,
                    pitch=f"{step}{acc}{octave}",
                    duration=duration,
                ))

            width_attr = m_node.get("width")
            if width_attr:
                try:
                    cumul_x += float(width_attr)
                except ValueError:
                    pass

    return out


def extract_xml_lyrics(
    musicxml_path: str,
    svg_systems_layout: List[List[StaffLine]],
) -> List[LyricAnchor]:
    """Walk the MusicXML and emit one LyricAnchor per <lyric> element.

    x is the parent note's x (page-margin + cumulative widths + default-x).
    y is computed as ``staff_top - lyric.default_y`` because MusicXML uses
    "+ up" while our XML coordinate system runs y-down. So the typical
    ``default-y="-80"`` puts the lyric 80 tenths *below* the staff top.
    """
    tree = etree.parse(musicxml_path)
    root = tree.getroot()
    parts = root.findall("score-partwise/part") if root.tag == "score-partwise" else root.findall("part")
    if not parts:
        parts = root.xpath("//part")
    if not parts:
        return []

    staff_tops: dict = {}
    sys_x_left: dict = {}
    for sys_idx, lines in enumerate(svg_systems_layout):
        for sl in lines:
            if sl.line_idx == 0:
                staff_tops[(sys_idx, sl.staff_idx)] = sl.y
            sys_x_left.setdefault(sys_idx, sl.x_left)

    out: List[LyricAnchor] = []
    for part_idx, part in enumerate(parts):
        cur_sys = 0
        cumul_x = 0.0
        first_print_seen = False

        for m_node in part.findall("measure"):
            print_el = m_node.find("print")
            if print_el is not None:
                new_sys = print_el.get("new-system") == "yes"
                if first_print_seen and new_sys:
                    cur_sys += 1
                    cumul_x = 0.0
                first_print_seen = True

            sys_left_tenths = sys_x_left.get(cur_sys, 0.0)
            measure_left_x = sys_left_tenths + cumul_x
            staff_top = staff_tops.get((cur_sys, part_idx), 0.0)

            for note in m_node.findall("note"):
                if note.find("rest") is not None:
                    continue
                dx_attr = note.get("default-x")
                if dx_attr is None:
                    continue
                try:
                    dx = float(dx_attr)
                except ValueError:
                    continue
                # A note may carry several lyrics (verses); emit each.
                for lyric in note.findall("lyric"):
                    ly_attr = lyric.get("default-y")
                    if ly_attr is None:
                        continue
                    try:
                        ly = float(ly_attr)
                    except ValueError:
                        continue
                    text = ""
                    text_el = lyric.find("text")
                    if text_el is not None and text_el.text:
                        text = text_el.text
                    out.append(LyricAnchor(
                        system_idx=cur_sys,
                        staff_idx=part_idx,
                        x_xml=measure_left_x + dx,
                        y_xml=staff_top - ly,
                        text=text,
                    ))

            width_attr = m_node.get("width")
            if width_attr:
                try:
                    cumul_x += float(width_attr)
                except ValueError:
                    pass
    return out


# --- Stage 3a: scan notehead detection via matchTemplate --------------------


def _make_notehead_template(spacing: float, filled: bool) -> np.ndarray:
    """Synthetic notehead template sized to the staff-line spacing.

    Filled: a solid ellipse, height = spacing × 0.9, width = spacing × 1.2,
    rotated −20°. The 0.9/1.2 sizing (vs the textbook 1.0/1.3) accounts for
    anti-aliasing eating the printed-notehead edge at low DPI.

    Open: outer filled ellipse minus an inner filled ellipse (ring), with
    ring thickness = spacing × 0.25. Built by subtraction so the inner edge
    is a clean ellipse rather than a constant-thickness offset.
    """
    h = max(7, int(round(spacing * 0.9)) | 1)
    w = max(9, int(round(spacing * 1.2)) | 1)
    pad = 2
    canvas = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.uint8)
    cx = canvas.shape[1] // 2
    cy = canvas.shape[0] // 2
    cv2.ellipse(
        canvas, (cx, cy), (w // 2, h // 2),
        -20.0, 0.0, 360.0, 255, -1,
    )
    if not filled:
        ring = max(1, int(round(spacing * 0.25)))
        inner_w = max(1, w // 2 - ring)
        inner_h = max(1, h // 2 - ring)
        cv2.ellipse(
            canvas, (cx, cy), (inner_w, inner_h),
            -20.0, 0.0, 360.0, 0, -1,
        )
    return canvas


def _nms(detections: List[NoteheadDetection], min_dist: float) -> List[NoteheadDetection]:
    """Greedy non-max suppression on detections, keeping the highest score.

    Uses ``<=`` (not ``<``) so two detections sitting exactly ``min_dist``
    apart get deduplicated — at low DPI matchTemplate often produces such
    boundary pairs around a single notehead.
    """
    detections = sorted(detections, key=lambda d: -d.score)
    kept: List[NoteheadDetection] = []
    md2 = min_dist * min_dist
    for d in detections:
        ok = True
        for k in kept:
            dx = d.x - k.x
            dy = d.y - k.y
            if dx * dx + dy * dy <= md2:
                ok = False
                break
        if ok:
            kept.append(d)
    return kept


def _staff_y_bands(band: SystemBand, pad_factor: float = 1.5) -> List[Tuple[int, int]]:
    """For each staff in a system, return (y0, y1) padded by pad_factor*spacing.

    Restricting detection to per-staff bands keeps lyric letters between staves
    out of the search area — at low DPI those letters score as ellipses.
    """
    spacing = band.line_spacing_px
    pad = int(round(spacing * pad_factor))
    out: List[Tuple[int, int]] = []
    by_staff: dict = {}
    for sl in band.staff_lines:
        by_staff.setdefault(sl.staff_idx, []).append(sl.y)
    for staff_idx in sorted(by_staff.keys()):
        ys = by_staff[staff_idx]
        out.append((int(min(ys) - pad), int(max(ys) + pad)))
    return out


def _pitch_grid_ys(band: SystemBand, ledger_steps: int = 4) -> List[float]:
    """Y-positions of every pitch slot (line OR space center) in this system.

    For each staff, emit its 5 line y's plus the 4 inter-line space centers,
    then extend the half-spacing grid up and down by ``ledger_steps`` slots
    so notes on ledger lines pass the snap-to-pitch filter.
    """
    spacing = band.line_spacing_px
    if spacing <= 0:
        return []
    out: List[float] = []
    by_staff: dict = {}
    for sl in band.staff_lines:
        by_staff.setdefault(sl.staff_idx, []).append(sl.y)
    for staff_idx, ys in by_staff.items():
        ys_sorted = sorted(ys)
        # Inside the staff: 5 lines + 4 inter-line spaces.
        for y in ys_sorted:
            out.append(y)
        for i in range(len(ys_sorted) - 1):
            out.append((ys_sorted[i] + ys_sorted[i + 1]) / 2.0)
        # Above + below: half-spacing extensions for ledger lines/spaces.
        top = ys_sorted[0]
        bot = ys_sorted[-1]
        step = spacing / 2.0
        for k in range(1, ledger_steps + 1):
            out.append(top - k * step)
            out.append(bot + k * step)
    out.sort()
    return out


def detect_noteheads(
    image: np.ndarray,
    system_band: SystemBand,
    threshold: float = 0.55,
    lyric_anchors_scan: Optional[List[Tuple[float, float]]] = None,
) -> List[NoteheadDetection]:
    """Detect noteheads via 2× upscale + matchTemplate.

    The scan is internally upscaled by 2× (cv2.resize, INTER_CUBIC) so the
    templates land in a regime where matchTemplate is reliable. Templates
    are regenerated at the upscaled spacing. Detected (x, y) coordinates
    are halved at the end so callers always work in the canonical
    original-scan coordinate system.

    Pipeline:
      1. scan_2x = upscale(scan, 2×, cubic), spacing_2x = spacing × 2.
      2. matchTemplate (TM_CCOEFF_NORMED) for filled and open templates,
         restricted to per-staff y-bands and skipping the clef+key-sig area.
      3. NMS within each template at min-distance = spacing_2x × 1.0.
      4. Combine filled+open and dedup overlaps at spacing_2x × 0.7,
         keeping the higher-scoring detection.
      5. Divide all (x, y) by 2.
    """
    spacing = system_band.line_spacing_px
    if spacing <= 0:
        return []

    # Step 1: upscale.
    scan_2x = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    binary_2x = _binarize(scan_2x)
    spacing_2x = spacing * 2.0
    h2, w2 = binary_2x.shape

    # Skip the clef + key signature area on the left (~4 spacings) and the
    # closing barline column on the right. Coordinates here are in 2× space.
    clef_skip_orig = spacing * 4.0
    x_min_orig = max(system_band.left_ink_x, system_band.left_barline_x) + clef_skip_orig
    x_max_orig = system_band.right_ink_x - 1.0
    x0_2x = int(max(0, x_min_orig * 2))
    x1_2x = int(min(w2, x_max_orig * 2))

    # Per-staff y-bands in 2× coordinates.
    staff_y_bands_2x: List[Tuple[int, int]] = [
        (max(0, int(y_top * 2)), min(h2, int(y_bot * 2) + 1))
        for (y_top, y_bot) in _staff_y_bands(system_band, pad_factor=1.5)
    ]

    # Step 2-3: matchTemplate per template type, NMS within type at 1.0 × spacing_2x.
    per_type: dict = {"filled": [], "open": []}
    for is_filled, label in ((True, "filled"), (False, "open")):
        template = _make_notehead_template(spacing_2x, filled=is_filled)
        for y0, y1 in staff_y_bands_2x:
            sub = binary_2x[y0:y1, x0_2x:x1_2x]
            if sub.size == 0 or template.shape[0] >= sub.shape[0] or template.shape[1] >= sub.shape[1]:
                continue
            result = cv2.matchTemplate(sub, template, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(result >= threshold)
            if ys.size == 0:
                continue
            th, tw = template.shape
            for ry, rx in zip(ys, xs):
                cx_2x = float(rx + tw / 2.0 + x0_2x)
                cy_2x = float(ry + th / 2.0 + y0)
                per_type[label].append(NoteheadDetection(
                    x=cx_2x, y=cy_2x,
                    note_type=label,
                    score=float(result[ry, rx]),
                    system_idx=system_band.system_idx,
                ))
        per_type[label] = _nms(per_type[label], min_dist=spacing_2x * 1.0)

    # Step 4: combine + cross-type dedup at spacing_2x × 0.7.
    combined = per_type["filled"] + per_type["open"]
    if not combined:
        return []
    combined = _nms(combined, min_dist=spacing_2x * 0.7)

    # Step 5: halve coordinates back to original scan space, then a final
    # dedup at ``spacing × 1.05`` to mop up sub-pixel boundary pairs that
    # the 2x-space NMS at exactly ``spacing_2x × 1.0`` left behind (a pair
    # 8.05 px apart in 2x squeaks through, becomes 4.025 in original).
    halved = [
        NoteheadDetection(
            x=d.x / 2.0,
            y=d.y / 2.0,
            note_type=d.note_type,
            score=d.score,
            system_idx=d.system_idx,
        )
        for d in combined
    ]
    halved = _nms(halved, min_dist=spacing * 1.05)

    # Lyric exclusion zones: each XML <lyric> is transformed to scan space
    # (caller does that with the system's similarity matrix) and produces a
    # rectangular forbidden zone around its anchor — ±1.5×spacing in x
    # (lyrics are roughly three-spacings wide), ±0.5×spacing in y (kept
    # narrow on purpose so a real notehead on the lowest line isn't rejected
    # by Stage 2's residual pixel-or-two of vertical error).
    if lyric_anchors_scan:
        spacing = system_band.line_spacing_px
        x_excl = spacing * 1.5
        y_excl = spacing * 0.5
        kept: List[NoteheadDetection] = []
        for d in halved:
            in_lyric = False
            for lx, ly in lyric_anchors_scan:
                if abs(d.x - lx) <= x_excl and abs(d.y - ly) <= y_excl:
                    in_lyric = True
                    break
            if not in_lyric:
                kept.append(d)
        return kept
    return halved


# --- Stage 1 debug visualization --------------------------------------------


def _draw_dashed_line(img, p1, p2, color, thickness=1, dash=8):
    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)
    total = float(np.linalg.norm(p2 - p1))
    if total < 1e-3:
        return
    n_segs = max(1, int(total // dash))
    for i in range(0, n_segs, 2):
        t0 = i / n_segs
        t1 = min(1.0, (i + 1) / n_segs)
        a = (1 - t0) * p1 + t0 * p2
        b = (1 - t1) * p1 + t1 * p2
        cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, thickness)


def visualize_stage1(
    deskewed: np.ndarray,
    scan_systems: List[SystemBand],
    svg_systems: List[List[StaffLine]],
    transforms: List[Optional[np.ndarray]],
    xml_noteheads: Optional[List[NoteHead]] = None,
    detected_noteheads: Optional[List[NoteheadDetection]] = None,
) -> np.ndarray:
    """One viz that answers Stage 1, Stage 2 (notehead y-positions), and
    Stage 3a (scan notehead detection) at once.

    Layers, top to bottom:
      - green  : detected scan staff lines
      - cyan   : detected leftmost / rightmost barlines (NOT used for the fit)
      - red    : XML-derived staff lines transformed via per-system similarity
      - magenta: XML noteheads transformed via per-system similarity (small "+")
      - orange : scan noteheads detected via matchTemplate (filled = solid,
                 open = unfilled circle)
    """
    if deskewed.ndim == 2:
        canvas = cv2.cvtColor(deskewed, cv2.COLOR_GRAY2BGR)
    else:
        canvas = deskewed.copy()

    # Green: detected scan staff lines.
    for band in scan_systems:
        for sl in band.staff_lines:
            cv2.line(
                canvas,
                (int(sl.x_left), int(sl.y)),
                (int(sl.x_right), int(sl.y)),
                (0, 200, 0), 1,
            )
        # Cyan: detected barlines (informational only).
        cv2.line(
            canvas,
            (int(band.left_barline_x), band.top_y),
            (int(band.left_barline_x), band.bot_y),
            (255, 255, 0), 2,
        )
        cv2.line(
            canvas,
            (int(band.right_barline_x), band.top_y),
            (int(band.right_barline_x), band.bot_y),
            (255, 255, 0), 2,
        )
        cv2.putText(
            canvas, f"S{band.system_idx + 1}",
            (int(band.left_barline_x) + 6, band.top_y - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
        )

    # Red dashed: XML-derived lines transformed via per-system similarity.
    for sys_idx, lines in enumerate(svg_systems):
        if sys_idx >= len(transforms) or transforms[sys_idx] is None:
            continue
        M = transforms[sys_idx]
        for sl in lines:
            p1 = np.array([sl.x_left, sl.y, 1.0])
            p2 = np.array([sl.x_right, sl.y, 1.0])
            t1 = M @ p1
            t2 = M @ p2
            _draw_dashed_line(
                canvas,
                (t1[0], t1[1]),
                (t2[0], t2[1]),
                (0, 0, 255), 1, dash=12,
            )

    # Magenta: XML noteheads transformed via per-system similarity.
    if xml_noteheads:
        for nh in xml_noteheads:
            if nh.system_idx >= len(transforms) or transforms[nh.system_idx] is None:
                continue
            M = transforms[nh.system_idx]
            p = np.array([nh.x_xml, nh.y_xml, 1.0])
            t = M @ p
            x, y = int(round(t[0])), int(round(t[1]))
            # small "+" mark (4-pixel arms)
            cv2.line(canvas, (x - 4, y), (x + 4, y), (255, 0, 200), 1)
            cv2.line(canvas, (x, y - 4), (x, y + 4), (255, 0, 200), 1)

    # Orange: scan-detected noteheads. Filled = solid disk, open = ring.
    if detected_noteheads:
        for d in detected_noteheads:
            x, y = int(round(d.x)), int(round(d.y))
            if d.note_type == "filled":
                cv2.circle(canvas, (x, y), 4, (0, 140, 255), -1)
            else:
                cv2.circle(canvas, (x, y), 5, (0, 140, 255), 2)

    return canvas


# --- entry point ------------------------------------------------------------


def main(scan_path: str, musicxml_path: str, output_path: str) -> int:
    scan = cv2.imread(scan_path)
    if scan is None:
        print(f"FATAL: cannot read scan: {scan_path}")
        return 2

    print(f"scan      : {scan_path} ({scan.shape[1]}×{scan.shape[0]})")
    print(f"musicxml  : {musicxml_path}")
    deskewed, angle = deskew(scan)
    print(f"deskew    : {angle:+.3f} deg")

    scan_systems = detect_systems(deskewed)
    print(f"systems   : {len(scan_systems)}")
    for band in scan_systems:
        print(
            f"  S{band.system_idx + 1}: "
            f"y=[{band.top_y}, {band.bot_y}]  "
            f"barlines x=[{band.left_barline_x:.1f}, {band.right_barline_x:.1f}]  "
            f"lines={len(band.staff_lines)}  "
            f"staves={len(set(sl.staff_idx for sl in band.staff_lines))}"
        )

    svg_systems = compute_svg_staff_lines(
        musicxml_path, n_systems=len(scan_systems) or None,
    )
    print(f"xml lines : {sum(len(s) for s in svg_systems)} across {len(svg_systems)} systems")

    # Validation: per-system similarity fit + residual.
    transforms: List[Optional[np.ndarray]] = []
    pair_count = min(len(scan_systems), len(svg_systems))
    print("residuals :")
    for i in range(pair_count):
        M, mean_r, max_r, n = fit_system_transform(svg_systems[i], scan_systems[i])
        transforms.append(M)
        print(
            f"  S{i + 1}: pairs={n:3d}  mean={mean_r:6.2f}px  max={max_r:6.2f}px"
            + ("  OK" if max_r <= 2.0 else "  WARN max>2px")
        )
    while len(transforms) < len(svg_systems):
        transforms.append(None)

    # XML noteheads (for Stage 2 y-correctness check + Stage 3 input).
    xml_heads = extract_xml_noteheads(musicxml_path, svg_systems)
    by_sys: dict = {}
    for nh in xml_heads:
        by_sys.setdefault(nh.system_idx, []).append(nh)
    print("xml notes :")
    for i in range(len(svg_systems)):
        print(f"  S{i + 1}: {len(by_sys.get(i, []))}")

    # Lyric anchors in scan space, indexed by system. Built from the XML
    # <lyric> default-y (positive UP, hence ``staff_top - default_y``) and
    # the parent note's default-x, then warped by the per-system similarity.
    xml_lyrics = extract_xml_lyrics(musicxml_path, svg_systems)
    lyrics_scan_by_sys: dict = {}
    for ly in xml_lyrics:
        if ly.system_idx >= len(transforms) or transforms[ly.system_idx] is None:
            continue
        M = transforms[ly.system_idx]
        p = np.array([ly.x_xml, ly.y_xml, 1.0])
        t = M @ p
        lyrics_scan_by_sys.setdefault(ly.system_idx, []).append((float(t[0]), float(t[1])))
    print("xml lyrics:")
    for i in range(len(svg_systems)):
        print(f"  S{i + 1}: {len(lyrics_scan_by_sys.get(i, []))}")

    # Stage 3a: threshold sweep with the new template + lyric exclusion.
    print("threshold sweep:")
    expected = [len(by_sys.get(i, [])) for i in range(len(svg_systems))]
    sweep_thresholds = (0.50, 0.55, 0.60)
    sweep_results: dict = {}
    for thr in sweep_thresholds:
        per_sys = []
        for band in scan_systems:
            d_no_filter = detect_noteheads(deskewed, band, threshold=thr)
            d_with_filter = detect_noteheads(
                deskewed, band, threshold=thr,
                lyric_anchors_scan=lyrics_scan_by_sys.get(band.system_idx),
            )
            per_sys.append((len(d_no_filter), len(d_with_filter)))
        sweep_results[thr] = per_sys
        cells = "  ".join(
            f"S{i + 1}={f}->{wf}/{exp}"
            for i, ((f, wf), exp) in enumerate(zip(per_sys, expected))
        )
        print(f"  thr={thr:.2f}  {cells}")

    # Pick the threshold whose lyric-filtered counts are within ±2 of XML.
    chosen_thr = None
    for thr in sweep_thresholds:
        in_tol = all(
            abs(wf - exp) <= 2
            for (_f, wf), exp in zip(sweep_results[thr], expected)
        )
        if in_tol:
            chosen_thr = thr
            break

    detected: List[NoteheadDetection] = []
    if chosen_thr is not None:
        print(f"chosen    : threshold {chosen_thr:.2f} (all systems within ±2)")
        for band in scan_systems:
            d = detect_noteheads(
                deskewed, band, threshold=chosen_thr,
                lyric_anchors_scan=lyrics_scan_by_sys.get(band.system_idx),
            )
            detected.extend(d)
            print(
                f"  S{band.system_idx + 1}: {len(d):3d}  "
                f"line_spacing={band.line_spacing_px:.2f}px"
            )
    else:
        # No single threshold satisfied tolerance — render at 0.55 for the viz
        # so the user can see what's happening. Caller prints the sweep above.
        print("chosen    : NONE within tolerance — rendering 0.55 for viz only")
        for band in scan_systems:
            d = detect_noteheads(
                deskewed, band, threshold=0.55,
                lyric_anchors_scan=lyrics_scan_by_sys.get(band.system_idx),
            )
            detected.extend(d)

    viz = visualize_stage1(
        deskewed, scan_systems, svg_systems, transforms, xml_heads, detected,
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, viz)
    print(f"wrote     : {output_path}")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    scan = args[0] if len(args) > 0 else "uploads/1777145439617_madlobeli_var_sruli.png"
    xml = args[1] if len(args) > 1 else "06 Madlobeli var.musicxml"
    out = args[2] if len(args) > 2 else "cache/stage1_debug.png"
    raise SystemExit(main(scan, xml, out))
