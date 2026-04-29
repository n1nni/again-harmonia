"""Notehead detection on a binarized scan.

For each detected staff, finds candidate notehead centers by:
  1. Cropping a band around the staff (with vertical padding for ledger lines).
  2. Subtracting the horizontal staff-line strokes so noteheads stand alone.
  3. Morphological opening with a small ellipse to drop thin stems and small
     noise (lyrics, dots, accidentals' thin strokes).
  4. Connected-component analysis, filtered by area + aspect ratio.

The candidates are coarse pixel coordinates of likely noteheads — used by
the aligner as snap targets for OSMD-derived note positions.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def detect_noteheads_in_band(
    binary_scan: np.ndarray,
    staff_top_y: int,
    staff_bot_y: int,
    staff_left_x: int,
    staff_right_x: int,
    line_spacing: float,
    pad_factor: float = 4.0,
    min_circularity: float = 0.55,
    min_solidity: float = 0.85,
) -> List[Tuple[float, float, float]]:
    """Return [(cx, cy, area), ...] notehead candidates in image coords.

    Pipeline per band:
      1. Strip horizontal staff-line strokes.
      2. Close with an ellipse > the hole in a half/whole notehead so hollow
         noteheads become solid blobs (otherwise they look like rings and
         get rejected by the circularity test).
      3. Open with a small ellipse to drop thin stems and noise.
      4. findContours + per-blob shape filtering: area envelope, aspect
         ratio, circularity (4πA/P²), and solidity (A / convex-hull area).

    Strict filtering removes clef loops (low circularity), the "3" tuplet
    digit (out of size envelope or low circularity), slur fragments
    (elongated aspect ratio), and lyrics (low circularity / solidity).
    """
    if binary_scan is None or binary_scan.size == 0:
        return []
    if line_spacing <= 0:
        return []

    h, w = binary_scan.shape[:2]
    pad_y = int(round(line_spacing * pad_factor))
    y1 = max(0, int(staff_top_y - pad_y))
    y2 = min(h, int(staff_bot_y + pad_y))
    x1 = max(0, int(staff_left_x - 4))
    x2 = min(w, int(staff_right_x + 4))
    if y2 - y1 <= 4 or x2 - x1 <= 4:
        return []

    band = binary_scan[y1:y2, x1:x2].copy()

    # Strip the staff lines.
    h_kernel_w = max(int(line_spacing * 5), 12)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_w, 1))
    h_lines = cv2.morphologyEx(band, cv2.MORPH_OPEN, h_kernel)
    no_lines = cv2.subtract(band, h_lines)

    # Fill the hole in hollow noteheads (half/whole notes) so they survive
    # the circularity test downstream. The closing kernel must exceed the
    # hole diameter (~1 line-spacing) but stay small enough not to fuse
    # adjacent noteheads.
    fill_size = max(3, int(round(line_spacing * 1.1)))
    if fill_size % 2 == 0:
        fill_size += 1
    fill_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fill_size, fill_size))
    filled = cv2.morphologyEx(no_lines, cv2.MORPH_CLOSE, fill_kernel)

    # Drop thin stems: open with a small ellipse.
    blob_size = max(2, int(round(line_spacing * 0.7)))
    if blob_size % 2 == 0:
        blob_size += 1
    blob_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (blob_size, blob_size))
    blobs = cv2.morphologyEx(filled, cv2.MORPH_OPEN, blob_kernel)

    contours, _ = cv2.findContours(blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    expected_area = (line_spacing * 1.2) * (line_spacing * 1.0)
    min_area = max(4.0, expected_area * 0.30)
    max_area = expected_area * 3.0
    max_dim = line_spacing * 2.8

    out: List[Tuple[float, float, float]] = []
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < min_area or area > max_area:
            continue
        perim = float(cv2.arcLength(c, True))
        if perim <= 1.0:
            continue
        # 4πA/P² — perfect circle = 1.0; ellipses with axis ratio 2 ≈ 0.79.
        circularity = 4.0 * np.pi * area / (perim * perim)
        if circularity < min_circularity:
            continue
        x, y, ww, hh = cv2.boundingRect(c)
        if ww > max_dim or hh > max_dim:
            continue
        ar = float(ww) / max(1.0, float(hh))
        if ar < 0.6 or ar > 2.0:
            continue
        # Solidity = area / convex hull area; rejects shapes with concavities
        # like the "3" digit, dynamic markings, brackets.
        hull = cv2.convexHull(c)
        hull_area = float(cv2.contourArea(hull))
        if hull_area > 0:
            solidity = area / hull_area
            if solidity < min_solidity:
                continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        out.append((float(cx + x1), float(cy + y1), area))

    return out


def build_pitch_grid(
    line_ys: Sequence[float],
    line_spacing: float,
    ledger_count: int = 5,
) -> List[float]:
    """Return the sorted list of valid notehead y positions for a staff.

    Includes the 5 staff lines, the 4 spaces between them, plus N ledger
    lines and ledger-spaces above and below.
    """
    if not line_ys or line_spacing <= 0:
        return []
    sorted_lines = sorted(float(y) for y in line_ys)
    half = line_spacing / 2.0
    grid: List[float] = []
    top = sorted_lines[0] - ledger_count * line_spacing
    bot = sorted_lines[-1] + ledger_count * line_spacing
    n_steps = int(round((bot - top) / half))
    for k in range(n_steps + 1):
        grid.append(top + k * half)
    return grid


def snap_y_to_grid(y: float, grid: Sequence[float]) -> float:
    """Snap y to the nearest grid position. If grid is empty, return y."""
    if not grid:
        return float(y)
    return float(min(grid, key=lambda gy: abs(gy - y)))


def filter_candidates_to_pitch_grid(
    candidates: Sequence[Tuple[float, float, float]],
    line_ys: Sequence[float],
    line_spacing: float,
    ledger_count: int = 5,
    tol_factor: float = 0.40,
) -> List[Tuple[float, float, float]]:
    """Drop candidates whose y is not near a valid notehead pitch position.

    Slurs, tuplet numbers, and dynamics fail this test.
    """
    if not candidates or not line_ys or line_spacing <= 0:
        return list(candidates)

    grid = build_pitch_grid(line_ys, line_spacing, ledger_count)
    tol = line_spacing * tol_factor
    out: List[Tuple[float, float, float]] = []
    for cx, cy, area in candidates:
        nearest = min(grid, key=lambda gy: abs(gy - cy))
        if abs(nearest - cy) <= tol:
            out.append((cx, cy, area))
    return out


def assign_notes_to_candidates(
    note_positions: Sequence[Tuple[float, float]],
    candidates: Sequence[Tuple[float, float, float]],
    radius_x: float,
    radius_y: float,
) -> Dict[int, Tuple[float, float, float]]:
    """Globally assign each note to at most one candidate (one-to-one).

    Uses the Hungarian algorithm (linear_sum_assignment) on a weighted
    distance cost. With y locked to the pitch grid by the caller,
    ``radius_y`` should be small (≤ half a line-spacing) so a candidate
    must sit at the same pitch as the note to be eligible.

    Returns ``{note_idx: (cx, cy, score)}`` only for notes that landed a
    real match within the radius. Notes without a feasible candidate are
    omitted (caller should fall back to IoU x-search for those).
    """
    if not note_positions or not candidates:
        return {}

    n_notes = len(note_positions)
    n_cands = len(candidates)
    INF = 1e9

    cost = np.full((n_notes, n_cands), INF, dtype=np.float64)
    rx = max(radius_x, 1e-3)
    ry = max(radius_y, 1e-3)
    for i, (nx, ny) in enumerate(note_positions):
        for j, (cx, cy, _area) in enumerate(candidates):
            dx = abs(cx - nx)
            dy = abs(cy - ny)
            if dx > radius_x or dy > radius_y:
                continue
            # X dominates the cost since y is essentially fixed already.
            cost[i, j] = (dx / rx) ** 2 + (2.0 * dy / ry) ** 2

    row_ind, col_ind = linear_sum_assignment(cost)

    out: Dict[int, Tuple[float, float, float]] = {}
    for i, j in zip(row_ind, col_ind):
        if cost[i, j] >= INF / 2:
            continue
        cx, cy, _ = candidates[j]
        out[int(i)] = (float(cx), float(cy), float(cost[i, j]))
    return out
