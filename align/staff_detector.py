"""Staff / system / barline detection on the scanned score."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np
from scipy.signal import find_peaks


@dataclass
class StaffLine:
    y: int  # row in image pixels


@dataclass
class Staff:
    lines: List[StaffLine]
    top_y: int
    bot_y: int
    left_x: int
    right_x: int
    line_spacing: float


@dataclass
class System:
    staves: List[Staff]
    top_y: int
    bot_y: int
    left_x: int
    right_x: int


# --- internal helpers -------------------------------------------------------


def _binarize(img: np.ndarray) -> np.ndarray:
    """Otsu binarize with staff strokes = 255."""
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _horizontal_strokes(binary: np.ndarray, min_run_frac: float = 0.3) -> np.ndarray:
    """Morphological opening to keep only long horizontal strokes."""
    w = max(20, int(binary.shape[1] * min_run_frac))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w, 1))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def _cluster_close_peaks(peaks: List[int], max_gap: int = 2) -> List[int]:
    """Collapse runs of adjacent peak rows into a single row (the mean)."""
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


def _group_lines_into_staves(
    lines: List[int], spacing_tol: float = 0.35
) -> List[List[int]]:
    """Form 5-line staves from detected staff-line rows.

    Noteheads often sit directly on staff lines and break the horizontal
    stroke signal, so some of the 5 lines may be missing. Rather than
    requiring 5 consecutive detections, we:
      1. Estimate the staff-line spacing from the tight gaps across all rows.
      2. For each candidate starting row, predict the remaining 4 line
         positions and snap to real detections where available; interpolate
         otherwise. A staff is accepted if at least 3 of the 5 predicted
         positions match a real detection within tolerance.
    """
    if len(lines) < 3:
        return []

    lines = sorted(lines)

    # Estimate the base line-to-line spacing: take the median of small gaps
    # (those below the overall gap median) — these are the intra-staff gaps.
    gaps = [lines[i + 1] - lines[i] for i in range(len(lines) - 1) if lines[i + 1] > lines[i]]
    if not gaps:
        return []
    median_gap = float(np.median(gaps))
    small = [g for g in gaps if g <= median_gap * 1.5]
    target_spacing = float(np.median(small)) if small else median_gap
    if target_spacing <= 0:
        return []

    tol = max(1.5, target_spacing * spacing_tol)

    staves: List[List[int]] = []
    used: set = set()

    for start_idx in range(len(lines)):
        if start_idx in used:
            continue
        top = lines[start_idx]
        predicted = [top + k * target_spacing for k in range(5)]

        matched_indices: List[int] = []
        final_rows: List[int] = []
        for k, pred in enumerate(predicted):
            best_j = -1
            best_dist = tol
            for j, ln in enumerate(lines):
                if j in used or j in matched_indices:
                    continue
                d = abs(ln - pred)
                if d < best_dist:
                    best_dist = d
                    best_j = j
            if best_j >= 0:
                matched_indices.append(best_j)
                final_rows.append(lines[best_j])
            else:
                final_rows.append(int(round(pred)))

        # Require the staff's anchor plus at least 2 more real detections.
        if len(matched_indices) >= 3:
            final_rows.sort()
            staves.append(final_rows)
            used.update(matched_indices)

    staves.sort(key=lambda s: s[0])
    return staves


def _group_staves_into_systems(
    staves: List[List[int]], line_spacings: List[float]
) -> List[List[int]]:
    """Cluster staves into systems by vertical gap.

    Uses an adaptive threshold: sort inter-staff gaps and split at the
    largest jump between them — works for choral scores with 3+ staves per
    system (where intra-system gaps can exceed 4× line spacing).
    """
    if not staves:
        return []
    if len(staves) == 1:
        return [[0]]

    gaps: List[int] = []
    for idx in range(1, len(staves)):
        gaps.append(staves[idx][0] - staves[idx - 1][-1])

    mean_spacing = float(np.mean(line_spacings)) if line_spacings else 5.0

    # Classify each gap as intra-system or inter-system.
    # Start from a safe floor (any gap <= ~3× line spacing is almost
    # certainly intra-system), then use the largest single jump in the
    # sorted gap distribution to decide where "inter-system" begins.
    floor = max(1.0, mean_spacing * 3.0)
    ceiling_guess = mean_spacing * 6.0

    if max(gaps) <= ceiling_guess:
        # All staves are roughly equally spaced — treat as one system.
        threshold = max(gaps) + 1
    else:
        # Find the largest absolute jump in the sorted gap distribution.
        # The midpoint of that jump becomes the intra/inter-system threshold.
        sorted_gaps = sorted(gaps)
        best_jump = 0.0
        split_at: float = sorted_gaps[-1]
        for i in range(1, len(sorted_gaps)):
            jump = sorted_gaps[i] - sorted_gaps[i - 1]
            if jump > best_jump:
                best_jump = jump
                split_at = (sorted_gaps[i] + sorted_gaps[i - 1]) / 2.0
        # Guard against a jump with the split landing inside intra-system
        # territory. The threshold must sit above the floor.
        threshold = max(split_at, floor)

    systems: List[List[int]] = [[0]]
    for idx, gap in enumerate(gaps, start=1):
        if gap <= threshold:
            systems[-1].append(idx)
        else:
            systems.append([idx])
    return systems


def _staff_horizontal_extent(
    horizontal: np.ndarray, top_y: int, bot_y: int, min_ink_frac: float = 0.2
) -> Tuple[int, int]:
    """Compute left/right x of a staff via column projection within the band."""
    band = horizontal[top_y : bot_y + 1, :]
    if band.size == 0:
        return 0, horizontal.shape[1] - 1
    col_sum = band.sum(axis=0)
    if col_sum.max() == 0:
        return 0, horizontal.shape[1] - 1
    threshold = col_sum.max() * min_ink_frac
    active = np.where(col_sum > threshold)[0]
    if active.size == 0:
        return 0, horizontal.shape[1] - 1
    return int(active[0]), int(active[-1])


# --- public API --------------------------------------------------------------


def detect_systems(image_path: str) -> List[System]:
    """Detect systems, staves, and staff-lines on a scanned score image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")

    binary = _binarize(img)
    horizontal = _horizontal_strokes(binary)

    row_sum = horizontal.sum(axis=1)
    max_val = row_sum.max()
    if max_val == 0:
        return []

    height_thresh = max_val * 0.3
    peaks, _ = find_peaks(row_sum, height=height_thresh, distance=2)
    clustered = _cluster_close_peaks(peaks.tolist(), max_gap=2)

    staves_lines = _group_lines_into_staves(clustered)
    if not staves_lines:
        return []

    line_spacings: List[float] = []
    for window in staves_lines:
        gaps = [window[k + 1] - window[k] for k in range(4)]
        line_spacings.append(float(sum(gaps) / 4.0))

    systems_idx = _group_staves_into_systems(staves_lines, line_spacings)

    systems: List[System] = []
    for group in systems_idx:
        staves: List[Staff] = []
        for idx in group:
            window = staves_lines[idx]
            spacing = line_spacings[idx]
            left_x, right_x = _staff_horizontal_extent(horizontal, window[0], window[-1])
            staves.append(
                Staff(
                    lines=[StaffLine(y=y) for y in window],
                    top_y=int(window[0]),
                    bot_y=int(window[-1]),
                    left_x=int(left_x),
                    right_x=int(right_x),
                    line_spacing=float(spacing),
                )
            )
        systems.append(
            System(
                staves=staves,
                top_y=int(min(s.top_y for s in staves)),
                bot_y=int(max(s.bot_y for s in staves)),
                left_x=int(min(s.left_x for s in staves)),
                right_x=int(max(s.right_x for s in staves)),
            )
        )
    return systems


def detect_barlines(image_path: str, system: System) -> List[int]:
    """Return x-coordinates of barlines within the given system's band."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")

    binary = _binarize(img)
    band = binary[system.top_y : system.bot_y + 1, system.left_x : system.right_x + 1]
    if band.size == 0:
        return []

    band_height = system.bot_y - system.top_y + 1
    kernel_h = max(5, int(band_height * 0.8))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_h))
    vertical = cv2.morphologyEx(band, cv2.MORPH_OPEN, kernel)

    col_sum = vertical.sum(axis=0)
    if col_sum.max() == 0:
        return []

    # Filter for tall vertical strokes, exclude noteheads/stems.
    threshold = col_sum.max() * 0.5
    min_distance = max(5, int(band_height * 0.2))
    peaks, _ = find_peaks(col_sum, height=threshold, distance=min_distance)

    # Offset back into full-image coordinates.
    return [int(p + system.left_x) for p in peaks.tolist()]
