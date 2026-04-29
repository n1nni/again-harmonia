"""IoU-based notehead refinement.

Given a coarse (x, y) estimate for a notehead center, search a small window
for the offset that maximizes overlap between a filled ellipse mask and the
dark pixels on the binarized scan. Binarized scans are cached across calls.
"""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np


_scan_cache: Dict[str, np.ndarray] = {}


def get_binary_scan(scan_path: str) -> np.ndarray:
    """Load and cache a binarized scan (noteheads/ink = 255)."""
    cached = _scan_cache.get(scan_path)
    if cached is not None:
        return cached
    img = cv2.imread(scan_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {scan_path}")
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _scan_cache[scan_path] = binary
    return binary


def clear_cache() -> None:
    _scan_cache.clear()


def notehead_iou(
    binary_scan: np.ndarray,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
) -> float:
    """IoU between a filled ellipse mask at (cx, cy) and dark pixels locally."""
    h, w = binary_scan.shape[:2]
    # A bit of padding around the ellipse.
    pad_x = max(1.0, rx * 2.0)
    pad_y = max(1.0, ry * 2.0)
    x1 = int(max(0, cx - pad_x))
    y1 = int(max(0, cy - pad_y))
    x2 = int(min(w, cx + pad_x))
    y2 = int(min(h, cy + pad_y))

    if x2 - x1 <= 2 or y2 - y1 <= 2:
        return 0.0

    patch = binary_scan[y1:y2, x1:x2]
    mask = np.zeros_like(patch)

    local_cx = int(cx) - x1
    local_cy = int(cy) - y1
    cv2.ellipse(
        mask,
        (local_cx, local_cy),
        (max(1, int(rx)), max(1, int(ry))),
        0, 0, 360, 255, -1,
    )

    patch_on = patch == 255
    mask_on = mask == 255
    intersection = int(np.logical_and(patch_on, mask_on).sum())
    union = int(np.logical_or(patch_on, mask_on).sum())
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def optimize_note_x(
    binary_scan: np.ndarray,
    coarse_x: float,
    coarse_y: float,
    rx: float,
    ry: float,
    search_radius: float = 20.0,
    step: float = 0.5,
) -> Tuple[float, float]:
    """Grid-search x offset maximizing notehead IoU. Returns (best_x, best_iou)."""
    best_iou = -1.0
    best_x = coarse_x
    dx = -search_radius
    while dx <= search_radius + 1e-9:
        iou = notehead_iou(binary_scan, coarse_x + dx, coarse_y, rx, ry)
        if iou > best_iou:
            best_iou = iou
            best_x = coarse_x + dx
        dx += step
    return best_x, max(0.0, best_iou)


def optimize_note_xy(
    binary_scan: np.ndarray,
    coarse_x: float,
    coarse_y: float,
    rx: float,
    ry: float,
    search_radius_x: float = 20.0,
    search_radius_y: float = 6.0,
    step: float = 1.0,
) -> Tuple[float, float, float]:
    """Grid-search both x and y offsets. Returns (best_x, best_y, best_iou)."""
    best_iou = -1.0
    best_x = coarse_x
    best_y = coarse_y
    dy = -search_radius_y
    while dy <= search_radius_y + 1e-9:
        dx = -search_radius_x
        while dx <= search_radius_x + 1e-9:
            iou = notehead_iou(binary_scan, coarse_x + dx, coarse_y + dy, rx, ry)
            if iou > best_iou:
                best_iou = iou
                best_x = coarse_x + dx
                best_y = coarse_y + dy
            dx += step
        dy += step
    return best_x, best_y, max(0.0, best_iou)
