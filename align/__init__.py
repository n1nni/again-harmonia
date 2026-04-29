"""Harmonia alignment pipeline."""

from .staff_detector import detect_systems, detect_barlines, Staff, StaffLine, System
from .svg_parser import parse_osmd_svg, SVGNote, SVGStaff, SVGSystem
from .aligner import align
from .notehead_detector import detect_noteheads_in_band, assign_notes_to_candidates

__all__ = [
    "detect_systems",
    "detect_barlines",
    "Staff",
    "StaffLine",
    "System",
    "parse_osmd_svg",
    "SVGNote",
    "SVGStaff",
    "SVGSystem",
    "align",
    "detect_noteheads_in_band",
    "assign_notes_to_candidates",
]
