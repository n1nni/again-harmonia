"""Harmonia — Flask backend.

Serves the split-view UI, exposes the static MusicXML file, accepts scan
uploads, and runs the alignment pipeline that maps OSMD-rendered notes to
pixel coordinates on the uploaded scan.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory, abort
from flask_cors import CORS
from werkzeug.utils import secure_filename

from align.aligner import align as run_alignment
from align.staff_detector import detect_barlines, detect_systems
from align.svg_parser import parse_osmd_svg


BASE_DIR = Path(__file__).parent.resolve()
UPLOAD_DIR = BASE_DIR / "uploads"
CACHE_DIR = BASE_DIR / "cache"
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# The MusicXML file is swappable; override via env var if needed.
MUSICXML_FILENAME = os.environ.get(
    "HARMONIA_MUSICXML",
    "06 Madlobeli var.musicxml",
)
MUSICXML_PATH = BASE_DIR / MUSICXML_FILENAME

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"}


app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR),
)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB uploads


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    # Keep things simple — serve templates/index.html directly.
    return send_from_directory(str(TEMPLATE_DIR), "index.html")


@app.route("/api/musicxml")
def musicxml():
    if not MUSICXML_PATH.exists():
        abort(404, description=f"MusicXML file not found: {MUSICXML_PATH.name}")
    with open(MUSICXML_PATH, "r", encoding="utf-8") as f:
        return f.read(), 200, {"Content-Type": "application/xml; charset=utf-8"}


@app.route("/api/upload", methods=["POST"])
def upload():
    if "scan" not in request.files:
        return jsonify(error="No file part named 'scan'"), 400
    file = request.files["scan"]
    if not file.filename:
        return jsonify(error="Empty filename"), 400
    if not _allowed(file.filename):
        return jsonify(error="Unsupported file type"), 400

    # Stamp the filename so repeated uploads don't collide.
    stamp = str(int(time.time() * 1000))
    safe = secure_filename(file.filename)
    stored_name = f"{stamp}_{safe}"
    dest = UPLOAD_DIR / stored_name
    file.save(dest)
    return jsonify(filename=stored_name, url=f"/uploads/{stored_name}")


@app.route("/uploads/<path:filename>")
def uploaded(filename: str):
    return send_from_directory(str(UPLOAD_DIR), filename)


@app.route("/api/detect_staves", methods=["POST"])
def detect_staves():
    payload = request.get_json(silent=True) or {}
    filename = payload.get("filename")
    if not filename:
        return jsonify(error="filename required"), 400
    scan_path = UPLOAD_DIR / filename
    if not scan_path.exists():
        return jsonify(error="scan not found"), 404

    try:
        systems = detect_systems(str(scan_path))
    except Exception as exc:
        traceback.print_exc()
        return jsonify(error=f"staff detection failed: {exc}"), 500

    return jsonify(
        systems=[
            {
                "system_idx": i,
                "top_y": s.top_y,
                "bot_y": s.bot_y,
                "left_x": s.left_x,
                "right_x": s.right_x,
                "staves": [
                    {
                        "staff_idx": j,
                        "top_y": st.top_y,
                        "bot_y": st.bot_y,
                        "left_x": st.left_x,
                        "right_x": st.right_x,
                        "line_spacing": st.line_spacing,
                        "lines": [line.y for line in st.lines],
                    }
                    for j, st in enumerate(s.staves)
                ],
            }
            for i, s in enumerate(systems)
        ]
    )


@app.route("/api/export_svg", methods=["POST"])
def export_svg():
    payload = request.get_json(silent=True) or {}
    svg_data = payload.get("svg_data")
    if not svg_data:
        return jsonify(error="svg_data required"), 400

    digest = hashlib.sha1(svg_data.encode("utf-8")).hexdigest()[:16]
    dest = CACHE_DIR / f"osmd_{digest}.svg"
    dest.write_text(svg_data, encoding="utf-8")
    return jsonify(cached_as=dest.name, bytes=len(svg_data))


@app.route("/api/align", methods=["POST"])
def align_endpoint():
    payload = request.get_json(silent=True) or {}
    filename = payload.get("filename")
    svg_data = payload.get("svg_data")
    osmd_notes = payload.get("osmd_notes") or []
    osmd_systems = payload.get("osmd_systems") or []
    osmd_measures = payload.get("osmd_measures") or []
    natural_width = payload.get("natural_width")
    natural_height = payload.get("natural_height")

    if not filename:
        return jsonify(error="filename required"), 400

    scan_path = UPLOAD_DIR / filename
    if not scan_path.exists():
        return jsonify(error="scan not found"), 404

    try:
        svg_systems = parse_osmd_svg(
            svg_data or "",
            osmd_notes=osmd_notes,
            osmd_systems=osmd_systems,
            osmd_measures=osmd_measures,
        )
    except Exception as exc:
        traceback.print_exc()
        return jsonify(error=f"SVG parse failed: {exc}"), 500

    try:
        scan_systems = detect_systems(str(scan_path))
    except Exception as exc:
        traceback.print_exc()
        return jsonify(error=f"staff detection failed: {exc}"), 500

    if not scan_systems:
        return jsonify(error="no systems detected in scan", notes=[]), 200
    if not svg_systems:
        return jsonify(error="no systems parsed from SVG", notes=[]), 200

    try:
        aligned = run_alignment(str(scan_path), svg_systems, scan_systems)
    except Exception as exc:
        traceback.print_exc()
        return jsonify(error=f"alignment failed: {exc}"), 500

    # Build scan-side measure boxes the same way the aligner does — split
    # each scan staff into N+1 measures using the detected barlines as
    # interior boundaries. The frontend uses these to render the
    # regenerated SVG overlay measure-by-measure.
    scan_measures = _build_scan_measures(
        str(scan_path), scan_systems, svg_systems,
    )

    # Persist alignment result alongside the scan so follow-up edits can
    # reference it.
    result_path = CACHE_DIR / f"{Path(filename).stem}_alignment.json"
    try:
        result_path.write_text(json.dumps({"notes": aligned}, indent=2), encoding="utf-8")
    except OSError:
        pass

    return jsonify(
        notes=aligned,
        natural_width=natural_width,
        natural_height=natural_height,
        scan_systems=len(scan_systems),
        svg_systems=len(svg_systems),
        scan_measures=scan_measures,
    )


def _build_scan_measures(scan_path, scan_systems, svg_systems):
    """For each scan (system, staff), split into measures by barlines.

    Pairs the resulting boxes positionally with the corresponding SVG
    measures (sorted by left_x within their staff) so the frontend can
    look them up by (system_idx, staff_idx, measure_idx). Falls back to a
    single measure per staff if no barlines are detected.
    """
    out = []
    pair_count = min(len(scan_systems), len(svg_systems))
    for sys_idx in range(pair_count):
        scan_sys = scan_systems[sys_idx]
        svg_sys = svg_systems[sys_idx]

        # Group SVG measures by staff_idx in left-to-right order.
        svg_by_staff = {}
        for m in svg_sys.measures:
            svg_by_staff.setdefault(m.staff_idx, []).append(m)
        for lst in svg_by_staff.values():
            lst.sort(key=lambda m: m.left_x)

        try:
            barline_xs = detect_barlines(scan_path, scan_sys)
        except Exception:
            barline_xs = []
        sorted_barlines = sorted(float(x) for x in barline_xs)

        staff_count = len(scan_sys.staves)
        for staff_idx in range(staff_count):
            scan_staff = scan_sys.staves[staff_idx]
            svg_measures = svg_by_staff.get(staff_idx, [])

            interior = [
                x for x in sorted_barlines
                if scan_staff.left_x < x < scan_staff.right_x
            ]
            boundaries = (
                [float(scan_staff.left_x)] + interior + [float(scan_staff.right_x)]
            )
            scan_boxes = []
            for i in range(len(boundaries) - 1):
                lx = boundaries[i]
                rx = boundaries[i + 1]
                if rx - lx < 2.0:
                    continue
                scan_boxes.append((lx, rx))

            n = min(len(svg_measures), len(scan_boxes))
            for i in range(n):
                lx, rx = scan_boxes[i]
                sm = svg_measures[i]
                out.append({
                    "system_idx": sys_idx,
                    "staff_idx": staff_idx,
                    "measure_idx": int(sm.measure_idx),
                    "left_x": float(lx),
                    "right_x": float(rx),
                    "top_y": float(scan_staff.top_y),
                    "bot_y": float(scan_staff.bot_y),
                })
    return out


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
