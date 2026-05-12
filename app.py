"""Harmonia — Flask backend.

Serves the split-view UI, exposes the static MusicXML file, accepts scan
uploads, and runs the alignment pipeline that maps OSMD-rendered notes to
pixel coordinates on the uploaded scan.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from flask import Flask, jsonify, request, send_from_directory, abort
from flask_cors import CORS
from werkzeug.utils import secure_filename

from align.aligner import align as run_alignment
from align.staff_detector import detect_barlines, detect_systems
from align.svg_parser import parse_osmd_svg


BASE_DIR = Path(__file__).parent.resolve()
UPLOAD_DIR = BASE_DIR / "uploads"
CACHE_DIR = BASE_DIR / "cache"
OMR_MUSICXML_DIR = BASE_DIR / "omr-musicxmls"
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

OMR_MUSICXML_DIR.mkdir(parents=True, exist_ok=True)

# The MusicXML file is swappable; override via env var if needed.
# This is the FALLBACK score used when no OMR job has been submitted yet
# (e.g. on first page load before any scan upload).
MUSICXML_FILENAME = os.environ.get(
    "HARMONIA_MUSICXML",
    "06 Madlobeli var.musicxml",
)
MUSICXML_PATH = BASE_DIR / MUSICXML_FILENAME

# External OMR service that turns an image into a rectified PNG + MusicXML.
# See OMR_Iliauni/API.md for the response schema.
OMR_API_URL = os.environ.get("OMR_API_URL", "http://127.0.0.1:5000").rstrip("/")
# OMR pipeline can take 2-10 s on CPU; allow plenty of headroom.
OMR_TIMEOUT = int(os.environ.get("OMR_TIMEOUT", "600"))

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
    # When the frontend has just uploaded a scan, it passes ?job=<stamp> to
    # fetch the MusicXML the OMR API produced for that scan. With no `job`
    # arg we serve the bundled fallback score (used on initial page load).
    job = request.args.get("job")
    if job:
        path = OMR_MUSICXML_DIR / f"{job}.musicxml"
        if not path.exists():
            abort(404, description=f"MusicXML for job {job} not found")
        return path.read_text(encoding="utf-8"), 200, {
            "Content-Type": "application/xml; charset=utf-8",
        }
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

    # Stamp the filename so repeated uploads don't collide. The same stamp
    # is reused as the OMR "job id" we expose to the frontend, so the
    # rectified PNG, the original upload, and the recognized MusicXML are
    # all addressable via that one identifier.
    stamp = str(int(time.time() * 1000))
    safe = secure_filename(file.filename)
    original_name = f"{stamp}_orig_{safe}"
    original_path = UPLOAD_DIR / original_name
    file.save(original_path)

    # Forward to the OMR API (POST /process) and read back the full
    # envelope: rectified PNG (base64) + detections + MusicXML.
    # Stream the upload so we don't double-buffer the file in memory.
    try:
        with open(original_path, "rb") as fh:
            resp = requests.post(
                f"{OMR_API_URL}/process",
                files={"image": (safe, fh, file.mimetype or "image/png")},
                timeout=OMR_TIMEOUT,
                stream=True,
            )
        resp.raise_for_status()
        omr_data = resp.json()
    except requests.exceptions.RequestException as exc:
        traceback.print_exc()
        return jsonify(
            error=f"OMR API call failed at {OMR_API_URL}/process: {exc}",
        ), 502

    if "rectified_image_b64" not in omr_data or "xml" not in omr_data:
        return jsonify(
            error="OMR response missing rectified_image_b64 or xml",
            payload_keys=list(omr_data.keys()),
        ), 502

    # Rectified PNG goes into uploads/ so the existing /uploads/<file>
    # route serves it; this is the image the frontend will overlay dots on.
    rectified_name = f"{stamp}_rectified.png"
    rectified_path = UPLOAD_DIR / rectified_name
    rectified_path.write_bytes(base64.b64decode(omr_data["rectified_image_b64"]))

    # Dev override: when omrResponse.musicxml sits in the repo root, use
    # its contents in place of the OMR-returned XML. Lets you iterate on
    # the XML side without re-running OMR. Delete the file to fall back
    # to the real response.
    override_xml_path = BASE_DIR / "omrResponse.musicxml"
    if override_xml_path.exists():
        omr_data["xml"] = override_xml_path.read_text(encoding="utf-8")

    # Recognized MusicXML lands in omr-musicxmls/, retrievable via
    # /api/musicxml?job=<stamp>.
    xml_path = OMR_MUSICXML_DIR / f"{stamp}.musicxml"
    xml_path.write_text(omr_data["xml"], encoding="utf-8")

    return jsonify(
        filename=rectified_name,
        url=f"/uploads/{rectified_name}",
        original_filename=original_name,
        original_url=f"/uploads/{original_name}",
        musicxml_url=f"/api/musicxml?job={stamp}",
        omr_notes_url=f"/api/omr_notes?job={stamp}",
        job_id=omr_data.get("job_id") or stamp,
    )


# Notehead classes the OMR pipeline emits (DeepScores naming). We dot only
# these — augmentation dots, slurs, beams, clefs, time/key sigs, etc. are
# excluded even though they share the same `det_XXXX` id space.
_NOTEHEAD_CLASSES = {
    "noteheadblack",
    "noteheadhalf",
    "noteheadwhole",
    "noteheaddoublewhole",
    "noteheadblacksmall",
    "noteheadhalfsmall",
    "noteheadwholesmall",
}


def _pitch_string(note_el):
    """Build a human-readable pitch label like 'F#4' from a <note> element."""
    pitch_el = note_el.find("pitch")
    if pitch_el is None:
        return "?"
    step_el = pitch_el.find("step")
    octave_el = pitch_el.find("octave")
    alter_el = pitch_el.find("alter")
    step = step_el.text if step_el is not None and step_el.text else "?"
    octave = octave_el.text if octave_el is not None and octave_el.text else "?"
    alter = alter_el.text if alter_el is not None and alter_el.text else None
    acc = {"1": "#", "2": "##", "-1": "b", "-2": "bb"}.get(alter, "")
    return f"{step}{acc}{octave}"


@app.route("/api/omr_notes")
def omr_notes():
    """Return notehead positions for a previously-uploaded scan.

    Reads the OMR-produced MusicXML (saved at upload time), parses the
    `omr-coordinates` JSON embedded in `<miscellaneous>`, then walks every
    `<note id="det_XXXX">` and emits the dot for any note whose detection
    class is a notehead. Coordinates are in rectified-image pixel space —
    same coordinate system as the PNG the frontend is displaying.
    """
    job = request.args.get("job")
    if not job:
        return jsonify(error="job query param required"), 400

    xml_path = OMR_MUSICXML_DIR / f"{job}.musicxml"
    if not xml_path.exists():
        return jsonify(error=f"MusicXML for job {job} not found"), 404

    try:
        root = ET.parse(str(xml_path)).getroot()
    except ET.ParseError as exc:
        return jsonify(error=f"MusicXML parse error: {exc}"), 500

    coords_by_id = {}
    image_width = None
    image_height = None
    for field in root.iter("miscellaneous-field"):
        name = field.get("name")
        text = field.text or ""
        if name == "omr-coordinates" and text:
            try:
                arr = json.loads(text)
                coords_by_id = {rec["id"]: rec for rec in arr if "id" in rec}
            except (json.JSONDecodeError, TypeError):
                pass
        elif name == "omr-image-width" and text:
            try:
                image_width = int(text)
            except ValueError:
                pass
        elif name == "omr-image-height" and text:
            try:
                image_height = int(text)
            except ValueError:
                pass

    notes = []
    # Walk per-part / per-measure so we can emit the same xml_key the JS
    # renderer uses internally — f"{part_idx}_{measure_idx}_{visible_idx}".
    # Visible-idx counts non-rest <note> elements within a measure (chord
    # members included), matching renderer.js _rebuildXmlNoteIndex
    # (visibleIdx increments per non-rest, regardless of <chord/> child).
    for part_idx, part in enumerate(root.findall("part")):
        for measure_idx, measure in enumerate(part.findall("measure")):
            visible_idx = 0
            for note_el in measure.findall("note"):
                # Rests don't get a key and don't increment visible_idx.
                if note_el.find("rest") is not None:
                    continue
                this_visible_idx = visible_idx
                visible_idx += 1

                det_id = note_el.get("id")
                if not det_id:
                    continue
                rec = coords_by_id.get(det_id)
                if not rec:
                    continue
                cls = (rec.get("class") or "").lower()
                if cls not in _NOTEHEAD_CLASSES:
                    continue

                type_el = note_el.find("type")
                duration = (
                    type_el.text if type_el is not None and type_el.text
                    else "quarter"
                )

                notes.append({
                    "note_id": det_id,
                    # The xml_key here is the bridge to OSMD's GraphicSheet
                    # walk; the frontend uses it to look up the note's
                    # rendered svgX/svgY/svgRx/svgRy for glyph cropping.
                    "xml_key": f"{part_idx}_{measure_idx}_{this_visible_idx}",
                    "part_idx": part_idx,
                    "measure_idx": measure_idx,
                    "visible_idx": this_visible_idx,
                    # The overlay renderer expects scan_x/scan_y in image
                    # pixels; OMR's cx/cy are exactly that for the rectified
                    # PNG.
                    "scan_x": float(rec.get("cx", 0)),
                    "scan_y": float(rec.get("cy", 0)),
                    "x1": float(rec.get("x1", 0)),
                    "y1": float(rec.get("y1", 0)),
                    "x2": float(rec.get("x2", 0)),
                    "y2": float(rec.get("y2", 0)),
                    "class": rec.get("class"),
                    "confidence": float(rec.get("conf", 0.0)),
                    "part_id": rec.get("part_id"),
                    "staff_in_part": rec.get("staff_in_part"),
                    "pitch": _pitch_string(note_el),
                    "duration": duration,
                })

    return jsonify(
        job_id=job,
        image_width=image_width,
        image_height=image_height,
        count=len(notes),
        notes=notes,
    )


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
    # Port 5001 because the OMR API squats on 5000 by default.
    app.run(host="127.0.0.1", port=5001, debug=True)
