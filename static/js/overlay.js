// overlay.js — draws interactive note dots over the uploaded scan.

class NoteOverlay {
  constructor(svgEl, imgEl) {
    this.svg = svgEl;
    this.img = imgEl;
    this.notes = [];
    this.dots = new Map();
    this.onNoteClick = null;
    // Click handler for the per-note glyph clones. Receives (omrNote, wrapper)
    // — host wires this up to surface the note's pitch + duration.
    this.onGlyphClick = null;
    this._naturalWidth = 0;
    this._naturalHeight = 0;
    this._regenGroup = null;
    this._staffGroup = null;
    this._glyphsGroup = null;
    this._dotsGroup = null;
    this._detectedSystems = null;
    this._showDetection = true;
    this._showRegen = true;
    this._showGlyphs = true;
    // Cached inputs so we can re-render the regen overlay on window resize.
    this._osmdSvgText = null;
    this._svgMeasures = null;
    this._scanMeasuresForRegen = null;
    // Cached inputs for the per-note glyph overlay (rerendered on resize).
    // Holds the live OSMD container DOM node so getBBox can find the
    // currently-rendered vf-stavenote groups.
    this._osmdContainerForGlyphs = null;
    this._omrNotesForGlyphs = null;

    window.addEventListener('resize', () => this._rescale());
  }

  _ensureGroups() {
    if (!this._regenGroup) {
      this._regenGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      this._regenGroup.setAttribute('class', 'regen-overlay');
      // Insert below the detection rects and the note dots so they remain on top.
      this.svg.insertBefore(this._regenGroup, this.svg.firstChild);
    }
    if (!this._staffGroup) {
      this._staffGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      this._staffGroup.setAttribute('class', 'staff-detection');
      this.svg.appendChild(this._staffGroup);
    }
    // Glyphs sit above staff outlines but below dots — when dots are
    // rendered they stay clickable on top.
    if (!this._glyphsGroup) {
      this._glyphsGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      this._glyphsGroup.setAttribute('class', 'note-glyphs');
      this.svg.appendChild(this._glyphsGroup);
    }
    if (!this._dotsGroup) {
      this._dotsGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      this._dotsGroup.setAttribute('class', 'note-dots');
      this.svg.appendChild(this._dotsGroup);
    }
  }

  setShowDetection(flag) {
    this._showDetection = !!flag;
    if (this._staffGroup) {
      this._staffGroup.style.display = flag ? '' : 'none';
    }
  }

  setShowRegen(flag) {
    this._showRegen = !!flag;
    if (this._regenGroup) {
      this._regenGroup.style.display = flag ? '' : 'none';
    }
  }

  setShowGlyphs(flag) {
    this._showGlyphs = !!flag;
    if (this._glyphsGroup) {
      this._glyphsGroup.style.display = flag ? '' : 'none';
    }
  }

  renderStaffDetection(systems, naturalWidth, naturalHeight) {
    this._detectedSystems = systems;
    this._naturalWidth = naturalWidth;
    this._naturalHeight = naturalHeight;
    this._updateSvgBox();
    this._ensureGroups();

    while (this._staffGroup.firstChild) {
      this._staffGroup.removeChild(this._staffGroup.firstChild);
    }

    const palette = ['#22c55e', '#f97316', '#3b82f6', '#a855f7', '#ec4899', '#14b8a6'];
    const dispW = this.img.offsetWidth;
    const dispH = this.img.offsetHeight;
    const sx = dispW / naturalWidth;
    const sy = dispH / naturalHeight;

    (systems || []).forEach((sys, sysIdx) => {
      const color = palette[sysIdx % palette.length];
      (sys.staves || []).forEach((staff) => {
        const x = staff.left_x * sx;
        const y = staff.top_y * sy;
        const w = (staff.right_x - staff.left_x) * sx;
        const h = (staff.bot_y - staff.top_y) * sy;
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', x);
        rect.setAttribute('y', y);
        rect.setAttribute('width', Math.max(1, w));
        rect.setAttribute('height', Math.max(1, h));
        rect.setAttribute('fill', color);
        rect.setAttribute('fill-opacity', '0.08');
        rect.setAttribute('stroke', color);
        rect.setAttribute('stroke-width', '1.5');
        rect.setAttribute('stroke-opacity', '0.85');
        this._staffGroup.appendChild(rect);
      });

      if (sys.staves && sys.staves.length) {
        const first = sys.staves[0];
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', first.left_x * sx);
        label.setAttribute('y', Math.max(12, first.top_y * sy - 4));
        label.setAttribute('fill', color);
        label.setAttribute('font-size', '11');
        label.setAttribute('font-family', 'system-ui, sans-serif');
        label.setAttribute('font-weight', '600');
        label.textContent = `S${sysIdx + 1}`;
        this._staffGroup.appendChild(label);
      }
    });

    this.setShowDetection(this._showDetection);
  }

  _updateSvgBox() {
    const dispW = this.img.offsetWidth;
    const dispH = this.img.offsetHeight;
    this.svg.setAttribute('width', dispW);
    this.svg.setAttribute('height', dispH);
    this.svg.setAttribute('viewBox', `0 0 ${dispW} ${dispH}`);
  }

  // Overlay the OSMD-rendered SVG on top of the scan, one *measure* at a
  // time. For each (svg_measure, scan_measure) pair, a nested <svg> with
  // `viewBox = svgMeasure bbox` and `x/y/width/height = scanMeasure bbox`
  // (in display coords) implements the same affine the dot positions use,
  // so the rendered glyph and the dot for the same OSMD note land on the
  // same scan pixel by construction.
  //
  // The viewBox is padded vertically by ~3× the staff-line spacing above
  // and below so lyrics, slurs, tempo marks, and beams above/below the
  // staff stay inside the visible region. overflow="visible" lets a long
  // beam or slur reach into the next measure's box if needed.
  renderRegeneratedScores(osmdSvgText, svgMeasures, scanMeasures, naturalWidth, naturalHeight) {
    this._osmdSvgText = osmdSvgText || null;
    this._svgMeasures = svgMeasures || null;
    this._scanMeasuresForRegen = scanMeasures || null;
    this._naturalWidth = naturalWidth;
    this._naturalHeight = naturalHeight;
    this._updateSvgBox();
    this._ensureGroups();

    while (this._regenGroup.firstChild) {
      this._regenGroup.removeChild(this._regenGroup.firstChild);
    }
    if (!osmdSvgText || !svgMeasures || !svgMeasures.length
        || !scanMeasures || !scanMeasures.length) {
      return;
    }

    let osmdRoot;
    try {
      const doc = new DOMParser().parseFromString(osmdSvgText, 'image/svg+xml');
      osmdRoot = doc.documentElement;
      if (!osmdRoot || osmdRoot.tagName.toLowerCase() !== 'svg') return;
    } catch (e) {
      console.error('renderRegeneratedScores: failed to parse OSMD SVG', e);
      return;
    }

    const dispW = this.img.offsetWidth;
    const dispH = this.img.offsetHeight;
    const sx = dispW / naturalWidth;
    const sy = dispH / naturalHeight;
    const SVG_NS = 'http://www.w3.org/2000/svg';

    // Index scan measures by (system, staff, measure) so we can pair them
    // up with SVG measures cheaply.
    const scanByKey = new Map();
    scanMeasures.forEach((sm) => {
      const key = `${sm.system_idx}|${sm.staff_idx}|${sm.measure_idx}`;
      scanByKey.set(key, sm);
    });

    svgMeasures.forEach((sv) => {
      const key = `${sv.system_idx}|${sv.staff_idx}|${sv.measure_idx}`;
      const sc = scanByKey.get(key);
      if (!sc) return;

      const svgW = sv.right_x - sv.left_x;
      const svgH = sv.bot_y - sv.top_y;
      const scanW = sc.right_x - sc.left_x;
      const scanH = sc.bot_y - sc.top_y;
      if (svgW <= 0 || svgH <= 0 || scanW <= 0 || scanH <= 0) return;

      // Pad the SVG viewBox vertically so lyrics / slurs / tempo
      // markings above and below the staff aren't clipped. Estimate
      // line-spacing from the measure box (5 lines = 4 gaps).
      const lineSpacing = svgH / 4.0;
      const padY = lineSpacing * 3.0;
      const vbY = sv.top_y - padY;
      const vbH = svgH + padY * 2.0;

      const nested = document.createElementNS(SVG_NS, 'svg');
      nested.setAttribute('viewBox', `${sv.left_x} ${vbY} ${svgW} ${vbH}`);
      nested.setAttribute('preserveAspectRatio', 'none');
      nested.setAttribute('x', sc.left_x * sx);
      // Push the destination box up by the same vertical pad ratio so the
      // staff lines themselves still align — pad is applied symmetrically
      // around the staff in the scan band too.
      const padYDisp = padY * (scanH / svgH);
      nested.setAttribute('y', sc.top_y * sy - padYDisp);
      nested.setAttribute('width', scanW * sx);
      nested.setAttribute('height', scanH * sy + padYDisp * 2.0);
      // Allow beams/slurs that legitimately reach into adjacent measures
      // to render. The viewBox already restricts what's visible by source.
      nested.setAttribute('overflow', 'visible');
      nested.setAttribute('data-system-idx', String(sv.system_idx));
      nested.setAttribute('data-staff-idx', String(sv.staff_idx));
      nested.setAttribute('data-measure-idx', String(sv.measure_idx));

      Array.from(osmdRoot.children).forEach((child) => {
        nested.appendChild(child.cloneNode(true));
      });

      this._regenGroup.appendChild(nested);
    });

    this.setShowRegen(this._showRegen !== false);
  }

  // Per-note glyph overlay using OSMD's VexFlow-style SVG output.
  //
  // OSMD wraps each rendered note in `<g class="vf-stavenote">`, with a
  // `<g class="vf-notehead">` (notehead) and a `<g class="vf-stem">` (stem)
  // inside. We pair each rendered stavenote with an OMR detection by
  // document-order index — both lists exclude rests, so element N in the
  // OSMD render is the same musical note as detection N coming back from
  // /api/omr_notes.
  //
  // Placement:
  //   - getBBox() on the original `vf-notehead` gives the notehead's center
  //     in OSMD-svg pixel space.
  //   - getBBox() on the original `vf-stavenote` gives the whole-glyph bbox
  //     so we know the stem extends a known amount above/below.
  //   - We deep-clone the stavenote, wrap it in a <g> with a transform that
  //     (a) shifts the OSMD notehead center to the origin,
  //     (b) scales OSMD pixels → display pixels by the ratio of OMR's
  //         detected notehead width to OSMD's rendered notehead width,
  //         multiplied by the scan→display scale, and
  //     (c) translates the origin to the OMR notehead center on the
  //         displayed scan.
  //
  // The cloned stavenote keeps its original path d-attributes; the wrapping
  // <g transform=...> does all the positioning. Result: each note's actual
  // SVG glyph from the right panel renders on top of the matching scan
  // notehead on the left panel.
  //
  //   osmdContainer — the live DOM container that OSMD rendered into.
  //                   Must be the right panel's `#osmdContainer` div with
  //                   the SVG attached, not a serialized string (we need
  //                   getBBox to work).
  //   omrNotes      — payload.notes from /api/omr_notes; one entry per
  //                   detected notehead with x1/y1/x2/y2 in rectified-PNG
  //                   pixel coords.
  renderNoteGlyphs(osmdContainer, omrNotes, naturalWidth, naturalHeight) {
    this._osmdContainerForGlyphs = osmdContainer || null;
    this._omrNotesForGlyphs = omrNotes || null;
    this._naturalWidth = naturalWidth;
    this._naturalHeight = naturalHeight;
    this._updateSvgBox();
    this._ensureGroups();

    while (this._glyphsGroup.firstChild) {
      this._glyphsGroup.removeChild(this._glyphsGroup.firstChild);
    }
    if (!osmdContainer || !omrNotes || !omrNotes.length) return;

    const osmdSvg = osmdContainer.querySelector('svg');
    if (!osmdSvg) return;

    // Each note OSMD renders is `<g class="vf-stavenote">` containing a
    // `<g class="vf-notehead">`. Rests are stavenotes without a notehead
    // (or with `vf-rest`), so filter by presence of vf-notehead — the
    // result is in document order and matches MusicXML's non-rest <note>
    // sequence, which is the same order /api/omr_notes returns.
    const stavenotes = Array.from(osmdSvg.querySelectorAll('g.vf-stavenote'))
      .filter((sn) => sn.querySelector('.vf-notehead'));
    if (!stavenotes.length) return;

    const dispW = this.img.offsetWidth;
    const dispH = this.img.offsetHeight;
    if (dispW <= 0 || dispH <= 0) return;
    const sx = dispW / naturalWidth;
    const sy = dispH / naturalHeight;
    const SVG_NS = 'http://www.w3.org/2000/svg';

    const pairCount = Math.min(stavenotes.length, omrNotes.length);
    let placed = 0;
    for (let i = 0; i < pairCount; i++) {
      const sn = stavenotes[i];
      const omr = omrNotes[i];
      const nhEl = sn.querySelector('.vf-notehead');
      if (!nhEl) continue;

      let nhBBox;
      try {
        nhBBox = nhEl.getBBox();
      } catch (e) {
        continue;
      }
      if (!nhBBox || nhBBox.width <= 0 || nhBBox.height <= 0) continue;

      const osmdCx = nhBBox.x + nhBBox.width / 2;
      const osmdCy = nhBBox.y + nhBBox.height / 2;

      const omrCx = (omr.x1 + omr.x2) / 2;
      const omrCy = (omr.y1 + omr.y2) / 2;
      const omrW = omr.x2 - omr.x1;
      if (omrW <= 0) continue;

      // Match notehead width 1:1 to the OMR-detected notehead in scan
      // pixels, then bake in the scan→display scale so the final transform
      // emits display-pixel coordinates.
      const noteScale = omrW / nhBBox.width;
      const finalScaleX = noteScale * sx;
      const finalScaleY = noteScale * sy;

      // Right-to-left: shift OSMD notehead center to origin → scale → put
      // origin at the OMR notehead center on the displayed scan.
      const tx = omrCx * sx;
      const ty = omrCy * sy;
      const transform = (
        `translate(${tx}, ${ty}) ` +
        `scale(${finalScaleX}, ${finalScaleY}) ` +
        `translate(${-osmdCx}, ${-osmdCy})`
      );

      const wrapper = document.createElementNS(SVG_NS, 'g');
      wrapper.setAttribute('transform', transform);
      wrapper.setAttribute('class', 'note-glyph');
      wrapper.setAttribute('data-det-id', omr.note_id || '');
      wrapper.setAttribute('data-pitch', omr.pitch || '');
      wrapper.setAttribute('data-duration', omr.duration || '');

      // Native browser tooltip on hover — shows up after the cursor sits
      // still on the glyph for a moment, no click required.
      const titleEl = document.createElementNS(SVG_NS, 'title');
      titleEl.textContent = `${omr.pitch || '?'} — ${omr.duration || '?'}`;
      wrapper.appendChild(titleEl);

      wrapper.appendChild(sn.cloneNode(true));

      // Click → fire onGlyphClick(omr, wrapper) so the host page can
      // surface the note's pitch + duration however it wants. The default
      // cursor styling lives in CSS; the JS just wires up the event.
      wrapper.addEventListener('click', (e) => {
        e.stopPropagation();
        if (typeof this.onGlyphClick === 'function') {
          this.onGlyphClick(omr, wrapper, e);
        }
      });

      this._glyphsGroup.appendChild(wrapper);
      placed++;
    }

    this.setShowGlyphs(this._showGlyphs !== false);
    return placed;
  }

  render(alignedNotes, naturalWidth, naturalHeight) {
    this._naturalWidth = naturalWidth;
    this._naturalHeight = naturalHeight;
    this._updateSvgBox();
    this._ensureGroups();

    // Clear dots only — keep the staff-detection overlay.
    while (this._dotsGroup.firstChild) this._dotsGroup.removeChild(this._dotsGroup.firstChild);
    this.dots.clear();
    this.notes = alignedNotes || [];

    const dispW = this.img.offsetWidth;
    const dispH = this.img.offsetHeight;
    const scaleX = dispW / naturalWidth;
    const scaleY = dispH / naturalHeight;

    this.notes.forEach((note) => {
      const displayX = note.scan_x * scaleX;
      const displayY = note.scan_y * scaleY;

      const conf = typeof note.confidence === 'number' ? note.confidence : 0.5;
      const color = conf > 0.7 ? '#22c55e' : conf > 0.4 ? '#f59e0b' : '#ef4444';

      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('cx', displayX);
      circle.setAttribute('cy', displayY);
      circle.setAttribute('r', 6);
      circle.setAttribute('fill', color);
      circle.setAttribute('fill-opacity', '0.75');
      circle.setAttribute('stroke', '#fff');
      circle.setAttribute('stroke-width', '1.5');
      circle.setAttribute('class', 'note-dot');
      circle.setAttribute('data-note-id', note.note_id);
      circle.style.cursor = 'pointer';

      const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
      title.textContent = `${note.pitch} | ${note.duration} | conf: ${conf.toFixed(2)}`;
      circle.appendChild(title);

      circle.addEventListener('click', (e) => {
        e.stopPropagation();
        if (this.onNoteClick) this.onNoteClick(note, circle);
      });

      this._dotsGroup.appendChild(circle);
      this.dots.set(note.note_id, circle);
    });
  }

  _rescale() {
    if (!this._naturalWidth || !this._naturalHeight) return;
    this._updateSvgBox();
    const dispW = this.img.offsetWidth;
    const dispH = this.img.offsetHeight;
    const scaleX = dispW / this._naturalWidth;
    const scaleY = dispH / this._naturalHeight;

    this.notes.forEach((note) => {
      const dot = this.dots.get(note.note_id);
      if (!dot) return;
      dot.setAttribute('cx', note.scan_x * scaleX);
      dot.setAttribute('cy', note.scan_y * scaleY);
    });

    if (this._detectedSystems) {
      this.renderStaffDetection(this._detectedSystems, this._naturalWidth, this._naturalHeight);
    }
    if (this._osmdSvgText && this._svgMeasures && this._scanMeasuresForRegen) {
      this.renderRegeneratedScores(
        this._osmdSvgText,
        this._svgMeasures,
        this._scanMeasuresForRegen,
        this._naturalWidth,
        this._naturalHeight,
      );
    }
    if (this._osmdContainerForGlyphs && this._omrNotesForGlyphs) {
      this.renderNoteGlyphs(
        this._osmdContainerForGlyphs,
        this._omrNotesForGlyphs,
        this._naturalWidth,
        this._naturalHeight,
      );
    }
  }

  updateDot(noteId, newPitch, newDuration, correctedX = null, correctedY = null) {
    const note = this.notes.find((n) => n.note_id === noteId);
    const dot = this.dots.get(noteId);
    if (!note || !dot) return;

    note.pitch = newPitch;
    note.duration = newDuration;
    if (correctedX !== null) note.scan_x = correctedX;
    if (correctedY !== null) note.scan_y = correctedY;

    dot.setAttribute('fill', '#3b82f6');
    const title = dot.querySelector('title');
    if (title) title.textContent = `${newPitch} | ${newDuration} | corrected`;
  }

  enableDragging() {
    let dragging = null;
    let startX = 0, startY = 0, origCX = 0, origCY = 0;

    this.svg.addEventListener('mousedown', (e) => {
      const dot = e.target.closest('.note-dot');
      if (!dot) return;
      dragging = dot;
      startX = e.clientX;
      startY = e.clientY;
      origCX = parseFloat(dot.getAttribute('cx'));
      origCY = parseFloat(dot.getAttribute('cy'));
      e.preventDefault();
    });

    window.addEventListener('mousemove', (e) => {
      if (!dragging) return;
      const dx = e.clientX - startX;
      const dy = e.clientY - startY;
      dragging.setAttribute('cx', origCX + dx);
      dragging.setAttribute('cy', origCY + dy);
    });

    window.addEventListener('mouseup', () => {
      if (!dragging) return;
      const noteId = dragging.getAttribute('data-note-id');
      const dispW = this.img.offsetWidth;
      const dispH = this.img.offsetHeight;
      const scaleX = dispW / this._naturalWidth;
      const scaleY = dispH / this._naturalHeight;
      const newScanX = parseFloat(dragging.getAttribute('cx')) / scaleX;
      const newScanY = parseFloat(dragging.getAttribute('cy')) / scaleY;
      const note = this.notes.find((n) => n.note_id === noteId);
      if (note) { note.scan_x = newScanX; note.scan_y = newScanY; }
      dragging = null;
    });
  }
}
