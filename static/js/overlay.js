// overlay.js — draws interactive note dots over the uploaded scan.

class NoteOverlay {
  constructor(svgEl, imgEl) {
    this.svg = svgEl;
    this.img = imgEl;
    this.notes = [];
    this.dots = new Map();
    this.onNoteClick = null;
    this._naturalWidth = 0;
    this._naturalHeight = 0;
    this._regenGroup = null;
    this._staffGroup = null;
    this._dotsGroup = null;
    this._detectedSystems = null;
    this._showDetection = true;
    this._showRegen = true;
    // Cached inputs so we can re-render the regen overlay on window resize.
    this._osmdSvgText = null;
    this._svgMeasures = null;
    this._scanMeasuresForRegen = null;

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
