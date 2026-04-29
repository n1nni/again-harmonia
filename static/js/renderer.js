// renderer.js — wraps OpenSheetMusicDisplay and extracts note positions.
//
// Also keeps the MusicXML as a parsed DOM so the editor can mutate notes
// and reload OSMD in place. Notes are keyed by (partIdx, measureIdx,
// noteInMeasureIdx), computed identically during the XML walk and the
// OSMD graphic walk.

class HarmoniaRenderer {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.osmd = null;
    this.svgData = null;
    this.xmlText = null;
    this.xmlDoc = null;
    // xmlKey → <note> DOM element
    this.xmlNoteIndex = new Map();
    // Layout numbers extracted from MusicXML <defaults>, in MusicXML tenths.
    this.layoutDefaults = null;
    // xmlKey → default-x (MusicXML tenths, offset from measure's left edge).
    this.xmlNotePositions = new Map();
    // "partIdx_measureIdx" → measure width in MusicXML tenths.
    this.xmlMeasureWidths = new Map();
    this.onAfterRender = null;
  }

  async init(musicxmlUrl) {
    this.osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay(this.container, {
      autoResize: false,
      backend: 'svg',
      drawingParameters: 'default',
      // Respect <print new-system="yes"> / <print new-page="yes">
      // so the rendered layout matches the scan's system breaks.
      newSystemFromXML: true,
      newPageFromXML: true,
      // The MusicXML carries <time print-object="no"> but OSMD doesn't honor
      // that attribute — suppress the 4/4 ourselves.
      drawTimeSignatures: false,
    });

    // Belt-and-braces for OSMD versions where constructor options don't
    // always propagate to the engraving rules.
    try {
      const rules = this.osmd.EngravingRules;
      if (rules) {
        rules.RenderTimeSignatures = false;
        // Force OSMD to honor <print new-system="yes"/> and
        // <print new-page="yes"/> from the MusicXML rather than auto-flowing
        // measures by available width.
        rules.NewSystemAtXMLNewSystemAttribute = true;
        rules.NewPageAtXMLNewPageAttribute = true;
        // Some 1.8.x builds also gate on these flags.
        if ('NewSystemAtXMLNewPageAttribute' in rules) {
          rules.NewSystemAtXMLNewPageAttribute = true;
        }
      }
    } catch (_) { /* noop */ }

    const response = await fetch(musicxmlUrl);
    this.xmlText = await response.text();
    this._parseXml(this.xmlText);
    this._applyLayoutDefaults();

    await this.osmd.load(this.xmlText);
    // Render is deferred — caller invokes render() once the panel is
    // visible, otherwise OSMD measures a 0-width container.
  }

  _parseXml(xmlText) {
    const parser = new DOMParser();
    this.xmlDoc = parser.parseFromString(xmlText, 'application/xml');
    this._rebuildXmlNoteIndex();
    this._extractLayoutDefaults();
    this._extractXmlPositions();
  }

  // Pull each note's `default-x` and each measure's `width` out of the
  // MusicXML so we can drive layout from the score's own coordinates
  // instead of OSMD's auto-spacing. `default-x` is the offset from the
  // measure's left edge to the *left edge of the notehead* in tenths.
  _extractXmlPositions() {
    this.xmlNotePositions.clear();
    this.xmlMeasureWidths.clear();
    if (!this.xmlDoc) return;
    const parts = this.xmlDoc.querySelectorAll('score-partwise > part');
    parts.forEach((part, partIdx) => {
      const measures = part.querySelectorAll('measure');
      measures.forEach((measure, measureIdx) => {
        const widthAttr = measure.getAttribute('width');
        if (widthAttr) {
          const w = parseFloat(widthAttr);
          if (Number.isFinite(w)) {
            this.xmlMeasureWidths.set(`${partIdx}_${measureIdx}`, w);
          }
        }
        const notes = measure.querySelectorAll('note');
        let visibleIdx = 0;
        notes.forEach((note) => {
          if (note.querySelector('rest')) return;
          const dxAttr = note.getAttribute('default-x');
          if (dxAttr) {
            const dx = parseFloat(dxAttr);
            if (Number.isFinite(dx)) {
              this.xmlNotePositions.set(`${partIdx}_${measureIdx}_${visibleIdx}`, dx);
            }
          }
          visibleIdx++;
        });
      });
    });
  }

  // Look up an XML measure width, falling back to part 0 if the requested
  // part is missing (parts share measure structure so widths agree across
  // parts in a well-formed score, but we guard against partial data).
  _xmlMeasureWidth(partIdx, measureIdx) {
    const direct = this.xmlMeasureWidths.get(`${partIdx}_${measureIdx}`);
    if (Number.isFinite(direct)) return direct;
    return this.xmlMeasureWidths.get(`0_${measureIdx}`);
  }

  // Walk OSMD's GraphicSheet but reposition each measure's x-extent using
  // XML measure widths within the system, and emit one entry per
  // (system, staff, measure). The y-extent stays from OSMD because pitch
  // placement is OSMD's job; only x is overridden. Returns enriched
  // entries that other extractors can reuse (carrying the OSMD measure
  // object for note iteration plus the system's tenths→pixel scale).
  _buildXmlMeasureLayout() {
    if (!this.osmd || !this.osmd.GraphicSheet) return [];
    const gs = this.osmd.GraphicSheet;
    const unitToPx = this._unitToPxFactor();
    const out = [];
    let globalSysIdx = 0;

    gs.MusicPages.forEach((page, pageIdx) => {
      page.MusicSystems.forEach((system) => {
        const sysIdx = globalSysIdx++;
        const staffLines = system.StaffLines || [];
        if (!staffLines.length) return;

        // System x-bounds from staff lines (OSMD's own engraving span).
        let sysLeftX = Infinity;
        let sysRightX = -Infinity;
        staffLines.forEach((staffLine) => {
          const pos = staffLine.PositionAndShape;
          if (!pos) return;
          const abs = pos.AbsolutePosition || { x: 0 };
          const size = pos.Size || { width: 0 };
          sysLeftX = Math.min(sysLeftX, abs.x * unitToPx);
          sysRightX = Math.max(sysRightX, (abs.x + size.width) * unitToPx);
        });
        if (!Number.isFinite(sysLeftX) || !Number.isFinite(sysRightX)) return;
        const sysWidthPx = sysRightX - sysLeftX;

        // Sum of XML measure widths for this system (use staff 0).
        const measureIndicesInSystem = [];
        (staffLines[0].Measures || []).forEach((m) => {
          if (!m) return;
          measureIndicesInSystem.push(this._measureIndex(m));
        });
        let totalXmlWidth = 0;
        measureIndicesInSystem.forEach((mIdx) => {
          const w = this._xmlMeasureWidth(0, mIdx);
          if (Number.isFinite(w)) totalXmlWidth += w;
        });
        // Tenths → OSMD pixels for this system. If we have no XML widths
        // (defensive), fall back to 1 so callers can detect & skip.
        const scale = totalXmlWidth > 0 ? sysWidthPx / totalXmlWidth : 0;

        staffLines.forEach((staffLine, staffIdxInSys) => {
          const partIdx = this._partIndexForStaffLine(staffLine, staffIdxInSys);
          let cumulX = 0;
          (staffLine.Measures || []).forEach((measure) => {
            if (!measure) return;
            const mIdx = this._measureIndex(measure);
            const xmlW = this._xmlMeasureWidth(partIdx, mIdx);
            if (!Number.isFinite(xmlW) || scale <= 0) return;

            const leftX = sysLeftX + cumulX * scale;
            const rightX = leftX + xmlW * scale;

            const pos = measure.PositionAndShape;
            const abs = pos ? (pos.AbsolutePosition || { y: 0 }) : { y: 0 };
            const size = pos ? (pos.Size || { height: 4 }) : { height: 4 };
            const topY = abs.y * unitToPx;
            const botY = (abs.y + size.height) * unitToPx;

            out.push({
              system_idx: sysIdx,
              page_idx: pageIdx,
              staff_idx: staffIdxInSys,
              part_idx: partIdx,
              measure_idx: mIdx,
              measure_obj: measure,
              left_x: leftX,
              right_x: rightX,
              top_y: topY,
              bot_y: botY,
              xml_width: xmlW,
              scale,
            });

            cumulX += xmlW;
          });
        });
      });
    });
    return out;
  }

  // Read <defaults> from the MusicXML so we can drive OSMD's engraving with
  // the score's own layout numbers instead of OSMD's hard-coded fallbacks.
  // Values stay in MusicXML tenths here; conversion happens at apply time.
  _extractLayoutDefaults() {
    this.layoutDefaults = null;
    if (!this.xmlDoc) return;
    const defaultsEl = this.xmlDoc.querySelector('score-partwise > defaults');
    if (!defaultsEl) return;
    const num = (sel) => {
      const el = defaultsEl.querySelector(sel);
      if (!el) return null;
      const n = parseFloat(el.textContent);
      return Number.isFinite(n) ? n : null;
    };
    this.layoutDefaults = {
      scalingMm: num('scaling > millimeters'),
      scalingTenths: num('scaling > tenths'),
      staffDistance: num('staff-layout > staff-distance'),
      systemDistance: num('system-layout > system-distance'),
      topSystemDistance: num('system-layout > top-system-distance'),
    };
  }

  // Push the MusicXML layout numbers into OSMD's EngravingRules so the
  // staves of different parts sit at the spacing the file declares (and
  // therefore line up parallel across the systems).
  _applyLayoutDefaults() {
    if (!this.osmd || !this.osmd.EngravingRules || !this.layoutDefaults) return;
    const rules = this.osmd.EngravingRules;
    const ld = this.layoutDefaults;
    // MusicXML spec: 10 tenths == 1 staff-space, and 1 OSMD unit == 1
    // staff-space. So tenths * 0.1 → OSMD units, regardless of the
    // <scaling> millimeters/tenths ratio (that only affects print size).
    const T = 0.1;
    const trySet = (key, value) => {
      if (value == null || !Number.isFinite(value)) return;
      if (key in rules) rules[key] = value;
    };

    if (ld.staffDistance != null) {
      const v = ld.staffDistance * T;
      // Cover both "between staves of one instrument" and "between
      // instruments" — different OSMD versions expose these under
      // different names; trySet skips ones that don't exist.
      trySet('BetweenStaffDistance', v);
      trySet('StaffDistance', v);
      trySet('InstrumentBracketHeightMargin', v);
    }
    if (ld.systemDistance != null) {
      const v = ld.systemDistance * T;
      trySet('MinimumDistanceBetweenSystems', v);
      trySet('MinSkyBottomDistBetweenSystems', v);
      trySet('SystemDistance', v);
    }
    if (ld.topSystemDistance != null) {
      const v = ld.topSystemDistance * T;
      trySet('PageTopMargin', v);
      trySet('PageTopMarginNarrow', v);
    }
  }

  _rebuildXmlNoteIndex() {
    this.xmlNoteIndex.clear();
    if (!this.xmlDoc) return;
    const parts = this.xmlDoc.querySelectorAll('score-partwise > part');
    parts.forEach((part, partIdx) => {
      const measures = part.querySelectorAll('measure');
      measures.forEach((measure, measureIdx) => {
        const notes = measure.querySelectorAll('note');
        let visibleIdx = 0;
        notes.forEach((note) => {
          // Skip rests for the index — we only mark pitched notes.
          if (note.querySelector('rest')) return;
          // Skip additional chord members — in MusicXML the first note of a
          // chord has no <chord/>, subsequent ones do. We treat each as its
          // own slot since OSMD also emits them as separate graphical notes.
          const key = `${partIdx}_${measureIdx}_${visibleIdx}`;
          this.xmlNoteIndex.set(key, note);
          visibleIdx++;
        });
      });
    });
  }

  async render(widthPx = null) {
    if (widthPx && widthPx > 0) {
      this.container.style.width = widthPx + 'px';
    }
    this.osmd.zoom = 1.0;
    await this.osmd.render();

    const svgEl = this.container.querySelector('svg');
    this.svgData = svgEl ? svgEl.outerHTML : null;
    if (typeof this.onAfterRender === 'function') {
      try { this.onAfterRender(); } catch (_) { /* noop */ }
    }
    return this.svgData;
  }

  getSvgViewBox() {
    const svgEl = this.container.querySelector('svg');
    if (!svgEl) return null;
    return svgEl.getAttribute('viewBox');
  }

  async matchScanWidth(scanImageEl) {
    const displayWidth = scanImageEl.offsetWidth || scanImageEl.naturalWidth;
    if (displayWidth > 0) {
      await this.render(displayWidth);
    }
  }

  // Walk OSMD's graphic model and emit system/staff bounding boxes.
  // This replaces fragile SVG line-parsing on the backend: we know the
  // positions exactly, in pixel space.
  extractSystems() {
    if (!this.osmd || !this.osmd.GraphicSheet) return [];
    const out = [];
    const gs = this.osmd.GraphicSheet;
    const unitToPx = this._unitToPxFactor();
    let globalSysIdx = 0;

    gs.MusicPages.forEach((page, pageIdx) => {
      page.MusicSystems.forEach((system) => {
        const staves = [];
        (system.StaffLines || []).forEach((staffLine, staffIdxInSys) => {
          const pos = staffLine.PositionAndShape;
          if (!pos) return;
          const abs = pos.AbsolutePosition || { x: 0, y: 0 };
          const size = pos.Size || { width: 0, height: 4 };
          staves.push({
            staff_idx: staffIdxInSys,
            left_x: abs.x * unitToPx,
            right_x: (abs.x + size.width) * unitToPx,
            top_y: abs.y * unitToPx,
            bot_y: (abs.y + size.height) * unitToPx,
          });
        });
        if (!staves.length) return;
        out.push({
          system_idx: globalSysIdx++,
          page_idx: pageIdx,
          staves,
          left_x: Math.min.apply(null, staves.map((s) => s.left_x)),
          right_x: Math.max.apply(null, staves.map((s) => s.right_x)),
          top_y: Math.min.apply(null, staves.map((s) => s.top_y)),
          bot_y: Math.max.apply(null, staves.map((s) => s.bot_y)),
        });
      });
    });
    return out;
  }

  // Per-(system, staff, measure) bbox in pixel space, with x-extents
  // computed from MusicXML measure widths (the source of truth) instead of
  // OSMD's auto-spacing. y-extents stay from OSMD because pitch-to-y is
  // OSMD's job. Falls back to OSMD's own measure bbox if the XML lacks
  // width attributes for this system.
  extractMeasures() {
    const layout = this._buildXmlMeasureLayout();
    if (layout.length) {
      return layout.map((m) => ({
        system_idx: m.system_idx,
        page_idx: m.page_idx,
        staff_idx: m.staff_idx,
        measure_idx: m.measure_idx,
        left_x: m.left_x,
        right_x: m.right_x,
        top_y: m.top_y,
        bot_y: m.bot_y,
      }));
    }
    // Fallback: OSMD-derived measure bboxes.
    if (!this.osmd || !this.osmd.GraphicSheet) return [];
    const out = [];
    const gs = this.osmd.GraphicSheet;
    const unitToPx = this._unitToPxFactor();
    let globalSysIdx = 0;
    gs.MusicPages.forEach((page, pageIdx) => {
      page.MusicSystems.forEach((system) => {
        const sysIdx = globalSysIdx++;
        (system.StaffLines || []).forEach((staffLine, staffIdxInSys) => {
          (staffLine.Measures || []).forEach((measure) => {
            if (!measure) return;
            const pos = measure.PositionAndShape;
            if (!pos) return;
            const abs = pos.AbsolutePosition || { x: 0, y: 0 };
            const size = pos.Size || { width: 0, height: 4 };
            out.push({
              system_idx: sysIdx,
              page_idx: pageIdx,
              staff_idx: staffIdxInSys,
              measure_idx: this._measureIndex(measure),
              left_x: abs.x * unitToPx,
              right_x: (abs.x + size.width) * unitToPx,
              top_y: abs.y * unitToPx,
              bot_y: (abs.y + size.height) * unitToPx,
            });
          });
        });
      });
    });
    return out;
  }

  // One entry per rendered notehead. The x-coord is computed from the
  // MusicXML `default-x` (the score's source of truth) plus a half-notehead
  // offset to land on the notehead center; the y-coord stays from OSMD
  // because pitch placement is OSMD's job. Falls back to OSMD's x for any
  // note that lacks `default-x` in the source XML.
  extractNotePositions() {
    if (!this.osmd || !this.osmd.GraphicSheet) return [];
    const unitToPx = this._unitToPxFactor();
    const layout = this._buildXmlMeasureLayout();
    // Index the layout by (system, staff, measure) for O(1) lookup.
    const layoutByKey = new Map();
    layout.forEach((m) => {
      layoutByKey.set(`${m.system_idx}|${m.staff_idx}|${m.measure_idx}`, m);
    });

    const notes = [];
    const gs = this.osmd.GraphicSheet;
    let globalSysIdx = 0;

    gs.MusicPages.forEach((page, pageIdx) => {
      page.MusicSystems.forEach((system) => {
        const sysIdx = globalSysIdx++;
        (system.StaffLines || []).forEach((staffLine, staffIdxInSys) => {
          const partIdx = this._partIndexForStaffLine(staffLine, staffIdxInSys);
          (staffLine.Measures || []).forEach((measure) => {
            if (!measure) return;
            const measureIdx = this._measureIndex(measure);
            const layoutEntry = layoutByKey.get(
              `${sysIdx}|${staffIdxInSys}|${measureIdx}`,
            );
            let visibleIdx = 0;
            (measure.staffEntries || []).forEach((se) => {
              (se.graphicalVoiceEntries || []).forEach((gve) => {
                (gve.notes || []).forEach((gNote) => {
                  const source = gNote.sourceNote;
                  const isRest = !!(source && source.isRest && source.isRest());
                  if (isRest) return;
                  const pos = gNote.PositionAndShape;
                  if (!pos) return;
                  const abs = pos.AbsolutePosition || { x: 0, y: 0 };
                  const size = pos.Size || { width: 1.2, height: 1.2 };
                  const xmlKey = `${partIdx}_${measureIdx}_${visibleIdx}`;
                  const ourVisIdx = visibleIdx;
                  visibleIdx++;

                  // y always from OSMD (pitch position).
                  const cyUnit = abs.y + size.height / 2;
                  const svgRx = (size.width / 2) * unitToPx;
                  const svgRy = (size.height / 2) * unitToPx;

                  // x from XML default-x when available. MusicXML defines
                  // default-x as the offset from the measure's left edge
                  // to the notehead's *left edge*; add half a notehead
                  // width to land on the center.
                  const defaultX = this.xmlNotePositions.get(xmlKey);
                  let svgX;
                  if (layoutEntry && Number.isFinite(defaultX)) {
                    svgX = layoutEntry.left_x + defaultX * layoutEntry.scale + svgRx;
                  } else {
                    svgX = (abs.x + size.width / 2) * unitToPx;
                  }
                  const svgY = cyUnit * unitToPx;

                  notes.push({
                    noteId: `p${pageIdx}_s${sysIdx}_st${staffIdxInSys}_m${measureIdx}_i${ourVisIdx + 1}`,
                    xmlKey,
                    pageIdx,
                    systemIdx: sysIdx,
                    staffIdx: staffIdxInSys,
                    partIdx,
                    measureIdx,
                    svgX,
                    svgY,
                    svgRx,
                    svgRy,
                    pitch: this._pitchName(source),
                    duration: this._durationName(source),
                  });
                });
              });
            });
          });
        });
      });
    });

    return notes;
  }

  // Modify an in-memory note and reload OSMD so the right panel updates.
  // `xmlKey` is the same key emitted by extractNotePositions.
  async applyNoteEdit(xmlKey, newPitch, newDuration) {
    if (!this.xmlDoc) return false;
    const noteEl = this.xmlNoteIndex.get(xmlKey);
    if (!noteEl) return false;

    if (newPitch) {
      this._applyPitchToNote(noteEl, newPitch);
    }
    if (newDuration) {
      this._applyDurationToNote(noteEl, newDuration);
    }

    const serialized = new XMLSerializer().serializeToString(this.xmlDoc);
    this.xmlText = serialized;

    try {
      await this.osmd.load(serialized);
      await this.render(parseFloat(this.container.style.width) || null);
      // Rebuild the index in case the edit shifted anything; the DOM
      // references still point at the same nodes so most entries stay stable.
      this._rebuildXmlNoteIndex();
      return true;
    } catch (err) {
      console.error('applyNoteEdit: OSMD reload failed', err);
      return false;
    }
  }

  _applyPitchToNote(noteEl, newPitch) {
    const match = /^([A-G])(##|#|bb|b|x)?(-?\d+)$/.exec(newPitch);
    if (!match) return;
    const step = match[1];
    const accidental = match[2] || '';
    const octave = match[3];
    let alter = 0;
    if (accidental === '#') alter = 1;
    else if (accidental === '##' || accidental === 'x') alter = 2;
    else if (accidental === 'b') alter = -1;
    else if (accidental === 'bb') alter = -2;

    let pitchEl = noteEl.querySelector('pitch');
    if (!pitchEl) {
      // Turning a rest into a pitched note — skip for now.
      return;
    }
    const doc = noteEl.ownerDocument;

    let stepEl = pitchEl.querySelector('step');
    if (!stepEl) {
      stepEl = doc.createElementNS(noteEl.namespaceURI, 'step');
      pitchEl.insertBefore(stepEl, pitchEl.firstChild);
    }
    stepEl.textContent = step;

    let alterEl = pitchEl.querySelector('alter');
    if (alter !== 0) {
      if (!alterEl) {
        alterEl = doc.createElementNS(noteEl.namespaceURI, 'alter');
        const octaveEl = pitchEl.querySelector('octave');
        pitchEl.insertBefore(alterEl, octaveEl);
      }
      alterEl.textContent = String(alter);
    } else if (alterEl) {
      alterEl.remove();
    }

    let octaveEl = pitchEl.querySelector('octave');
    if (!octaveEl) {
      octaveEl = doc.createElementNS(noteEl.namespaceURI, 'octave');
      pitchEl.appendChild(octaveEl);
    }
    octaveEl.textContent = octave;
  }

  _applyDurationToNote(noteEl, newDuration) {
    const typeEl = noteEl.querySelector('type');
    if (typeEl) {
      typeEl.textContent = newDuration;
    }
    // Also update the <duration> element (in divisions). Without reflowing
    // the measure we can't rebalance durations, so leave the divisions
    // alone — OSMD will render the <type> but the math may warn.
  }

  _partIndexForStaffLine(staffLine, fallback) {
    // OSMD's staff line has ParentStaff → ParentInstrument; the instrument
    // index in MusicSheet.Instruments is the part index.
    try {
      const instrument = staffLine.ParentStaff && staffLine.ParentStaff.ParentInstrument;
      if (instrument && this.osmd && this.osmd.Sheet && this.osmd.Sheet.Instruments) {
        const idx = this.osmd.Sheet.Instruments.indexOf(instrument);
        if (idx >= 0) return idx;
      }
    } catch (_) { /* fall through */ }
    return fallback;
  }

  _measureIndex(graphicalMeasure) {
    // Prefer the 0-based MeasureListIndex off the parent source measure.
    const src = graphicalMeasure.parentSourceMeasure;
    if (src && typeof src.MeasureListIndex === 'number') return src.MeasureListIndex;
    if (typeof graphicalMeasure.MeasureNumber === 'number') {
      return Math.max(0, graphicalMeasure.MeasureNumber - 1);
    }
    return 0;
  }

  _unitToPxFactor() {
    const candidates = [
      this.osmd && this.osmd.unitInPixels,
      this.osmd && this.osmd.EngravingRules && this.osmd.EngravingRules.UnitInPixels,
    ];
    for (const c of candidates) {
      if (typeof c === 'number' && c > 0) return c;
    }
    return 10;
  }

  _pitchName(sourceNote) {
    if (!sourceNote) return '?';
    if (sourceNote.Pitch) {
      const p = sourceNote.Pitch;
      const step = p.FundamentalNote !== undefined ? this._stepLetter(p.FundamentalNote) : '?';
      const alter = p.AccidentalHalfTones || 0;
      const octave = (typeof p.Octave === 'number') ? p.Octave : 4;
      const acc = alter === 1 ? '#' : alter === -1 ? 'b' : alter === 2 ? 'x' : alter === -2 ? 'bb' : '';
      return step + acc + octave;
    }
    if (typeof sourceNote.halfTone === 'number') {
      const names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
      const octave = Math.floor(sourceNote.halfTone / 12) - 1;
      return names[sourceNote.halfTone % 12] + octave;
    }
    return '?';
  }

  _stepLetter(idx) {
    const letters = ['C', 'D', 'E', 'F', 'G', 'A', 'B'];
    return letters[idx] || '?';
  }

  _durationName(sourceNote) {
    if (!sourceNote) return 'quarter';
    const len = sourceNote.Length && sourceNote.Length.realValue;
    if (typeof len !== 'number') return 'quarter';
    if (len >= 1) return 'whole';
    if (len >= 0.5) return 'half';
    if (len >= 0.25) return 'quarter';
    if (len >= 0.125) return 'eighth';
    if (len >= 0.0625) return '16th';
    return '32nd';
  }
}
