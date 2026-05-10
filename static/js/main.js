// main.js — bootstrap.
//
// Flow:
//  1. On page load, render MusicXML with OSMD (no scan yet).
//  2. On scan upload:
//     a) Display the scan in the left panel at its natural size.
//     b) Call /api/detect_staves to draw the system detection overlay.
//     c) Call /api/align to produce the blue note dots over the scan.
//  3. Editing a dot updates both the overlay AND the rendered MusicXML.
//
// We intentionally DO NOT resize the OSMD container to match the scan —
// that was breaking the rendered layout. The alignment pipeline already
// warps from SVG space into scan space internally, so the two panels can
// be rendered at whatever sizes look good on their own.

document.addEventListener('DOMContentLoaded', async () => {
  const renderer = new HarmoniaRenderer('osmdContainer');
  const overlay = new NoteOverlay(
    document.getElementById('overlaysvg'),
    document.getElementById('scanImage')
  );
  const editor = new NoteEditor('noteEditor', overlay, renderer);

  overlay.onNoteClick = (note, dot) => editor.open(note, dot);
  overlay.enableDragging();

  let uploadedFilename = null;
  let naturalWidth = null;
  let naturalHeight = null;
  // Cached scan-system bboxes from the most recent /api/detect_staves call,
  // used to overlay the regenerated SVG on the scan in runAlignment.
  let scanSystemsCache = null;
  // Cached OMR notehead detections so post-edit re-renders don't re-fetch.
  // Stored as { url, notes, width, height } — these are the notes used to
  // place per-note glyph crops over the scan.
  let lastOmrGlyphs = null;

  const statusEl = document.getElementById('status');
  const setStatus = (msg) => { statusEl.textContent = msg; };

  try {
    await renderer.init('/api/musicxml');
    setStatus('Upload a scan to begin.');
  } catch (e) {
    console.error(e);
    setStatus('Error loading score: ' + e.message);
  }

  // After the user corrects a note, OSMD reloads — the live SVG in the
  // right panel is replaced with new vf-stavenote elements, so we just
  // re-run the glyph overlay against the (now-updated) container.
  editor.onAfterSave = async () => {
    if (!uploadedFilename) return;
    if (lastOmrGlyphs) {
      overlay.renderNoteGlyphs(
        renderer.container,
        lastOmrGlyphs.notes,
        lastOmrGlyphs.width,
        lastOmrGlyphs.height,
      );
    } else {
      setStatus('Re-aligning after edit...');
      await runAlignment();
    }
  };

  // "Show detection" toggle.
  const toggleBtn = document.getElementById('toggleDetection');
  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      const next = toggleBtn.dataset.on !== 'true';
      toggleBtn.dataset.on = next ? 'true' : 'false';
      toggleBtn.textContent = next ? 'Hide Detection' : 'Show Detection';
      overlay.setShowDetection(next);
    });
    // Default: on.
    toggleBtn.dataset.on = 'true';
    toggleBtn.textContent = 'Hide Detection';
  }

  // "Show regenerated SVG" toggle (the affine overlay of the rendered score).
  const toggleRegenBtn = document.getElementById('toggleRegen');
  if (toggleRegenBtn) {
    toggleRegenBtn.addEventListener('click', () => {
      const next = toggleRegenBtn.dataset.on !== 'true';
      toggleRegenBtn.dataset.on = next ? 'true' : 'false';
      toggleRegenBtn.textContent = next ? 'Hide Regenerated' : 'Show Regenerated';
      overlay.setShowRegen(next);
    });
    toggleRegenBtn.dataset.on = 'true';
    toggleRegenBtn.textContent = 'Hide Regenerated';
  }

  // "Show glyphs" toggle for the per-note OSMD-glyph overlay.
  const toggleGlyphsBtn = document.getElementById('toggleGlyphs');
  if (toggleGlyphsBtn) {
    toggleGlyphsBtn.addEventListener('click', () => {
      const next = toggleGlyphsBtn.dataset.on !== 'true';
      toggleGlyphsBtn.dataset.on = next ? 'true' : 'false';
      toggleGlyphsBtn.textContent = next ? 'Hide Glyphs' : 'Show Glyphs';
      overlay.setShowGlyphs(next);
    });
    toggleGlyphsBtn.dataset.on = 'true';
    toggleGlyphsBtn.textContent = 'Hide Glyphs';
  }

  document.getElementById('scanUpload').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setStatus('Uploading scan to OMR...');
    const formData = new FormData();
    formData.append('scan', file);

    let data;
    try {
      const res = await fetch('/api/upload', { method: 'POST', body: formData });
      data = await res.json();
      if (!res.ok) throw new Error(data.error || 'upload failed');
    } catch (err) {
      setStatus('Upload failed: ' + err.message);
      return;
    }

    // /api/upload now forwards the image to the OMR API, so `data.filename`
    // points to the *rectified* PNG and `data.musicxml_url` points to the
    // recognized score. Both are keyed by the same job stamp.
    uploadedFilename = data.filename;
    const imgEl = document.getElementById('scanImage');
    imgEl.src = data.url || ('/uploads/' + uploadedFilename);

    imgEl.onload = async () => {
      naturalWidth = imgEl.naturalWidth;
      naturalHeight = imgEl.naturalHeight;
      document.getElementById('alignBtn').disabled = false;

      // Reveal the rendered-score panel.
      const osmdPanel = document.getElementById('osmdPanel');
      osmdPanel.style.display = '';

      // Swap the right-hand score from the bundled fallback to the
      // OMR-recognized MusicXML for this upload.
      if (data.musicxml_url) {
        setStatus('Loading recognized score...');
        try {
          await renderer.loadFromUrl(data.musicxml_url);
        } catch (err) {
          console.error(err);
          setStatus('Failed to load recognized score: ' + err.message);
          return;
        }
      }

      await renderer.render();

      setStatus('Scan loaded. Detecting staves...');
      await runDetection();
      // Two overlays go up in parallel:
      //   1) per-note glyph crops keyed by OMR det_id (precise notehead
      //      placement using rectified-PNG pixel coords)
      //   2) the per-measure regenerated SVG overlay (the proven
      //      whole-measure render on top of the scan, drawn by
      //      runAlignment via /api/align)
      // Both populate independent SVG groups in overlay.js, so they
      // coexist and either one can be toggled off via the header buttons.
      if (data.omr_notes_url) {
        await runOmrGlyphs(data.omr_notes_url);
      }
      await runAlignment();
    };
  });

  document.getElementById('alignBtn').addEventListener('click', async () => {
    await runAlignment();
  });

  async function runOmrGlyphs(omrNotesUrl) {
    if (!omrNotesUrl) return;
    setStatus('Loading OMR notehead coordinates...');
    try {
      const res = await fetch(omrNotesUrl);
      const payload = await res.json();
      if (!res.ok) throw new Error(payload.error || 'omr_notes failed');

      // OMR returns coords in the rectified PNG's own pixel space — the same
      // space the overlay scales to via naturalWidth/naturalHeight.
      const w = payload.image_width || naturalWidth;
      const h = payload.image_height || naturalHeight;
      lastOmrGlyphs = {
        url: omrNotesUrl,
        notes: payload.notes || [],
        width: w,
        height: h,
      };

      // Clone each `<g class="vf-stavenote">` from the OSMD-rendered SVG
      // in the right panel and place it on the scan at the matching OMR
      // notehead bbox. Pairing is by document order — both lists exclude
      // rests so the Nth stavenote is the Nth /api/omr_notes entry.
      const placed = overlay.renderNoteGlyphs(
        renderer.container,
        payload.notes || [],
        w,
        h,
      );

      const total = (payload.notes || []).length;
      setStatus(`Overlaid ${placed || 0}/${total} note glyph(s) on scan.`);
    } catch (err) {
      console.error(err);
      setStatus('OMR glyphs failed: ' + err.message);
    }
  }

  async function runDetection() {
    if (!uploadedFilename) return;
    try {
      const res = await fetch('/api/detect_staves', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: uploadedFilename }),
      });
      const payload = await res.json();
      if (!res.ok) throw new Error(payload.error || 'detection failed');
      scanSystemsCache = payload.systems || [];
      overlay.renderStaffDetection(scanSystemsCache, naturalWidth, naturalHeight);
      const count = (payload.systems || []).length;
      const staves = (payload.systems || []).reduce((n, s) => n + (s.staves || []).length, 0);
      setStatus(`Detected ${count} system(s), ${staves} stave(s).`);
    } catch (err) {
      console.error(err);
      setStatus('Detection failed: ' + err.message);
    }
  }

  async function runAlignment() {
    if (!uploadedFilename) return;
    setStatus('Aligning notes...');

    const osmdNotes = renderer.extractNotePositions();
    const osmdSystems = renderer.extractSystems();
    const osmdMeasures = renderer.extractMeasures();
    const svgData = renderer.svgData;

    try {
      const res = await fetch('/api/align', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          filename: uploadedFilename,
          svg_data: svgData,
          osmd_notes: osmdNotes,
          osmd_systems: osmdSystems,
          osmd_measures: osmdMeasures,
          natural_width: naturalWidth,
          natural_height: naturalHeight,
        }),
      });
      const payload = await res.json();
      if (!res.ok) {
        throw new Error(payload.error || 'alignment failed');
      }

      // Server returns error field on partial failure with an empty notes
      // list — surface that message rather than silently showing 0 dots.
      if (payload.error) {
        setStatus('Align warning: ' + payload.error);
      } else {
        const n = (payload.notes || []).length;
        const ss = payload.scan_systems ?? 0;
        const vs = payload.svg_systems ?? 0;
        setStatus(`Aligned ${n} note(s). Scan systems: ${ss}, SVG systems: ${vs}.`);
      }

      // The OMR-driven per-note glyph overlay is now the primary visual
      // (see runOmrGlyphs); we deliberately don't re-render alignment-pipeline
      // dots here — they would compete with the glyph layer. The Align button
      // is kept for the regenerated-SVG overlay below.

      // Affine-overlay the regenerated SVG on the scan, one *measure* at a
      // time. The same per-measure affine drives the dot positions in the
      // aligner, so dots and rendered glyphs land on the same scan pixel.
      overlay.renderRegeneratedScores(
        svgData,
        osmdMeasures,
        payload.scan_measures || [],
        naturalWidth,
        naturalHeight,
      );
    } catch (err) {
      console.error(err);
      setStatus('Align failed: ' + err.message);
    }
  }
});
