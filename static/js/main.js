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

  const statusEl = document.getElementById('status');
  const setStatus = (msg) => { statusEl.textContent = msg; };

  try {
    await renderer.init('/api/musicxml');
    setStatus('Upload a scan to begin.');
  } catch (e) {
    console.error(e);
    setStatus('Error loading score: ' + e.message);
  }

  // After the user corrects a note, re-render leaves dots stale; snap them
  // back with a fresh align pass so the blue dots match the new layout.
  editor.onAfterSave = async () => {
    if (!uploadedFilename) return;
    setStatus('Re-aligning after edit...');
    await runAlignment();
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

  document.getElementById('scanUpload').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setStatus('Uploading scan...');
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

    uploadedFilename = data.filename;
    const imgEl = document.getElementById('scanImage');
    imgEl.src = '/uploads/' + uploadedFilename;

    imgEl.onload = async () => {
      naturalWidth = imgEl.naturalWidth;
      naturalHeight = imgEl.naturalHeight;
      document.getElementById('alignBtn').disabled = false;

      // Reveal the rendered-score panel and render OSMD now that the
      // container has a non-zero width.
      const osmdPanel = document.getElementById('osmdPanel');
      osmdPanel.style.display = '';
      await renderer.render();

      setStatus('Scan loaded. Detecting staves...');
      await runDetection();
      await runAlignment();
    };
  });

  document.getElementById('alignBtn').addEventListener('click', async () => {
    await runAlignment();
  });

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

      // Attach xml_key so the editor can tell the renderer which note to edit.
      const notesWithKey = (payload.notes || []).map((n) => {
        const src = osmdNotes.find((o) => o.noteId === n.note_id);
        return Object.assign({}, n, { xml_key: src && src.xmlKey });
      });
      overlay.render(notesWithKey, naturalWidth, naturalHeight);

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
