// editor.js — pitch/duration correction modal.
//
// On save we update the overlay dot locally AND ask the renderer to apply
// the change to the MusicXML DOM + reload OSMD, so the right-hand rendered
// score reflects the correction.

class NoteEditor {
  constructor(modalId, overlay, renderer) {
    this.modal = document.getElementById(modalId);
    this.overlay = overlay;
    this.renderer = renderer;
    this.currentNote = null;
    this.currentDot = null;
    this.onAfterSave = null;

    document.getElementById('saveNote').addEventListener('click', () => this.save());
    document.getElementById('cancelEdit').addEventListener('click', () => this.close());

    const pitchSel = document.getElementById('editPitch');
    const names = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'];
    for (let oct = 2; oct <= 6; oct++) {
      names.forEach((n) => {
        const opt = document.createElement('option');
        opt.value = n + oct;
        opt.textContent = n + oct;
        pitchSel.appendChild(opt);
      });
    }
  }

  open(note, dotEl) {
    this.currentNote = note;
    this.currentDot = dotEl;

    const pitchSel = document.getElementById('editPitch');
    const durSel = document.getElementById('editDuration');
    if ([...pitchSel.options].some((o) => o.value === note.pitch)) {
      pitchSel.value = note.pitch;
    }
    if ([...durSel.options].some((o) => o.value === note.duration)) {
      durSel.value = note.duration;
    }

    const rect = dotEl.getBoundingClientRect();
    const modalWidth = 240;
    const viewportWidth = window.innerWidth;
    const left = rect.right + 10 + modalWidth < viewportWidth
      ? rect.right + 10
      : Math.max(10, rect.left - modalWidth - 10);
    this.modal.style.left = left + 'px';
    this.modal.style.top = Math.max(10, rect.top) + 'px';

    this.modal.classList.remove('hidden');
  }

  close() {
    this.modal.classList.add('hidden');
    this.currentNote = null;
    this.currentDot = null;
  }

  async save() {
    if (!this.currentNote) return;
    const newPitch = document.getElementById('editPitch').value;
    const newDuration = document.getElementById('editDuration').value;
    const note = this.currentNote;

    // 1) update the dot locally
    this.overlay.updateDot(note.note_id, newPitch, newDuration);

    // 2) push the edit into the MusicXML + re-render the right panel
    let applied = false;
    if (this.renderer && note.xml_key) {
      applied = await this.renderer.applyNoteEdit(note.xml_key, newPitch, newDuration);
    }

    this.close();
    if (typeof this.onAfterSave === 'function') {
      try { this.onAfterSave(note, { applied }); } catch (_) { /* noop */ }
    }
  }
}
