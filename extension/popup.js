const backendInput = document.getElementById('backendUrl');
const snapshotSelect = document.getElementById('snapshotSelect');
const statusDiv = document.getElementById('status');

function setStatus(text, ok) {
  statusDiv.textContent = text;
  statusDiv.className = ok ? 'ok' : (ok === false ? 'err' : '');
}

// Load saved settings
chrome.storage.local.get(['backendUrl', 'snapshotPath'], (items) => {
  backendInput.value = items.backendUrl || 'http://localhost:8000';
  if (items.snapshotPath) {
    // Add current snapshot as an option if list not yet loaded
    const opt = document.createElement('option');
    opt.value = items.snapshotPath;
    opt.textContent = items.snapshotPath.split('/').pop().split('\\').pop();
    opt.selected = true;
    snapshotSelect.appendChild(opt);
  }
  loadSnapshots();
});

async function loadSnapshots() {
  const url = backendInput.value.replace(/\/+$/, '');
  setStatus('Loading snapshots...', null);
  try {
    const resp = await fetch(`${url}/api/snapshots`);
    if (!resp.ok) {
      setStatus(`Backend error: ${resp.status}`, false);
      return;
    }
    const snapshots = await resp.json();
    // Preserve current selection
    const currentVal = snapshotSelect.value;
    snapshotSelect.innerHTML = '<option value="">-- none (no model) --</option>';
    for (const s of snapshots) {
      const opt = document.createElement('option');
      opt.value = s.path;
      opt.textContent = `${s.name}  (${s.directory})`;
      if (s.path === currentVal) opt.selected = true;
      snapshotSelect.appendChild(opt);
    }
    if (snapshots.length > 0) {
      setStatus(`${snapshots.length} snapshots found`, true);
      // Auto-select the last (newest) snapshot if nothing was previously selected
      if (!currentVal && snapshots.length > 0) {
        snapshotSelect.value = snapshots[snapshots.length - 1].path;
      }
    } else {
      setStatus('No .pt files found in snapshot dirs', false);
    }
  } catch (e) {
    setStatus('Cannot reach backend — is it running?', false);
  }
}

document.getElementById('refreshSnapshots').addEventListener('click', async () => {
  await loadSnapshots();
});

document.getElementById('testBtn').addEventListener('click', async () => {
  const url = backendInput.value.replace(/\/+$/, '');
  try {
    const resp = await fetch(`${url}/api/snapshots`);
    if (resp.ok) {
      const snapshots = await resp.json();
      setStatus(`Connected! ${snapshots.length} snapshots available`, true);
    } else {
      setStatus('Backend responded with error ' + resp.status, false);
    }
  } catch (e) {
    setStatus('Cannot reach backend at ' + url, false);
  }
});

document.getElementById('saveBtn').addEventListener('click', () => {
  const settings = {
    backendUrl: backendInput.value.replace(/\/+$/, ''),
    snapshotPath: snapshotSelect.value
  };
  chrome.storage.local.set(settings, () => {
    setStatus('Settings saved!', true);
    setTimeout(() => setStatus(''), 2000);
  });
});
