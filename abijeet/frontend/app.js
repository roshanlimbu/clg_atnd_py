/* ────────────────────────────────────────────────────
   FaceTrack Dashboard — app.js
   Public / Event Attendance System
   Internal team: detected but NEVER logged.
   Random users: captured, stored, shown here.
──────────────────────────────────────────────────── */

const POLL_MS = 10_000;
const MAX_SAMPLES = 12;

const state = {
  currentTab: 'visitors',
  viewMode: 'grid',      // 'grid' | 'table'
  samples: [],
  countdown: POLL_MS / 1000,
  timer: null,
  lastData: null,
};

/* ── Element refs ── */
const $ = id => document.getElementById(id);

/* ── Tab switching ── */
function switchTab(name) {
  state.currentTab = name;
  document.querySelectorAll('.tab-panel').forEach(p => p.hidden = true);
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
  const panel = $(`panel-${name}`);
  const tab = $(`tab-${name}`);
  if (panel) panel.hidden = false;
  if (tab) tab.classList.add('active');
  if (name === 'pending') fetchPendingFaces();
}

/* ── View mode (grid / table) ── */
function setView(mode) {
  state.viewMode = mode;
  $('visitor-grid').hidden = mode !== 'grid';
  $('visitor-table').hidden = mode !== 'table';
  $('view-grid').classList.toggle('active', mode === 'grid');
  $('view-table').classList.toggle('active', mode === 'table');
  if (state.lastData) renderVisitors(state.lastData.faces || []);
}

/* ── Toast notification ── */
function showToast(msg, type = 'ok') {
  const t = $('toast');
  t.textContent = msg;
  t.className = `toast toast-${type} show`;
  t.hidden = false;
  clearTimeout(t._timer);
  t._timer = setTimeout(() => {
    t.classList.remove('show');
    setTimeout(() => { t.hidden = true; }, 320);
  }, 3500);
}

/* ════════════════════════════════════════════════════
   SYNC / FETCH
════════════════════════════════════════════════════ */

async function fetchSummary() {
  setSyncState('pending', 'Syncing');
  try {
    const res = await fetch(`/api/summary?ts=${Date.now()}`, { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || 'Database read failed');

    state.lastData = data;
    renderStats(data);
    renderVisitors(data.faces || []);
    renderInternalTeam(data.internal_team || []);
    recordSample(data);

    setSyncState('ok', 'Live');
    resetCountdown(data.poll_seconds || 10);

    $('today-date').textContent = data.date || '–';
    $('info-last-sync').textContent = data.server_time || '–';
    $('info-window').textContent = `${data.active_window_minutes} min`;
    $('info-row-limit').textContent = data.row_limit;
    $('info-poll').textContent = `${data.poll_seconds || 10}s`;
  } catch (err) {
    setSyncState('error', 'Offline');
    console.error('Fetch summary failed:', err);
  }
  // Always refresh pending count
  fetchPendingFaces();
}

/* ════════════════════════════════════════════════════
   RENDER STATS
════════════════════════════════════════════════════ */

function renderStats(data) {
  animateNumber($('total-count'), parseInt($('total-count').textContent) || 0, data.total_count);
  animateNumber($('active-faces'), parseInt($('active-faces').textContent) || 0, data.active_faces);
  animateNumber($('unique-faces'), parseInt($('unique-faces').textContent) || 0, data.unique_faces);
  const teamCount = (data.internal_team || []).length;
  animateNumber($('team-count'), parseInt($('team-count').textContent) || 0, teamCount);
}

function animateNumber(el, from, to) {
  if (from === to) { el.textContent = to; return; }
  const steps = 20, delta = (to - from) / steps;
  let current = from, step = 0;
  const tick = () => {
    step++;
    current += delta;
    el.textContent = step >= steps ? to : Math.round(current);
    if (step < steps) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

/* ════════════════════════════════════════════════════
   RENDER VISITORS
════════════════════════════════════════════════════ */

function renderVisitors(faces) {
  if (state.viewMode === 'grid') renderGrid(faces);
  else renderTable(faces);
}

function renderGrid(faces) {
  const grid = $('visitor-grid');
  if (!faces.length) {
    grid.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">
          <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2">
            <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>
            <circle cx="9" cy="7" r="4"/>
            <path d="M23 21v-2a4 4 0 0 0-3-3.87"/>
            <path d="M16 3.13a4 4 0 0 1 0 7.75"/>
          </svg>
        </div>
        <p>No visitors detected yet.</p>
        <p class="empty-sub">Faces detected by the live camera will appear here automatically.</p>
      </div>`;
    return;
  }

  const now = new Date();
  grid.innerHTML = faces.map(face => {
    const isActive = isRecentlyActive(face.last_seen, 5);
    const photoHtml = face.photo_url
      ? `<img src="${escAttr(face.photo_url)}" alt="Visitor ${escHtml(face.person_id)}" loading="lazy">`
      : `<div class="visitor-no-photo">
           <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2">
             <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>
           </svg>
         </div>`;
    const activeBadge = isActive ? `<span class="active-badge">Active</span>` : '';
    const displayName = face.display_name ? escHtml(face.display_name) : '';
    return `
      <article class="visitor-card ${isActive ? 'active-visitor' : ''}" title="${escAttr(face.person_id)}">
        <div class="visitor-photo-wrap">
          ${photoHtml}
          ${activeBadge}
        </div>
        <div class="visitor-info">
          <div class="visitor-id">${escHtml(face.person_id)}</div>
          ${displayName ? `<div class="visitor-name">${displayName}</div>` : ''}
          <div class="visitor-meta">
            <span class="visitor-count">×${face.count}</span>
            <span class="visitor-time">${escHtml(face.last_seen || '')}</span>
          </div>
        </div>
      </article>`;
  }).join('');
}

function renderTable(faces) {
  const tbody = $('faces-body');
  if (!faces.length) {
    tbody.innerHTML = `<tr><td colspan="6" class="empty">No visitor records yet.</td></tr>`;
    return;
  }
  tbody.innerHTML = faces.map(face => {
    const photoHtml = face.photo_url
      ? `<a href="${escAttr(face.photo_url)}" target="_blank" rel="noreferrer">
           <img class="face-photo" src="${escAttr(face.photo_url)}" alt="${escAttr(face.person_id)}">
         </a>`
      : `<span class="photo-empty">No photo</span>`;
    const conf = face.confidence != null ? `<span class="conf-badge">${Math.round(face.confidence * 100)}%</span>` : 'n/a';
    return `
      <tr>
        <td>${photoHtml}</td>
        <td><span class="face-id">${escHtml(face.person_id)}</span></td>
        <td>${escHtml(face.first_seen || '–')}</td>
        <td>${escHtml(face.last_seen || '–')}</td>
        <td>${face.count}</td>
        <td>${conf}</td>
      </tr>`;
  }).join('');
}

function isRecentlyActive(lastSeen, minutes) {
  if (!lastSeen) return false;
  try {
    const t = new Date(`${new Date().toDateString()} ${lastSeen}`);
    return (Date.now() - t.getTime()) < minutes * 60 * 1000;
  } catch { return false; }
}

/* ════════════════════════════════════════════════════
   PENDING FACES
════════════════════════════════════════════════════ */

async function fetchPendingFaces() {
  try {
    const res = await fetch(`/api/pending-faces?ts=${Date.now()}`, { cache: 'no-store' });
    if (!res.ok) return;
    const data = await res.json();
    if (!data.ok) return;
    const count = data.pending_count || 0;
    $('pending-count').textContent = `${count} pending`;
    // Badge on nav tab
    const badge = $('pending-badge');
    badge.textContent = count;
    badge.hidden = count === 0;
    if (state.currentTab === 'pending') renderPendingFaces(data.faces || []);
  } catch (err) {
    console.warn('Failed to fetch pending faces:', err);
  }
}

function renderPendingFaces(faces) {
  const container = $('pending-faces');
  if (!faces.length) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">
          <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2">
            <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
          </svg>
        </div>
        <p>No pending faces.</p>
        <p class="empty-sub">New unregistered visitors will queue here for your review.</p>
      </div>`;
    return;
  }
  container.innerHTML = faces.map(face => `
    <article class="pending-card" data-id="${face.id}">
      <div class="pending-card-photo">
        <img src="${escAttr(face.photo_url)}" alt="Pending face #${face.id}" loading="lazy">
      </div>
      <div class="pending-card-body">
        <div class="pending-meta">
          <span>Quality: ${face.quality_score}</span>
          <span>Similarity: ${face.best_similarity}</span>
          <span>Frames: ${face.frames_seen}</span>
        </div>
        <div class="pending-time">${escHtml(face.created_at || '')}</div>
        <input class="pending-name-input" type="text" placeholder="Enter visitor name..."
               data-id="${face.id}" aria-label="Visitor name for face ${face.id}">
        <div class="pending-actions">
          <button class="btn-register" onclick="registerFace(${face.id})" id="reg-btn-${face.id}">
            ✓ Register
          </button>
          <button class="btn-dismiss" onclick="dismissFace(${face.id})" id="dis-btn-${face.id}">
            ✕ Dismiss
          </button>
        </div>
      </div>
    </article>`).join('');
}

async function registerFace(id) {
  const card = document.querySelector(`.pending-card[data-id="${id}"]`);
  if (!card) return;
  const nameInput = card.querySelector('.pending-name-input');
  const name = (nameInput?.value || '').trim();
  if (!name) {
    nameInput.classList.add('error');
    nameInput.placeholder = 'Name is required!';
    nameInput.focus();
    return;
  }
  nameInput.classList.remove('error');
  const btn = $(`reg-btn-${id}`);
  btn.disabled = true; btn.textContent = 'Saving...';
  try {
    const res = await fetch('/api/register-face', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id, name }),
    });
    const data = await res.json();
    if (data.ok) {
      card.classList.add('resolved');
      card.innerHTML = `<div class="pending-resolved">✓ Registered as <strong>${escHtml(data.person_id)}</strong> — ${escHtml(data.display_name)}</div>`;
      showToast(`Registered: ${data.display_name}`, 'ok');
      setTimeout(() => fetchPendingFaces(), 1500);
      fetchSummary();
    } else {
      showToast(data.error || 'Registration failed.', 'err');
      btn.disabled = false; btn.textContent = '✓ Register';
    }
  } catch (err) {
    showToast('Network error: ' + err.message, 'err');
    btn.disabled = false; btn.textContent = '✓ Register';
  }
}

async function dismissFace(id) {
  const card = document.querySelector(`.pending-card[data-id="${id}"]`);
  if (!card) return;
  const btn = $(`dis-btn-${id}`);
  btn.disabled = true; btn.textContent = 'Removing...';
  try {
    const res = await fetch('/api/dismiss-face', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id }),
    });
    const data = await res.json();
    if (data.ok) {
      card.classList.add('dismissed');
      showToast('Face dismissed.', 'ok');
      setTimeout(() => fetchPendingFaces(), 600);
    } else {
      showToast(data.error || 'Dismiss failed.', 'err');
      btn.disabled = false; btn.textContent = '✕ Dismiss';
    }
  } catch (err) {
    showToast('Network error: ' + err.message, 'err');
    btn.disabled = false; btn.textContent = '✕ Dismiss';
  }
}

/* ════════════════════════════════════════════════════
   INTERNAL TEAM
════════════════════════════════════════════════════ */

function renderInternalTeam(members) {
  $('internal-count').textContent = `${members.length} members`;
  $('team-count').textContent = members.length;
  const list = $('internal-list');
  if (!members.length) {
    list.innerHTML = `<li class="team-empty">No team members registered yet.</li>`;
    return;
  }
  list.innerHTML = members.map(m => {
    const name = m.display_name || m.person_id;
    const initial = (name[0] || '?').toUpperCase();
    const avatarHtml = m.photo_url
      ? `<img class="team-avatar" src="${escAttr(m.photo_url)}" alt="${escAttr(name)}">`
      : `<div class="team-avatar-placeholder" aria-hidden="true">${escHtml(initial)}</div>`;
    return `
      <li>
        ${avatarHtml}
        <div class="team-member-info">
          <div class="team-member-name">${escHtml(name)}</div>
          <div class="team-member-id">${escHtml(m.person_id)}</div>
        </div>
        <svg class="team-shield" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" title="Attendance ignored">
          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
        </svg>
      </li>`;
  }).join('');
}

/* ── Internal team form ── */
$('internal-form').addEventListener('submit', async e => {
  e.preventDefault();
  const formData = new FormData(e.target);
  const name = String(formData.get('name') || '').trim();
  const photo = formData.get('photo');
  if (!name || !(photo instanceof File) || photo.size === 0) {
    setStatus('Name and photo are required.', 'error');
    return;
  }
  setStatus('Saving...', 'pending');
  $('submit-btn').disabled = true;
  try {
    const res = await fetch('/api/internal-team', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok || !data.ok) throw new Error(data.error || `HTTP ${res.status}`);
    e.target.reset();
    clearPhotoPreview();
    setStatus('✓ Saved. This face will NOT be counted in attendance.', 'ok');
    showToast(`Team member added: ${data.member?.display_name}`, 'ok');
    await fetchSummary();
  } catch (err) {
    setStatus(err.message, 'error');
    showToast(err.message, 'err');
  } finally {
    $('submit-btn').disabled = false;
  }
});

function setStatus(msg, cls) {
  const el = $('internal-status');
  el.textContent = msg;
  el.className = `form-status ${cls}`;
}

/* ── Photo preview ── */
$('internal-photo').addEventListener('change', e => {
  const file = e.target.files?.[0];
  if (!file) return clearPhotoPreview();
  const url = URL.createObjectURL(file);
  $('photo-preview-img').src = url;
  $('photo-drop').hidden = true;
  $('photo-preview-wrap').hidden = false;
});

$('photo-clear').addEventListener('click', () => {
  $('internal-form').reset();
  clearPhotoPreview();
});

function clearPhotoPreview() {
  $('photo-preview-img').src = '';
  $('photo-preview-wrap').hidden = true;
  $('photo-drop').hidden = false;
}

// Drag-and-drop hint
const drop = $('photo-drop');
drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('drag-over'); });
drop.addEventListener('dragleave', () => drop.classList.remove('drag-over'));
drop.addEventListener('drop', e => {
  e.preventDefault();
  drop.classList.remove('drag-over');
  const file = e.dataTransfer?.files?.[0];
  if (file && file.type.startsWith('image/')) {
    const dt = new DataTransfer();
    dt.items.add(file);
    $('internal-photo').files = dt.files;
    $('internal-photo').dispatchEvent(new Event('change'));
  }
});

/* ════════════════════════════════════════════════════
   ANALYTICS — Sample history + mini chart
════════════════════════════════════════════════════ */

function recordSample(data) {
  state.samples.unshift({ time: (data.server_time || '').split(' ')[1] || '', total: data.total_count, active: data.active_faces });
  state.samples = state.samples.slice(0, MAX_SAMPLES);
  renderSampleList();
  renderChart();
}

function renderSampleList() {
  const list = $('sample-list');
  if (!state.samples.length) {
    list.innerHTML = `<li class="empty">No data yet.</li>`;
    return;
  }
  list.innerHTML = state.samples.map(s => `
    <li>
      <span class="s-time">${escHtml(s.time)}</span>
      <span class="s-active">${s.active} active</span>
      <span class="s-total">${s.total}</span>
    </li>`).join('');
}

function renderChart() {
  const canvas = $('chart-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.offsetWidth || 560;
  const H = 160;
  canvas.width = W;
  canvas.height = H;
  ctx.clearRect(0, 0, W, H);

  const samples = [...state.samples].reverse();
  if (samples.length < 2) return;

  const maxVal = Math.max(...samples.map(s => s.total), 1);
  const padX = 10, padY = 16;
  const chartW = W - padX * 2, chartH = H - padY * 2;

  const pts = samples.map((s, i) => ({
    x: padX + (i / (samples.length - 1)) * chartW,
    y: padY + (1 - s.total / maxVal) * chartH,
  }));

  // Gradient fill
  const grad = ctx.createLinearGradient(0, padY, 0, H);
  grad.addColorStop(0, 'rgba(59,130,246,0.25)');
  grad.addColorStop(1, 'rgba(59,130,246,0)');
  ctx.beginPath();
  ctx.moveTo(pts[0].x, H);
  pts.forEach(p => ctx.lineTo(p.x, p.y));
  ctx.lineTo(pts[pts.length - 1].x, H);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // Line
  ctx.beginPath();
  pts.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
  ctx.strokeStyle = '#3b82f6';
  ctx.lineWidth = 2.5;
  ctx.lineJoin = 'round';
  ctx.stroke();

  // Dots
  pts.forEach(p => {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
    ctx.fillStyle = '#3b82f6';
    ctx.fill();
  });
}

/* ════════════════════════════════════════════════════
   SYNC UI
════════════════════════════════════════════════════ */

function setSyncState(status, label) {
  $('sync-dot').className = `pulse-dot ${status}`;
  $('sync-status').textContent = label;
}

function resetCountdown(secs) {
  state.countdown = secs;
  $('next-sync').textContent = `${secs}s`;
}

function startCountdown() {
  clearInterval(state.timer);
  state.timer = setInterval(() => {
    state.countdown = Math.max(0, state.countdown - 1);
    $('next-sync').textContent = `${state.countdown}s`;
  }, 1000);
}

/* ════════════════════════════════════════════════════
   HELPERS
════════════════════════════════════════════════════ */

function escHtml(v) {
  return String(v ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#039;');
}
function escAttr(v) { return escHtml(v).replace(/`/g,'&#096;'); }

/* ════════════════════════════════════════════════════
   INIT
════════════════════════════════════════════════════ */

fetchSummary();
startCountdown();
setInterval(fetchSummary, POLL_MS);
window.addEventListener('resize', renderChart);
