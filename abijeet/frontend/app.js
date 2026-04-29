const POLL_MS = 10_000;
const MAX_SAMPLES = 12;

const state = {
  samples: [],
  countdown: POLL_MS / 1000,
  timer: null,
};

const elements = {
  syncLabel: document.querySelector("#sync-label"),
  syncDot: document.querySelector("#sync-dot"),
  syncStatus: document.querySelector("#sync-status"),
  totalCount: document.querySelector("#total-count"),
  activeFaces: document.querySelector("#active-faces"),
  uniqueFaces: document.querySelector("#unique-faces"),
  nextSync: document.querySelector("#next-sync"),
  activeWindow: document.querySelector("#active-window"),
  facesBody: document.querySelector("#faces-body"),
  sampleCount: document.querySelector("#sample-count"),
  sampleList: document.querySelector("#sample-list"),
};

async function fetchSummary() {
  setSyncState("pending", "Syncing");

  try {
    const response = await fetch(`/api/summary?ts=${Date.now()}`, {
      cache: "no-store",
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    if (!data.ok) {
      throw new Error(data.error || "Database read failed");
    }

    renderSummary(data);
    setSyncState("ok", "Live");
    resetCountdown(data.poll_seconds || 10);
  } catch (error) {
    setSyncState("error", "Offline");
    elements.syncLabel.textContent = error.message;
    resetCountdown(POLL_MS / 1000);
  }
}

function renderSummary(data) {
  elements.totalCount.textContent = data.total_count;
  elements.activeFaces.textContent = data.active_faces;
  elements.uniqueFaces.textContent = data.unique_faces;
  elements.activeWindow.textContent =
    `${data.active_window_minutes} min active window, latest ${data.row_limit} rows`;
  elements.syncLabel.textContent = `Last synced ${data.server_time}`;

  renderFaces(data.faces);
  recordSample(data);
}

function renderFaces(faces) {
  if (!faces.length) {
    elements.facesBody.innerHTML = `
      <tr>
        <td colspan="6" class="empty">No face records yet.</td>
      </tr>
    `;
    return;
  }

  elements.facesBody.innerHTML = faces.map((face) => `
    <tr>
      <td><span class="face-id">${escapeHtml(face.person_id)}</span></td>
      <td>${renderPhoto(face)}</td>
      <td>${face.count}</td>
      <td>${escapeHtml(face.first_seen)}</td>
      <td>${escapeHtml(face.last_seen)}</td>
      <td>${formatConfidence(face.confidence)}</td>
    </tr>
  `).join("");
}

function renderPhoto(face) {
  if (!face.photo_url) {
    return `<span class="photo-empty">No photo</span>`;
  }

  return `
    <a class="face-photo-link" href="${escapeAttribute(face.photo_url)}" target="_blank" rel="noreferrer">
      <img class="face-photo" src="${escapeAttribute(face.photo_url)}" alt="Saved face photo for ${escapeAttribute(face.person_id)}">
    </a>
  `;
}

function recordSample(data) {
  state.samples.unshift({
    time: data.server_time.split(" ")[1],
    total: data.total_count,
    active: data.active_faces,
  });
  state.samples = state.samples.slice(0, MAX_SAMPLES);

  elements.sampleCount.textContent = `${state.samples.length} samples`;
  elements.sampleList.innerHTML = state.samples.map((sample) => `
    <li>
      <span>${escapeHtml(sample.time)}</span>
      <strong>${sample.total}</strong>
      <em>${sample.active} active</em>
    </li>
  `).join("");
}

function setSyncState(status, label) {
  elements.syncDot.className = `dot ${status}`;
  elements.syncStatus.textContent = label;
}

function resetCountdown(seconds) {
  state.countdown = seconds;
  elements.nextSync.textContent = `${state.countdown}s`;
}

function startCountdown() {
  window.clearInterval(state.timer);
  state.timer = window.setInterval(() => {
    state.countdown = Math.max(0, state.countdown - 1);
    elements.nextSync.textContent = `${state.countdown}s`;
  }, 1000);
}

function formatConfidence(value) {
  if (value === null || value === undefined) {
    return "n/a";
  }
  return `${Math.round(value * 100)}%`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttribute(value) {
  return escapeHtml(value).replaceAll("`", "&#096;");
}

fetchSummary();
startCountdown();
window.setInterval(fetchSummary, POLL_MS);
