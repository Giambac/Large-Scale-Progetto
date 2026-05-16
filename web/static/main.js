// main.js — WebSocket client for Clustering Agent Debug UI
// Connects to Flask-SocketIO server; re-renders cluster cards on each 'state_update' event.
// No external dependencies. ES6 syntax only (no transpilation).
//
// soft_probs encoding: dict-of-dicts keyed by cluster_id (not positional index).
// Server payload: { str(item_id): { str(cluster_id): float } }
// Access pattern: softProbs[String(itemId)][String(cluster.id)]
// This is correct after split/merge when cluster IDs become non-contiguous.

const socket = io();

// ── Session list (UI-V2-01, D-28) ────────────────────────────────
function loadSessionsList() {
    fetch('/sessions')
        .then(function (r) { return r.json(); })
        .then(function (sessions) { renderSessionsList(sessions); })
        .catch(function (err) {
            console.error('Failed to load sessions:', err);
        });
}

function renderSessionsList(sessions) {
    const list = document.getElementById('sessions-list');
    if (!list) return;
    list.innerHTML = '';
    if (!sessions || sessions.length === 0) {
        list.innerHTML = '<li class="placeholder">No sessions yet.</li>';
        return;
    }
    sessions.forEach(function (s) {
        const li = document.createElement('li');
        li.className = 'session-item';
        li.title = 'Click to resume session ' + s.session_id;
        li.innerHTML =
            '<span class="session-ts">' + escapeHtml(s.name || s.timestamp) + '</span> ' +
            '<span class="session-meta">' + s.cluster_count + ' clusters, turn ' + s.turn_count + '</span>';
        li.addEventListener('click', function () { resumeSession(s.session_id); });
        list.appendChild(li);
    });
}

function resumeSession(sessionId) {
    document.getElementById('status-banner').textContent = 'Status: Resuming session ' + sessionId + '...';
    fetch('/resume/' + encodeURIComponent(sessionId), { method: 'POST' })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            document.getElementById('status-banner').textContent =
                'Status: Resumed — ' + sessionId + ' (turn ' + data.turn_index + ')';
            // state_update will be emitted by server via SocketIO; UI updates automatically
            loadSessionsList();  // refresh session list to show updated turn count
        })
        .catch(function (err) {
            document.getElementById('status-banner').textContent =
                'Status: Resume failed — ' + err;
        });
}

// ── Connection events ─────────────────────────────────────────────
socket.on('connect', function () {
    document.getElementById('status-banner').textContent = 'Status: Connected';
    loadSessionsList();  // UI-V2-01: refresh session list on (re)connect
    socket.emit('request_progress');
});

socket.on('disconnect', function () {
    document.getElementById('status-banner').textContent = 'Status: Disconnected';
});

// ── State update: re-render cluster cards and metrics ─────────────
socket.on('state_update', function (data) {
    // data = { turn_index, clusters, soft_probs, cognitive_load }
    // soft_probs: { str(item_id): { str(cluster_id): float } }
    document.getElementById('turn-index').textContent = 'Turn: ' + data.turn_index;
    document.getElementById('cognitive-load').textContent =
        'Cognitive Load: ' + (data.cognitive_load || 0).toFixed(3);
    renderClusterCards(data.clusters, data.soft_probs);
    appendHistoryEntry(data.turn_index);
});

socket.on('session_stopped', function (data) {
    document.getElementById('status-banner').textContent =
        'Status: Stopped — ' + data.reason;
});

// ── UMAP projection renderer (VIZ-V2-01) ─────────────────────────
socket.on('projection_update', function (data) {
    // data = { coords, cluster_ids, max_probs, cluster_colors }
    // coords: list of N [x, y] pairs
    // cluster_ids: list of N cluster_id ints
    // max_probs: list of N floats (used for point opacity)
    // cluster_colors: { str(cluster_id): "#rrggbb" }
    drawProjection(data.coords, data.cluster_ids, data.max_probs, data.cluster_colors);
});

// ── Progress update during session initialisation ─────────────────
socket.on('progress_update', function (data) {
    console.log('progress_update received:', data);
    const stages = { embeddings: 1, clustering: 2, umap: 3 };
    const stageNum = stages[data.stage] || '?';
    const pctStr = data.pct > 0 ? ' (' + data.pct + '%)' : '';
    document.getElementById('cluster-cards').innerHTML =
        '<p class="placeholder">[' + stageNum + '/3] ' + escapeHtml(data.msg) + pctStr + '</p>';
});

function drawProjection(coords, clusterIds, maxProbs, clusterColors) {
    const canvas = document.getElementById('projection-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    if (!coords || coords.length === 0) return;

    // Compute bounding box for normalization
    const xs = coords.map(function (p) { return p[0]; });
    const ys = coords.map(function (p) { return p[1]; });
    const xMin = Math.min.apply(null, xs);
    const xMax = Math.max.apply(null, xs);
    const yMin = Math.min.apply(null, ys);
    const yMax = Math.max.apply(null, ys);
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;
    const margin = 20;  // pixels

    function toCanvas(x, y) {
        return [
            margin + ((x - xMin) / xRange) * (W - 2 * margin),
            margin + ((y - yMin) / yRange) * (H - 2 * margin),
        ];
    }

    // Draw each point
    coords.forEach(function (coord, idx) {
        const clusterId = String(clusterIds[idx]);
        const color = clusterColors[clusterId] || '#888888';
        const opacity = 0.3 + 0.7 * (maxProbs[idx] || 0.5);  // range [0.3, 1.0]
        const [cx, cy] = toCanvas(coord[0], coord[1]);

        ctx.beginPath();
        ctx.arc(cx, cy, 3, 0, 2 * Math.PI);
        ctx.fillStyle = hexToRgba(color, opacity);
        ctx.fill();
    });
}

function hexToRgba(hex, alpha) {
    // Convert "#rrggbb" to "rgba(r,g,b,alpha)"
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha.toFixed(2) + ')';
}

// ── Cluster card renderer ─────────────────────────────────────────
function renderClusterCards(clusters, softProbs) {
    const container = document.getElementById('cluster-cards');
    container.innerHTML = '';
    clusters.forEach(function (cluster) {
        const card = document.createElement('div');
        card.className = 'cluster-card';
        card.innerHTML =
            '<h3>Cluster ' + cluster.id + ': ' + escapeHtml(cluster.name) + '</h3>' +
            '<p>' + escapeHtml(cluster.description) + '</p>' +
            '<p>Items: ' + cluster.item_ids.length + '</p>' +
            renderTopItems(cluster, softProbs);
        container.appendChild(card);
    });
}

function renderTopItems(cluster, softProbs) {
    // Show top-5 items by confidence (highest soft_prob for this cluster).
    // softProbs is a dict-of-dicts: { str(item_id): { str(cluster_id): float } }.
    // Always look up by String(cluster.id) — correct after split/merge when IDs are non-contiguous.
    const clusterKey = String(cluster.id);
    const itemScores = cluster.item_ids.map(function (itemId) {
        const itemProbs = softProbs[String(itemId)] || {};
        const confidence = itemProbs[clusterKey] !== undefined ? itemProbs[clusterKey] : 0;
        return { itemId: itemId, confidence: confidence };
    });
    itemScores.sort(function (a, b) { return b.confidence - a.confidence; });
    const top5 = itemScores.slice(0, 5);
    let html = '<ul class="item-list">';
    top5.forEach(function (item) {
        html += '<li>Item ' + item.itemId + ' <span class="prob-bar" style="width:' +
            Math.round(item.confidence * 100) + 'px"></span> ' +
            (item.confidence * 100).toFixed(1) + '%</li>';
    });
    html += '</ul>';
    return html;
}

// ── History entry ─────────────────────────────────────────────────
function appendHistoryEntry(turnIndex) {
    const list = document.getElementById('history-list');
    const li = document.createElement('li');
    li.textContent = 'Turn ' + turnIndex + ' — state updated';
    list.appendChild(li);
    list.scrollTop = list.scrollHeight;
}

// ── Upload form submission ─────────────────────────────────────────
document.getElementById('upload-form').addEventListener('submit', function (e) {
    e.preventDefault();
    const fileInput = document.getElementById('file-input');
    if (!fileInput.files.length) {
        alert('Please select a file first.');
        return;
    }
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('backend', document.getElementById('backend-select').value);
    document.getElementById('status-banner').textContent = 'Status: Uploading...';
    fetch('/upload', { method: 'POST', body: formData })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            document.getElementById('status-banner').textContent =
                'Status: Session started — ' + data.records + ' records';
        })
        .catch(function (err) {
            document.getElementById('status-banner').textContent = 'Status: Upload failed — ' + err;
        });
});

// ── Utilities ─────────────────────────────────────────────────────
function escapeHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;')
        .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

