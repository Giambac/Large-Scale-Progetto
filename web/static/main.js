// main.js — WebSocket client for Clustering Agent Debug UI
// Connects to Flask-SocketIO server; re-renders cluster cards on each 'state_update' event.
// No external dependencies. ES6 syntax only (no transpilation).
//
// soft_probs encoding: dict-of-dicts keyed by cluster_id (not positional index).
// Server payload: { str(item_id): { str(cluster_id): float } }
// Access pattern: softProbs[String(itemId)][String(cluster.id)]
// This is correct after split/merge when cluster IDs become non-contiguous.

const socket = io();

// Debug: log tutti gli eventi socket in arrivo
var _origOnevent = socket.onevent;
socket.onevent = function(packet) {
    console.log('[socket event]', packet.data[0], packet.data.length > 1 ? '(data)' : '');
    _origOnevent.call(this, packet);
};

// D-28: pairwise accuracy history for sparkline (last 10 turns)
const _pairwiseHistory = [];
const _SPARKLINE_MAX = 10;

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
            '<span class="session-ts">' + escapeHtml(s.name || s.timestamp) + '</span>' +
            '<br><span class="session-meta">' + s.cluster_count + ' clusters, turn ' + s.turn_count + '</span>';
        li.addEventListener('click', function () { resumeSession(s.session_id); });
        list.appendChild(li);
    });
}

function resumeSession(sessionId) {
    document.getElementById('status-banner').textContent = 'Status: Resuming...';
    fetch('/resume/' + encodeURIComponent(sessionId), { method: 'POST' })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            document.getElementById('status-banner').textContent =
                'Status: Resumed';
            // state_update will be emitted by server via SocketIO; UI updates automatically
            loadSessionsList();  // refresh session list to show updated turn count
        })
        .catch(function (err) {
            document.getElementById('status-banner').textContent =
                'Status: Resume failed';
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
    // data = { turn_index, clusters, soft_probs, cognitive_load, pairwise_accuracy, convergence_signal, contradiction_count }
    // soft_probs: { str(item_id): { str(cluster_id): float } }
    document.getElementById('turn-index').textContent = 'Turn: ' + data.turn_index;
    document.getElementById('cognitive-load').textContent =
        'Cognitive Load: ' + (data.cognitive_load || 0).toFixed(3);
    renderClusterCards(data.clusters, data.soft_probs);
    appendHistoryEntry(data.turn_index);
    // D-28: Judge Agent metrics
    updateJudgeMetrics(data);
});

socket.on('session_stopped', function (data) {
    document.getElementById('status-banner').textContent =
        'Status: Stopped';
    // Update convergence signal display on stop
    const convEl = document.getElementById('convergence-signal');
    if (convEl) {
        convEl.textContent = 'Convergence: ' + data.reason;
        convEl.style.fontWeight = 'bold';
    }
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
    // Render global UMAP into #mini-plots-container as a single full-width canvas.
    const container = document.getElementById('mini-plots-container');
    if (!container) return;
    if (!coords || coords.length === 0) return;

    // Reuse or create the canvas
    var canvas = document.getElementById('projection-canvas-global');
    if (!canvas) {
        canvas = document.createElement('canvas');
        canvas.id = 'projection-canvas-global';
        canvas.style.cssText = 'width:100%;height:100%;border:1px solid #ddd;border-radius:4px;background:#fff;display:block;';
        container.innerHTML = '';
        container.appendChild(canvas);
    }

    // Size to container
    canvas.width  = container.clientWidth  || 600;
    canvas.height = container.clientHeight || 400;

    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    // Bounding box for normalization
    const xs = coords.map(function (p) { return p[0]; });
    const ys = coords.map(function (p) { return p[1]; });
    const xMin = Math.min.apply(null, xs);
    const xMax = Math.max.apply(null, xs);
    const yMin = Math.min.apply(null, ys);
    const yMax = Math.max.apply(null, ys);
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;
    const margin = 24;

    function toCanvas(x, y) {
        return [
            margin + ((x - xMin) / xRange) * (W - 2 * margin),
            margin + ((y - yMin) / yRange) * (H - 2 * margin),
        ];
    }

    coords.forEach(function (coord, idx) {
        const clusterId = String(clusterIds[idx]);
        const color = clusterColors[clusterId] || '#888888';
        const opacity = 0.3 + 0.7 * (maxProbs[idx] || 0.5);
        const pos = toCanvas(coord[0], coord[1]);

        ctx.beginPath();
        ctx.arc(pos[0], pos[1], 3, 0, 2 * Math.PI);
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

// ── D-28: Judge Agent metrics (contradiction, convergence, pairwise accuracy) ─
function updateJudgeMetrics(data) {
    // Contradiction count
    const cCount = document.getElementById('contradiction-count');
    if (cCount) {
        cCount.textContent = 'Contradictions: ' + (data.contradiction_count !== undefined ? data.contradiction_count : '—');
    }

    // Convergence signal
    const convEl = document.getElementById('convergence-signal');
    if (convEl) {
        const signal = data.convergence_signal || 'running';
        convEl.textContent = 'Convergence: ' + signal;
        // Visual indicator: highlight when stopped
        convEl.style.fontWeight = (signal !== 'running') ? 'bold' : 'normal';
        convEl.style.color = (signal === 'oracle_satisfied') ? '#3cb44b'
            : (signal === 'turn_budget') ? '#f58231'
            : (signal === 'diminishing_returns') ? '#911eb4'
            : '';
    }

    // Pairwise accuracy value
    const paEl = document.getElementById('pairwise-accuracy');
    if (paEl) {
        const pa = data.pairwise_accuracy;
        paEl.textContent = 'Pairwise Accuracy: ' + (pa !== undefined && pa !== null ? (pa * 100).toFixed(1) + '%' : '—');
    }

    // Sparkline — push to history, keep last 10
    if (data.pairwise_accuracy !== undefined && data.pairwise_accuracy !== null) {
        _pairwiseHistory.push(data.pairwise_accuracy);
        if (_pairwiseHistory.length > _SPARKLINE_MAX) {
            _pairwiseHistory.shift();
        }
        renderSparkline(_pairwiseHistory);
    }
}

function renderSparkline(history) {
    const container = document.getElementById('sparkline');
    if (!container) return;
    container.innerHTML = '';
    if (history.length === 0) return;
    const maxVal = Math.max.apply(null, history) || 1;
    const H = 20; // px height
    history.forEach(function (v) {
        const bar = document.createElement('div');
        const h = Math.max(2, Math.round((v / maxVal) * H));
        bar.style.cssText = 'width:6px;background:#4363d8;border-radius:1px;height:' + h + 'px;';
        bar.title = (v * 100).toFixed(1) + '%';
        container.appendChild(bar);
    });
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
    formData.append('backend', document.getElementById('watch-backend').value);
    document.getElementById('status-banner').textContent = 'Status: Uploading...';
    fetch('/upload', { method: 'POST', body: formData })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            document.getElementById('status-banner').textContent =
                'Status: Started';
            // Aggiorna il dataset_path per Start Watch
            if (data.dataset_path) {
                var dsField = document.getElementById('watch-dataset');
                if (dsField) dsField.value = data.dataset_path;
            }
        })
        .catch(function (err) {
            document.getElementById('status-banner').textContent = 'Status: Upload failed';
        });
});

// ── Utilities ─────────────────────────────────────────────────────
function escapeHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;')
        .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}


// ── Watch mode: cluster e projection ─────────────────────────────
socket.on('study_state', function (data) {
    // data = { clusters: [{ id, name, description, sample_items }] }
    renderWatchClusterCards(data.clusters);
});

socket.on('study_projection', function (data) {
    // data = { coords, cluster_colors, global_bounds, per_cluster }
    console.log('[study_projection] received, clusters:', data.per_cluster ? Object.keys(data.per_cluster).length : 'null');
    renderWatchMiniPlots(data);
});

socket.on('watch_agent_message', function (data) {
    appendWatchBubble('agent', data.turn, data.text);
});

socket.on('watch_oracle_turn', function (data) {
    appendWatchBubble('oracle', data.turn, data.text);
    if (data.satisfied) {
        var banner = document.getElementById('session-complete-banner');
        if (banner) banner.classList.add('visible');
    }
});

socket.on('study_ended', function (data) {
    document.getElementById('status-banner').textContent = 'Status: Stopped';
    var banner = document.getElementById('session-complete-banner');
    if (banner) banner.classList.add('visible');
    var chatStatus = document.getElementById('chat-status');
    if (chatStatus) chatStatus.textContent = 'Session ended: ' + (data.convergence_reason || '');
});

// ── Watch cluster cards renderer ──────────────────────────────────
function renderWatchClusterCards(clusters) {
    var container = document.getElementById('cluster-cards-container');
    if (!container) return;
    container.innerHTML = '';
    // Svuota i cluster iniziali dell'upload
    var uploadCards = document.getElementById('cluster-cards');
    if (uploadCards) uploadCards.innerHTML = '';
    clusters.forEach(function (cluster) {
        var card = document.createElement('div');
        card.className = 'study-cluster-card';

        var header = document.createElement('div');
        header.className = 'study-cluster-header';

        var name = document.createElement('span');
        name.className = 'study-cluster-name';
        name.textContent = cluster.name;

        var toggle = document.createElement('span');
        toggle.textContent = '+';

        header.appendChild(name);
        header.appendChild(toggle);

        var desc = document.createElement('div');
        desc.className = 'study-cluster-desc';
        desc.textContent = cluster.description || '';

        var itemsDiv = document.createElement('div');
        itemsDiv.className = 'study-cluster-items';
        if (cluster.sample_items && cluster.sample_items.length) {
            var ul = document.createElement('ul');
            cluster.sample_items.forEach(function (item) {
                var li = document.createElement('li');
                var span = document.createElement('span');
                span.className = 'study-item';
                span.textContent = item.text_preview || ('item ' + item.item_id);
                li.appendChild(span);
                ul.appendChild(li);
            });
            itemsDiv.appendChild(ul);
        }

        header.addEventListener('click', function () {
            var isOpen = itemsDiv.classList.contains('open');
            itemsDiv.classList.toggle('open', !isOpen);
            toggle.textContent = isOpen ? '+' : '-';
        });

        card.appendChild(header);
        card.appendChild(desc);
        card.appendChild(itemsDiv);
        container.appendChild(card);
    });
}

// ── Watch mini-plot renderer ──────────────────────────────────────
function renderWatchMiniPlots(data) {
    var container = document.getElementById('mini-plots-container');
    if (!container) { console.error('[renderWatchMiniPlots] container not found'); return; }

    var allCoords = data.coords;
    var clusterColors = data.cluster_colors;
    var globalBounds = data.global_bounds;
    var perCluster = data.per_cluster;

    console.log('[renderWatchMiniPlots] allCoords:', allCoords ? allCoords.length : 'null',
        'globalBounds:', !!globalBounds, 'perCluster keys:', perCluster ? Object.keys(perCluster).length : 'null');

    if (!allCoords || !globalBounds || !perCluster) {
        console.warn('[renderWatchMiniPlots] missing data, skipping render');
        return;
    }

    // Ridisegna tutto da zero — i cluster IDs cambiano ad ogni split/merge
    // quindi un diff per ID non funziona. Il redraw è solo canvas, non UMAP.
    container.innerHTML = '';

    Object.keys(perCluster).forEach(function (clusterId) {
        var clusterData = perCluster[clusterId];
        var color = clusterColors[clusterId] || '#888888';

        var wrapper = document.createElement('div');
        wrapper.className = 'mini-plot-wrapper';
        wrapper.setAttribute('data-cluster-id', clusterId);

        var label = document.createElement('div');
        label.className = 'mini-plot-label';
        label.textContent = clusterData.name + ' (' + clusterData.item_ids.length + ')';

        var canvas = document.createElement('canvas');
        canvas.className = 'mini-plot-canvas';
        canvas.width = 200;
        canvas.height = 200;

        wrapper.appendChild(label);
        wrapper.appendChild(canvas);
        container.appendChild(wrapper);

        drawWatchMiniPlot(canvas, clusterData.item_ids, allCoords, color, globalBounds);
    });
}

function drawWatchMiniPlot(canvas, clusterItemIds, allCoords, clusterColor, globalBounds) {
    var ctx = canvas.getContext('2d');
    var W = canvas.width;
    var H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    var xMin = globalBounds.x_min, xMax = globalBounds.x_max;
    var yMin = globalBounds.y_min, yMax = globalBounds.y_max;
    var xRange = xMax - xMin || 1;
    var yRange = yMax - yMin || 1;
    var margin = 10;

    function toCanvas(x, y) {
        return [
            margin + ((x - xMin) / xRange) * (W - 2 * margin),
            margin + ((y - yMin) / yRange) * (H - 2 * margin),
        ];
    }

    var clusterSet = {};
    clusterItemIds.forEach(function (id) { clusterSet[id] = true; });

    // sfondo grigio
    for (var i = 0; i < allCoords.length; i++) {
        if (clusterSet[i]) continue;
        var pt = toCanvas(allCoords[i][0], allCoords[i][1]);
        ctx.beginPath();
        ctx.arc(pt[0], pt[1], 2, 0, 2 * Math.PI);
        ctx.fillStyle = 'rgba(150,150,150,0.15)';
        ctx.fill();
    }

    // cluster items
    var r = parseInt(clusterColor.slice(1, 3), 16);
    var g = parseInt(clusterColor.slice(3, 5), 16);
    var b = parseInt(clusterColor.slice(5, 7), 16);
    clusterItemIds.forEach(function (itemId) {
        var coord = allCoords[itemId];
        if (!coord) return;
        var pt = toCanvas(coord[0], coord[1]);
        ctx.beginPath();
        ctx.arc(pt[0], pt[1], 3, 0, 2 * Math.PI);
        ctx.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
        ctx.fill();
    });
}

// ── Chat bubble renderer ──────────────────────────────────────────
function appendWatchBubble(role, turn, text) {
    var log = document.getElementById('chat-log');
    if (!log) return;

    var wrap = document.createElement('div');
    var meta = document.createElement('div');
    meta.className = 'bubble-meta';
    meta.textContent = role + ' · turn ' + turn;

    var bubble = document.createElement('div');
    bubble.className = 'bubble ' + (role === 'agent' ? 'bubble-agent' : 'bubble-oracle');
    bubble.textContent = text;

    wrap.appendChild(meta);
    wrap.appendChild(bubble);
    log.appendChild(wrap);
    log.scrollTop = log.scrollHeight;

    var chatStatus = document.getElementById('chat-status');
    if (chatStatus) chatStatus.textContent = role === 'agent' ? 'Agent is thinking...' : 'Oracle replied · turn ' + turn;
}