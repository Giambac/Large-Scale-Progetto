// study.js — WebSocket client for Conversational Clustering Study UI (EXP-V2-01)
// Handles: study_state, study_projection, study_awaiting_feedback,
//          study_satisfaction_detected, study_ended events.
// Canvas drawing uses plain HTML5 Canvas — no external charting library.
// All user-provided text rendered via textContent (never innerHTML) per T-06-03-05.

(function () {
    'use strict';

    // ── Session ID from URL ───────────────────────────────────────────────────
    var sessionId = window.location.pathname.split('/').pop();

    // ── Global projection state ───────────────────────────────────────────────
    var _allCoords = [];           // list of [x, y] pairs for all items
    var _globalBounds = null;      // {x_min, x_max, y_min, y_max}
    var _clusterColors = {};       // { str(cluster_id): "#hex" }
    var _perCluster = {};          // { str(cluster_id): { item_ids: [...], name, description } }
    var _highlightedItemId = null; // currently highlighted item_id (int or null)

    // ── SocketIO connection ───────────────────────────────────────────────────
    var socket = io();

    socket.on('connect', function () {
        document.getElementById('study-status').textContent = 'Connected. Initializing session...';
    });

    socket.on('disconnect', function () {
        document.getElementById('study-status').textContent = 'Disconnected from server.';
    });

    // ── study_state: render cluster cards ────────────────────────────────────
    socket.on('study_state', function (data) {
        // data = { clusters: [{ id, name, description, sample_items: [{ item_id, text_preview }] }] }
        renderClusterCards(data.clusters);
    });

    // ── study_projection: draw mini-plot canvases ─────────────────────────────
    socket.on('study_projection', function (data) {
        // data = { coords, cluster_colors, global_bounds, per_cluster }
        _allCoords = data.coords;
        _clusterColors = data.cluster_colors;
        _globalBounds = data.global_bounds;
        _perCluster = data.per_cluster;
        renderMiniPlots();
    });

    // ── study_awaiting_feedback: enable input ─────────────────────────────────
    socket.on('study_awaiting_feedback', function () {
        enableFeedback();
        document.getElementById('study-status').textContent = 'Please type your feedback and click Send.';
    });

    // ── study_satisfaction_detected: show confirmation banner ─────────────────
    socket.on('study_satisfaction_detected', function (data) {
        disableFeedback();
        var banner = document.getElementById('satisfaction-banner');
        var msgEl = document.getElementById('satisfaction-message');
        msgEl.textContent = data.message || 'It looks like you are satisfied. End session?';
        banner.classList.add('visible');
        document.getElementById('study-status').textContent = 'Satisfaction detected — please confirm.';
    });

    // ── study_ended: disable input, show complete banner ─────────────────────
    socket.on('study_ended', function (data) {
        disableFeedback();
        hideSatisfactionBanner();
        var completeBanner = document.getElementById('session-complete-banner');
        completeBanner.classList.add('visible');
        var reason = (data && data.convergence_reason) ? data.convergence_reason : 'complete';
        document.getElementById('study-status').textContent = 'Session ended: ' + reason;
    });

    // ── Send feedback button ──────────────────────────────────────────────────
    document.getElementById('send-feedback').addEventListener('click', function () {
        sendFeedback();
    });

    document.getElementById('study-feedback').addEventListener('keydown', function (e) {
        // Ctrl+Enter or Cmd+Enter submits
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            sendFeedback();
        }
    });

    function sendFeedback() {
        var textarea = document.getElementById('study-feedback');
        var text = textarea.value.trim();
        if (!text) return;
        socket.emit('study_feedback', { session_id: sessionId, text: text });
        textarea.value = '';
        disableFeedback();
        document.getElementById('study-status').textContent = 'Feedback sent. Waiting for response...';
    }

    // ── Satisfaction banner actions ───────────────────────────────────────────
    document.getElementById('satisfaction-yes').addEventListener('click', function () {
        hideSatisfactionBanner();
        socket.emit('study_feedback', { session_id: sessionId, text: 'yes' });
        document.getElementById('study-status').textContent = 'Confirmed. Ending session...';
    });

    document.getElementById('satisfaction-no').addEventListener('click', function () {
        hideSatisfactionBanner();
        socket.emit('study_feedback', { session_id: sessionId, text: 'no' });
        document.getElementById('study-status').textContent = 'Continuing session...';
    });

    function hideSatisfactionBanner() {
        document.getElementById('satisfaction-banner').classList.remove('visible');
    }

    // ── Enable / disable feedback input ──────────────────────────────────────
    function enableFeedback() {
        document.getElementById('study-feedback').disabled = false;
        document.getElementById('send-feedback').disabled = false;
        document.getElementById('study-feedback').focus();
    }

    function disableFeedback() {
        document.getElementById('study-feedback').disabled = true;
        document.getElementById('send-feedback').disabled = true;
    }

    // ── Cluster card renderer ─────────────────────────────────────────────────
    function renderClusterCards(clusters) {
        var container = document.getElementById('cluster-cards-container');
        container.innerHTML = '';

        clusters.forEach(function (cluster) {
            var card = document.createElement('div');
            card.className = 'study-cluster-card';

            // Header (clickable to expand/collapse)
            var header = document.createElement('div');
            header.className = 'study-cluster-header';
            header.setAttribute('data-cluster-id', cluster.id);

            var nameSpan = document.createElement('span');
            nameSpan.className = 'study-cluster-name';
            nameSpan.textContent = cluster.name;

            var toggleSpan = document.createElement('span');
            toggleSpan.textContent = '+';
            toggleSpan.style.cssText = 'font-size:1rem;color:#666;';

            header.appendChild(nameSpan);
            header.appendChild(toggleSpan);

            // Description
            var desc = document.createElement('div');
            desc.className = 'study-cluster-desc';
            desc.textContent = cluster.description;

            // Item list (collapsed by default)
            var itemsDiv = document.createElement('div');
            itemsDiv.className = 'study-cluster-items';

            var ul = document.createElement('ul');
            var items = cluster.sample_items || [];
            items.forEach(function (item) {
                var li = document.createElement('li');
                var span = document.createElement('span');
                span.className = 'study-item';
                span.setAttribute('data-item-id', item.item_id);
                span.setAttribute('data-cluster-id', cluster.id);
                // textContent prevents XSS (T-06-03-05: no innerHTML with user text)
                span.textContent = item.text_preview;
                span.title = item.text_preview;

                span.addEventListener('mouseenter', function () {
                    highlightItem(parseInt(span.getAttribute('data-item-id')), parseInt(span.getAttribute('data-cluster-id')));
                });
                span.addEventListener('mouseleave', function () {
                    clearHighlight(parseInt(span.getAttribute('data-cluster-id')));
                });

                li.appendChild(span);
                ul.appendChild(li);
            });
            itemsDiv.appendChild(ul);

            // Toggle expand/collapse on header click
            header.addEventListener('click', function () {
                var isOpen = itemsDiv.classList.contains('open');
                if (isOpen) {
                    itemsDiv.classList.remove('open');
                    toggleSpan.textContent = '+';
                } else {
                    itemsDiv.classList.add('open');
                    toggleSpan.textContent = '-';
                }
            });

            card.appendChild(header);
            card.appendChild(desc);
            card.appendChild(itemsDiv);
            container.appendChild(card);
        });
    }

    // ── Mini-plot renderer ────────────────────────────────────────────────────
    function renderMiniPlots() {
        var container = document.getElementById('mini-plots-container');
        container.innerHTML = '';

        if (!_globalBounds || !_allCoords || _allCoords.length === 0) return;

        Object.keys(_perCluster).forEach(function (clusterId) {
            var clusterData = _perCluster[clusterId];
            var color = _clusterColors[clusterId] || '#888888';

            var wrapper = document.createElement('div');
            wrapper.className = 'mini-plot-wrapper';
            wrapper.setAttribute('data-cluster-id', clusterId);

            var label = document.createElement('div');
            label.className = 'mini-plot-label';
            label.textContent = clusterData.name;
            label.title = clusterData.name;

            var canvas = document.createElement('canvas');
            canvas.className = 'mini-plot-canvas';
            canvas.width = 200;
            canvas.height = 200;
            canvas.setAttribute('data-cluster-id', clusterId);

            wrapper.appendChild(label);
            wrapper.appendChild(canvas);
            container.appendChild(wrapper);

            // Draw initial state (no highlight)
            drawMiniPlot(canvas, clusterData.item_ids, _allCoords, color, _globalBounds, null);
        });
    }

    // ── Highlight item in mini-plot ───────────────────────────────────────────
    function highlightItem(itemId, clusterId) {
        _highlightedItemId = itemId;
        var canvas = document.querySelector('canvas.mini-plot-canvas[data-cluster-id="' + clusterId + '"]');
        if (!canvas) return;
        var clusterData = _perCluster[String(clusterId)];
        if (!clusterData) return;
        var color = _clusterColors[String(clusterId)] || '#888888';
        drawMiniPlot(canvas, clusterData.item_ids, _allCoords, color, _globalBounds, itemId);
    }

    function clearHighlight(clusterId) {
        _highlightedItemId = null;
        var canvas = document.querySelector('canvas.mini-plot-canvas[data-cluster-id="' + clusterId + '"]');
        if (!canvas) return;
        var clusterData = _perCluster[String(clusterId)];
        if (!clusterData) return;
        var color = _clusterColors[String(clusterId)] || '#888888';
        drawMiniPlot(canvas, clusterData.item_ids, _allCoords, color, _globalBounds, null);
    }

    // ── Canvas drawing (plain HTML5 Canvas — no external library) ────────────
    /**
     * drawMiniPlot(canvas, clusterItemIds, allCoords, clusterColor, globalBounds, highlightItemId)
     *
     * All items (outside cluster): small circles, radius 2, opacity 0.15, gray.
     * Cluster items: radius 3, full opacity, clusterColor.
     * Highlighted item (if in cluster): radius 6, brighter fill.
     *
     * Scaling: global bounds -> canvas pixel space with margin.
     */
    function drawMiniPlot(canvas, clusterItemIds, allCoords, clusterColor, globalBounds, highlightItemId) {
        var ctx = canvas.getContext('2d');
        var W = canvas.width;
        var H = canvas.height;
        ctx.clearRect(0, 0, W, H);

        if (!allCoords || allCoords.length === 0) return;

        var xMin = globalBounds.x_min;
        var xMax = globalBounds.x_max;
        var yMin = globalBounds.y_min;
        var yMax = globalBounds.y_max;
        var xRange = xMax - xMin || 1;
        var yRange = yMax - yMin || 1;
        var margin = 10;

        function toCanvas(x, y) {
            return [
                margin + ((x - xMin) / xRange) * (W - 2 * margin),
                margin + ((y - yMin) / yRange) * (H - 2 * margin),
            ];
        }

        // Build a Set of this cluster's item_ids for fast lookup
        var clusterSet = {};
        clusterItemIds.forEach(function (id) { clusterSet[id] = true; });

        // Pass 1: draw all non-cluster items (background, low opacity)
        for (var i = 0; i < allCoords.length; i++) {
            if (clusterSet[i]) continue;
            var pt = toCanvas(allCoords[i][0], allCoords[i][1]);
            ctx.beginPath();
            ctx.arc(pt[0], pt[1], 2, 0, 2 * Math.PI);
            ctx.fillStyle = 'rgba(150,150,150,0.15)';
            ctx.fill();
        }

        // Pass 2: draw cluster items (foreground, full opacity)
        var r = parseInt(clusterColor.slice(1, 3), 16);
        var g = parseInt(clusterColor.slice(3, 5), 16);
        var b = parseInt(clusterColor.slice(5, 7), 16);

        clusterItemIds.forEach(function (itemId) {
            var coord = allCoords[itemId];
            if (!coord) return;
            var pt = toCanvas(coord[0], coord[1]);

            var isHighlighted = (highlightItemId !== null && highlightItemId === itemId);
            if (isHighlighted) {
                // Brighter fill: add 60 to each channel, clamped at 255
                var br = Math.min(255, r + 60);
                var bg = Math.min(255, g + 60);
                var bb = Math.min(255, b + 60);
                ctx.beginPath();
                ctx.arc(pt[0], pt[1], 6, 0, 2 * Math.PI);
                ctx.fillStyle = 'rgb(' + br + ',' + bg + ',' + bb + ')';
                ctx.fill();
                ctx.strokeStyle = '#222';
                ctx.lineWidth = 1.5;
                ctx.stroke();
            } else {
                ctx.beginPath();
                ctx.arc(pt[0], pt[1], 3, 0, 2 * Math.PI);
                ctx.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
                ctx.fill();
            }
        });
    }

}());
