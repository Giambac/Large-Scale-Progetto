// study.js — WebSocket client for Conversational Clustering UI (study and watch modes).
// Study mode (default): study_state, study_projection, study_awaiting_feedback,
//                       study_satisfaction_detected, study_ended.
// Watch mode (body[data-mode="watch"]): same cluster/projection rendering, plus
//                                       watch_agent_message and watch_oracle_turn
//                                       chat bubbles. Human-input controls are absent
//                                       on the watch page, so their handlers are skipped.
// Canvas drawing uses plain HTML5 Canvas — no external charting library.
// All user-provided text rendered via textContent (never innerHTML) per T-06-03-05.

(function () {
    'use strict';

    // ── Mode detection (study | watch) ────────────────────────────────────────
    var MODE = (document.body && document.body.dataset && document.body.dataset.mode) || 'study';
    var IS_WATCH = (MODE === 'watch');
    console.log('[study.js v=2026-05-18-5] mode=' + MODE + ' path=' + window.location.pathname);

    // Global error reporter — any uncaught error from this script is surfaced
    // in the chat-status (watch mode) or study-status (study mode) bar so it's
    // visible without opening DevTools.
    window.addEventListener('error', function (e) {
        var sb = document.getElementById('chat-status') || document.getElementById('study-status');
        if (sb) sb.textContent = '[js error] ' + (e.message || e) + ' @ ' + (e.filename || '?') + ':' + (e.lineno || '?');
        console.error('[uncaught]', e);
    });

    // ── Session ID from URL ───────────────────────────────────────────────────
    var sessionId = window.location.pathname.split('/').pop();

    // ── Global projection state ───────────────────────────────────────────────
    var _allCoords = [];           // list of [x, y] pairs for all items
    var _globalBounds = null;      // {x_min, x_max, y_min, y_max}
    var _clusterColors = {};       // { str(cluster_id): "#hex" }
    var _perCluster = {};          // { str(cluster_id): { item_ids: [...], name, description } }
    var _highlightedItemId = null; // currently highlighted item_id (int or null)

    // ── Status writer (works in both modes) ──────────────────────────────────
    function setStatus(text) {
        var el = document.getElementById(IS_WATCH ? 'chat-status' : 'study-status');
        if (el) el.textContent = text;
    }

    // ── SocketIO connection ───────────────────────────────────────────────────
    var socket = io();

    socket.on('connect', function () {
        setStatus(IS_WATCH ? 'Connected. Waiting for agent...' : 'Connected. Initializing session...');
        if (IS_WATCH) hydrateWatchFromReplay();
    });

    // In watch mode, fetch any already-emitted projection/state/chat from the
    // server cache. SocketIO has no replay-on-connect, so the page would
    // otherwise miss everything emitted before the browser opened the socket.
    function hydrateWatchFromReplay() {
        function showHydrateError(stage, err) {
            console.error('[hydrate] ' + stage, err);
            var msg = '[hydrate ' + stage + ' error] ' +
                (err && err.message ? err.message : String(err));
            setStatus(msg);
            var cContainer = document.getElementById('cluster-cards-container');
            if (cContainer) cContainer.innerHTML = '<p style="color:#a00;font-size:0.78rem;">' + msg + '</p>';
            var pContainer = document.getElementById('mini-plots-container');
            if (pContainer) pContainer.innerHTML = '<p style="color:#a00;font-size:0.78rem;">' + msg + '</p>';
        }

        setStatus('Hydrating from /replay...');
        fetch('/watch/' + sessionId + '/replay')
            .then(function (r) {
                if (!r.ok) throw new Error('HTTP ' + r.status);
                return r.json();
            })
            .then(function (data) {
                console.log('[hydrate] replay payload', {
                    projection: !!data.projection,
                    state_clusters: data.state ? data.state.clusters.length : null,
                    chat: data.chat ? data.chat.length : 0,
                    ended: !!data.ended,
                });
                setStatus('Hydrated. projection=' + (!!data.projection) +
                          ' clusters=' + (data.state ? data.state.clusters.length : 0) +
                          ' chat=' + (data.chat ? data.chat.length : 0));
                if (data.projection) {
                    _allCoords = data.projection.coords;
                    _clusterColors = data.projection.cluster_colors;
                    _globalBounds = data.projection.global_bounds;
                    _perCluster = data.projection.per_cluster;
                    renderMiniPlots();
                }
                if (data.state && data.state.clusters) {
                    renderClusterCards(data.state.clusters);
                }
                if (data.chat && data.chat.length) {
                    var log = document.getElementById('chat-log');
                    if (log) log.innerHTML = '';
                    data.chat.forEach(function (b) {
                        appendBubble(b.role, b.turn, b.text, !!b.satisfied);
                    });
                }
                if (data.ended) {
                    var banner = document.getElementById('session-complete-banner');
                    if (banner) banner.classList.add('visible');
                }
            })
            .catch(function (err) { showHydrateError('fetch', err); });
    }

    socket.on('disconnect', function () {
        setStatus('Disconnected from server.');
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

    // ── Study-mode handlers (skipped in watch mode) ──────────────────────────
    if (!IS_WATCH) {
        // study_awaiting_feedback: enable input
        socket.on('study_awaiting_feedback', function () {
            enableFeedback();
            setStatus('Please type your feedback and click Send.');
        });

        // study_satisfaction_detected: show confirmation banner
        socket.on('study_satisfaction_detected', function (data) {
            disableFeedback();
            var banner = document.getElementById('satisfaction-banner');
            var msgEl = document.getElementById('satisfaction-message');
            msgEl.textContent = data.message || 'It looks like you are satisfied. End session?';
            banner.classList.add('visible');
            setStatus('Satisfaction detected — please confirm.');
        });

        document.getElementById('send-feedback').addEventListener('click', sendFeedback);
        document.getElementById('study-feedback').addEventListener('keydown', function (e) {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') sendFeedback();
        });
        document.getElementById('satisfaction-yes').addEventListener('click', function () {
            hideSatisfactionBanner();
            socket.emit('study_feedback', { session_id: sessionId, text: 'yes' });
            setStatus('Confirmed. Ending session...');
        });
        document.getElementById('satisfaction-no').addEventListener('click', function () {
            hideSatisfactionBanner();
            socket.emit('study_feedback', { session_id: sessionId, text: 'no' });
            setStatus('Continuing session...');
        });
    }

    // ── Watch-mode handlers: chat bubbles ────────────────────────────────────
    if (IS_WATCH) {
        socket.on('watch_agent_message', function (data) {
            appendBubble('agent', data.turn, data.text, false);
        });
        socket.on('watch_oracle_turn', function (data) {
            appendBubble('oracle', data.turn, data.text, !!data.satisfied);
        });
    }

    function appendBubble(role, turn, text, satisfied) {
        var log = document.getElementById('chat-log');
        if (!log) return;
        var wrap = document.createElement('div');
        wrap.className = 'bubble bubble-' + role + (satisfied ? ' satisfied' : '');
        var meta = document.createElement('div');
        meta.className = 'bubble-meta';
        meta.textContent = (role === 'agent' ? 'Clustering Agent' : 'Oracle') +
            ' · turn ' + turn + (satisfied ? ' · [SATISFIED]' : '');
        var body = document.createElement('div');
        body.textContent = text;
        wrap.appendChild(meta);
        wrap.appendChild(body);
        log.appendChild(wrap);
        log.scrollTop = log.scrollHeight;
    }

    // ── study_ended: shared between modes ────────────────────────────────────
    socket.on('study_ended', function (data) {
        if (!IS_WATCH) {
            disableFeedback();
            hideSatisfactionBanner();
        }
        var completeBanner = document.getElementById('session-complete-banner');
        if (completeBanner) completeBanner.classList.add('visible');
        var reason = (data && data.convergence_reason) ? data.convergence_reason : 'complete';
        setStatus('Session ended: ' + reason);
    });

    function sendFeedback() {
        var textarea = document.getElementById('study-feedback');
        var text = textarea.value.trim();
        if (!text) return;
        socket.emit('study_feedback', { session_id: sessionId, text: text });
        textarea.value = '';
        disableFeedback();
        setStatus('Feedback sent. Waiting for response...');
    }

    function hideSatisfactionBanner() {
        var b = document.getElementById('satisfaction-banner');
        if (b) b.classList.remove('visible');
    }

    function enableFeedback() {
        var ta = document.getElementById('study-feedback');
        var btn = document.getElementById('send-feedback');
        if (!ta || !btn) return;
        ta.disabled = false;
        btn.disabled = false;
        ta.focus();
    }

    function disableFeedback() {
        var ta = document.getElementById('study-feedback');
        var btn = document.getElementById('send-feedback');
        if (ta) ta.disabled = true;
        if (btn) btn.disabled = true;
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
        if (!container) {
            console.error('[renderMiniPlots] #mini-plots-container not found');
            return;
        }

        // Diagnostic line so we can tell what the page received without browser dev tools.
        var diag = '[diag] coords=' + (_allCoords ? _allCoords.length : 'null') +
                   ' clusters=' + (_perCluster ? Object.keys(_perCluster).length : 'null') +
                   ' bounds=' + (_globalBounds ? 'ok' : 'null');
        console.log('[renderMiniPlots]', diag);

        if (!_globalBounds || !_allCoords || _allCoords.length === 0 ||
            !_perCluster || Object.keys(_perCluster).length === 0) {
            container.innerHTML = '<p style="color:#a00;font-size:0.78rem;">' +
                'No projection data yet. ' + diag + '</p>';
            return;
        }

        container.innerHTML = '';
        try {
            Object.keys(_perCluster).forEach(function (clusterId) {
                var clusterData = _perCluster[clusterId];
                var color = _clusterColors[clusterId] || '#888888';

                var wrapper = document.createElement('div');
                wrapper.className = 'mini-plot-wrapper';
                wrapper.setAttribute('data-cluster-id', clusterId);

                var label = document.createElement('div');
                label.className = 'mini-plot-label';
                label.textContent = clusterData.name + ' (' + clusterData.item_ids.length + ')';
                label.title = clusterData.name;

                var canvas = document.createElement('canvas');
                canvas.className = 'mini-plot-canvas';
                canvas.width = 200;
                canvas.height = 200;
                canvas.setAttribute('data-cluster-id', clusterId);

                wrapper.appendChild(label);
                wrapper.appendChild(canvas);
                container.appendChild(wrapper);

                drawMiniPlot(canvas, clusterData.item_ids, _allCoords, color, _globalBounds, null);
            });
        } catch (err) {
            console.error('[renderMiniPlots] error', err);
            var p = document.createElement('p');
            p.style.cssText = 'color:#a00;font-size:0.78rem;white-space:pre-wrap;';
            p.textContent = '[render error] ' + (err && err.message ? err.message : String(err)) +
                '\n' + diag;
            container.appendChild(p);
        }
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
