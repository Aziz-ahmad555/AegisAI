// Global status bar: one threat level for the whole building, derived from
// live data on every page. Red (CRITICAL) only when there is actual danger:
// a declared fire, or fused sensor risk at CRITICAL.
(function () {
    var socket = window.aegisSocket;
    if (!socket) return;

    var ORDER = ["NORMAL", "ELEVATED", "HIGH", "CRITICAL"];
    var state = { fires: [], blocked: 0, sensorLevel: "NORMAL", sensorScore: 0, pending: 0, connected: false };

    function $(id) { return document.getElementById(id); }
    function max(a, b) { return ORDER.indexOf(a) >= ORDER.indexOf(b) ? a : b; }

    function render() {
        var level = state.sensorLevel;
        var detail;
        if (state.fires.length) {
            level = "CRITICAL";
            detail = "Fire declared: " + state.fires.join(", ");
        } else if (state.pending) {
            level = max(level, "HIGH");
            detail = state.pending + " unconfirmed alert" + (state.pending === 1 ? "" : "s") + " - review now";
        } else if (level !== "NORMAL") {
            detail = "Sensor risk " + state.sensorScore.toFixed(1);
        } else {
            detail = "No active incidents";
        }
        if (!state.connected) detail = "Live link down - data may be stale";

        var threat = $("threat");
        threat.dataset.level = level;
        $("threat-level").textContent = level;
        $("threat-detail").textContent = detail;
        document.title = (level === "NORMAL" ? "" : level + " · ") + document.title.replace(/^(ELEVATED|HIGH|CRITICAL) · /, "");

        $("status-fires").textContent = state.fires.length ? state.fires.join(", ") : "None";
        $("cell-fires").classList.toggle("is-danger", state.fires.length > 0);
        $("status-pending").textContent = String(state.pending);
        $("cell-pending").classList.toggle("is-warn", state.pending > 0);
        $("status-sensors").textContent = state.sensorLevel + " · " + state.sensorScore.toFixed(1);
    }

    socket.on("state_update", function (s) {
        state.fires = s.nodes.filter(function (n) { return n.status === "FIRE"; }).map(function (n) { return n.id; });
        state.blocked = s.blocked_edges.length;
        render();
    });
    socket.on("sensor_update", function (d) {
        state.sensorLevel = d.risk_level;
        state.sensorScore = d.risk_score;
        render();
    });
    socket.on("pending_actions", function (a) { state.pending = a.length; render(); });

    function setConn(on) {
        state.connected = on;
        $("conn-dot").classList.toggle("offline", !on);
        $("conn-text").textContent = on ? "live" : "offline";
        var dot = $("brand-dot");
        if (dot) dot.classList.toggle("offline", !on);
        render();
    }
    socket.on("connect", function () { setConn(true); });
    socket.on("disconnect", function () { setConn(false); });
    if (socket.connected) setConn(true);

    window.aegisStatus = state;   // read-only snapshot for page scripts
})();
