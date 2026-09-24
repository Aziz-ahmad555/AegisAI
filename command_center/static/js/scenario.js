// Guided scenario: global banner (always labelled SCENARIO / SIMULATED while a
// scenario is active), step progress, countdown and controls. Other pages
// react to the same state via the "aegis:scenario" DOM event.
(function () {
    var socket = window.aegisSocket;
    if (!socket) return;
    var banner = document.getElementById("scenario-banner");
    var runBtn = document.getElementById("scenario-run");
    var last = null;

    function cmd(action) { socket.emit("scenario_command", { action: action }); }
    if (runBtn) runBtn.onclick = function () { cmd("start"); };
    banner.querySelector('[data-cmd="pause"]').onclick = function () { cmd(last && last.status === "paused" ? "resume" : "pause"); };
    banner.querySelector('[data-cmd="stop"]').onclick = function () { cmd("stop"); };
    banner.querySelector('[data-cmd="reset"]').onclick = function () { cmd("reset"); };

    var labels = { running: "Running", paused: "Paused", finished: "Complete", stopped: "Stopped" };

    function render(st) {
        last = st;
        banner.hidden = !st.active;
        document.body.classList.toggle("is-simulated", !!st.active);
        if (runBtn) runBtn.disabled = st.status === "running" || st.status === "paused";
        if (!st.active) return;

        var total = st.steps.length;
        var current = st.step >= 0 ? st.steps[Math.min(st.step, total - 1)].title : "Starting";
        banner.querySelector(".sc-status").textContent = labels[st.status] || st.status;
        banner.querySelector(".sc-step").textContent = st.step >= 0
            ? "Step " + (Math.min(st.step, total - 1) + 1) + "/" + total + " · " + current
            : "Starting…";
        banner.querySelector(".sc-progress").innerHTML = st.steps.map(function (s) {
            return '<span class="sc-seg sc-' + s.state + '" title="' + window.aegisEscape(s.title) + '"></span>';
        }).join("");

        var extra = "";
        if (st.countdown != null) extra = "Auto-confirming in " + st.countdown + "s — or decide in the panel";
        else if (st.note) extra = st.note;
        else if (st.status === "finished") extra = "Press Reset to return everything to normal";
        banner.querySelector(".sc-extra").textContent = extra;

        var pause = banner.querySelector('[data-cmd="pause"]');
        pause.textContent = st.status === "paused" ? "Resume" : "Pause";
        pause.disabled = st.status !== "running" && st.status !== "paused";
        banner.querySelector('[data-cmd="stop"]').disabled = pause.disabled;
    }

    socket.on("scenario_state", function (st) {
        render(st);
        document.dispatchEvent(new CustomEvent("aegis:scenario", { detail: st }));
    });
})();
