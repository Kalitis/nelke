// Live self-improvement cycle card: streams agent work (tools, edits, tokens)
// over SSE and drives a progress bar + event log.
(function () {
  "use strict";

  function el(id) { return document.getElementById(id); }

  var statusEl = el("cycle-status");
  var objectiveEl = el("cycle-objective");
  var progressEl = el("cycle-progress");
  var logEl = el("improve-log");
  var thoughtEl = el("cycle-thought");

  // Accumulated live agent text (streamed tokens). Reset at each step so the
  // user sees the agent's current "thought" refresh as it works.
  var thought = "";
  function resetThought() {
    thought = "";
    if (thoughtEl) thoughtEl.textContent = "";
  }
  function appendToken(tok) {
    thought += tok;
    if (thoughtEl) {
      thoughtEl.textContent = thought.length > 240 ? thought.slice(-240) : thought;
    }
  }

  // ---- Progress ----------------------------------------------------------
  function setProgress(pct) {
    if (pct < 0) pct = 0;
    if (pct > 100) pct = 100;
    progressEl.style.width = pct + "%";
  }

  // Trail of recent event lines (log grows downward, tail stays visible).
  var lines = [];
  function appendLine(text) {
    lines.push(text);
    if (lines.length > 200) lines.shift();
    logEl.textContent = lines.join("\n");
    logEl.scrollTop = logEl.scrollHeight;
  }

  // ---- Event rendering ---------------------------------------------------
  function render(ev) {
    var kind = ev.kind;
    var msg = ev.message || "";
    var payload = ev.payload || {};
    if (kind === "cycle_start") {
      statusEl.textContent = "running";
      objectiveEl.textContent = ev.payload && ev.payload.objective ? ev.payload.objective : msg;
      appendLine("cycle started — " + msg);
    } else if (kind === "step_start") {
      statusEl.textContent = "step " + (payload.step || "");
      appendLine("—— step " + payload.step + " ——");
      resetThought();
    } else if (kind === "agent_tool") {
      var tool = payload.tool || "";
      var argsText = "";
      if (payload.args) {
        argsText = Object.keys(payload.args).slice(0, 2).map(function (k) {
          var v = String(payload.args[k]);
          if (v.length > 40) v = v.slice(0, 40) + "…";
          return k + "=" + v;
        }).join(", ");
      }
      appendLine("🔧 " + tool + "(" + argsText + ")");
      // files the agent edits/reads show up as distinct highlights
      if (payload.args && payload.args.path) {
        appendLine("📄 " + payload.args.path);
      }
    } else if (kind === "agent_tool_result") {
      appendLine("✔ " + (payload.snippet || msg));
    } else if (kind === "agent_token") {
      // Streamed token of the agent's current reply — show live in the
      // "thought" line instead of flooding the event log.
      appendToken(payload.token || msg || "");
    } else if (kind === "commit") {
      appendLine("✅ committed " + (payload.sha || "") + " — " + msg);
      if (payload.step && payload.total_steps) {
        setProgress((payload.step / payload.total_steps) * 100);
      }
    } else if (kind === "gate") {
      appendLine("⛩ gate — " + msg.split("\n")[0]);
      if (payload.step && payload.total_steps) {
        setProgress((payload.step / payload.total_steps) * 100);
      }
    } else if (kind === "step_ok") {
      appendLine("✅ " + msg);
    } else if (kind === "boot_check_failed") {
      appendLine("❌ " + msg);
    } else if (kind === "ai_review") {
      appendLine("🤖 AI review: " + (payload.verdict || msg));
    } else if (kind === "awaiting_human") {
      statusEl.textContent = "awaiting human";
      appendLine("👤 " + msg);
    } else if (kind === "merged") {
      statusEl.textContent = "merged";
      setProgress(100);
      appendLine("🎉 " + msg);
    } else if (kind === "cycle_error") {
      statusEl.textContent = "error";
      appendLine("❌ " + msg);
    } else if (kind === "human_rejected") {
      statusEl.textContent = "rejected";
      appendLine("❌ " + msg);
    } else {
      if (msg) appendLine("· " + msg);
    }
  }

  // ---- SSE subscription --------------------------------------------------
  function connect() {
    var es = new EventSource("/api/cycles/stream");
    es.addEventListener("cycle_event", function (e) {
      var ev = JSON.parse(e.data);
      render(ev);
    });
    es.addEventListener("cycle_result", function (e) {
      var ev = JSON.parse(e.data);
      statusEl.textContent = ev.status;
      if (ev.status === "merged") setProgress(100);
    });
    es.onerror = function () {
      // transient; try again
    };
  }

  // Prime the card with the currently persisted trace on load.
  function prime() {
    fetch("/api/cycles")
      .then(function (r) { return r.json(); })
      .then(function (cycles) {
        if (!cycles.length) return;
        var c = cycles[cycles.length - 1];
        objectiveEl.textContent = c.objective;
        statusEl.textContent = c.status;
        (c.events || []).forEach(render);
      })
      .catch(function () {});
  }

  connect();
  prime();
})();
