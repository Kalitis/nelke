// Nelke web UI: minimal vanilla JS. SSE for streaming, fetch for actions.
(function () {
  "use strict";

  function el(id) { return document.getElementById(id); }

  // ---- Chat -----------------------------------------------------------------
  var form = el("chat-form");
  if (form) {
    form.addEventListener("submit", function (e) {
      e.preventDefault();
      var text = el("chat-text").value.trim();
      if (!text) return;
      var profile = el("profile") ? el("profile").value : null;
      sendChat(text, profile);
      el("chat-text").value = "";
    });
  }

  function sendChat(text, profile) {
    var transcript = el("transcript");
    var userLine = document.createElement("div");
    userLine.textContent = "you: " + text;
    transcript.appendChild(userLine);
    var answerLine = document.createElement("div");
    answerLine.className = "answer";
    transcript.appendChild(answerLine);

    fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text, profile: profile }),
    }).then(function (resp) {
      // The response is an SSE stream; read it manually from the body reader.
      var reader = resp.body.getReader();
      var decoder = new TextDecoder();
      var buffer = "";
      function pump() {
        reader.read().then(function (chunk) {
          if (chunk.done) return;
          buffer += decoder.decode(chunk.value, { stream: true });
          var parts = buffer.split("\n\n");
          buffer = parts.pop();
          parts.forEach(function (part) {
            handleSsePart(part, answerLine);
          });
          pump();
        });
      }
      pump();
    });
  }

  function handleSsePart(part, answerLine) {
    var eventName = "message";
    var data = "";
    part.split("\n").forEach(function (line) {
      if (line.startsWith("event:")) eventName = line.slice(6).trim();
      else if (line.startsWith("data:")) data += line.slice(5).trim();
    });
    var payload = {};
    try { payload = JSON.parse(data); } catch (e) { payload = { text: data }; }
    if (eventName === "token") {
      answerLine.textContent += payload.text;
    } else if (eventName === "tool") {
      var t = document.createElement("div");
      t.className = "tool";
      t.textContent = "-> " + payload.name + "(" + JSON.stringify(payload.args) + ")";
      answerLine.appendChild(t);
    } else if (eventName === "tool_result") {
      var tr = document.createElement("div");
      tr.className = "tool-result";
      tr.textContent = "=> " + payload.name + ": " + payload.snippet;
      answerLine.appendChild(tr);
    } else if (eventName === "done") {
      var d = document.createElement("div");
      d.className = "tool";
      d.textContent = "tokens: " + (payload.usage ? payload.usage.total_tokens : 0);
      answerLine.appendChild(d);
    } else if (eventName === "error") {
      var err = document.createElement("div");
      err.style.color = "#e5534b";
      err.textContent = "error: " + payload.message;
      answerLine.appendChild(err);
    }
  }

  // ---- Improve --------------------------------------------------------------
  var improveForm = el("improve-form");
  if (improveForm) {
    improveForm.addEventListener("submit", function (e) {
      e.preventDefault();
      var objective = el("improve-objective").value.trim();
      if (!objective) return;
      var autoApprove = el("improve-auto") ? el("improve-auto").checked : false;
      el("improve-log").textContent = "starting cycle…\n";
      fetch("/api/improve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ objective: objective, auto_approve: autoApprove }),
      });
    });
  }

  // ---- Review page ----------------------------------------------------------
  function resolveReview(id, decision) {
    fetch("/api/review/" + id, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decision: decision }),
    }).then(function (r) { return r.json(); }).then(function (data) {
      el("review-result").textContent = data.status || data.error || "done";
    });
  }
  var approveBtn = el("approve-btn");
  var rejectBtn = el("reject-btn");
  if (approveBtn) approveBtn.addEventListener("click", function () { resolveReview(approveBtn.dataset.id, "approved"); });
  if (rejectBtn) rejectBtn.addEventListener("click", function () { resolveReview(rejectBtn.dataset.id, "rejected"); });
})();
