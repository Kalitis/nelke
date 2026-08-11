// Nelke web UI: minimal vanilla JS. SSE for streaming, fetch for actions.
(function () {
  "use strict";

  function el(id) { return document.getElementById(id); }

  var activeChatId = null;

  // Live/per-turn token usage (mirrors the per-call usage_events in the DB).
  var liveUsage = { prompt: 0, completion: 0, total: 0, calls: 0 };
  function resetLiveUsage() {
    liveUsage = { prompt: 0, completion: 0, total: 0, calls: 0 };
    updateLiveUsage();
  }
  function updateLiveUsage() {
    var el_ = el("live-usage");
    if (el_) el_.textContent = "tokens: " + liveUsage.total;
  }

  // ---- Chat list -----------------------------------------------------------
  function loadChats(selectId) {
    fetch("/api/chats").then(function (r) { return r.json(); }).then(function (chats) {
      var list = el("chat-list");
      list.innerHTML = "";
      chats.forEach(function (chat) {
        var item = document.createElement("button");
        item.type = "button";
        item.className = "chat-item" + (chat.id === activeChatId ? " active" : "");
        item.dataset.id = chat.id;
        item.title = chat.id;

        var title = document.createElement("span");
        title.className = "chat-item-title";
        title.textContent = chat.title || "New chat";

        var meta = document.createElement("span");
        meta.className = "chat-item-count";
        meta.textContent = (chat.message_count || 0) + " msgs";

        item.appendChild(title);
        item.appendChild(meta);
        item.addEventListener("click", function () { selectChat(chat.id); });
        list.appendChild(item);
      });
      if (selectId) selectChat(selectId);
    }).catch(function () {});
  }

  function selectChat(id) {
    activeChatId = id;
    highlightChat(id);
    resetLiveUsage();
    fetch("/api/chats/" + id).then(function (r) { return r.json(); }).then(function (chat) {
      renderTranscript(chat.messages || []);
      el("chat-title").textContent = chat.title || "New chat";
      el("chat-rename").hidden = false;
      el("chat-delete").hidden = false;
      renderChatMemory(chat.memory || []);
      var input = el("chat-text");
      input.disabled = false;
      input.focus();
      el("chat-form").querySelector("button").disabled = false;
    }).catch(function () {});
  }

  function highlightChat(id) {
    var items = document.querySelectorAll(".chat-item");
    items.forEach(function (it) { it.classList.toggle("active", it.dataset.id === id); });
  }

  function renderTranscript(messages) {
    var transcript = el("transcript");
    transcript.innerHTML = "";
    if (!messages.length) {
      var p = document.createElement("p");
      p.className = "empty";
      p.textContent = "No messages yet — say hi.";
      transcript.appendChild(p);
      return;
    }
    messages.forEach(function (m) {
      if (m.role === "user") {
        var u = document.createElement("div");
        u.className = "user-line";
        u.textContent = "you: " + m.content;
        transcript.appendChild(u);
      } else if (m.role === "assistant") {
        var a = document.createElement("div");
        a.className = "answer";
        a.textContent = m.content || "(no answer)";
        transcript.appendChild(a);
      }
    });
    transcript.scrollTop = transcript.scrollHeight;
  }

  function renderChatMemory(files) {
    var panel = el("chat-memory");
    panel.innerHTML = "";
    if (!files || !files.length) { panel.hidden = true; return; }
    panel.hidden = false;
    var h = document.createElement("div");
    h.className = "memory-head";
    h.textContent = "Chat memory (" + files.length + " file" + (files.length === 1 ? "" : "s") + ")";
    panel.appendChild(h);
    files.forEach(function (f) {
      var li = document.createElement("div");
      li.className = "memory-entry";
      li.textContent = f.name + " (" + f.size + " B)";
      panel.appendChild(li);
    });
  }

  // ---- New / rename / delete ------------------------------------------------
  el("new-chat").addEventListener("click", function () {
    fetch("/api/chats", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    }).then(function (r) { return r.json(); }).then(function (data) {
      activeChatId = data.id;
      loadChats(data.id);
    }).catch(function () {});
  });

  el("chat-rename").addEventListener("click", function () {
    var title = prompt("Rename chat:", el("chat-title").textContent);
    if (!title) return;
    fetch("/api/chats/" + activeChatId, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: title }),
    }).then(function () {
      el("chat-title").textContent = title;
      loadChats();
    }).catch(function () {});
  });

  el("chat-delete").addEventListener("click", function () {
    if (!confirm("Delete this chat and its history?")) return;
    fetch("/api/chats/" + activeChatId, { method: "DELETE" }).then(function () {
      activeChatId = null;
      el("chat-title").textContent = "Chats";
      el("chat-rename").hidden = true;
      el("chat-delete").hidden = true;
      el("chat-text").disabled = true;
      el("chat-form").querySelector("button").disabled = true;
      renderChatMemory([]);
      renderTranscript([]);
      loadChats();
    }).catch(function () {});
  });

  // ---- Chat send (SSE) ------------------------------------------------------
  var form = el("chat-form");
  if (form) {
    form.addEventListener("submit", function (e) {
      e.preventDefault();
      var text = el("chat-text").value.trim();
      if (!text || !activeChatId) return;
      var profile = el("profile") ? el("profile").value : null;
      sendChat(text, profile, activeChatId);
      el("chat-text").value = "";
    });
  }

  function sendChat(text, profile, chatId) {
    resetLiveUsage();
    var transcript = el("transcript");
    var userLine = document.createElement("div");
    userLine.className = "user-line";
    userLine.textContent = "you: " + text;
    transcript.appendChild(userLine);
    var answerLine = document.createElement("div");
    answerLine.className = "answer";
    transcript.appendChild(answerLine);
    transcript.scrollTop = transcript.scrollHeight;

    fetch("/api/chats/" + chatId + "/messages", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text, profile: profile }),
    }).then(function (resp) {
      var reader = resp.body.getReader();
      var decoder = new TextDecoder();
      var buffer = "";
      function pump() {
        reader.read().then(function (chunk) {
          if (chunk.done) {
            // history grew — refresh labels/title derived from first user message
            loadChats();
            return;
          }
          buffer += decoder.decode(chunk.value, { stream: true });
          var parts = buffer.split("\n\n");
          buffer = parts.pop();
          parts.forEach(function (part) { handleSsePart(part, answerLine); });
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
    } else if (eventName === "usage") {
      liveUsage.total += payload.total_tokens || 0;
      liveUsage.prompt += payload.prompt_tokens || 0;
      liveUsage.completion += payload.completion_tokens || 0;
      liveUsage.calls += 1;
      updateLiveUsage();
    } else if (eventName === "done") {
      // finalise the counter with the run's aggregate (same cumulative total)
      if (payload.usage) {
        liveUsage.total = payload.usage.total_tokens || liveUsage.total;
      }
      updateLiveUsage();
      var d = document.createElement("div");
      d.className = "tool";
      d.textContent = "tokens: " + liveUsage.total + " (" + liveUsage.calls + " call" + (liveUsage.calls === 1 ? "" : "s") + ")";
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

  // ---- boot ----------------------------------------------------------------
  loadChats();
})();
