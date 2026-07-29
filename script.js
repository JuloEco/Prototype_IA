const promptInput = document.getElementById("prompt");
const lengthInput = document.getElementById("length");
const lengthValue = document.getElementById("length-value");
const tempInput = document.getElementById("temperature");
const tempValue = document.getElementById("temp-value");
const topKInput = document.getElementById("top-k");
const topKValue = document.getElementById("top-k-value");
const topPInput = document.getElementById("top-p");
const topPValue = document.getElementById("top-p-value");
const generateBtn = document.getElementById("generate-btn");
const output = document.getElementById("output");

lengthInput.addEventListener("input", () => {
  lengthValue.textContent = lengthInput.value;
});
tempInput.addEventListener("input", () => {
  tempValue.textContent = parseFloat(tempInput.value).toFixed(1);
});
topKInput.addEventListener("input", () => {
  topKValue.textContent = topKInput.value;
});
topPInput.addEventListener("input", () => {
  topPValue.textContent = parseFloat(topPInput.value).toFixed(2);
});

async function generateText() {
  generateBtn.disabled = true;
  generateBtn.textContent = "Génération en cours...";
  output.textContent = "...";
  output.classList.remove("filled");

  try {
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: promptInput.value,
        length: parseInt(lengthInput.value, 10),
        temperature: parseFloat(tempInput.value),
        top_k: parseInt(topKInput.value, 10),
        top_p: parseFloat(topPInput.value),
      }),
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Erreur serveur");

    output.textContent = data.text;
    output.classList.add("filled");
  } catch (err) {
    output.textContent = "⚠️ " + err.message;
  } finally {
    generateBtn.disabled = false;
    generateBtn.textContent = "Générer";
  }
}

generateBtn.addEventListener("click", generateText);

// ===================== Onglets =====================
const tabButtons = document.querySelectorAll(".tab-btn");
const tabPanels = document.querySelectorAll(".tab-panel");
tabButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    tabButtons.forEach((b) => b.classList.remove("active"));
    tabPanels.forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
  });
});

// ===================== Chat documentaire (/api/ask) =====================
const chatWindow = document.getElementById("chat-window");
const chatInput = document.getElementById("chat-input");
const chatSendBtn = document.getElementById("chat-send-btn");
const modeSelect = document.getElementById("mode");
const chatTopK = document.getElementById("chat-top-k");
const chatTopKValue = document.getElementById("chat-top-k-value");
const minScore = document.getElementById("min-score");
const minScoreValue = document.getElementById("min-score-value");

chatTopK.addEventListener("input", () => {
  chatTopKValue.textContent = chatTopK.value;
});
minScore.addEventListener("input", () => {
  minScoreValue.textContent = parseFloat(minScore.value).toFixed(2);
});

function scrollChatToBottom() {
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function addUserBubble(text) {
  const msg = document.createElement("div");
  msg.className = "msg user";
  msg.innerHTML = `<div class="bubble"></div>`;
  msg.querySelector(".bubble").textContent = text;
  chatWindow.appendChild(msg);
  scrollChatToBottom();
  return msg;
}

function addBotBubble(html) {
  const msg = document.createElement("div");
  msg.className = "msg bot";
  msg.innerHTML = `<div class="bubble">${html}</div>`;
  chatWindow.appendChild(msg);
  scrollChatToBottom();
  return msg;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function renderSources(sources) {
  if (!sources || sources.length === 0) return "";
  const items = sources
    .map(
      (s) =>
        `<div class="source-item">📄 <span class="doc-id">${escapeHtml(s.doc_id)}</span> ` +
        `<span class="score">(score ${s.score.toFixed(2)})</span></div>`
    )
    .join("");
  return `<div class="sources">Sources :${items}</div>`;
}

async function askQuestion() {
  const question = chatInput.value.trim();
  if (!question) return;

  addUserBubble(question);
  chatInput.value = "";
  chatSendBtn.disabled = true;
  const thinkingMsg = addBotBubble(`<span class="typing">Recherche dans les documents...</span>`);

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        mode: modeSelect.value,
        top_k: parseInt(chatTopK.value, 10),
        min_score: parseFloat(minScore.value),
      }),
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Erreur serveur");

    const bubbleEl = thinkingMsg.querySelector(".bubble");
    bubbleEl.classList.remove("typing");

    if (!data.answer) {
      bubbleEl.innerHTML = `🤷 ${escapeHtml(data.note || "Aucune réponse trouvée.")}`;
    } else {
      bubbleEl.innerHTML =
        escapeHtml(data.answer) +
        renderSources(data.sources) +
        (data.note ? `<div class="note">${escapeHtml(data.note)}</div>` : "");
    }
  } catch (err) {
    const bubbleEl = thinkingMsg.querySelector(".bubble");
    bubbleEl.classList.remove("typing");
    bubbleEl.classList.add("error");
    bubbleEl.textContent = "⚠️ " + err.message;
  } finally {
    chatSendBtn.disabled = false;
    scrollChatToBottom();
  }
}

chatSendBtn.addEventListener("click", askQuestion);
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    askQuestion();
  }
});
