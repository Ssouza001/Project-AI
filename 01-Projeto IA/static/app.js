const messages = document.querySelector("#messages");
const form = document.querySelector("#chatForm");
const input = document.querySelector("#claimInput");
const sendButton = document.querySelector(".send-button");
const statusPills = document.querySelectorAll("[data-status-pill]");
const brNewsList = document.querySelector("#brNewsList");
const initialMessagesHtml = messages.innerHTML;

const labelMap = {
  false: {
    title: "Provavelmente falsa",
    chip: "Falsa",
    className: "label-false",
  },
  true: {
    title: "Possivelmente verdadeira",
    chip: "Verdadeira",
    className: "label-true",
  },
  mixed: {
    title: "Precisa de contexto",
    chip: "Contexto",
    className: "label-mixed",
  },
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function safeNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatSource(value) {
  if (value === "google_fact_check") {
    return "Google Fact Check";
  }

  if (value === "ml_model_v2") {
    return "Modelo_V2 local";
  }

  if (value === "ml_model_v3") {
    return "Modelo_V3 local";
  }

  return value || "Fonte nao informada";
}

function setStatus(text) {
  statusPills.forEach((statusPill) => {
    statusPill.textContent = text;
  });
}

function scrollToBottom() {
  messages.scrollTop = messages.scrollHeight;
}

function appendMessage(role, content) {
  const article = document.createElement("article");
  article.className = `message ${role === "user" ? "user-message" : "assistant-message"}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = role === "user" ? "EU" : "V";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  if (typeof content === "string") {
    bubble.innerHTML = `<p>${escapeHtml(content)}</p>`;
  } else {
    bubble.append(content);
  }

  article.append(avatar, bubble);
  messages.append(article);
  scrollToBottom();
  return article;
}

function appendLoading() {
  const wrapper = document.createElement("div");
  wrapper.className = "typing";
  wrapper.setAttribute("aria-label", "Analisando");
  wrapper.innerHTML = "<span></span><span></span><span></span>";
  return appendMessage("assistant", wrapper);
}

function createMetaItem(label, value) {
  return `
    <div class="meta-item">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value || "Nao informado")}</strong>
    </div>
  `;
}

function createContextSection(data) {
  const articles = Array.isArray(data.context_articles) ? data.context_articles : [];

  if (!data.context_used || !articles.length) {
    return "";
  }

  const articleItems = articles
    .map((article) => {
      const trusted = article.trusted_for_training
        ? '<span class="context-badge trusted">Pode reforcar treino</span>'
        : '<span class="context-badge review">Revisar antes do treino</span>';
      const link = article.url
        ? `<a href="${escapeHtml(article.url)}" target="_blank" rel="noreferrer">Abrir fonte</a>`
        : "";

      return `
        <li>
          <strong>${escapeHtml(article.title || "Noticia relacionada")}</strong>
          <span>${escapeHtml(article.source_name || "Fonte nao informada")}</span>
          ${article.description ? `<p>${escapeHtml(article.description)}</p>` : ""}
          <div class="context-actions">
            ${trusted}
            ${link}
          </div>
        </li>
      `;
    })
    .join("");

  return `
    <section class="context-box">
      <div class="context-heading">
        <strong>Contexto buscado automaticamente</strong>
        <span>Busca: ${escapeHtml(data.context_keyword || "termos relacionados")}</span>
      </div>
      <ul>${articleItems}</ul>
      <p>${escapeHtml(data.training_note || "As noticias foram salvas como candidatas para auditoria.")}</p>
    </section>
  `;
}

function createResultCard(data) {
  const label = labelMap[data.label] || labelMap.mixed;
  const card = document.createElement("div");
  card.className = "result-card";

  const metaItems = [
    createMetaItem("Fonte", formatSource(data.source)),
  ];

  if (data.publisher) {
    metaItems.push(createMetaItem("Publicador", data.publisher));
  }

  if (data.textual_rating) {
    metaItems.push(createMetaItem("Veredito original", data.textual_rating));
  }

  if (data.context_used) {
    metaItems.push(createMetaItem("Contexto", "Noticias recentes consultadas"));
  }

  const reviewTitle = data.review_title
    ? `<p class="review-title">${escapeHtml(data.review_title)}</p>`
    : "";

  const reviewLink = data.review_url
    ? `<p><a class="review-link" href="${escapeHtml(data.review_url)}" target="_blank" rel="noreferrer">Abrir checagem original</a></p>`
    : "";

  const contextSection = createContextSection(data);

  card.innerHTML = `
    <div class="result-top">
      <div>
        <h3 class="result-title">${label.title}</h3>
        ${reviewTitle}
      </div>
      <span class="label-chip ${label.className}">${label.chip}</span>
    </div>
    <div class="meta-grid">
      ${metaItems.join("")}
    </div>
    <p class="warning">${escapeHtml(data.warning || "Classificacao experimental. Use como apoio, nao como garantia absoluta de veracidade.")}</p>
    ${contextSection}
    ${reviewLink}
  `;

  return card;
}

async function analyzeClaim(claim) {
  const response = await fetch("/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ claim }),
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.error || "Nao foi possivel analisar a afirmacao.");
  }

  return data;
}

function autoResize() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 130)}px`;
}

function resetChat() {
  messages.innerHTML = initialMessagesHtml;
  input.value = "";
  autoResize();
  input.focus();
  scrollToBottom();
}

function renderNewsList(articles) {
  if (!brNewsList) {
    return;
  }

  if (!articles.length) {
    brNewsList.innerHTML = '<span class="news-empty">Nenhuma noticia BR retornada agora.</span>';
    return;
  }

  brNewsList.innerHTML = articles
    .map((article) => {
      const claim = article.claim_text || article.title || "";
      return `
        <button class="news-item" type="button" data-example="${escapeHtml(claim)}">
          <span class="news-source">${escapeHtml(article.source_name || article.source_api || "Fonte")}</span>
          <strong>${escapeHtml(article.title || "Sem titulo")}</strong>
        </button>
      `;
    })
    .join("");
}

async function loadBrazilNews() {
  if (!brNewsList) {
    return;
  }

  brNewsList.innerHTML = '<span class="news-empty">Buscando noticias brasileiras...</span>';

  try {
    const response = await fetch("/news/br?q=eleicao&limit=8");
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Nao foi possivel buscar noticias.");
    }

    renderNewsList(data.articles || []);
  } catch (error) {
    brNewsList.innerHTML = `<span class="news-empty">Falha ao buscar noticias: ${escapeHtml(error.message)}</span>`;
  }
}

async function submitClaim(claim) {
  const cleanClaim = claim.trim();
  if (!cleanClaim || sendButton.disabled) {
    return;
  }

  appendMessage("user", cleanClaim);
  input.value = "";
  autoResize();
  sendButton.disabled = true;

  const loadingMessage = appendLoading();

  try {
    const data = await analyzeClaim(cleanClaim);
    loadingMessage.remove();
    appendMessage("assistant", createResultCard(data));

    if (navigator.vibrate) {
      navigator.vibrate(20);
    }
  } catch (error) {
    loadingMessage.remove();
    appendMessage("assistant", `Nao consegui analisar agora: ${error.message}`);
  } finally {
    sendButton.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  submitClaim(input.value);
});

input.addEventListener("input", autoResize);

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.addEventListener("click", (event) => {
  const exampleButton = event.target.closest("[data-example]");
  const newChatButton = event.target.closest("[data-new-chat]");
  const loadNewsButton = event.target.closest("[data-load-news]");

  if (exampleButton) {
    submitClaim(exampleButton.dataset.example || "");
  }

  if (newChatButton) {
    resetChat();
  }

  if (loadNewsButton) {
    loadBrazilNews();
  }
});

fetch("/health")
  .then((response) => response.json())
  .then((data) => {
    setStatus(data.google_factcheck_enabled ? "API + Modelo" : "Modelo local");
  })
  .catch(() => {
    setStatus("Offline?");
  });

autoResize();
loadBrazilNews();
