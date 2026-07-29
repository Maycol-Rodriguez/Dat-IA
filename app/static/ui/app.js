// Dat-IA · UI de una sola consulta (estilo OpenCode). Vanilla JS, sin build step.
// Servida same-origin desde FastAPI (/ui), así que las llamadas van a rutas relativas
// ("/query/answer", etc.) sin necesidad de CORS.

const API_BASE = "";

const NUMERIC_TYPES = new Set(["integer", "decimal", "percentage"]);

// ---------------------------------------------------------------------------
// Tema
// ---------------------------------------------------------------------------

(function themeToggle() {
  const KEY = "dat-ia-ui-theme";
  const root = document.documentElement;
  const saved = localStorage.getItem(KEY) || "dark";
  root.setAttribute("data-theme", saved);

  document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("theme-toggle");
    const sync = () => {
      btn.textContent = root.getAttribute("data-theme") === "light" ? "🌙" : "☀️";
    };
    sync();
    btn.addEventListener("click", () => {
      const next = root.getAttribute("data-theme") === "light" ? "dark" : "light";
      root.setAttribute("data-theme", next);
      localStorage.setItem(KEY, next);
      sync();
    });
  });
})();

// ---------------------------------------------------------------------------
// Utilidades de red
// ---------------------------------------------------------------------------

async function getJson(path) {
  try {
    const res = await fetch(API_BASE + path);
    const data = await res.json().catch(() => ({}));
    return { ok: res.ok, status: res.status, data };
  } catch (err) {
    return { ok: false, status: 0, data: { detail: "No se pudo conectar con el backend." } };
  }
}

async function postJson(path, body) {
  try {
    const res = await fetch(API_BASE + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    return { ok: res.ok, status: res.status, data };
  } catch (err) {
    return { ok: false, status: 0, data: { detail: "No se pudo conectar con el backend." } };
  }
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// ---------------------------------------------------------------------------
// Formateo de SQL (sin dependencias): quiebra líneas en cláusulas de primer
// nivel y en cada columna/condición, para que el SQL generado sea legible
// de un vistazo en vez de una sola línea larga.
// ---------------------------------------------------------------------------

const SQL_CLAUSE_KEYWORDS = [
  "SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY", "HAVING", "LIMIT", "OFFSET",
  "UNION ALL", "UNION",
  "LEFT OUTER JOIN", "RIGHT OUTER JOIN", "FULL OUTER JOIN",
  "LEFT JOIN", "RIGHT JOIN", "INNER JOIN", "FULL JOIN", "CROSS JOIN", "JOIN",
].sort((a, b) => b.length - a.length);

const SQL_INDENT = "    ";

function matchClauseKeywordAt(sql, pos) {
  const boundary = /[\s(]/;
  for (const keyword of SQL_CLAUSE_KEYWORDS) {
    const slice = sql.slice(pos, pos + keyword.length);
    if (slice.toUpperCase() !== keyword) continue;

    const before = pos === 0 ? " " : sql[pos - 1];
    const after = sql[pos + keyword.length] || " ";
    if (boundary.test(before) && boundary.test(after)) return keyword;
  }
  return null;
}

function matchWordAt(sql, pos, word) {
  const slice = sql.slice(pos, pos + word.length);
  if (slice.toUpperCase() !== word) return false;

  const before = pos === 0 ? " " : sql[pos - 1];
  const after = sql[pos + word.length] || " ";
  return /[\s(]/.test(before) && /[\s)]/.test(after);
}

/** Reformatea SQL de una sola línea en líneas indentadas por cláusula.
 *  No es un parser completo: respeta paréntesis y comillas para no romper
 *  llamadas a función ni literales, pero no valida sintaxis. */
function formatSql(rawSql) {
  const sql = (rawSql || "").trim();
  if (!sql) return "";

  let depth = 0;
  let inString = false;
  let stringChar = "";
  let output = "";
  let i = 0;

  while (i < sql.length) {
    const ch = sql[i];

    if (inString) {
      output += ch;
      if (ch === stringChar) inString = false;
      i += 1;
      continue;
    }

    if (ch === "'" || ch === '"') {
      inString = true;
      stringChar = ch;
      output += ch;
      i += 1;
      continue;
    }

    if (ch === "(") {
      depth += 1;
      output += ch;
      i += 1;
      continue;
    }

    if (ch === ")") {
      depth = Math.max(0, depth - 1);
      output += ch;
      i += 1;
      continue;
    }

    if (depth === 0) {
      const clause = matchClauseKeywordAt(sql, i);
      if (clause) {
        output = output.replace(/[ \t]+$/, "");
        output += (output ? "\n" : "") + clause.toUpperCase();
        i += clause.length;
        continue;
      }

      if (ch === ",") {
        output += ",\n" + SQL_INDENT;
        i += 1;
        while (sql[i] === " ") i += 1;
        continue;
      }

      if (matchWordAt(sql, i, "AND") || matchWordAt(sql, i, "OR")) {
        const word = sql.slice(i, i + 2).toUpperCase() === "OR" ? "OR" : "AND";
        output = output.replace(/[ \t]+$/, "");
        output += "\n" + SQL_INDENT + word;
        i += word.length;
        continue;
      }
    }

    output += ch;
    i += 1;
  }

  return output
    .split("\n")
    .map((line) => {
      const [, lead, rest] = line.match(/^(\s*)([\s\S]*)$/);
      return lead + rest.replace(/[ \t]+/g, " ").trim();
    })
    .join("\n")
    .trim();
}

// ---------------------------------------------------------------------------
// Estado del backend (pill dentro del prompt card)
// ---------------------------------------------------------------------------

async function refreshStatus() {
  const dot = document.querySelector("#backend-status .status-dot");
  const label = document.querySelector("#backend-status [data-status-label]");

  const health = await getJson("/health");
  if (!health.ok) {
    dot.setAttribute("data-ok", "false");
    label.textContent = "Backend no disponible";
    return;
  }

  const ready = await getJson("/ready");
  if (ready.ok && ready.data.database === "connected") {
    dot.setAttribute("data-ok", "true");
    label.textContent = "API conectada · BD lista";
  } else {
    dot.setAttribute("data-ok", "warn");
    label.textContent = "API conectada · BD no configurada";
  }
}

// ---------------------------------------------------------------------------
// Render del panel de resultado (uno solo, se reemplaza en cada consulta)
// ---------------------------------------------------------------------------

const resultPanel = document.getElementById("result-panel");
const resultQuestionEl = resultPanel.querySelector(".result-question");
const pipelineDetails = resultPanel.querySelector(".pipeline");
const summaryText = resultPanel.querySelector(".pipeline-summary-text");
const stepsContainer = resultPanel.querySelector(".pipeline-steps");
const answerTextEl = resultPanel.querySelector(".answer-text");

function resetResultPanel(question) {
  resultQuestionEl.textContent = question;
  stepsContainer.innerHTML = "";
  answerTextEl.textContent = "";
  pipelineDetails.setAttribute("open", "");
  summaryText.textContent = "Ejecutando pipeline…";
  resultPanel.hidden = false;
}

function addStep(key, title) {
  const tpl = document.getElementById("tpl-step");
  const node = tpl.content.cloneNode(true);
  const step = node.querySelector(".step");
  step.dataset.step = key;
  step.dataset.status = "running";
  step.querySelector(".step-title").textContent = title;
  step.querySelector(".step-state").textContent = "ejecutando…";
  stepsContainer.appendChild(node);
  return stepsContainer.querySelector(`.step[data-step="${key}"]`);
}

function setStepStatus(stepEl, status, stateLabel) {
  stepEl.dataset.status = status;
  stepEl.querySelector(".step-state").textContent = stateLabel;
}

function pill(text, variant) {
  return `<span class="pill ${variant || ""}">${escapeHtml(text)}</span>`;
}

// ---------------------------------------------------------------------------
// Paso 1 — Filtro de seguridad
// Paso 2 — Optimizador de consulta
// Paso 3 — Recuperación de esquema (RAG)
//
// Los tres se pintan a partir de la MISMA respuesta de /query/answer (que
// ya calcula shield/optimizer/retrieval internamente): la UI ya no llama
// /query/shield ni /query/optimize por separado, para no duplicar la
// clasificación de seguridad ni la llamada al optimizer (Gemini) por
// pregunta. Esos dos endpoints siguen existiendo para uso independiente.
// ---------------------------------------------------------------------------

function renderShieldStep(d) {
  const step = addStep("shield", "Filtro de seguridad");
  const shield = d.shield;

  if (!shield) {
    setStepStatus(step, "skipped", "sin datos");
    step.querySelector(".step-body").innerHTML =
      `<p class="step-note">El backend no devolvió datos del filtro de seguridad.</p>`;
    return step;
  }

  const isMalicious = shield.label === "MALICIOUS";
  setStepStatus(step, isMalicious ? "error" : "ok", isMalicious ? "bloqueado" : "aprobado");
  step.querySelector(".step-body").innerHTML = `
    <div class="pill-row">
      ${pill(shield.label, isMalicious ? "danger" : "success")}
      ${pill(`score: ${shield.score.toFixed(4)}`, "")}
    </div>
    <p class="step-note">Modelo: SQLPromptShield (salmane11/SQLPromptShield)</p>
  `;
  return step;
}

function renderOptimizerStep(d) {
  const step = addStep("optimizer", "Optimizador de consulta");
  const opt = d.optimized;

  if (!opt) {
    setStepStatus(step, "skipped", "omitido");
    step.querySelector(".step-body").innerHTML =
      `<p class="step-note">No se llegó a ejecutar el optimizador.</p>`;
    return step;
  }

  setStepStatus(step, "ok", opt.optimizer || "listo");

  const filtersHtml = (opt.filters || [])
    .map((f) => `${escapeHtml(f.field)} ${escapeHtml(f.operator)} ${escapeHtml(f.value)}`)
    .join(", ") || "—";

  const dateRangeHtml = opt.date_range
    ? `${escapeHtml(opt.date_range.start_date)} → ${escapeHtml(opt.date_range.end_date)}`
    : "—";

  step.querySelector(".step-body").innerHTML = `
    <div class="pill-row">
      ${pill(opt.intent || "-", "accent")}
      ${pill(opt.operation || "-", "")}
      ${pill(`optimizer: ${opt.optimizer || "-"}`, "")}
    </div>
    <dl class="kv-grid">
      <dt>Pregunta normalizada</dt><dd>${escapeHtml(opt.normalized_question || "—")}</dd>
      <dt>Métricas</dt><dd>${(opt.metrics || []).map(escapeHtml).join(", ") || "—"}</dd>
      <dt>Filtros</dt><dd>${filtersHtml}</dd>
      <dt>Rango de fechas</dt><dd>${dateRangeHtml}</dd>
      <dt>Agrupación</dt><dd>${(opt.group_by || []).map(escapeHtml).join(", ") || "—"}</dd>
      <dt>Tablas sugeridas</dt><dd>${(opt.suggested_tables || []).map(escapeHtml).join(", ") || "—"}</dd>
    </dl>
  `;
  return step;
}

function renderRetrievalStep(d) {
  const step = addStep("retrieval", "Recuperación de esquema (RAG)");
  const retrieval = d.retrieval;

  if (!retrieval) {
    setStepStatus(step, "skipped", "omitido");
    step.querySelector(".step-body").innerHTML =
      `<p class="step-note">No se ejecutó la recuperación de esquema.</p>`;
    return step;
  }

  const selected = retrieval.selected_tables || [];
  const noneSelected = selected.length === 0;
  setStepStatus(
    step,
    noneSelected ? "warn" : "ok",
    noneSelected ? "0 tablas superaron el umbral" : `${selected.length} tabla(s) seleccionada(s)`
  );

  const sorted = [...(retrieval.candidates || [])].sort((a, b) => a.distance - b.distance);
  const rows = sorted
    .map(
      (c) => `
      <tr>
        <td>${escapeHtml(c.table)}</td>
        <td class="numeric">${c.distance.toFixed(4)}</td>
        <td>${escapeHtml(c.source)}</td>
        <td>${c.passed_threshold ? pill("sí", "success") : pill("no", "danger")}</td>
      </tr>`
    )
    .join("");

  step.querySelector(".step-body").innerHTML = `
    <div class="pill-row">
      ${pill(`umbral: ${retrieval.distance_threshold}`, "accent")}
      ${pill(`seleccionadas: ${selected.join(", ") || "ninguna"}`, "")}
    </div>
    <div class="result-table-wrap">
      <table class="result-table">
        <thead>
          <tr><th>Tabla</th><th class="numeric">Distancia</th><th>Origen</th><th>¿Pasó umbral?</th></tr>
        </thead>
        <tbody>${rows || `<tr><td colspan="4">Sin candidatos.</td></tr>`}</tbody>
      </table>
    </div>
  `;
  return step;
}

// ---------------------------------------------------------------------------
// Paso 4 — Generación + ejecución SQL, y paso 5 — respuesta (/query/answer)
// ---------------------------------------------------------------------------

function renderResultTable(table) {
  if (!table || !table.rows || table.rows.length === 0) return "";

  const head = table.columns
    .map((c) => `<th class="${NUMERIC_TYPES.has(c.type) ? "numeric" : ""}">${escapeHtml(c.label)}</th>`)
    .join("");

  const body = table.rows
    .map((row) => {
      const cells = table.columns
        .map((c) => {
          const cls = NUMERIC_TYPES.has(c.type) ? "numeric" : "";
          return `<td class="${cls}">${escapeHtml(row[c.key] ?? "—")}</td>`;
        })
        .join("");
      return `<tr>${cells}</tr>`;
    })
    .join("");

  return `
    <div class="result-table-wrap">
      <table class="result-table">
        <thead><tr>${head}</tr></thead>
        <tbody>${body}</tbody>
      </table>
    </div>
    <p class="step-note">${table.row_count} fila(s)</p>
  `;
}

function statusPillVariant(status) {
  if (status === "success") return "success";
  if (status === "error" || status === "rejected" || status === "blocked") return "danger";
  if (status === "unknown" || status === "prototype" || status === "no_context") return "warning";
  return "";
}

// ---------------------------------------------------------------------------
// Paso de observabilidad — bucle generar/validar/juzgar y
// guardrail de resultado/groundedness. No son datos que exponga
// ningún otro endpoint: solo viven en la respuesta de /query/answer.
// ---------------------------------------------------------------------------

function renderValidationStep(validationStep, d) {
  const attempts = d.attempts ?? 1;
  const rejected = d.validation === "rejected";
  const warnings = d.warnings || [];

  let status = "ok";
  let stateLabel = attempts > 1 ? `aprobado en el intento ${attempts}` : "aprobado a la primera";
  if (rejected) {
    status = "error";
    stateLabel = `rechazado tras ${attempts} intento(s)`;
  } else if (warnings.length) {
    status = "warn";
    stateLabel = `aprobado con ${warnings.length} advertencia(s)`;
  }
  setStepStatus(validationStep, status, stateLabel);

  const warningsHtml = warnings.length
    ? `<ul class="warnings-list">${warnings.map((w) => `<li>${escapeHtml(w)}</li>`).join("")}</ul>`
    : `<p class="step-note">Sin advertencias del guardrail.</p>`;

  validationStep.querySelector(".step-body").innerHTML = `
    <div class="pill-row">
      ${pill(`intentos: ${attempts}`, attempts > 1 ? "warning" : "success")}
      ${rejected ? pill("rechazado por el juez", "danger") : pill("SQL aprobado", "success")}
    </div>
    ${warningsHtml}
  `;
}

async function runAnswerSteps(question) {
  const result = await postJson("/query/answer", { question });

  if (!result.ok) {
    const detail = result.data.detail || "No se pudo completar la consulta.";
    const failStep = addStep("sql", "Generación y ejecución SQL");
    setStepStatus(failStep, "error", `HTTP ${result.status || "?"}`);
    failStep.querySelector(".step-body").innerHTML = `<p class="step-note">${escapeHtml(detail)}</p>`;
    return { ok: false, answer: detail, status: "error" };
  }

  const d = result.data;

  renderShieldStep(d);

  // El shield y la falta de contexto se resuelven en un solo paso: no
  // tiene sentido pintar SQL/validación/respuesta vacíos cuando el
  // pipeline se detuvo antes de generar nada (igual que ya hacía la UI
  // cuando el shield bloqueaba, antes de tener múltiples llamadas).
  if (d.status === "blocked") {
    return { ok: true, answer: d.answer, status: d.status };
  }

  renderOptimizerStep(d);
  renderRetrievalStep(d);

  if (d.status === "no_context") {
    return { ok: true, answer: d.answer, status: d.status };
  }

  const sqlStep = addStep("sql", "Generación y ejecución SQL");
  const validationStep = addStep("validation", "Validación y guardrail");
  const answerStep = addStep("answer", "Respuesta sintetizada");

  const formattedSql = formatSql(d.sql) || "—";

  setStepStatus(sqlStep, statusPillVariant(d.status) === "danger" ? "error" : "ok", d.status);
  sqlStep.querySelector(".step-body").innerHTML = `
    <div class="pill-row">
      ${pill(d.status, statusPillVariant(d.status))}
      ${d.sources ? pill(`fuentes: ${d.sources}`, "") : ""}
    </div>
    <div class="code-block">
      <button class="copy-btn" type="button">Copiar</button>
      <pre><code>${escapeHtml(formattedSql)}</code></pre>
    </div>
  `;
  sqlStep.querySelector(".copy-btn")?.addEventListener("click", (ev) => {
    navigator.clipboard.writeText(formattedSql);
    ev.target.textContent = "Copiado";
    setTimeout(() => (ev.target.textContent = "Copiar"), 1200);
  });

  renderValidationStep(validationStep, d);

  setStepStatus(answerStep, statusPillVariant(d.status) === "danger" ? "error" : "ok", `${(d.data || []).length} fila(s)`);
  answerStep.querySelector(".step-body").innerHTML = renderResultTable(d.table);

  return { ok: true, answer: d.answer, status: d.status };
}

// ---------------------------------------------------------------------------
// Orquestación de una consulta completa
//
// Una sola llamada a /query/answer basta para todo el pipeline; los pasos
// se pintan todos juntos cuando esa llamada responde (sin reveal
// progresivo por-paso, que exigiría streaming del backend).
// ---------------------------------------------------------------------------

async function handleQuestion(question) {
  resetResultPanel(question);
  resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });

  summaryText.textContent = "Ejecutando pipeline…";
  const answerResult = await runAnswerSteps(question);

  pipelineDetails.removeAttribute("open");

  if (answerResult.status === "blocked") {
    summaryText.textContent = "Bloqueado por el filtro de seguridad";
  } else if (answerResult.status === "no_context") {
    summaryText.textContent = "No se encontró contexto relevante";
  } else {
    summaryText.textContent = "Pipeline completado";
  }

  answerTextEl.textContent = answerResult.answer || "No se recibió respuesta.";
}

// ---------------------------------------------------------------------------
// Composer
// ---------------------------------------------------------------------------

const composer = document.getElementById("composer");
const questionInput = document.getElementById("question-input");
const sendBtn = document.getElementById("send-btn");

function autoGrow() {
  questionInput.style.height = "auto";
  questionInput.style.height = Math.min(questionInput.scrollHeight, 220) + "px";
}

questionInput.addEventListener("input", autoGrow);

questionInput.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey) {
    ev.preventDefault();
    composer.requestSubmit();
  }
});

let isSending = false;

composer.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const question = questionInput.value.trim();
  if (!question || isSending) return;

  isSending = true;
  sendBtn.disabled = true;

  try {
    await handleQuestion(question);
  } finally {
    isSending = false;
    sendBtn.disabled = false;
    questionInput.focus();
  }
});

document.getElementById("clear-result").addEventListener("click", () => {
  resultPanel.hidden = true;
  questionInput.value = "";
  autoGrow();
  questionInput.focus();
});

// ---------------------------------------------------------------------------
// Arranque
// ---------------------------------------------------------------------------

refreshStatus();
questionInput.focus();
