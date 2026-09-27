// Угадайка: рисуем — модель (onnxruntime-web, WASM) угадывает в браузере. Без сети в рантайме.
import * as ort from "./vendor/ort.wasm.bundle.min.mjs";

const C = window.DoodleCore;
const $ = (s) => document.querySelector(s);
// wasmPaths не задаём: загрузчик и .wasm лежат рядом с бандлом и находятся по его import.meta.url
ort.env.wasm.numThreads = 1;                  // без COOP/COEP кросс-origin-isolation нет → один поток

const state = { session: null, labels: null, prompts: [], mode: "free", pre: { size: 64, pad: 2, line_width: 2, epsilon: 2 },
                busy: false, pending: false, timer: 0, score: 0, streak: 0, prompt: null, deadline: 0, tick: 0, ended: false };

async function loadModel() {
  const labels = await (await fetch("model/labels.json")).json();
  state.labels = labels;
  state.byId = Object.fromEntries(labels.classes.map((c) => [c.id, c]));
  Object.assign(state.pre, labels.preprocess || {});
  try {
    state.prompts = (await (await fetch("model/lesson_prompts.json")).json()).prompts || [];
  } catch { state.prompts = []; }
  const buf = new Uint8Array(await (await fetch("model/model.onnx")).arrayBuffer());
  state.session = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  $("#status").textContent = `Модель готова: ${labels.classes.length} классов (прогон ${labels.run || "?"}).`;
}

async function infer(drawing) {
  const p = state.pre;
  const img = C.preprocess(drawing, p.size, p.pad, p.line_width, p.epsilon);
  const t0 = performance.now();
  const out = await state.session.run({ input: new ort.Tensor("float32", img, [1, 1, p.size, p.size]) });
  const probs = out.probs.data;
  return { probs, img, ms: performance.now() - t0 };
}

function preview(img, size) {
  const cv = document.createElement("canvas");
  cv.width = cv.height = size;
  const g = cv.getContext("2d"), id = g.createImageData(size, size);
  for (let i = 0; i < img.length; i++) {
    const v = 255 - Math.round(Math.min(1, Math.max(0, img[i])) * 255);
    id.data[i * 4] = id.data[i * 4 + 1] = id.data[i * 4 + 2] = v; id.data[i * 4 + 3] = 255;
  }
  g.putImageData(id, 0, 0);
  return cv.toDataURL();
}

function top5(probs) {
  return [...probs.keys()].sort((a, b) => probs[b] - probs[a]).slice(0, 5)
    .map((i) => ({ ...state.labels.classes[i], p: probs[i] }));
}

const pct = (v) => Math.round(v * 100) + "%";

function renderGuesses(top) {
  $("#top5").innerHTML = top.map((t) =>
    `<li class="${t.group || t.kind || ""}"><span>${esc(t.ru)}<span class="id">${esc(t.id)}</span></span>` +
    `<span class="pct">${(t.p * 100).toFixed(1)}%</span><span class="track"><i style="width:${Math.max(1, t.p * 100)}%"></i></span></li>`).join("");
}

function verdict(top) {
  const best = top[0], el = $("#verdict");
  const accepted = best.threshold != null && best.p >= best.threshold && best.id !== "other";
  if (best.id === "rude" && accepted) { el.className = "verdict card rude"; el.innerHTML = `<b>Ой-ой…</b><small>давай нарисуем что-нибудь другое</small>`; return "rude"; }
  if (accepted) { el.className = "verdict card ok"; el.innerHTML = `Это <b>${esc(best.ru).toLowerCase()}</b>! <small>уверен на ${pct(best.p)}</small>`; return best.id; }
  el.className = "verdict card no"; el.innerHTML = `<b>Хм, не разобрал…</b><small>дорисуй ещё немного</small>`; return null;
}

function esc(s) { return String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

// ─────────── петля предсказаний
let sketch, lastRun = 0;
async function predict(final) {
  if (!state.session || sketch.empty) return;
  if (state.busy) { state.pending = true; return; }
  state.busy = true; lastRun = performance.now();
  try {
    const { probs, img, ms } = await infer(sketch.drawing().map((s) => s.slice(0, 2)));
    if (sketch.empty) return;
    $("#preview").src = preview(img, state.pre.size);
    $("#ms").textContent = `${ms.toFixed(0)} мс`;
    const top = top5(probs);
    renderGuesses(top);
    const win = verdict(top);
    if (state.mode === "game" && !state.ended) checkGame(top, win);
  } finally {
    state.busy = false;
    if (state.pending) { state.pending = false; schedule(true); }
  }
}
function schedule(final) {
  clearTimeout(state.deb);
  const wait = final ? 0 : Math.max(0, 250 - (performance.now() - lastRun));
  state.deb = setTimeout(() => predict(final), wait);
}

// ─────────── игра
function newPrompt() {
  const p = state.prompts;
  state.prompt = p.length ? p[Math.floor(Math.random() * p.length)] : { id: "banana", ru: "Банан", ru_acc: "банан" };
  $("#g-word").textContent = state.prompt.ru_acc || state.prompt.ru.toLowerCase();
}
function startGame() {
  sketch.clear(); state.ended = false; newPrompt();
  state.deadline = performance.now() + 20000;
  $("#big-card").hidden = true;
  clearInterval(state.tick);
  state.tick = setInterval(() => {
    const left = Math.max(0, state.deadline - performance.now());
    $("#g-timer").textContent = (left / 1000).toFixed(0);
    if (left <= 0) endGame(false);
  }, 100);
}
function checkGame(top, win) {
  if (win === "rude") { showBig("rude", "🙈", "К директору!"); endGame(false, true); return; }
  if (win && state.prompt && win === state.prompt.id) {
    state.score += 10; state.streak += 1;
    $("#g-score").textContent = state.score; $("#g-streak").textContent = state.streak;
    showBig("win", "🎉", `Это ${(state.prompt.ru_acc || state.prompt.ru).toLowerCase()}!`);
    endGame(true);
  }
}
function endGame(success, silent) {
  if (state.ended) return;
  state.ended = true; clearInterval(state.tick);
  if (!success && !silent) { state.streak = 0; $("#g-streak").textContent = 0; showBig("", "🙈", "Время вышло!"); }
  setTimeout(() => { if (state.mode === "game") startGame(); }, 1800);
}
function showBig(cls, emoji, text) {
  const c = $("#big-card"); c.className = "big-card " + cls; c.hidden = false;
  $("#big-emoji").textContent = emoji; $("#big-text").textContent = text;
}

function setMode(m) {
  state.mode = m;
  document.querySelectorAll("#mode button").forEach((b) => b.classList.toggle("on", b.dataset.m === m));
  $("#game-bar").hidden = m !== "game";
  $("#start").hidden = m !== "game";
  $("#big-card").hidden = true;
  clearInterval(state.tick);
  sketch.clear();
  if (m === "game") { $("#g-score").textContent = state.score = 0; $("#g-streak").textContent = state.streak = 0; startGame(); }
}

// ─────────── старт
window.addEventListener("DOMContentLoaded", async () => {
  sketch = new Sketch($("#canvas"), { onChange: (s, final) => {
    $("#hint").classList.toggle("gone", !s.empty);
    if (s.empty) { $("#top5").innerHTML = ""; $("#verdict").className = "verdict card"; $("#verdict").innerHTML = '<span class="muted">Нарисуй что-нибудь…</span>'; $("#preview").removeAttribute("src"); return; }
    if (!state.ended || state.mode === "free") schedule(final);
  } });
  sketch.color = "#1f1a17";
  $("#undo").addEventListener("click", () => sketch.undo());
  $("#clear").addEventListener("click", () => { sketch.clear(); if (state.mode === "game") startGame(); });
  $("#start").addEventListener("click", startGame);
  document.querySelectorAll("#mode button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.m)));
  document.addEventListener("keydown", (e) => {
    if (/^(INPUT|SELECT|TEXTAREA)$/.test(document.activeElement?.tagName || "")) return;
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { e.preventDefault(); sketch.undo(); }
    else if (e.key === "Escape") sketch.clear();
  });
  try { await loadModel(); } catch (e) { $("#status").textContent = "Не удалось загрузить модель: " + e.message; }
  window.__ready = true;                       // для headless-проверки
});
