// Doodle Lab — редактор датасета рисунков. Без сборки и зависимостей.
// Рисунок хранится в формате Quick Draw raw: штрих = [[xs], [ys], [ts]], ts — мс от первой точки рисунка.
"use strict";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const store = {
  get(k, d) { try { const v = localStorage.getItem("doodle." + k); return v === null ? d : v; } catch { return d; } },
  set(k, v) { try { localStorage.setItem("doodle." + k, v); } catch { /* приватный режим */ } },
};

async function api(path, body) {
  const opt = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch(path, opt);
  let j = null;
  try { j = await r.json(); } catch { /* не JSON */ }
  if (!r.ok) throw new Error((j && j.error) || `HTTP ${r.status}`);
  return j;
}

let toastTimer = 0;
function toast(msg, err = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.toggle("err", err);
  t.classList.add("on");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("on"), err ? 3500 : 1600);
}

// ───────────────────────────────────────────── мелок: восковая текстура (непрозрачная — стыки не темнеют)
const PAPER_RGB = [255, 254, 249];
const BOARD_RGB = [47, 74, 58];
const patternCache = new Map();
function crayonPattern(ctx, hex, bgRgb) {
  const key = hex + bgRgb.join();
  if (patternCache.has(key)) return patternCache.get(key);
  const n = parseInt(hex.slice(1), 16), c = [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  const S = 96, off = document.createElement("canvas");
  off.width = off.height = S;
  const g = off.getContext("2d"), img = g.createImageData(S, S);
  let seed = 1234567;
  const rnd = () => { seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t ^= t + Math.imul(t ^ (t >>> 7), 61 | t); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };   // mulberry32
  for (let i = 0; i < S * S; i++) {
    const r = rnd();
    // зерно: большая часть пикселей — цвет мелка, часть — «просвет» бумаги
    const k = r < 0.12 ? 0.35 + rnd() * 0.3 : r < 0.3 ? 0.08 + rnd() * 0.12 : rnd() * 0.06;
    for (let ch = 0; ch < 3; ch++) img.data[i * 4 + ch] = Math.round(c[ch] + (bgRgb[ch] - c[ch]) * k);
    img.data[i * 4 + 3] = 255;
  }
  g.putImageData(img, 0, 0);
  const p = ctx.createPattern(off, "repeat");
  patternCache.set(key, p);
  return p;
}

// ───────────────────────────────────────────── холст
const LOGICAL_W = 800;                   // логическая ширина листа; координаты рисунка — в этих единицах
const FORMATS = { square: 1, sheet: 706 / 480, board: 418 / 1024 };   // высота / ширина (лист и доска — как в игре)

class Sketch {
  constructor(canvas, opts = {}) {
    this.c = canvas;
    this.ctx = canvas.getContext("2d");
    this.opts = opts;
    this.strokes = [];          // {pts: [[x, y, t]], color}
    this.cur = null;
    this.t0 = null;
    this.pointer = null;
    this.device = null;
    this.color = "#2d3a8c";
    this.format = "square";
    this.bg = PAPER_RGB;
    new ResizeObserver(() => this.resize()).observe(canvas);
    canvas.addEventListener("pointerdown", e => this.down(e));
    canvas.addEventListener("pointermove", e => this.move(e));
    canvas.addEventListener("pointerup", e => this.up(e));
    canvas.addEventListener("pointercancel", e => this.up(e));
    canvas.addEventListener("contextmenu", e => e.preventDefault());
    this.resize();
  }
  get H() { return LOGICAL_W * FORMATS[this.format]; }
  setFormat(f) {
    this.format = f;
    this.c.style.aspectRatio = `1 / ${FORMATS[f]}`;
    this.c.classList.toggle("board", f === "board");
    this.bg = f === "board" ? BOARD_RGB : PAPER_RGB;
    this.clear();
    this.resize();
  }
  resize() {
    const r = this.c.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
    if (this.c.width !== w || this.c.height !== h) {
      this.c.width = w;
      this.c.height = h;
      this.redraw();
    }
  }
  scale() { return this.c.width / LOGICAL_W; }
  pos(e) {
    const r = this.c.getBoundingClientRect();
    return [(e.clientX - r.left) * LOGICAL_W / r.width, (e.clientY - r.top) * this.H / r.height];
  }
  down(e) {
    if (this.pointer !== null) return;
    if (e.pointerType === "mouse" && e.button !== 0) return;
    if (this.device === "pen" && e.pointerType === "touch") return;   // ладонь при рисовании пером
    e.preventDefault();
    this.pointer = e.pointerId;
    try { this.c.setPointerCapture(e.pointerId); } catch { /* ок */ }
    if (!this.device || e.pointerType === "pen") this.device = e.pointerType;
    const now = performance.now();
    if (this.t0 === null) this.t0 = now;
    const [x, y] = this.pos(e);
    this.cur = { pts: [[x, y, Math.round(now - this.t0)]], color: this.color };
    this.strokes.push(this.cur);
    this.drawDot(this.cur.pts[0], this.cur.color);
    this.changed(false);
  }
  move(e) {
    if (e.pointerId !== this.pointer || !this.cur) return;
    e.preventDefault();
    const evs = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
    for (const ce of (evs.length ? evs : [e])) {
      const [x, y] = this.pos(ce);
      const p = this.cur.pts, last = p[p.length - 1];
      if ((x - last[0]) ** 2 + (y - last[1]) ** 2 < 1) continue;      // как в игре: точки ближе 1 ед. — не пишем
      const t = Math.max(last[2], Math.round((ce.timeStamp || performance.now()) - this.t0));
      p.push([x, y, t]);
      this.drawSeg(p, p.length - 1, this.cur.color);
    }
    this.changed(false);
  }
  up(e) {
    if (e.pointerId !== this.pointer) return;
    this.pointer = null;
    this.cur = null;
    this.changed(true);
  }
  lineWidth() { return (this.format === "board" ? 7 : 8) * this.scale(); }
  style(color) {
    const g = this.ctx;
    g.lineCap = "round";
    g.lineJoin = "round";
    g.lineWidth = this.lineWidth();
    const pat = crayonPattern(g, color, this.bg);
    g.strokeStyle = pat;
    g.fillStyle = pat;
  }
  drawDot(p, color) {
    const g = this.ctx, s = this.scale();
    this.style(color);
    g.beginPath();
    g.arc(p[0] * s, p[1] * s, this.lineWidth() / 2, 0, Math.PI * 2);
    g.fill();
  }
  // Сглаживание как в игре: квадратичные сплайны по серединам отрезков.
  drawSeg(pts, i, color) {
    const g = this.ctx, s = this.scale();
    this.style(color);
    g.beginPath();
    if (i === 1) {
      g.moveTo(pts[0][0] * s, pts[0][1] * s);
      g.lineTo((pts[0][0] + pts[1][0]) / 2 * s, (pts[0][1] + pts[1][1]) / 2 * s);
    } else {
      const a = pts[i - 2], c = pts[i - 1], b = pts[i];
      g.moveTo((a[0] + c[0]) / 2 * s, (a[1] + c[1]) / 2 * s);
      g.quadraticCurveTo(c[0] * s, c[1] * s, (c[0] + b[0]) / 2 * s, (c[1] + b[1]) / 2 * s);
    }
    g.stroke();
  }
  drawTail(pts, color) {
    if (pts.length < 2) return;
    const g = this.ctx, s = this.scale(), a = pts[pts.length - 2], b = pts[pts.length - 1];
    this.style(color);
    g.beginPath();
    g.moveTo((a[0] + b[0]) / 2 * s, (a[1] + b[1]) / 2 * s);
    g.lineTo(b[0] * s, b[1] * s);
    g.stroke();
  }
  redraw() {
    this.ctx.clearRect(0, 0, this.c.width, this.c.height);
    for (const st of this.strokes) {
      this.drawDot(st.pts[0], st.color);
      for (let i = 1; i < st.pts.length; i++) this.drawSeg(st.pts, i, st.color);
      if (st !== this.cur) this.drawTail(st.pts, st.color);
    }
  }
  undo() {
    if (this.pointer !== null) return;
    this.strokes.pop();
    if (!this.strokes.length) { this.t0 = null; this.device = null; }
    this.redraw();
    this.changed(true);
  }
  clear() {
    this.strokes = [];
    this.cur = null;
    this.pointer = null;
    this.t0 = null;
    this.device = null;
    this.ctx.clearRect(0, 0, this.c.width, this.c.height);
    this.changed(true);
  }
  get empty() { return this.strokes.length === 0; }
  drawing() {
    const r = v => Math.round(v * 10) / 10;
    return this.strokes.map(s => [s.pts.map(p => r(p[0])), s.pts.map(p => r(p[1])), s.pts.map(p => p[2])]);
  }
  changed(final) {
    if (this.strokes.length && final && this.cur === null && this.strokes[this.strokes.length - 1].pts.length > 1) {
      // хвост сплайна последнего штриха дорисовываем при отпускании
      const st = this.strokes[this.strokes.length - 1];
      this.drawTail(st.pts, st.color);
    }
    this.opts.onChange && this.opts.onChange(this, final);
  }
}

// ───────────────────────────────────────────── состояние
let CFG = null;
const byId = id => CFG.classes.find(c => c.id === id);
const nameOf = c => `${c.ru} (${c.id})`;

function fillClassSelect(sel, current, withCounts = true) {
  const groups = [
    ["Свои классы", CFG.classes.filter(c => c.kind === "custom")],
    ["Задания урока", CFG.classes.filter(c => c.kind === "quickdraw")],
    ["Отказ", CFG.classes.filter(c => c.kind === "other")],
  ];
  sel.innerHTML = "";
  for (const [label, list] of groups) {
    if (!list.length) continue;
    const og = document.createElement("optgroup");
    og.label = label;
    for (const c of list) {
      const o = document.createElement("option");
      o.value = c.id;
      const n = CFG.counts[c.id] ? CFG.counts[c.id].active : 0;
      o.textContent = withCounts ? `${nameOf(c)} — ${n}/${c.target}` : nameOf(c);
      og.appendChild(o);
    }
    sel.appendChild(og);
  }
  if (current && byId(current)) sel.value = current;
}

// ───────────────────────────────────────────── вкладка «Рисовать»
const D = {};
function initDraw() {
  D.sketch = new Sketch($("#d-canvas"), {
    onChange: (s) => {
      $("#d-save").disabled = s.empty;
      $("#d-hint").classList.toggle("gone", !s.empty);
    },
  });
  D.cls = store.get("class", CFG.default_class);
  if (!byId(D.cls)) D.cls = CFG.default_class;
  fillClassSelect($("#d-class"), D.cls);
  $("#d-class").addEventListener("change", e => { D.cls = e.target.value; store.set("class", D.cls); D.sketch.clear(); nextPrompt(); renderTask(); });
  const au = $("#d-author");
  au.value = store.get("author", "");
  au.addEventListener("input", () => store.set("author", au.value.trim()));
  au.addEventListener("keydown", e => { if (e.key === "Enter") { e.preventDefault(); au.blur(); } });

  const colors = ["#2d3a8c", "#b8322a", "#2f7d3b", "#6b3fa0", "#1f1a17"];
  const box = $("#d-colors");
  colors.forEach((c, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.style.background = c;
    b.title = "цвет мелка";
    b.setAttribute("aria-label", "цвет " + (i + 1));
    b.addEventListener("click", () => { D.crayon = D.sketch.color = c; $$("button", box).forEach(x => x.classList.toggle("on", x === b)); });
    box.appendChild(b);
  });
  box.firstChild.classList.add("on");
  D.crayon = colors[0];

  $$("#d-format button").forEach(b => b.addEventListener("click", () => {
    if (!D.sketch.empty && !confirm("Сменить формат? Текущий рисунок будет очищен.")) return;
    $$("#d-format button").forEach(x => x.classList.toggle("on", x === b));
    D.sketch.setFormat(b.dataset.f);
    box.style.visibility = b.dataset.f === "board" ? "hidden" : "";
    D.sketch.color = b.dataset.f === "board" ? "#f4f1e8" : D.crayon;          // на доске — мел
    store.set("format", b.dataset.f);
  }));
  const f = store.get("format", "square");
  if (f !== "square" && FORMATS[f]) $(`#d-format button[data-f="${f}"]`).click();

  $("#d-undo").addEventListener("click", () => D.sketch.undo());
  $("#d-clear").addEventListener("click", () => D.sketch.clear());
  $("#d-skip").addEventListener("click", () => { D.sketch.clear(); nextPrompt(); });
  $("#d-save").addEventListener("click", save);
  $("#d-newprompt").addEventListener("click", nextPrompt);
  nextPrompt();
  renderTask();
}

function nextPrompt() {
  const list = CFG.variety_prompts;
  if (!list.length) { D.prompt = ""; $("#d-prompt").textContent = "как хочется"; return; }
  let p;
  do { p = list[Math.floor(Math.random() * list.length)]; } while (list.length > 1 && p === D.prompt);
  D.prompt = p;
  $("#d-prompt").textContent = p;
}

function renderTask() {
  const c = byId(D.cls);
  $("#d-word").textContent = c.ru_acc || c.ru.toLowerCase();
  const n = CFG.counts[c.id] ? CFG.counts[c.id].active : 0;
  $("#d-count").textContent = n;
  $("#d-target").textContent = c.target;
  $("#d-bar").style.width = Math.min(100, 100 * n / Math.max(1, c.target)) + "%";
  $("#d-note").textContent = c.kind === "custom"
    ? "Эти рисунки нужны, чтобы игра научилась их отсеивать. Рисуйте по-разному — следуйте подсказке под заданием."
    : c.kind === "other"
      ? "Любой предмет или каракули, которых нет среди заданий: так модель учится говорить «не разобрал»."
      : "Свои рисунки заданий мешаются с Quick Draw при обучении — особенно полезны мышью, как в игре.";
}

async function save() {
  if (D.sketch.empty || D.saving) return;
  D.saving = true;
  $("#d-save").disabled = true;
  try {
    const r = await api("/api/samples", {
      word: D.cls, author: $("#d-author").value.trim(), prompt: D.prompt, device: D.sketch.device || "mouse",
      canvas: { w: LOGICAL_W, h: Math.round(D.sketch.H), format: D.sketch.format }, drawing: D.sketch.drawing(),
    });
    CFG.counts[D.cls] = r.count;
    toast(`Сохранено · ${r.count.active} / ${byId(D.cls).target}`);
    D.sketch.clear();
    nextPrompt();
    renderTask();
    fillClassSelect($("#d-class"), D.cls);
    G.dirty = true;
  } catch (e) {
    toast("Не сохранилось: " + e.message, true);
    $("#d-save").disabled = D.sketch.empty;
  } finally {
    D.saving = false;
  }
}

// ───────────────────────────────────────────── вкладка «Галерея»
const G = { offset: 0, dirty: true };
function initGallery() {
  G.cls = new URLSearchParams(location.search).get("class") || store.get("gclass", CFG.default_class);
  if (!byId(G.cls)) G.cls = CFG.default_class;
  fillClassSelect($("#g-class"), G.cls, false);
  for (const id of ["#g-class", "#g-author", "#g-status"]) {
    $(id).addEventListener("change", () => {
      if (id === "#g-class") { $("#g-author").value = ""; store.set("gclass", $("#g-class").value); }
      loadGallery(true);
    });
  }
  $("#g-more").addEventListener("click", () => loadGallery(false));
}

function renderStats() {
  const c = CFG.counts[$("#g-class").value] || { active: 0, flagged: 0, deleted: 0 };
  const cls = byId($("#g-class").value);
  $("#g-stats").innerHTML =
    `<span>в датасете <b>${c.active}</b> / ${cls.target}</span><span>помечено <b>${c.flagged}</b></span><span>удалено <b>${c.deleted}</b></span>`;
}

async function loadGallery(reset) {
  const word = $("#g-class").value, author = $("#g-author").value, status = $("#g-status").value;
  if (reset) { G.offset = 0; $("#g-grid").innerHTML = ""; }
  try {
    const q = new URLSearchParams({ word, author, status, offset: G.offset, limit: 60 });
    const r = await api("/api/samples?" + q);
    const sel = $("#g-author"), cur = sel.value;
    sel.innerHTML = '<option value="">все</option>' + r.authors.map(a => `<option>${esc(a)}</option>`).join("");
    sel.value = cur;
    for (const it of r.items) $("#g-grid").appendChild(tile(word, it));
    G.offset += r.items.length;
    $("#g-more").hidden = G.offset >= r.total;
    $("#g-empty").hidden = r.total > 0;
    G.dirty = false;
    const cfg = await api("/api/config");
    CFG.counts = cfg.counts;
    renderStats();
  } catch (e) {
    toast("Галерея: " + e.message, true);
  }
}

function esc(s) { return String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

function fmtDate(ts) {
  if (!ts) return "";
  const d = new Date(ts);
  return isNaN(d) ? ts : d.toLocaleString("ru-RU", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });
}

function tile(word, it) {
  const el = document.createElement("div");
  const render = () => {
    el.className = "card tile " + (it.status === "ok" ? "" : it.status);
    const badge = it.status === "flag" ? '<span class="badge flag">помечен</span>' : it.status === "delete" ? '<span class="badge delete">удалён</span>' : "";
    const btns = it.status === "delete"
      ? `<button class="btn small" data-a="ok">Вернуть</button>`
      : `${it.status === "flag" ? '<button class="btn small" data-a="ok">Норм</button>' : '<button class="btn small" data-a="flag">Пометить</button>'}<button class="btn small danger" data-a="delete">Удалить</button>`;
    el.innerHTML = `<img loading="lazy" alt="" src="/api/thumb?word=${encodeURIComponent(word)}&key=${encodeURIComponent(it.key_id)}">
      <div class="meta"><b>${esc(it.author)}</b> ${badge}<br>${esc(it.prompt || "")}<br>${fmtDate(it.timestamp)} · ${esc(it.device)}</div>
      <div class="row">${btns}</div>`;
    $$("button", el).forEach(b => b.addEventListener("click", async () => {
      try {
        const r = await api("/api/review", { word, key_id: it.key_id, action: b.dataset.a, author: $("#d-author").value.trim() });
        it.status = r.status;
        CFG.counts[word] = r.count;
        renderStats();
        const f = $("#g-status").value;
        const visible = f === "all" || (f === "active" && it.status === "ok") || (f === "flagged" && it.status === "flag") || (f === "deleted" && it.status === "delete");
        if (visible) render(); else el.remove();
        if (word === D.cls) { renderTask(); fillClassSelect($("#d-class"), D.cls); }
      } catch (e) { toast(e.message, true); }
    }));
  };
  render();
  return el;
}

// ───────────────────────────────────────────── вкладка «Тест»
const T = { busy: false, pending: false, last: 0, timer: 0 };
function initTest() {
  T.sketch = new Sketch($("#t-canvas"), {
    onChange: (s, final) => {
      $("#t-hint").classList.toggle("gone", !s.empty);
      if (s.empty) { resetTest(); return; }
      schedulePredict(final);
    },
  });
  T.sketch.color = "#1f1a17";
  $("#t-undo").addEventListener("click", () => T.sketch.undo());
  $("#t-clear").addEventListener("click", () => T.sketch.clear());
  renderModelInfo(CFG.model);
}

function renderModelInfo(m) {
  const el = $("#t-model");
  el.classList.toggle("err", !m.available);
  el.textContent = m.available ? `Модель: ${m.run} · ${m.classes} классов · onnxruntime на сервере` : (m.error || "Модель не найдена");
}

function resetTest() {
  $("#t-verdict").className = "verdict";
  $("#t-verdict").innerHTML = '<span class="muted">Нарисуйте что-нибудь</span>';
  $("#t-top").innerHTML = "";
  $("#t-preview").removeAttribute("src");
  $("#t-ms").textContent = "";
}

function schedulePredict(final) {
  clearTimeout(T.timer);
  const wait = final ? 0 : Math.max(0, 300 - (performance.now() - T.last));
  T.timer = setTimeout(predict, wait);
}

async function predict() {
  if (T.sketch.empty) return;
  if (T.busy) { T.pending = true; return; }
  T.busy = true;
  T.last = performance.now();
  try {
    const r = await api("/api/predict", { drawing: T.sketch.drawing().map(s => s.slice(0, 2)) });
    if (T.sketch.empty) return;
    $("#t-preview").src = r.preview;
    if (!r.available) {
      $("#t-verdict").className = "verdict no";
      $("#t-verdict").innerHTML = "<b>Модели нет</b><small>Экспортируйте: python scripts/export_onnx.py --run runs/…</small>";
      renderModelInfo({ available: false, error: r.error });
      return;
    }
    const v = r.verdict, pct = p => Math.round(p * 100) + "%";
    const el = $("#t-verdict");
    if (v.accepted) {
      el.className = "verdict " + (v.id === "rude" ? "rude" : "ok");
      el.innerHTML = `Похоже на: <b>${esc(v.ru)}</b> ${pct(v.p)}<small>порог класса ${pct(v.threshold)}</small>`;
    } else if (v.id === "other") {
      el.className = "verdict no";
      el.innerHTML = `<b>Не разобрал</b><small>модель думает: «другое» (${pct(v.p)})</small>`;
    } else {
      el.className = "verdict no";
      el.innerHTML = `<b>Не разобрал</b><small>лучшее: ${esc(v.ru)} ${pct(v.p)}, нужно ≥ ${pct(v.threshold ?? 1)}</small>`;
    }
    $("#t-top").innerHTML = r.top.map(t =>
      `<li><span>${esc(t.ru)}<span class="id">${esc(t.id)}</span></span><span class="pct">${(t.p * 100).toFixed(1)}%</span>` +
      `<span class="track"><i style="width:${Math.max(1, t.p * 100)}%"></i></span></li>`).join("");
    $("#t-ms").textContent = `предобработка ${r.ms.preprocess} мс · модель ${r.ms.inference} мс`;
    renderModelInfo({ available: true, run: r.run, classes: CFG.model.classes || "?" });
  } catch (e) {
    toast("Тест: " + e.message, true);
  } finally {
    T.busy = false;
    if (T.pending) { T.pending = false; schedulePredict(true); }
  }
}

// ───────────────────────────────────────────── вкладки и клавиатура
function showTab(name) {
  $$(".tabs button").forEach(b => b.classList.toggle("on", b.dataset.tab === name));
  $$(".tab").forEach(t => t.classList.toggle("on", t.id === "tab-" + name));
  store.set("tab", name);
  if (name === "gallery" && G.dirty) loadGallery(true);
  if (name === "test") api("/api/config").then(c => { CFG.model = c.model; renderModelInfo(c.model); }).catch(() => {});
  (name === "draw" ? D.sketch : name === "test" ? T.sketch : null)?.resize();
}

document.addEventListener("keydown", e => {
  const tab = store.get("tab", "draw");
  const typing = /^(INPUT|SELECT|TEXTAREA)$/.test(document.activeElement?.tagName || "");
  const sk = tab === "draw" ? D.sketch : tab === "test" ? T.sketch : null;
  if (!sk) return;
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z" && !typing) { e.preventDefault(); sk.undo(); }
  else if (e.key === "Escape" && !typing) { sk.clear(); }
  else if (e.key === "Enter" && !typing && tab === "draw") { e.preventDefault(); save(); }
});

(async function main() {
  try {
    CFG = await api("/api/config");
  } catch (e) {
    document.body.innerHTML = `<p style="padding:24px">Сервер недоступен: ${esc(e.message)}</p>`;
    return;
  }
  initDraw();
  initGallery();
  initTest();
  $$(".tabs button").forEach(b => b.addEventListener("click", () => showTab(b.dataset.tab)));
  const q = new URLSearchParams(location.search).get("tab");
  showTab(q || store.get("tab", "draw"));
})();
