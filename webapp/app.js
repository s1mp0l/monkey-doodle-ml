// Doodle Lab (веб): рисуем, храним в IndexedDB, отдаём файлом .ndjson. Никаких запросов на сервер.
"use strict";

const { CONFIG, makeRecord, toNdjson, exportFilename } = window.DoodleCore;
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

// ───────────────────────────────────────────── хранилище: IndexedDB (запасной вариант — память)
const DB = (() => {
  let dbp = null, mem = null;
  function open() {
    if (dbp) return dbp;
    dbp = new Promise((resolve, reject) => {
      let req;
      try { req = indexedDB.open("doodle-lab", 1); } catch (e) { reject(e); return; }
      req.onupgradeneeded = () => {
        const db = req.result;
        db.createObjectStore("drawings", { keyPath: "key_id" });
        db.createObjectStore("kv");
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    }).catch(e => {
      console.warn("IndexedDB недоступна, рисунки живут до закрытия вкладки", e);
      mem = { drawings: new Map(), kv: new Map() };
      return null;
    });
    return dbp;
  }
  async function tx(store, mode, fn) {
    const db = await open();
    if (!db) return fn(null, mem[store]);
    return new Promise((resolve, reject) => {
      const t = db.transaction(store, mode), s = t.objectStore(store);
      let out;
      Promise.resolve(fn(s, null)).then(v => { out = v; });
      t.oncomplete = () => resolve(out);
      t.onerror = () => reject(t.error);
      t.onabort = () => reject(t.error);
    });
  }
  const req = r => new Promise((res, rej) => { r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error); });
  return {
    async persistent() { return !!(await open()); },
    put: rec => tx("drawings", "readwrite", (s, m) => m ? m.set(rec.key_id, rec) : req(s.put(rec))),
    del: key => tx("drawings", "readwrite", (s, m) => m ? m.delete(key) : req(s.delete(key))),
    all: () => tx("drawings", "readonly", (s, m) => m ? [...m.values()] : req(s.getAll())),
    get: key => tx("kv", "readonly", (s, m) => m ? m.get(key) : req(s.get(key))),
    set: (key, v) => tx("kv", "readwrite", (s, m) => m ? m.set(key, v) : req(s.put(v, key))),
  };
})();

let toastTimer = 0;
function toast(msg, err = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.toggle("err", err);
  t.classList.add("on");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("on"), err ? 3500 : 1600);
}

// ───────────────────────────────────────────── состояние
const S = { cls: CONFIG.defaultClass, prompt: "", records: [], sent: new Set(), saving: false, crayon: "#2d3a8c" };
const cls = () => CONFIG.classes.find(c => c.id === S.cls) || CONFIG.classes[0];
const nick = () => $("#nick").value.trim();

async function load() {
  S.records = ((await DB.all()) || []).sort((a, b) => (a.timestamp < b.timestamp ? 1 : -1));
  S.sent = new Set((await DB.get("sent")) || []);
}

function renderCounts() {
  const c = cls(), mine = S.records.filter(r => r.word === c.id);
  const unsent = S.records.filter(r => !S.sent.has(r.key_id));
  $("#word").textContent = c.ru_acc || c.ru.toLowerCase();
  $("#count").textContent = mine.length;
  $("#target").textContent = c.target;
  $("#bar").style.width = Math.min(100, 100 * mine.length / Math.max(1, c.target)) + "%";
  $("#mine-count").textContent = S.records.length;
  $("#export").disabled = S.records.length === 0;
  $("#export").textContent = unsent.length && unsent.length !== S.records.length
    ? `Скачать мои рисунки (${S.records.length}, новых ${unsent.length})` : `Скачать мои рисунки (${S.records.length})`;
  $("#clear-sent").hidden = ![...S.sent].some(k => S.records.some(r => r.key_id === k));
  $("#gal-count").textContent = S.records.length ? `· ${S.records.length}` : "";
}

function nextPrompt() {
  const list = CONFIG.variety;
  let p;
  do { p = list[Math.floor(Math.random() * list.length)]; } while (list.length > 1 && p === S.prompt);
  S.prompt = p;
  $("#prompt").textContent = p;
}

// ───────────────────────────────────────────── сохранение
async function save() {
  if (sketch.empty || S.saving) return;
  S.saving = true;
  $("#save").disabled = true;
  try {
    const rec = makeRecord({
      word: S.cls, author: nick(), prompt: S.prompt, device: sketch.device || "mouse",
      canvas: { w: LOGICAL_W, h: Math.round(sketch.H), format: sketch.format }, drawing: sketch.drawing(),
    });
    if (!rec.simplified.length) throw new Error("пустой рисунок");
    await DB.put(rec);
    S.records.unshift(rec);
    sketch.clear();
    nextPrompt();
    renderCounts();
    if (!$("#grid").hidden) renderGallery();
    toast(`Сохранено · ${S.records.filter(r => r.word === S.cls).length} / ${cls().target}`);
  } catch (e) {
    toast("Не сохранилось: " + e.message, true);
    $("#save").disabled = sketch.empty;
  } finally {
    S.saving = false;
  }
}

// ───────────────────────────────────────────── экспорт
function exportFile() {
  if (!S.records.length) return;
  const author = nick() || "аноним";
  // ник, введённый позже, проставляется в записи без ника
  const recs = S.records.slice().reverse().map(r => (r.author === "аноним" && author !== "аноним" ? { ...r, author } : r));
  const text = toNdjson(recs);
  const name = exportFilename(S.cls, nick(), recs.length);
  const url = URL.createObjectURL(new Blob([text], { type: "application/x-ndjson" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
  S.sent = new Set(recs.map(r => r.key_id));
  DB.set("sent", [...S.sent]);
  $("#send-status").textContent = `Файл «${name}» скачан (${recs.length} рис.). ${CONFIG.sendHint}`;
  renderCounts();
  if (!$("#grid").hidden) renderGallery();
  return { name, text };
}

async function clearSent() {
  const keys = S.records.filter(r => S.sent.has(r.key_id)).map(r => r.key_id);
  if (!keys.length) return;
  if (!confirm(`Удалить из браузера ${keys.length} уже скачанных рисунков? Сделайте это, только если файл отправлен.`)) return;
  for (const k of keys) await DB.del(k);
  S.records = S.records.filter(r => !keys.includes(r.key_id));
  S.sent = new Set();
  await DB.set("sent", []);
  $("#send-status").textContent = "";
  renderCounts();
  renderGallery();
  toast("Отправленные удалены");
}

// ───────────────────────────────────────────── галерея
function drawThumb(canvas, rec) {
  const px = 160, g = canvas.getContext("2d");
  canvas.width = canvas.height = px;
  g.fillStyle = "#fff";
  g.fillRect(0, 0, px, px);
  const simp = rec.simplified || [];
  if (!simp.length) return;
  let mx = 0, my = 0;
  for (const s of simp) { mx = Math.max(mx, ...s[0]); my = Math.max(my, ...s[1]); }
  const pad = 14, f = (px - 2 * pad) / 255, ox = (px - mx * f) / 2, oy = (px - my * f) / 2;
  g.strokeStyle = g.fillStyle = "#2b2118";
  g.lineWidth = 4;
  g.lineCap = g.lineJoin = "round";
  for (const s of simp) {
    g.beginPath();
    g.moveTo(ox + s[0][0] * f, oy + s[1][0] * f);
    if (s[0].length === 1) g.lineTo(ox + s[0][0] * f + 0.01, oy + s[1][0] * f);
    for (let i = 1; i < s[0].length; i++) g.lineTo(ox + s[0][i] * f, oy + s[1][i] * f);
    g.stroke();
  }
}

function renderGallery() {
  const grid = $("#grid");
  grid.innerHTML = "";
  $("#gal-empty").hidden = grid.hidden || S.records.length > 0;
  for (const rec of S.records) {
    const el = document.createElement("div");
    el.className = "card tile" + (S.sent.has(rec.key_id) ? " sent" : "");
    const cv = document.createElement("canvas");
    drawThumb(cv, rec);
    const meta = document.createElement("div");
    meta.className = "meta";
    const d = new Date(rec.timestamp);
    meta.textContent = `${rec.prompt || ""}${S.sent.has(rec.key_id) ? " · скачан" : ""}\n${isNaN(d) ? "" : d.toLocaleString("ru-RU", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" })}`;
    meta.style.whiteSpace = "pre-line";
    const del = document.createElement("button");
    del.className = "btn small danger";
    del.type = "button";
    del.textContent = "Удалить";
    del.addEventListener("click", async () => {
      await DB.del(rec.key_id);
      S.records = S.records.filter(r => r.key_id !== rec.key_id);
      el.remove();
      renderCounts();
      $("#gal-empty").hidden = S.records.length > 0;
    });
    el.append(cv, meta, del);
    grid.appendChild(el);
  }
}

// ───────────────────────────────────────────── интерфейс
let sketch;
function initUi() {
  sketch = new Sketch($("#canvas"), {
    onChange: s => {
      $("#save").disabled = s.empty;
      $("#hint").classList.toggle("gone", !s.empty);
    },
  });
  const colors = ["#2d3a8c", "#b8322a", "#2f7d3b", "#6b3fa0", "#1f1a17"];
  const box = $("#colors");
  colors.forEach((c, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.style.background = c;
    b.setAttribute("aria-label", "цвет " + (i + 1));
    b.addEventListener("click", () => { S.crayon = sketch.color = c; $$("button", box).forEach(x => x.classList.toggle("on", x === b)); });
    box.appendChild(b);
  });
  box.firstChild.classList.add("on");
  $$("#format button").forEach(b => b.addEventListener("click", () => {
    if (!sketch.empty && !confirm("Сменить формат? Текущий рисунок будет очищен.")) return;
    $$("#format button").forEach(x => x.classList.toggle("on", x === b));
    sketch.setFormat(b.dataset.f);
    box.style.visibility = b.dataset.f === "board" ? "hidden" : "";
    sketch.color = b.dataset.f === "board" ? "#f4f1e8" : S.crayon;
  }));

  if (CONFIG.classes.length > 1) {
    $("#class-field").hidden = false;
    $("#cls").innerHTML = CONFIG.classes.map(c => `<option value="${c.id}">${c.ru}</option>`).join("");
    $("#cls").value = S.cls;
    $("#cls").addEventListener("change", e => { S.cls = e.target.value; sketch.clear(); renderCounts(); });
  }
  $("#undo").addEventListener("click", () => sketch.undo());
  $("#clear").addEventListener("click", () => sketch.clear());
  $("#skip").addEventListener("click", () => { sketch.clear(); nextPrompt(); });
  $("#save").addEventListener("click", save);
  $("#newprompt").addEventListener("click", nextPrompt);
  $("#export").addEventListener("click", exportFile);
  $("#clear-sent").addEventListener("click", clearSent);
  $("#send-hint").textContent = CONFIG.sendHint;
  $("#gal-toggle").addEventListener("click", () => {
    const g = $("#grid");
    g.hidden = !g.hidden;
    $("#gal-toggle").textContent = g.hidden ? "показать" : "скрыть";
    if (!g.hidden) renderGallery(); else $("#gal-empty").hidden = true;
  });
  for (const id of ["#nick", "#nick-intro"]) {
    $(id).addEventListener("input", e => {
      const v = e.target.value;
      $("#nick").value = $("#nick-intro").value = v;
      DB.set("nick", v.trim());
    });
  }
  $("#agree").addEventListener("click", () => { DB.set("agreed", true); showDraw(); });
  $("#show-intro").addEventListener("click", () => { $("#intro").hidden = false; $("#draw").hidden = true; window.scrollTo(0, 0); });

  document.addEventListener("keydown", e => {
    if ($("#draw").hidden) return;
    const typing = /^(INPUT|SELECT|TEXTAREA)$/.test(document.activeElement?.tagName || "");
    if (typing) { if (e.key === "Enter") document.activeElement.blur(); return; }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { e.preventDefault(); sketch.undo(); }
    else if (e.key === "Escape") sketch.clear();
    else if (e.key === "Enter") { e.preventDefault(); save(); }
  });
}

function showDraw() {
  $("#intro").hidden = true;
  $("#draw").hidden = false;
  sketch.resize();
}

let ready;
window.DoodleApp = { save, exportFile, sketch: () => sketch, records: () => S.records, ready: () => ready };   // для самопроверки
ready = (async function main() {
  initUi();
  await load();
  const n = (await DB.get("nick")) || "";
  $("#nick").value = $("#nick-intro").value = n;
  nextPrompt();
  renderCounts();
  if (!(await DB.persistent())) toast("Браузер не даёт хранилище — рисунки пропадут при закрытии вкладки. Скачивайте файл!", true);
  if (await DB.get("agreed")) showDraw(); else $("#intro").hidden = false;
})();
