// Холст с мелком — копия класса Sketch из editor/static/app.js (держать одинаковыми).
// Рисунок: штрих = [[xs], [ys], [ts]], ts — мс от первой точки рисунка; координаты в логических единицах листа.
"use strict";

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

