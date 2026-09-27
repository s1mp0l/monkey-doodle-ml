// Ядро веб-сборщика: конфиг, каноническое упрощение штрихов (порт doodle/preprocess.normalize),
// сборка записи в формате локального редактора и сериализация ndjson.
// Работает и в браузере (window.DoodleCore), и под node (module.exports) — для тестов.
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.DoodleCore = api;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // ───────────────────────────────────────────── конфиг
  const CONFIG = {
    // Классы, которые можно рисовать. Сейчас только rude; чтобы включить другие — добавить сюда
    // (id должен совпадать с config/classes.yaml).
    classes: [
      { id: "rude", ru: "Неприличное", ru_acc: "неприличный рисунок", target: 3000 },
    ],
    defaultClass: "rude",
    // Как отправить файл — поменяйте под свой канал.
    sendHint: "Отправьте скачанный файл разработчику — в Telegram или на почту, как договаривались.",
    epsilon: 2.0,
    variety: [
      "маленький", "большой, на весь лист", "в углу листа", "сбоку", "повёрнутый", "вверх ногами",
      "одной линией, не отрывая", "криво и быстро", "медленно и аккуратно", "очень просто, 2–3 штриха",
      "подробно, много деталей", "только контур", "с закраской (штриховка)", "вытянутый",
      "толстый и короткий", "как ребёнок", "рядом с другим рисунком", "с подписью или буквами рядом",
      "левой рукой", "как на заборе",
    ],
  };

  // ───────────────────────────────────────────── упрощение (бит-в-бит как doodle/preprocess.py)
  function toPointArrays(drawing) {
    const out = [];
    for (const s of drawing || []) {
      if (!Array.isArray(s) || s.length < 2) continue;
      const n = Math.min(s[0].length, s[1].length);
      const p = [];
      for (let i = 0; i < n; i++) {
        const x = Number(s[0][i]), y = Number(s[1][i]);
        if (Number.isFinite(x) && Number.isFinite(y)) p.push([x, y]);
      }
      if (p.length) out.push(p);
    }
    return out;
  }

  function dedupe(p) {
    if (p.length < 2) return p;
    const out = [p[0]];
    for (let i = 1; i < p.length; i++) {
      if (p[i][0] !== p[i - 1][0] || p[i][1] !== p[i - 1][1]) out.push(p[i]);
    }
    return out;
  }

  // Расстояние от точки до ОТРЕЗКА ab (тот же порядок операций, что в numpy-эталоне).
  function segDist(p, a, b) {
    const ab0 = b[0] - a[0], ab1 = b[1] - a[1];
    const l2 = ab0 * ab0 + ab1 * ab1;
    const ap0 = p[0] - a[0], ap1 = p[1] - a[1];
    if (l2 === 0) return Math.sqrt(ap0 * ap0 + ap1 * ap1);
    let t = (ap0 * ab0 + ap1 * ab1) / l2;
    t = Math.min(Math.max(t, 0), 1);
    const dx = ap0 - t * ab0, dy = ap1 - t * ab1;
    return Math.sqrt(dx * dx + dy * dy);
  }

  function rdp(p, eps) {
    const n = p.length;
    if (n < 3) return p.slice();
    const keep = new Uint8Array(n);
    keep[0] = keep[n - 1] = 1;
    const stack = [[0, n - 1]];
    while (stack.length) {
      const [i, j] = stack.pop();
      if (j <= i + 1) continue;
      let best = -1, m = -1;
      for (let k = i + 1; k < j; k++) {
        const d = segDist(p[k], p[i], p[j]);
        if (d > best) { best = d; m = k; }          // первый максимум
      }
      if (best > eps) {
        keep[m] = 1;
        stack.push([i, m]);
        stack.push([m, j]);
      }
    }
    return p.filter((_, k) => keep[k]);
  }

  /** Сырые штрихи [[xs],[ys],(ts)] → [[xs],[ys]] целых 0..255 (Quick Draw simplified). */
  function normalize(drawing, eps = CONFIG.epsilon) {
    const pts = toPointArrays(drawing);
    if (!pts.length) return [];
    let mnx = Infinity, mny = Infinity, mxx = -Infinity, mxy = -Infinity;
    for (const p of pts) for (const [x, y] of p) {
      if (x < mnx) mnx = x; if (x > mxx) mxx = x;
      if (y < mny) mny = y; if (y > mxy) mxy = y;
    }
    const ext = Math.max(mxx - mnx, mxy - mny);
    const s = ext > 0 ? 255.0 / ext : 1.0;
    return pts.map(p => {
      let q = dedupe(p.map(([x, y]) => [(x - mnx) * s, (y - mny) * s]));
      q = rdp(q, eps);
      q = dedupe(q.map(([x, y]) => [Math.floor(x + 0.5), Math.floor(y + 0.5)]));
      return [q.map(v => v[0]), q.map(v => v[1])];
    });
  }

  // ───────────────────────────────────────────── растеризация (порт doodle/preprocess.rasterize, docs/PREPROCESS.md)
  const SIZE = 64, PAD = 2.0, LINE_WIDTH = 2.0;

  /** Нормализованные штрихи [[xs],[ys]] → отрезки [ax,ay,bx,by] в пикселях выхода. */
  function toPixelSegments(strokes, size, pad) {
    const pts = toPointArrays(strokes);
    if (!pts.length) return [];
    let mnx = Infinity, mny = Infinity, mxx = -Infinity, mxy = -Infinity;
    for (const p of pts) for (const [x, y] of p) {
      if (x < mnx) mnx = x; if (x > mxx) mxx = x;
      if (y < mny) mny = y; if (y > mxy) mxy = y;
    }
    const cx = (mnx + mxx) / 2, cy = (mny + mxy) / 2, f = (size - 2 * pad) / 255.0;
    const segs = [];
    for (const p of pts) {
      const q = p.map(([x, y]) => [size / 2 + (x - cx) * f, size / 2 + (y - cy) * f]);
      if (q.length === 1) segs.push([q[0][0], q[0][1], q[0][0], q[0][1]]);
      else for (let i = 0; i < q.length - 1; i++) segs.push([q[i][0], q[i][1], q[i + 1][0], q[i + 1][1]]);
    }
    return segs;
  }

  /** Растеризация «расстояние до ближайшего отрезка» → Float32Array size*size (row-major, 1 = чернила). */
  function rasterize(strokes, size = SIZE, pad = PAD, lineWidth = LINE_WIDTH) {
    const img = new Float32Array(size * size);
    const segs = toPixelSegments(strokes, size, pad);
    if (!segs.length) return img;
    const r = lineWidth / 2;
    for (let row = 0; row < size; row++) {
      const py = row + 0.5;
      for (let col = 0; col < size; col++) {
        const px = col + 0.5;
        let best = Infinity;
        for (let k = 0; k < segs.length; k++) {
          const [ax, ay, bx, by] = segs[k];
          const abx = bx - ax, aby = by - ay, l2 = abx * abx + aby * aby;
          const apx = px - ax, apy = py - ay;
          let t = l2 === 0 ? 0 : (apx * abx + apy * aby) / l2;
          t = t < 0 ? 0 : t > 1 ? 1 : t;
          const dx = apx - t * abx, dy = apy - t * aby, d = Math.sqrt(dx * dx + dy * dy);
          if (d < best) best = d;
        }
        let v = r + 0.5 - best;
        img[row * size + col] = v < 0 ? 0 : v > 1 ? 1 : v;
      }
    }
    return img;
  }

  /** Сырой рисунок → Float32Array size*size (нормализация + растеризация). */
  function preprocess(drawing, size = SIZE, pad = PAD, lineWidth = LINE_WIDTH, eps = CONFIG.epsilon) {
    return rasterize(normalize(drawing, eps), size, pad, lineWidth);
  }

  // ───────────────────────────────────────────── запись
  function isoLocal(d = new Date()) {
    const pad = (v, n = 2) => String(v).padStart(n, "0");
    const off = -d.getTimezoneOffset(), sign = off >= 0 ? "+" : "-", a = Math.abs(off);
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}:` +
      `${pad(d.getSeconds())}.${pad(d.getMilliseconds(), 3)}${sign}${pad(Math.floor(a / 60))}:${pad(a % 60)}`;
  }

  function uuid() {
    if (typeof crypto !== "undefined" && crypto.randomUUID) return crypto.randomUUID();
    const b = new Uint8Array(16);
    crypto.getRandomValues(b);
    b[6] = (b[6] & 0x0f) | 0x40; b[8] = (b[8] & 0x3f) | 0x80;
    const h = [...b].map(v => v.toString(16).padStart(2, "0")).join("");
    return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`;
  }

  const r1 = v => { const r = Math.round(v * 10) / 10; return Object.is(r, -0) ? 0 : r; };

  /** Запись в формате data/custom/<класс>.ndjson локального редактора (порядок полей тот же). */
  function makeRecord({ word, author, prompt, device, canvas, drawing, now, keyId }) {
    const raw = drawing.map(s => [s[0].map(r1), s[1].map(r1), s[2].map(v => Math.round(v))]);
    return {
      key_id: keyId || uuid(),
      word,
      author: String(author || "").trim().slice(0, 40) || "аноним",
      timestamp: isoLocal(now),
      prompt: String(prompt || "").slice(0, 80),
      device: ["mouse", "pen", "touch"].includes(device) ? device : "unknown",
      canvas: canvas || {},
      source: "web",
      drawing: raw,
      simplified: normalize(raw),
    };
  }

  function toNdjson(records) {
    return records.map(r => JSON.stringify(r)).join("\n") + (records.length ? "\n" : "");
  }

  function exportFilename(word, nickname, count, now = new Date()) {
    const nick = String(nickname || "").trim().replace(/[^\p{L}\p{N}_-]+/gu, "_").replace(/^_+|_+$/g, "").slice(0, 24) || "аноним";
    const pad = v => String(v).padStart(2, "0");
    const date = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
    return `${word}_${nick}_${date}_${count}.ndjson`;
  }

  return { CONFIG, normalize, rdp, segDist, rasterize, preprocess, toPixelSegments, makeRecord, toNdjson, exportFilename, isoLocal, uuid };
});
