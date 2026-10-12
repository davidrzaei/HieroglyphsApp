// hiero_ocr.js - offline hieroglyph recognition for photos and scans. Dependency-free ES module (browser, Node >= 20).
//
//   import { loadModel, recognise } from './hiero_ocr.js';
//   await loadModel('model.json');            // or loadModel({ json: <object|string>, bin: <ArrayBuffer|Uint8Array> })
//   const r = recognise(imageData, { direction: 'ltr' });   // 'ltr' | 'rtl' | 'auto' (auto = ltr, see below)
//   r = { layout: 'rows'|'columns', direction, polarity, lines: [ { signs: [ { char, codepoint, conf, uncertain,
//         alts: [ { char, codepoint, conf } x5 ], box: [x, y, w, h], group } ] } ] }
//
// imageData: { width, height, data } with RGBA (browser ImageData), RGB or grey (1 byte per pixel) data.
// The segmentation is a line-by-line port of segment.py (the Python reference) and gives the same boxes; the CNN is the
// forward pass of ref_forward.py (3x3 conv with folded BatchNorm, ReLU, 2x2 max-pool, global average pool, linear,
// softmax over logits + corpus prior). 'auto' direction: no facing detector yet, falls back to 'ltr'.

export const PARAMS = Object.freeze({
  max_side: 1600, stretch_lo: 0.01, stretch_hi: 0.99,
  w1: 0.15, wmin: 9, cwin: 1.0, kl: 2.0, kh: 4.0, dmin: 4.0, kt: 0.0,
  tail: 0.02, rdiv: 4.0, pair: 0.55, pair_lr: 4.0, pol_sure: 6.0,
  min_px: 10, speck_frac: 0.06, frame_len: 3.0, frame_thin: 0.10, frame_fill: 0.06, overlap: 0.5, skew_max: 8.0, skew_step: 0.5,
  stroke_aspect: 1.8, stroke_hmax: 0.7, stroke_ratio: 1.35, stroke_gap: 1.25, frag_size: 0.3, frag_gap: 0.04,
  group_overlap: 0.3, cut_tol: 0.12, crop_pad: 0.04,
  rm_gap: 0.25, rm_size: 1.3, rm_other: 0.5, rm_margin: 0.0, rm_theta: 0.5,
  sel_floor: 0.3, fb_min_signs: 3, fb_factor: 0.75,
});

// ------------------------------------------------------------------ small helpers
function pyRound(x) {               // Python round(): half to even
  const f = Math.floor(x), d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return (f % 2 === 0) ? f : f + 1;
}
const bw = (b) => b[2] - b[0];
const bh = (b) => b[3] - b[1];
function inter(a, b) {
  const w = Math.min(a[2], b[2]) - Math.max(a[0], b[0]), h = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  return (w > 0 && h > 0) ? w * h : 0;
}
const union = (a, b) => [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[2], b[2]), Math.max(a[3], b[3]), a[4] + b[4]];
function boxGap(a, b) {
  const gx = Math.max(0, Math.max(a[0], b[0]) - Math.min(a[2], b[2]));
  const gy = Math.max(0, Math.max(a[1], b[1]) - Math.min(a[3], b[3]));
  return Math.max(gx, gy);
}
function cmpTuple(a, b) {           // Python tuple order
  for (let i = 0; i < a.length; i++) { if (a[i] < b[i]) return -1; if (a[i] > b[i]) return 1; }
  return 0;
}

// ------------------------------------------------------------------ pixels
export function toGrey(img) {
  const { width: W, height: H, data } = img;
  const n = W * H, g = new Uint8Array(n);
  const ch = data.length / n;
  if (ch === 1) { g.set(data); return { w: W, h: H, d: g }; }
  if (ch !== 3 && ch !== 4) throw new Error('image data must have 1, 3 or 4 channels');
  for (let i = 0, j = 0; i < n; i++, j += ch) g[i] = (77 * data[j] + 150 * data[j + 1] + 29 * data[j + 2] + 128) >> 8;
  return { w: W, h: H, d: g };
}

function downscale(g, maxSide) {
  const { w: W, h: H, d } = g;
  const f = Math.max(1, Math.ceil(Math.max(H, W) / maxSide));
  if (f === 1) return [{ w: W, h: H, d: d.slice() }, 1];
  const h = Math.floor(H / f), w = Math.floor(W / f), out = new Uint8Array(w * h), ff = f * f, half = Math.floor(ff / 2);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let s = 0;
    for (let dy = 0; dy < f; dy++) { const row = (y * f + dy) * W + x * f; for (let dx = 0; dx < f; dx++) s += d[row + dx]; }
    out[y * w + x] = Math.floor((s + half) / ff);
  }
  return [{ w, h, d: out }, f];
}

function rankValue(d, q) {
  const hist = new Float64Array(256);
  for (let i = 0; i < d.length; i++) hist[d[i]]++;
  const r = Math.floor(q * (d.length - 1));
  let c = 0;
  for (let v = 0; v < 256; v++) { c += hist[v]; if (c >= r + 1) return v; }
  return 256;
}

function stretch(g, p) {
  const lo = rankValue(g.d, p.stretch_lo), hi = rankValue(g.d, p.stretch_hi);
  if (hi - lo < 1) return { w: g.w, h: g.h, d: g.d.slice() };
  const out = new Uint8Array(g.d.length), den = 2 * (hi - lo);
  for (let i = 0; i < g.d.length; i++) {
    const v = Math.floor(((g.d[i] - lo) * 510 + (hi - lo)) / den);
    out[i] = v < 0 ? 0 : (v > 255 ? 255 : v);
  }
  return { w: g.w, h: g.h, d: out };
}

function box3(g) {
  const { w: W, h: H, d } = g, out = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let s = 0, n = 0;
    for (let yy = Math.max(0, y - 1); yy <= Math.min(H - 1, y + 1); yy++)
      for (let xx = Math.max(0, x - 1); xx <= Math.min(W - 1, x + 1); xx++) { s += d[yy * W + xx]; n++; }
    out[y * W + x] = Math.floor((2 * s + n) / (2 * n));
  }
  return { w: W, h: H, d: out };
}

function residual(gb, win) {
  const { w: W, h: H, d } = gb, W1 = W + 1, rr = Math.floor(win / 2);
  const I = new Float64Array((H + 1) * W1);
  for (let y = 0; y < H; y++) {
    let row = 0;
    for (let x = 0; x < W; x++) { row += d[y * W + x]; I[(y + 1) * W1 + x + 1] = I[y * W1 + x + 1] + row; }
  }
  const x0 = new Int32Array(W), x1 = new Int32Array(W);
  for (let x = 0; x < W; x++) { x0[x] = Math.min(W, Math.max(0, x - rr)); x1[x] = Math.min(W, Math.max(0, x + rr + 1)); }
  const r = new Float64Array(W * H);
  for (let y = 0; y < H; y++) {
    const y0 = Math.min(H, Math.max(0, y - rr)), y1 = Math.min(H, Math.max(0, y + rr + 1)), a = y1 * W1, b = y0 * W1;
    for (let x = 0; x < W; x++) {
      const n = (y1 - y0) * (x1[x] - x0[x]);
      const s1 = I[a + x1[x]] - I[b + x1[x]] - I[a + x0[x]] + I[b + x0[x]];
      r[y * W + x] = d[y * W + x] - s1 / n;
    }
  }
  return r;
}

function absQuantile(r, q) {
  const c = new Float64Array(4096);
  for (let i = 0; i < r.length; i++) { let k = Math.floor(Math.abs(r[i]) * 4); if (k > 4095) k = 4095; c[k]++; }
  const target = Math.floor(q * (r.length - 1)) + 1;
  let acc = 0;
  for (let k = 0; k < 4096; k++) { acc += c[k]; if (acc >= target) return (k + 0.5) / 4.0; }
  return (4096 + 0.5) / 4.0;
}
const robustSigma = (r) => 1.4826 * absQuantile(r, 0.5);

function maxfilter(b, W, H, r) {         // binary dilation with a (2r+1)^2 square, outside = false
  const tmp = new Uint8Array(W * H), out = new Uint8Array(W * H);
  for (let x = 0; x < W; x++) for (let y = 0; y < H; y++) {
    if (!b[y * W + x]) continue;
    for (let yy = Math.max(0, y - r); yy <= Math.min(H - 1, y + r); yy++) tmp[yy * W + x] = 1;
  }
  for (let y = 0; y < H; y++) {
    const o = y * W;
    for (let x = 0; x < W; x++) {
      if (!tmp[o + x]) continue;
      for (let xx = Math.max(0, x - r); xx <= Math.min(W - 1, x + r); xx++) out[o + xx] = 1;
    }
  }
  return out;
}

// 8-connected components in raster order of their first pixel (same numbering as scipy.ndimage.label)
function components(m, W, H) {
  const lab = new Int32Array(W * H), comps = [];
  const stack = new Int32Array(W * H);
  let n = 0;
  for (let i = 0; i < W * H; i++) {
    if (!m[i] || lab[i]) continue;
    n++;
    let sp = 0; stack[sp++] = i; lab[i] = n;
    let x0 = W, y0 = H, x1 = -1, y1 = -1, area = 0;
    while (sp > 0) {
      const k = stack[--sp], y = (k / W) | 0, x = k - y * W;
      area++;
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy; if (yy < 0 || yy >= H) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx; if (xx < 0 || xx >= W) continue;
          const kk = yy * W + xx;
          if (m[kk] && !lab[kk]) { lab[kk] = n; stack[sp++] = kk; }
        }
      }
    }
    comps.push([x0, y0, x1 + 1, y1 + 1, area]);
  }
  return [comps, lab, n];
}

function hysteresis(lo, hi, W, H) {
  const [, lab, n] = components(lo, W, H);
  if (n === 0) return lo;
  const keep = new Uint8Array(n + 1);
  for (let i = 0; i < W * H; i++) if (hi[i] && lo[i]) keep[lab[i]] = 1;
  keep[0] = 0;
  const out = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) out[i] = keep[lab[i]];
  return out;
}

function weightedMedianSide(comps, lim) {
  const items = [];
  for (const c of comps) { const s = Math.max(c[2] - c[0], c[3] - c[1]); if (s <= lim) items.push([s, c[4]]); }
  if (!items.length) return 0.0;
  items.sort(cmpTuple);
  let tot = 0; for (const it of items) tot += it[1];
  let acc = 0;
  for (const [s, a] of items) { acc += a; if (2 * acc >= tot) return s; }
  return items[items.length - 1][0];
}

function polarityCandidates(r, W, H, win, p) {
  const T = absQuantile(r, 1.0 - p.tail);
  const D = new Uint8Array(W * H), L = new Uint8Array(W * H);
  let nd = 0, nl = 0;
  for (let i = 0; i < r.length; i++) { if (r[i] < -T) { D[i] = 1; nd++; } if (r[i] > T) { L[i] = 1; nl++; } }
  if (nd + 1 >= p.pol_sure * (nl + 1)) return [['dark'], { nd, nl }];
  if (nl + 1 >= p.pol_sure * (nd + 1)) return [['light'], { nd, nl }];
  const rho = Math.max(2, pyRound(win / p.rdiv));
  const Ld = maxfilter(L, W, H, rho), Dd = maxfilter(D, W, H, rho);
  let cd = 0, cl = 0;
  for (let i = 0; i < r.length; i++) { if (Ld[i] && D[i]) cd++; if (Dd[i] && L[i]) cl++; }
  const first = nd >= nl ? 'dark' : 'light', other = first === 'dark' ? 'light' : 'dark';
  const paired = cd >= p.pair * nd && cl >= p.pair * nl;
  const balanced = nd + 1 < p.pair_lr * (nl + 1) && nl + 1 < p.pair_lr * (nd + 1);
  return [(paired && balanced) ? ['relief', first, other] : [first, 'relief', other], { nd, nl, cd, cl }];
}

function inkMask(r, W, H, pol, p) {
  const sg = robustSigma(r);
  let lo = Math.max(p.dmin, p.kl * sg), hi = Math.max(p.dmin, p.kh * sg);
  if (p.kt > 0) { const T = absQuantile(r, 1.0 - p.tail); hi = Math.max(p.dmin, Math.min(hi, p.kt * T)); lo = Math.min(lo, hi); }
  const m = new Uint8Array(W * H), n = W * H;
  if (pol === 'dark' || pol === 'relief') {
    const a = new Uint8Array(n), b = new Uint8Array(n);
    for (let i = 0; i < n; i++) { a[i] = r[i] < -lo ? 1 : 0; b[i] = r[i] < -hi ? 1 : 0; }
    const h = hysteresis(a, b, W, H); for (let i = 0; i < n; i++) m[i] |= h[i];
  }
  if (pol === 'light' || pol === 'relief') {
    const a = new Uint8Array(n), b = new Uint8Array(n);
    for (let i = 0; i < n; i++) { a[i] = r[i] > lo ? 1 : 0; b[i] = r[i] > hi ? 1 : 0; }
    const h = hysteresis(a, b, W, H); for (let i = 0; i < n; i++) m[i] |= h[i];
  }
  return m;
}

function binarise(gb, r1, pol, p) {
  const { w: W, h: H } = gb;
  let mask = inkMask(r1, W, H, pol, p);
  const comps = components(mask, W, H)[0].filter((c) => Math.max(bw(c), bh(c)) >= p.min_px);
  const scale = weightedMedianSide(comps, 0.9 * Math.max(H, W));
  const win1 = Math.max(p.wmin, pyRound(p.w1 * Math.min(H, W))) | 1;
  if (scale > 0) {
    const win2 = Math.max(p.wmin, pyRound(p.cwin * scale)) | 1;
    if (win2 !== win1) mask = inkMask(residual(gb, win2), W, H, pol, p);
  }
  return mask;
}

// ------------------------------------------------------------------ boxes
function mergeOverlapping(boxes, frac) {
  const B = boxes.map((b) => b.slice());
  let changed = true;
  while (changed) {
    changed = false;
    let i = 0;
    while (i < B.length) {
      let j = i + 1;
      while (j < B.length) {
        const a = B[i], b = B[j];
        if (inter(a, b) >= frac * Math.min(bw(a) * bh(a), bw(b) * bh(b))) { B[i] = union(a, b); B.splice(j, 1); changed = true; j = i + 1; }
        else j++;
      }
      i++;
    }
  }
  return B;
}

function skewAngle(boxes, axis, scale, p) {
  if (boxes.length < 3) return 0.0;
  const n = boxes.length, cx = new Float64Array(n), cy = new Float64Array(n), wt = new Float64Array(n);
  for (let i = 0; i < n; i++) { const b = boxes[i]; cx[i] = (b[0] + b[2]) / 2.0; cy[i] = (b[1] + b[3]) / 2.0; wt[i] = b[4]; }
  const binw = Math.max(2.0, scale / 6.0);
  let best = -1.0, bestA = 0.0;
  const na = pyRound(2 * p.skew_max / p.skew_step);
  const proj = new Float64Array(n);
  for (let k = 0; k <= na; k++) {
    const a = -p.skew_max + k * p.skew_step;
    const t = a * (Math.PI / 180.0), c = Math.cos(t), s = Math.sin(t);
    let mn = Infinity;
    for (let i = 0; i < n; i++) { proj[i] = axis === 'rows' ? (cy[i] * c - cx[i] * s) : (cx[i] * c + cy[i] * s); if (proj[i] < mn) mn = proj[i]; }
    const h = new Map();
    let mx = 0;
    for (let i = 0; i < n; i++) { const idx = Math.floor((proj[i] - mn) / binw); h.set(idx, (h.get(idx) || 0) + wt[i]); if (idx > mx) mx = idx; }
    let sc = 0;
    for (const v of h.values()) sc += v * v;
    if (sc > best * (1 + 1e-9)) { best = sc; bestA = a; }
  }
  return bestA;
}

function rotCoords(b, axis, ang) {
  const t = ang * (Math.PI / 180.0), c = Math.cos(t), s = Math.sin(t);
  const cx = (b[0] + b[2]) / 2.0, cy = (b[1] + b[3]) / 2.0;
  if (axis === 'rows') {
    const ac = cy * c - cx * s, al = cx * c + cy * s;
    return [al - bw(b) / 2.0, al + bw(b) / 2.0, ac - bh(b) / 2.0, ac + bh(b) / 2.0];
  }
  const ac = cx * c + cy * s, al = cy * c - cx * s;
  return [al - bh(b) / 2.0, al + bh(b) / 2.0, ac - bw(b) / 2.0, ac + bw(b) / 2.0];
}

function bands(boxes, axis, ang, shrink = 0.15) {
  const iv = boxes.map((b, i) => { const r = rotCoords(b, axis, ang), d = (r[3] - r[2]) * shrink; return [r[2] + d, r[3] - d, i]; });
  iv.sort(cmpTuple);
  const out = []; let cur = [], end = -1e18;
  for (const [a0, a1, i] of iv) {
    if (cur.length && a0 > end) { out.push(cur); cur = []; end = -1e18; }
    cur.push(i); end = Math.max(end, a1);
  }
  if (cur.length) out.push(cur);
  return out;
}

function elongation(boxes, lines, axis, ang) {
  let num = 0.0, den = 0.0;
  for (const L of lines) {
    const rc = L.map((i) => rotCoords(boxes[i], axis, ang));
    const along = Math.max(...rc.map((r) => r[1])) - Math.min(...rc.map((r) => r[0]));
    const thick = Math.max(...rc.map((r) => r[3])) - Math.min(...rc.map((r) => r[2]));
    let w = 0; for (const i of L) w += boxes[i][4];
    num += w * along / Math.max(thick, 1.0); den += w;
  }
  return num / Math.max(den, 1.0);
}

function mergeStrokes(boxesIn, lh, p) {
  const boxes = boxesIn.map((b) => b.slice());
  const isStroke = (b) => bh(b) >= p.stroke_aspect * Math.max(bw(b), 1) && bh(b) <= p.stroke_hmax * lh;
  const S = [];
  boxes.forEach((b, i) => { if (isStroke(b)) S.push(i); });
  S.sort((i, j) => cmpTuple([boxes[i][0], boxes[i][1], i], [boxes[j][0], boxes[j][1], j]));
  const parent = boxes.map((_, i) => i);
  const find = (x) => { while (parent[x] !== x) x = parent[x]; return x; };
  for (let ii = 0; ii < S.length; ii++) {
    const i = S[ii], a = boxes[i];
    for (let jj = ii + 1; jj < S.length; jj++) {
      const j = S[jj], b = boxes[j];
      const gap = b[0] - a[2];
      if (gap < 0) continue;
      const ha = bh(a), hb = bh(b);
      if (gap > p.stroke_gap * Math.max(ha, hb)) break;
      if (Math.max(ha, hb) > p.stroke_ratio * Math.min(ha, hb)) continue;
      if (Math.min(a[3], b[3]) - Math.max(a[1], b[1]) < 0.7 * Math.min(ha, hb)) continue;
      const y0 = Math.min(a[1], b[1]), y1 = Math.max(a[3], b[3]);
      let blocked = false;
      for (let k = 0; k < boxes.length; k++) {
        if (k === i || k === j) continue;
        const c = boxes[k], cx = (c[0] + c[2]) / 2.0;
        if (a[2] <= cx && cx <= b[0] && Math.min(c[3], y1) - Math.max(c[1], y0) > 0) { blocked = true; break; }
      }
      if (blocked) continue;
      const ra = find(i), rb = find(j);
      if (ra !== rb) parent[Math.max(ra, rb)] = Math.min(ra, rb);
      break;
    }
  }
  const runs = new Map();
  for (let i = 0; i < boxes.length; i++) { const r = find(i); if (!runs.has(r)) runs.set(r, []); runs.get(r).push(i); }
  const keys = [...runs.keys()].sort((a, b) => a - b);
  const out = [];
  for (const r of keys) {
    const members = runs.get(r);
    let u = boxes[members[0]];
    for (const m of members.slice(1)) u = union(u, boxes[m]);
    if (members.length > 1 && bw(u) > 1.3 * lh) for (const m of members) out.push(boxes[m]);
    else out.push(u);
  }
  return out;
}

function mergeFragments(boxesIn, lh, p) {
  const boxes = boxesIn.map((b) => b.slice());
  let changed = true;
  while (changed) {
    changed = false;
    const n = boxes.length;
    for (let i = 0; i < n; i++) {
      const a = boxes[i];
      if (Math.max(bw(a), bh(a)) >= p.frag_size * lh) continue;
      let best = null, bj = -1;
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const g = boxGap(a, boxes[j]);
        if (g <= p.frag_gap * lh && (best === null || g < best)) { best = g; bj = j; }
      }
      if (bj >= 0) { boxes[bj] = union(boxes[bj], a); boxes.splice(i, 1); changed = true; break; }
    }
  }
  return boxes;
}

function xyCut(idx, boxes, direction, preferY, tol, depth = 0) {
  if (idx.length <= 1 || depth > 8) return idx.slice();
  for (const useY of (preferY ? [true, false] : [false, true])) {
    const lo = useY ? (b) => b[1] : (b) => b[0];
    const hi = useY ? (b) => b[3] : (b) => b[2];
    const items = idx.slice().sort((i, j) => (lo(boxes[i]) + hi(boxes[i])) / 2.0 - (lo(boxes[j]) + hi(boxes[j])) / 2.0);
    let parts = [], cur = [], end = null;
    for (const i of items) {
      const b = boxes[i], ext = hi(b) - lo(b), a0 = lo(b) + tol * ext;
      if (cur.length && a0 >= end) { parts.push(cur); cur = []; end = null; }
      cur.push(i);
      const e = hi(b) - tol * ext;
      end = end === null ? e : Math.max(end, e);
    }
    parts.push(cur);
    if (parts.length > 1) {
      if (!useY && direction === 'rtl') parts = parts.reverse();
      const out = [];
      for (const q of parts) out.push(...xyCut(q, boxes, direction, !useY, tol, depth + 1));
      return out;
    }
  }
  const sgn = direction === 'rtl' ? -1 : 1;
  return idx.slice().sort((i, j) => cmpTuple([(boxes[i][1] + boxes[i][3]) / 2.0, sgn * (boxes[i][0] + boxes[i][2]) / 2.0],
                                             [(boxes[j][1] + boxes[j][3]) / 2.0, sgn * (boxes[j][0] + boxes[j][2]) / 2.0]));
}

function orderLine(boxes, axis, ang, direction, p) {
  const rc = boxes.map((b) => rotCoords(b, axis, ang));
  const n = boxes.length, parent = boxes.map((_, i) => i);
  const find = (x) => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    const ov = Math.min(rc[i][1], rc[j][1]) - Math.max(rc[i][0], rc[j][0]);
    if (ov >= p.group_overlap * Math.min(rc[i][1] - rc[i][0], rc[j][1] - rc[j][0])) {
      const a = find(i), b = find(j);
      if (a !== b) parent[Math.max(a, b)] = Math.min(a, b);
    }
  }
  const groups = new Map();
  for (let i = 0; i < n; i++) { const r = find(i); if (!groups.has(r)) groups.set(r, []); groups.get(r).push(i); }
  const glist = [...groups.values()];
  const sgn = (axis === 'rows' && direction === 'rtl') ? -1 : 1;
  const key = (g) => sgn > 0 ? sgn * Math.min(...g.map((i) => rc[i][0])) : sgn * Math.max(...g.map((i) => rc[i][1]));
  const keyed = glist.map((g) => [key(g), g]);
  keyed.sort((a, b) => a[0] - b[0]);
  const out = [];
  keyed.forEach(([, g], gi) => { for (const i of xyCut(g, boxes, direction, true, p.cut_tol)) out.push([boxes[i], gi]); });
  return out;
}

function recMerge(boxesIn, lh, score, p) {
  const boxes = boxesIn.map((b) => b.slice());
  for (;;) {
    let best = null;
    const n = boxes.length;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      const a = boxes[i], b = boxes[j];
      if (boxGap(a, b) > p.rm_gap * lh) continue;
      const u = union(a, b);
      if (Math.max(bw(u), bh(u)) > p.rm_size * lh) continue;
      let bad = false;
      for (let k = 0; k < n; k++) {
        if (k !== i && k !== j && inter(u, boxes[k]) > p.rm_other * bw(boxes[k]) * bh(boxes[k])) { bad = true; break; }
      }
      if (bad) continue;
      const gain = p.rm_theta > 0 ? score(u) - score(a) - score(b) + p.rm_theta : score(u) - Math.max(score(a), score(b));
      if (gain > p.rm_margin && (best === null || gain > best[0])) best = [gain, i, j, u];
    }
    if (best === null) return boxes;
    const [, i, j, u] = best;
    boxes[i] = u; boxes.splice(j, 1);
  }
}

function segmentPolarity(g0, gb, r1, pol, f, direction, layout, p, score) {
  const { w: W, h: H } = gb;
  const mask = binarise(gb, r1, pol, p);
  const comps = components(mask, W, H)[0].filter((c) => Math.max(bw(c), bh(c)) >= p.min_px);
  const res = { layout: 'rows', polarity: pol, factor: f, scale: 0.0, lines: [] };
  if (!comps.length) return res;
  const scale = weightedMedianSide(comps, 0.9 * Math.max(H, W));
  res.scale = scale;
  const keep = [];
  for (const c of comps) {
    const L = Math.max(bw(c), bh(c)), T = Math.min(bw(c), bh(c));
    if (L < p.speck_frac * scale) continue;
    if (L >= p.frame_len * scale && (T <= p.frame_thin * scale || c[4] < p.frame_fill * bw(c) * bh(c))) continue;
    keep.push(c);
  }
  const boxes = mergeOverlapping(keep, p.overlap);
  if (!boxes.length) return res;
  const hyp = {};
  for (const ax of ['rows', 'columns']) {
    const ang = skewAngle(boxes, ax, scale, p);
    const lines = bands(boxes, ax, ang);
    hyp[ax] = [elongation(boxes, lines, ax, ang), ang, lines];
  }
  const axis = (layout === 'rows' || layout === 'columns') ? layout : (hyp.rows[0] >= hyp.columns[0] ? 'rows' : 'columns');
  const ang = hyp[axis][1];
  let lines = hyp[axis][2];
  res.layout = axis; res.angle = ang;
  const linePos = (L) => { const rc = L.map((i) => rotCoords(boxes[i], axis, ang)); return (Math.min(...rc.map((r) => r[2])) + Math.max(...rc.map((r) => r[3]))) / 2.0; };
  lines = lines.map((L) => [linePos(L), L]).sort((a, b) => a[0] - b[0]).map((x) => x[1]);
  if (axis === 'columns' && direction === 'rtl') lines = lines.reverse();
  for (const L of lines) {
    let lb = L.map((i) => boxes[i]);
    const rc = lb.map((b) => rotCoords(b, axis, ang));
    const lh = Math.max(...rc.map((r) => r[3])) - Math.min(...rc.map((r) => r[2]));
    lb = mergeStrokes(lb, lh, p);
    lb = mergeFragments(lb, lh, p);
    lb = mergeOverlapping(lb, p.overlap);
    if (score) lb = recMerge(lb, lh, score, p);
    const seq = orderLine(lb, axis, ang, direction, p);
    res.lines.push(seq.map(([b, gi]) => ({ box: [b[0] * f, b[1] * f, bw(b) * f, bh(b) * f], wbox: b.slice(0, 4), group: gi,
                                           conf: score ? score(b) : null })));
  }
  return res;
}

// segmentation only (the Python reference is segment.segment). classify(crop Uint8Array 64*64) -> top-1 probability
export function segment(img, opts = {}) {
  const p = Object.assign({}, PARAMS, opts.params || {});
  const direction = opts.direction === 'rtl' ? 'rtl' : 'ltr';
  const layout = opts.layout || 'auto';
  const [g0, f] = downscale(img.d ? img : toGrey(img), p.max_side);
  const gb = box3(stretch(g0, p));
  const { w: W, h: H } = gb;
  const win1 = Math.max(p.wmin, pyRound(p.w1 * Math.min(H, W))) | 1;
  const r1 = residual(gb, win1);
  let [cands, stats] = polarityCandidates(r1, W, H, win1, p);
  let score = null;
  if (opts.classify) {
    const cache = new Map();
    score = (b) => {
      const k = b[0] + ',' + b[1] + ',' + b[2] + ',' + b[3];
      if (!cache.has(k)) cache.set(k, opts.classify(prepCrop(g0, [b[0], b[1], b[2], b[3]], p.crop_pad)));
      return cache.get(k);
    };
  } else cands = cands.slice(0, 1);
  const run = (q) => {
    let best = null, bestSc = null;
    for (const pol of cands) {
      const res = segmentPolarity(g0, gb, r1, pol, f, direction, layout, q, score);
      let sc = 0.0;
      if (score) for (const L of res.lines) for (const s of L) sc += s.conf - q.sel_floor;
      res.selection_score = sc;
      if (best === null || sc > bestSc) { best = res; bestSc = sc; }
    }
    return best;
  };
  const count = (r) => r.lines.reduce((a, L) => a + L.length, 0);
  let best = run(p);
  const nsig = count(best);
  if (nsig < p.fb_min_signs && p.fb_factor < 1.0) {          // weak ink: once more with lower thresholds
    const q = Object.assign({}, p, { kl: p.kl * p.fb_factor, kh: p.kh * p.fb_factor });
    const alt = run(q);
    if (score ? alt.selection_score > best.selection_score : count(alt) > nsig) { alt.fallback = true; best = alt; }
  }
  best.candidates = cands; best.pol_stats = stats; best.work = g0;
  return best;
}

// ------------------------------------------------------------------ crops (exactly like segment.prep_crop / build_real.to64)
function axisTaps(nIn, nOut) {
  const taps = [], sc = nIn / nOut;
  if (nOut <= nIn) {
    for (let j = 0; j < nOut; j++) {
      const a = j * sc, b = (j + 1) * sc, k0 = Math.floor(a), k1 = Math.min(nIn, Math.ceil(b)), t = [];
      for (let k = k0; k < k1; k++) { const w = Math.min(b, k + 1) - Math.max(a, k); if (w > 0) t.push([k, w / sc]); }
      taps.push(t);
    }
  } else {
    for (let j = 0; j < nOut; j++) {
      let x = (j + 0.5) * sc - 0.5; if (x < 0) x = 0.0;
      const k0 = Math.floor(x), fr = x - k0, k1 = Math.min(k0 + 1, nIn - 1);
      taps.push([[k0, 1.0 - fr], [k1, fr]]);
    }
  }
  return taps;
}

export function prepCrop(work, wbox, padFrac = PARAMS.crop_pad, size = 64, inner = 54) {
  const { w: W, h: H, d } = work;
  let [x0, y0, x1, y1] = wbox;
  const pad = pyRound(padFrac * Math.max(x1 - x0, y1 - y0));
  x0 = Math.max(0, x0 - pad); y0 = Math.max(0, y0 - pad); x1 = Math.min(W, x1 + pad); y1 = Math.min(H, y1 + pad);
  const h = y1 - y0, w = x1 - x0;
  const px = (y, x) => d[(y0 + y) * W + x0 + x];
  const border = [];
  for (let x = 0; x < w; x++) border.push(px(0, x));
  for (let x = 0; x < w; x++) border.push(px(h - 1, x));
  for (let y = 0; y < h; y++) border.push(px(y, 0));
  for (let y = 0; y < h; y++) border.push(px(y, w - 1));
  border.sort((a, b) => a - b);
  const bgv = border[Math.floor((border.length - 1) / 2)];
  const s = inner / Math.max(h, w);
  const nw = Math.max(1, Math.floor(w * s + 0.5)), nh = Math.max(1, Math.floor(h * s + 0.5));
  const tx = axisTaps(w, nw), ty = axisTaps(h, nh);
  const tmp = new Float64Array(h * nw);
  for (let j = 0; j < nw; j++) for (let y = 0; y < h; y++) {
    let acc = 0; for (const [k, wt] of tx[j]) acc = acc + wt * px(y, k); tmp[y * nw + j] = acc;
  }
  const out = new Uint8Array(size * size).fill(bgv);
  const yo = Math.floor((size - nh) / 2), xo = Math.floor((size - nw) / 2);
  for (let i = 0; i < nh; i++) for (let x = 0; x < nw; x++) {
    let acc = 0; for (const [k, wt] of ty[i]) acc = acc + wt * tmp[k * nw + x];
    let v = Math.floor(acc + 0.5); v = v < 0 ? 0 : (v > 255 ? 255 : v);
    out[(yo + i) * size + xo + x] = v;
  }
  return out;
}

// ------------------------------------------------------------------ CNN
let CURRENT = null;

function parseModel(meta, buf) {
  const u8 = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  if (u8.byteLength !== meta.total_bytes) throw new Error('model.bin has ' + u8.byteLength + ' bytes, expected ' + meta.total_bytes);
  const ab = u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength);
  const T = {};
  for (const t of meta.tensors) {
    if ((t.dtype || 'float32') === 'int8') T[t.name] = new Int8Array(ab, t.offset, t.length);
    else T[t.name] = new Float32Array(ab, t.offset, t.length / 4);
    T[t.name].shape = t.shape;
  }
  for (const name of Object.keys(T)) {           // dequantise: w = float32(q * scale[out])
    if (!T[name + '_scale']) continue;
    const q = T[name], sc = T[name + '_scale'], per = q.length / sc.length, w = new Float32Array(q.length);
    for (let i = 0; i < q.length; i++) w[i] = q[i] * sc[Math.floor(i / per)];
    w.shape = q.shape; T[name] = w;
  }
  const layers = meta.layers.map((L) => Object.assign({}, L, L.weight ? { W: T[L.weight], B: T[L.bias] } : {}));
  const prior = T.prior_logit || new Float32Array(meta.classes.length);
  return { meta, layers, prior, classes: meta.classes, threshold: meta.threshold != null ? meta.threshold : 0.56,
           norm: meta.normalisation, bytes: u8.byteLength };
}

export async function loadModel(src) {
  let meta, bin;
  if (typeof src === 'string') {
    const isNode = typeof process !== 'undefined' && process.versions && process.versions.node && !/^https?:/.test(src);
    if (isNode) {
      const fs = await import('node:fs/promises'), path = await import('node:path');
      const file = src.startsWith('file:') ? new URL(src).pathname : src;
      meta = JSON.parse(await fs.readFile(file, 'utf8'));
      bin = await fs.readFile(path.join(path.dirname(file), meta.weights_file));
    } else {
      meta = await (await fetch(src)).json();
      bin = await (await fetch(new URL(meta.weights_file, new URL(src, typeof location !== 'undefined' ? location.href : undefined)))).arrayBuffer();
    }
  } else {
    meta = typeof src.json === 'string' ? JSON.parse(src.json) : src.json;
    bin = src.bin;
  }
  CURRENT = parseModel(meta, bin);
  return CURRENT;
}

function conv3x3ReluPool(x, C, H, Wd, w, b, Cout) {
  // zero padding 1, stride 1, then ReLU and 2x2 max-pool stride 2. Four output channels per pass over the input
  // (each input pixel is loaded once for four channels); per output the arithmetic is the same as one channel at a time.
  const Hp = H + 2, Wp = Wd + 2, pad = new Float32Array(C * Hp * Wp);
  for (let c = 0; c < C; c++) for (let y = 0; y < H; y++) pad.set(x.subarray((c * H + y) * Wd, (c * H + y + 1) * Wd), (c * Hp + y + 1) * Wp + 1);
  const HW = H * Wd, Ho = H >> 1, Wo = Wd >> 1, out = new Float32Array(Cout * Ho * Wo);
  const conv = new Float32Array(4 * HW);
  for (let c0 = 0; c0 < Cout; c0 += 4) {
    const nb = Math.min(4, Cout - c0);
    for (let j = 0; j < 4; j++) conv.fill(j < nb ? b[c0 + j] : 0, j * HW, (j + 1) * HW);
    for (let ci = 0; ci < C; ci++) {
      const base = ci * Hp * Wp;
      const k0 = (c0 * C + ci) * 9, k1 = k0 + C * 9, k2 = k1 + C * 9, k3 = k2 + C * 9;
      if (nb < 4) {                                   // generic tail (not used by the shipped model)
        for (let j = 0; j < nb; j++) {
          const k = ((c0 + j) * C + ci) * 9, o = j * HW;
          for (let y = 0; y < H; y++) {
            const r0 = base + y * Wp, r1 = r0 + Wp, r2 = r1 + Wp, oy = o + y * Wd;
            for (let xx = 0; xx < Wd; xx++) {
              conv[oy + xx] += w[k] * pad[r0 + xx] + w[k + 1] * pad[r0 + xx + 1] + w[k + 2] * pad[r0 + xx + 2]
                             + w[k + 3] * pad[r1 + xx] + w[k + 4] * pad[r1 + xx + 1] + w[k + 5] * pad[r1 + xx + 2]
                             + w[k + 6] * pad[r2 + xx] + w[k + 7] * pad[r2 + xx + 1] + w[k + 8] * pad[r2 + xx + 2];
            }
          }
        }
        continue;
      }
      const a0 = w[k0], a1 = w[k0 + 1], a2 = w[k0 + 2], a3 = w[k0 + 3], a4 = w[k0 + 4], a5 = w[k0 + 5], a6 = w[k0 + 6], a7 = w[k0 + 7], a8 = w[k0 + 8];
      const b0 = w[k1], b1 = w[k1 + 1], b2 = w[k1 + 2], b3 = w[k1 + 3], b4 = w[k1 + 4], b5 = w[k1 + 5], b6 = w[k1 + 6], b7 = w[k1 + 7], b8 = w[k1 + 8];
      const c0_ = w[k2], c1 = w[k2 + 1], c2 = w[k2 + 2], c3 = w[k2 + 3], c4 = w[k2 + 4], c5 = w[k2 + 5], c6 = w[k2 + 6], c7 = w[k2 + 7], c8 = w[k2 + 8];
      const d0 = w[k3], d1 = w[k3 + 1], d2 = w[k3 + 2], d3 = w[k3 + 3], d4 = w[k3 + 4], d5 = w[k3 + 5], d6 = w[k3 + 6], d7 = w[k3 + 7], d8 = w[k3 + 8];
      for (let y = 0; y < H; y++) {
        const r0 = base + y * Wp, r1 = r0 + Wp, r2 = r1 + Wp, o = y * Wd;
        for (let xx = 0; xx < Wd; xx++) {
          const p0 = pad[r0 + xx], p1 = pad[r0 + xx + 1], p2 = pad[r0 + xx + 2];
          const p3 = pad[r1 + xx], p4 = pad[r1 + xx + 1], p5 = pad[r1 + xx + 2];
          const p6 = pad[r2 + xx], p7 = pad[r2 + xx + 1], p8 = pad[r2 + xx + 2];
          const i = o + xx;
          conv[i] += a0 * p0 + a1 * p1 + a2 * p2 + a3 * p3 + a4 * p4 + a5 * p5 + a6 * p6 + a7 * p7 + a8 * p8;
          conv[i + HW] += b0 * p0 + b1 * p1 + b2 * p2 + b3 * p3 + b4 * p4 + b5 * p5 + b6 * p6 + b7 * p7 + b8 * p8;
          conv[i + 2 * HW] += c0_ * p0 + c1 * p1 + c2 * p2 + c3 * p3 + c4 * p4 + c5 * p5 + c6 * p6 + c7 * p7 + c8 * p8;
          conv[i + 3 * HW] += d0 * p0 + d1 * p1 + d2 * p2 + d3 * p3 + d4 * p4 + d5 * p5 + d6 * p6 + d7 * p7 + d8 * p8;
        }
      }
    }
    for (let j = 0; j < nb; j++) {
      const ob = (c0 + j) * Ho * Wo, cb = j * HW;
      for (let y = 0; y < Ho; y++) for (let xx = 0; xx < Wo; xx++) {
        const i = cb + 2 * y * Wd + 2 * xx;
        const m = Math.max(conv[i], conv[i + 1], conv[i + Wd], conv[i + Wd + 1]);
        out[ob + y * Wo + xx] = m > 0 ? m : 0;           // ReLU and max-pool commute
      }
    }
  }
  return out;
}

// raw logits (without the prior) of one 64x64 crop
export function logits(crop, model = CURRENT) {
  if (!model) throw new Error('call loadModel() first');
  const n = model.norm, sc = Math.fround(n.scale);
  let x = new Float32Array(64 * 64);
  for (let i = 0; i < x.length; i++) x[i] = Math.fround(Math.fround(Math.fround(crop[i] * sc) - n.mean) / n.std);
  let C = 1, H = 64, W = 64;
  for (const L of model.layers) {
    if (L.type === 'conv3x3') { x = conv3x3ReluPool(x, C, H, W, L.W, L.B, L.out_ch); C = L.out_ch; H >>= 1; W >>= 1; }
    else if (L.type === 'global_avg_pool') {
      const f = new Float32Array(C), hw = H * W;
      for (let c = 0; c < C; c++) { let s = 0; for (let i = 0; i < hw; i++) s += x[c * hw + i]; f[c] = s / hw; }
      x = f;
    } else if (L.type === 'linear') {
      const o = new Float32Array(L.out_features), fin = L.in_features;
      for (let k = 0; k < o.length; k++) { let s = L.B[k]; const r = k * fin; for (let i = 0; i < fin; i++) s += L.W[r + i] * x[i]; o[k] = s; }
      x = o;
    }
  }
  return x;
}

// top-5 of softmax(logits + prior) -> { idx: [5], prob: [5] }
export function classify(crop, model = CURRENT) {
  const lo = logits(crop, model), pr = model.prior, n = lo.length, z = new Float64Array(n);
  let mx = -Infinity;
  for (let i = 0; i < n; i++) { z[i] = lo[i] + pr[i]; if (z[i] > mx) mx = z[i]; }
  let s = 0;
  for (let i = 0; i < n; i++) { z[i] = Math.exp(z[i] - mx); s += z[i]; }
  const idx = [], prob = [];
  for (let t = 0; t < 5; t++) {
    let bi = -1;
    for (let i = 0; i < n; i++) if ((bi < 0 || z[i] > z[bi]) && !idx.includes(i)) bi = i;
    idx.push(bi); prob.push(z[bi] / s);
  }
  return { idx, prob };
}

// full pipeline: photo -> lines of recognised signs in reading order
export function recognise(imageData, opts = {}) {
  const model = opts.model || CURRENT;
  if (!model) throw new Error('call loadModel() first');
  const dirIn = opts.direction || 'ltr';
  const direction = dirIn === 'rtl' ? 'rtl' : 'ltr';          // 'auto': no facing detector yet -> ltr
  const cache = new Map();
  const t0 = (typeof performance !== 'undefined' ? performance : Date).now();
  let nCnn = 0, msCnn = 0;
  const cls = (crop) => {
    let h = 0; for (let i = 0; i < crop.length; i += 7) h = (h * 31 + crop[i]) | 0;
    const key = h + ':' + crop.length;
    let hit = cache.get(key);
    if (hit) for (const e of hit) if (e.crop.every((v, i) => v === crop[i])) return e.res;
    const tc = (typeof performance !== 'undefined' ? performance : Date).now();
    const res = classify(crop, model);
    msCnn += (typeof performance !== 'undefined' ? performance : Date).now() - tc; nCnn++;
    if (!hit) cache.set(key, hit = []);
    hit.push({ crop, res });
    return res;
  };
  const seg = segment(imageData, { direction, layout: opts.layout, params: opts.params, classify: (crop) => cls(crop).prob[0] });
  const C = model.classes;
  const lines = seg.lines.map((L) => ({
    signs: L.map((s) => {
      const r = cls(prepCrop(seg.work, s.wbox, PARAMS.crop_pad));
      const alts = r.idx.map((k, t) => ({ char: C[k].char, codepoint: C[k].codepoint, conf: r.prob[t], cls: k }));
      return { char: alts[0].char, codepoint: alts[0].codepoint, conf: r.prob[0], uncertain: r.prob[0] < model.threshold,
               alts, box: s.box, group: s.group };
    }),
  }));
  return { layout: seg.layout, direction, direction_source: dirIn === 'auto' ? 'default (no facing detector)' : 'user',
           polarity: seg.polarity, scale: seg.scale * seg.factor, lines,
           stats: { cnn_crops: nCnn, cnn_ms: msCnn, total_ms: (typeof performance !== 'undefined' ? performance : Date).now() - t0 } };
}
