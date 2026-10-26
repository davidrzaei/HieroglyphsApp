// hiero_ocr.js - offline hieroglyph recognition for photos and scans. Dependency-free ES module (browser, Node >= 20).
//
//   import { loadModel, loadFacing, recognise, reorder } from './hiero_ocr.js';
//   await loadModel('model.json');            // or loadModel({ json: <object|string>, bin: <ArrayBuffer|Uint8Array> })
//   await loadFacing('facing.json');          // optional (reading direction); or loadFacing({ json, bin })
//   await loadDetector('det.json');           // optional (P3 B sign detector for walls); or loadDetector({ json, bin })
//   await loadSignLM('signlm.json');          // optional (P3 C2: each line decoded with the corpus' sign sequences)
//   const r = recognise(imageData, { direction: 'auto' });  // 'ltr' | 'rtl' | 'auto' (from the signs' facing)
//   r = { layout: 'rows'|'columns', direction, direction_source: 'auto'|'user'|'default (no facing model)',
//         direction_certainty, direction_votes, direction_flag, direction_auto: {...}, polarity, angle,
//         lines: [ { signs: [ { char, codepoint, conf, uncertain, alts: [ { char, codepoint, conf, cls } x5 ],
//         box: [x, y, w, h], group, wbox, k } ] } ], segmenter: 'classic'|'detector', segmenter_mode, segmenter_scores, stats }
//   with loadSignLM(): each sign also has rec = { cls, prob } (the recogniser's own top-k) and p_rec; char/conf/uncertain/
//   alts come from the joint decoding of its line (see "dictionary decoding"); r.decode = { applied, params, changed };
//   opts.signlm: null switches the decoding off, opts.lm overrides its parameters.
//   detector lines get in-line merges (P3 C1, DET_PARAMS.det_merge: overlapping boxes, plural-stroke runs read as one sign).
//   opts.segmenter: 'auto' (default with a detector: detector; if it finds <= 30 signs also the classical segmentation,
//   the larger quality score wins) | 'classic' | 'detector'; without loadDetector() always 'classic'.
//   const r2 = reorder(r, 'rtl');             // the same signs in the other reading direction (no new recognition)
//
// imageData: { width, height, data } with RGBA (browser ImageData), RGB or grey (1 byte per pixel) data.
// The segmentation is a line-by-line port of segment.py (the Python reference) and gives the same boxes; the CNN is the
// forward pass of ref_forward.py (3x3 conv with folded BatchNorm, ReLU, 2x2 max-pool, global average pool, linear,
// softmax over logits + corpus prior). 'auto' direction (P3 A0): see "reading direction" below; without loadFacing()
// it falls back to 'ltr' (direction_source 'default (no facing model)').

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
                                           conf: score ? score(b) : null, k: lb.indexOf(b) })));
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

function parseTensors(meta, buf) {
  const u8 = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  if (u8.byteLength !== meta.total_bytes) throw new Error(meta.weights_file + ' has ' + u8.byteLength + ' bytes, expected ' + meta.total_bytes);
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
  return { T, bytes: u8.byteLength };
}

function parseModel(meta, buf) {
  const { T, bytes } = parseTensors(meta, buf);
  const u8 = { byteLength: bytes };
  const layers = meta.layers.map((L) => Object.assign({}, L, L.weight ? { W: T[L.weight], B: T[L.bias] } : {}));
  const prior = T.prior_logit || new Float32Array(meta.classes.length);
  return { meta, layers, prior, classes: meta.classes, threshold: meta.threshold != null ? meta.threshold : 0.56,
           norm: meta.normalisation, bytes: u8.byteLength };
}

export async function loadModel(src) {
  const { meta, bin } = await readModelFiles(src);
  CURRENT = parseModel(meta, bin);
  return CURRENT;
}

async function readModelFiles(src) {
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
  return { meta, bin };
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

// raw logits (without the prior) of one 64x64 crop. tap = { after: n } also keeps the output of the n-th conv layer
// (after ReLU + max-pool) in tap.map (the facing head branches off the classifier's conv3 map).
export function logits(crop, model = CURRENT, tap = null) {
  if (!model) throw new Error('call loadModel() first');
  const n = model.norm, sc = Math.fround(n.scale);
  let x = new Float32Array(64 * 64);
  for (let i = 0; i < x.length; i++) x[i] = Math.fround(Math.fround(Math.fround(crop[i] * sc) - n.mean) / n.std);
  let C = 1, H = 64, W = 64, nConv = 0;
  for (const L of model.layers) {
    if (L.type === 'conv3x3') {
      x = conv3x3ReluPool(x, C, H, W, L.W, L.B, L.out_ch); C = L.out_ch; H >>= 1; W >>= 1;
      if (tap && ++nConv === tap.after) tap.map = x;
    }
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

// top-k (default 5) of softmax(logits + prior) -> { idx: [k], prob: [k] }
export function classify(crop, model = CURRENT, tap = null, k = 5) {
  const lo = logits(crop, model, tap), pr = model.prior, n = lo.length, z = new Float64Array(n);
  let mx = -Infinity;
  for (let i = 0; i < n; i++) { z[i] = lo[i] + pr[i]; if (z[i] > mx) mx = z[i]; }
  let s = 0;
  for (let i = 0; i < n; i++) { z[i] = Math.exp(z[i] - mx); s += z[i]; }
  const idx = [], prob = [];
  for (let t = 0; t < k; t++) {
    let bi = -1;
    for (let i = 0; i < n; i++) if ((bi < 0 || z[i] > z[bi]) && !idx.includes(i)) bi = i;
    idx.push(bi); prob.push(z[bi] / s);
  }
  return { idx, prob };
}

// ------------------------------------------------------------------ reading direction from the signs' facing (P3 A0)
// Signs face the beginning of the line. The font drawings (standard orientation) face left = read left to right; a
// mirrored line (signs facing right) is read right to left, in rows and in columns alike. The facing head (facing.json +
// facing.bin, int8 like model.bin) branches off the classifier's conv3 map and has one logit per asymmetric class
// (> 0 = mirrored = faces right); the classifier's top-1 class picks the logit. recognise(img, {direction: 'auto'})
// segments and classifies once, lets the confident asymmetric signs vote (weight = classifier confidence x clipped
// logit), and re-orders the boxes it already has (reorder(); the boxes never depend on the direction).
let FACING = null;
const VOTE_DEFAULTS = Object.freeze({ min_conf: 0.56, clip: 4.0, temperature: 0.3329, sure: 0.95, min_certainty: 0.55,
                                      min_votes: 1, max_voters: 48 });   // = facing.json "vote" (chosen on dev pages, facing/tune_vote.py)
const now = () => (typeof performance !== 'undefined' ? performance : Date).now();

function parseFacing(meta, buf) {
  const { T, bytes } = parseTensors(meta, buf);
  const layers = meta.layers.map((L) => Object.assign({}, L, L.weight ? { W: T[L.weight], B: T[L.bias] } : {}));
  const headOf = new Map();
  meta.head_classes.forEach((c, h) => headOf.set(c, h));
  const voters = new Set(meta.vote_classes);
  return { meta, layers, headOf, voters, after: meta.branch_after_conv, vote: Object.assign({}, VOTE_DEFAULTS, meta.vote || {}),
           bytes };
}

// loadFacing('facing.json') or loadFacing({ json, bin }); without it, direction 'auto' falls back to 'ltr'.
export async function loadFacing(src) {
  const { meta, bin } = await readModelFiles(src);
  FACING = parseFacing(meta, bin);
  return FACING;
}

// facing logit of one sign from the classifier's conv3 map (Float32Array C*H*W) and the classifier's class index;
// null when the class has no facing output.
export function facingFromMap(map, cls, fm = FACING) {
  if (!fm) throw new Error('call loadFacing() first');
  const h = fm.headOf.get(cls);
  if (h === undefined) return null;
  let x = map, C = fm.meta.input_shape[0], H = fm.meta.input_shape[1], W = fm.meta.input_shape[2];
  for (const L of fm.layers) {
    if (L.type === 'conv3x3') { x = conv3x3ReluPool(x, C, H, W, L.W, L.B, L.out_ch); C = L.out_ch; H >>= 1; W >>= 1; }
    else if (L.type === 'linear') {
      const o = new Float32Array(L.out_features), fin = L.in_features;
      for (let k = 0; k < o.length; k++) { let s = L.B[k]; const r = k * fin; for (let i = 0; i < fin; i++) s += L.W[r + i] * x[i]; o[k] = L.relu && s < 0 ? 0 : s; }
      x = o;
    } else if (L.type === 'head') {                       // only the row of the class
      const fin = L.in_features, r = h * fin; let s = L.B[h];
      for (let i = 0; i < fin; i++) s += L.W[r + i] * x[i];
      return s;
    }
  }
  throw new Error('facing model without a head layer');
}

// convenience (tests): crop -> { cls, conf, logit } with the loaded classifier and facing head
export function facing(crop, model = CURRENT, fm = FACING) {
  const tap = { after: fm.after }, r = classify(crop, model, tap);
  return { cls: r.idx[0], conf: r.prob[0], logit: facingFromMap(tap.map, r.idx[0], fm) };
}

// the vote. signs: recognised signs with .conf, .alts[0].cls and .fmap (conv3 map, kept by recognise for candidates)
export function voteDirection(signs, fm = FACING, vote = {}) {
  const V = Object.assign({}, fm ? fm.vote : VOTE_DEFAULTS, vote);
  const cand = signs.filter((s) => s.fmap && s.conf >= V.min_conf && fm.voters.has(s.alts[0].cls));
  cand.sort((a, b) => b.conf - a.conf);
  const used = cand.slice(0, V.max_voters);
  let E = 0, right = 0;
  for (const s of used) {
    const z = facingFromMap(s.fmap, s.alts[0].cls, fm);
    s.facing = z;
    E += s.conf * Math.max(-V.clip, Math.min(V.clip, z));
    if (z > 0) right++;
  }
  const pRtl = 1 / (1 + Math.exp(-V.temperature * E));
  const cert = Math.max(pRtl, 1 - pRtl);
  const flagged = used.length < V.min_votes || cert < V.min_certainty;
  const direction = flagged ? 'ltr' : (pRtl > 0.5 ? 'rtl' : 'ltr');
  return { direction, certainty: direction === 'rtl' ? pRtl : 1 - pRtl, level: flagged ? 'unsure' : (cert >= V.sure ? 'sure' : 'likely'),
           flagged, votes: used.length, facing_right: right, facing_left: used.length - right, evidence: E, p_rtl: pRtl };
}

// lines in the order of direction `from` -> the same signs in the order of direction `to` (exactly what segment() gives
// with `to`: the line order flips for columns, each line is re-ordered by orderLine from its boxes in segment order k)
function orderSigns(lines, axis, ang, from, to, p = PARAMS) {
  if (from === to) return lines;
  const out = lines.map((L) => {
    const byK = L.signs.slice().sort((a, b) => a.k - b.k);
    const boxes = byK.map((s) => [s.wbox[0], s.wbox[1], s.wbox[2], s.wbox[3], 0]);
    const seq = orderLine(boxes, axis, ang, to, p);
    return Object.assign({}, L, { signs: seq.map(([b, gi]) => Object.assign(byK[boxes.indexOf(b)], { group: gi })) });
  });
  return axis === 'columns' ? out.reverse() : out;
}

// a recognise() result in the other reading direction (cheap: no segmentation, no CNN). The detected direction stays
// in r.direction_auto. source: 'user' (an override) or 'auto'.
export function reorder(r, direction, source = 'user') {
  const to = direction === 'rtl' ? 'rtl' : 'ltr';
  const lines = orderSigns(r.lines.map((L) => ({ signs: L.signs.map((s) => Object.assign({}, s)) })), r.layout, r.angle || 0,
                           r.direction, to, Object.assign({}, PARAMS, r.params || {}));
  const out = Object.assign({}, r, { lines, direction: to, direction_source: source });
  if (source === 'auto' && r.direction_auto) out.direction_certainty = to === 'rtl' ? r.direction_auto.p_rtl : 1 - r.direction_auto.p_rtl;
  if (r.decode && r.decode.applied && to !== r.direction && SIGNLM) {      // P3 C2: the decoding depends on the order
    const d = decodeLines(out.lines, SIGNLM, r.decode.params);
    out.decode = Object.assign({}, r.decode, { changed: d.changed });
  }
  return out;
}

// ------------------------------------------------------------------ sign detector (P3 B)
// A small fully convolutional net (det.json + det.bin, int8 like model.bin; detect.py is the Python reference) finds
// the signs on walls, where the classical binarisation fails: the work image (as in segment()) is area-resampled to a
// long side <= 640, the net gives stride-4 maps (sign-centre heat, offset, log size), the 3x3 peaks above a threshold
// are the signs (boxes the classifier cannot read, top-1 < det_floor, are dropped), then the classical layout/ordering
// steps run on these boxes (no binarisation merges). recognise() with segmenter 'auto' (the default once loadDetector()
// was called): the detector runs first; more than det_gate signs = a wall or a dense page -> detector result; otherwise
// the classical segmentation runs too and the larger quality score sum(top-1 probability - sel_floor) wins.
let DETECTOR = null;
export const DET_PARAMS = Object.freeze({ det_side: 640, det_thr: 0.40, det_nms: 0.6, det_min_px: 3, det_floor: 0.2, det_gate: 30,
                                         det_merge: 6, pl_wide: 1.2, pl_min: 0.4, pl_gain: -9.0, pl_count: 1 });   // P3 C1: det_merge, pl_*
// (= detect.DET_DEFAULTS, chosen on the dev sets only)

export async function loadDetector(src) {
  const { meta, bin } = await readModelFiles(src);
  const { T, bytes } = parseTensors(meta, bin);
  const ops = meta.ops.map((o) => Object.assign({}, o, o.op === 'conv' ? { W: T[o.name + '.weight'], B: T[o.name + '.bias'] } : {}));
  DETECTOR = { meta, ops, bytes, decode: meta.decode || {} };
  return DETECTOR;
}

function areaResize(g, nw, nh) {                    // = det_common.area_resize (same taps and float64 order as prepCrop)
  const { w: W, h: H, d } = g, tx = axisTaps(W, nw), ty = axisTaps(H, nh);
  const tmp = new Float64Array(H * nw);
  for (let j = 0; j < nw; j++) for (let y = 0; y < H; y++) {
    let acc = 0; for (const [k, wt] of tx[j]) acc = acc + wt * d[y * W + k]; tmp[y * nw + j] = acc;
  }
  const out = new Uint8Array(nw * nh);
  for (let i = 0; i < nh; i++) for (let x = 0; x < nw; x++) {
    let acc = 0; for (const [k, wt] of ty[i]) acc = acc + wt * tmp[k * nw + x];
    let v = Math.floor(acc + 0.5); out[i * nw + x] = v < 0 ? 0 : (v > 255 ? 255 : v);
  }
  return { w: nw, h: nh, d: out };
}

function conv3x3(x, C, H, Wd, w, b, Cout, stride, relu) {   // zero padding 1; stride 1 or 2; 4 output channels per pass
  const Ho = stride === 2 ? H >> 1 : H, Wo = stride === 2 ? Wd >> 1 : Wd, HWo = Ho * Wo;
  const Hp = H + 2, Wp = Wd + 2, pad = new Float32Array(C * Hp * Wp);
  for (let c = 0; c < C; c++) for (let y = 0; y < H; y++) pad.set(x.subarray((c * H + y) * Wd, (c * H + y + 1) * Wd), (c * Hp + y + 1) * Wp + 1);
  const out = new Float32Array(Cout * HWo), acc = new Float64Array(4 * HWo);
  for (let c0 = 0; c0 < Cout; c0 += 4) {
    const nb = Math.min(4, Cout - c0);
    for (let j = 0; j < 4; j++) acc.fill(j < nb ? b[c0 + j] : 0, j * HWo, (j + 1) * HWo);
    for (let ci = 0; ci < C; ci++) {
      const base = ci * Hp * Wp;
      const k = [0, 1, 2, 3].map((j) => ((Math.min(c0 + j, Cout - 1)) * C + ci) * 9);
      const wa = w.subarray(k[0], k[0] + 9), wb = w.subarray(k[1], k[1] + 9), wc = w.subarray(k[2], k[2] + 9), wd = w.subarray(k[3], k[3] + 9);
      const a0 = wa[0], a1 = wa[1], a2 = wa[2], a3 = wa[3], a4 = wa[4], a5 = wa[5], a6 = wa[6], a7 = wa[7], a8 = wa[8];
      const b0 = wb[0], b1 = wb[1], b2 = wb[2], b3 = wb[3], b4 = wb[4], b5 = wb[5], b6 = wb[6], b7 = wb[7], b8 = wb[8];
      const e0 = wc[0], e1 = wc[1], e2 = wc[2], e3 = wc[3], e4 = wc[4], e5 = wc[5], e6 = wc[6], e7 = wc[7], e8 = wc[8];
      const d0 = wd[0], d1 = wd[1], d2 = wd[2], d3 = wd[3], d4 = wd[4], d5 = wd[5], d6 = wd[6], d7 = wd[7], d8 = wd[8];
      for (let y = 0; y < Ho; y++) {
        const r0 = base + (stride * y) * Wp, r1 = r0 + Wp, r2 = r1 + Wp, o = y * Wo;
        for (let xx = 0; xx < Wo; xx++) {
          const q = stride * xx;
          const p0 = pad[r0 + q], p1 = pad[r0 + q + 1], p2 = pad[r0 + q + 2];
          const p3 = pad[r1 + q], p4 = pad[r1 + q + 1], p5 = pad[r1 + q + 2];
          const p6 = pad[r2 + q], p7 = pad[r2 + q + 1], p8 = pad[r2 + q + 2];
          const i = o + xx;
          acc[i] += a0 * p0 + a1 * p1 + a2 * p2 + a3 * p3 + a4 * p4 + a5 * p5 + a6 * p6 + a7 * p7 + a8 * p8;
          acc[i + HWo] += b0 * p0 + b1 * p1 + b2 * p2 + b3 * p3 + b4 * p4 + b5 * p5 + b6 * p6 + b7 * p7 + b8 * p8;
          acc[i + 2 * HWo] += e0 * p0 + e1 * p1 + e2 * p2 + e3 * p3 + e4 * p4 + e5 * p5 + e6 * p6 + e7 * p7 + e8 * p8;
          acc[i + 3 * HWo] += d0 * p0 + d1 * p1 + d2 * p2 + d3 * p3 + d4 * p4 + d5 * p5 + d6 * p6 + d7 * p7 + d8 * p8;
        }
      }
    }
    for (let j = 0; j < nb; j++) {
      const ob = (c0 + j) * HWo, ab = j * HWo;
      for (let i = 0; i < HWo; i++) { const v = acc[ab + i]; out[ob + i] = relu && v < 0 ? 0 : v; }
    }
  }
  return out;
}

function conv1x1(x, C, HW, w, b, Cout, relu) {
  const out = new Float32Array(Cout * HW), acc = new Float64Array(HW);
  for (let co = 0; co < Cout; co++) {
    acc.fill(b[co]);
    for (let ci = 0; ci < C; ci++) {
      const wv = w[co * C + ci], xb = ci * HW;
      for (let i = 0; i < HW; i++) acc[i] += wv * x[xb + i];
    }
    const ob = co * HW;
    for (let i = 0; i < HW; i++) { const v = acc[i]; out[ob + i] = relu && v < 0 ? 0 : v; }
  }
  return out;
}

function up2(x, C, H, W) {
  const out = new Float32Array(C * 4 * H * W), W2 = 2 * W;
  for (let c = 0; c < C; c++) for (let y = 0; y < 2 * H; y++) {
    const src = (c * H + (y >> 1)) * W, dst = (c * 2 * H + y) * W2;
    for (let xx = 0; xx < W2; xx++) out[dst + xx] = x[src + (xx >> 1)];
  }
  return out;
}

// raw detector output for a work image: { out: Float32Array 5*h*w, h, w, sx, sy, vh, vw } (stride-4 maps)
export function detectorForward(work, side = DET_PARAMS.det_side, det = DETECTOR) {
  if (!det) throw new Error('call loadDetector() first');
  const s = Math.min(1.0, side / Math.max(work.h, work.w));
  let g = work;
  if (s < 1.0) g = areaResize(work, Math.max(1, pyRound(work.w * s)), Math.max(1, pyRound(work.h * s)));
  const Hp = Math.ceil(g.h / 32) * 32, Wp = Math.ceil(g.w / 32) * 32;
  const x = new Float32Array(Hp * Wp);
  for (let y = 0; y < Hp; y++) {
    const sy = Math.min(y, g.h - 1);
    for (let xx = 0; xx < Wp; xx++) {
      const v = g.d[sy * g.w + Math.min(xx, g.w - 1)];
      x[y * Wp + xx] = Math.fround(Math.fround(Math.fround(v / 255) - 0.5) / 0.25);
    }
  }
  const env = { x: { t: x, C: 1, H: Hp, W: Wp } };
  for (const o of det.ops) {
    if (o.op === 'conv') {
      const a = env[o.in];
      const t = o.k === 3 ? conv3x3(a.t, a.C, a.H, a.W, o.W, o.B, o.cout, o.s, o.relu) : conv1x1(a.t, a.C, a.H * a.W, o.W, o.B, o.cout, o.relu);
      env[o.out] = { t, C: o.cout, H: o.s === 2 ? a.H >> 1 : a.H, W: o.s === 2 ? a.W >> 1 : a.W };
    } else if (o.op === 'up2') {
      const a = env[o.in]; env[o.out] = { t: up2(a.t, a.C, a.H, a.W), C: a.C, H: 2 * a.H, W: 2 * a.W };
    } else if (o.op === 'add') {
      const a = env[o.a], b = env[o.b], t = new Float32Array(a.t.length);
      for (let i = 0; i < t.length; i++) t[i] = a.t[i] + b.t[i];
      env[o.out] = { t, C: a.C, H: a.H, W: a.W };
    }
  }
  const r = env[det.meta.output];
  return { out: r.t, h: r.H, w: r.W, sx: g.w / work.w, sy: g.h / work.h, vh: g.h, vw: g.w };
}

function iouXYWH(a, b) {
  const x0 = Math.max(a[0], b[0]), y0 = Math.max(a[1], b[1]), x1 = Math.min(a[0] + a[2], b[0] + b[2]), y1 = Math.min(a[1] + a[3], b[1] + b[3]);
  const i = Math.max(0, x1 - x0) * Math.max(0, y1 - y0), u = a[2] * a[3] + b[2] * b[3] - i;
  return u > 0 ? i / u : 0.0;
}

// sign boxes in work pixels: [[x0, y0, x1, y1, area, score]] (= detect.detect_boxes)
export function detectBoxes(work, p = Object.assign({}, PARAMS, DET_PARAMS), det = DETECTOR) {
  const F = detectorForward(work, p.det_side, det);
  const { out, h, w } = F, hw = h * w, S = det.meta.stride || 4;
  const heat = new Float64Array(hw);
  for (let i = 0; i < hw; i++) heat[i] = 1 / (1 + Math.exp(-out[i]));
  const cand = [];
  for (let y = 0; y < h; y++) {
    if (y * S >= F.vh) break;
    for (let x = 0; x < w; x++) {
      if (x * S >= F.vw) continue;
      const v = heat[y * w + x];
      if (v < p.det_thr) continue;
      let peak = true;
      for (let dy = -1; dy <= 1 && peak; dy++) for (let dx = -1; dx <= 1; dx++) {
        const yy = y + dy, xx = x + dx;
        if (yy < 0 || yy >= h || xx < 0 || xx >= w) continue;
        if (heat[yy * w + xx] > v) { peak = false; break; }
      }
      if (!peak) continue;
      const i = y * w + x;
      const cx = (x + out[hw + i]) * S, cy = (y + out[2 * hw + i]) * S, bwv = Math.exp(out[3 * hw + i]) * S, bhv = Math.exp(out[4 * hw + i]) * S;
      cand.push([v, [(cx - bwv / 2) / F.sx, (cy - bhv / 2) / F.sy, bwv / F.sx, bhv / F.sy]]);
    }
  }
  cand.sort((a, b) => b[0] - a[0]);                 // stable: ties keep the row-major order (like Python)
  const keep = [];
  for (const c of cand) if (keep.every((k) => iouXYWH(c[1], k[1]) < p.det_nms)) keep.push(c);
  const res = [];
  for (const [s, [bx, by, bwv, bhv]] of keep) {
    const x0 = Math.max(0, Math.floor(bx + 0.5)), y0 = Math.max(0, Math.floor(by + 0.5));
    const x1 = Math.min(work.w, Math.floor(bx + bwv + 0.5)), y1 = Math.min(work.h, Math.floor(by + bhv + 0.5));
    if (x1 - x0 >= p.det_min_px && y1 - y0 >= p.det_min_px) res.push([x0, y0, x1, y1, (x1 - x0) * (y1 - y0), s]);
  }
  return res;
}

// P3 C1: in-line merges on the detector's lines (= detect.merge_det_line); det_merge bit flags in this order: 8 drop a box
// lying inside a larger one, 1 classical plural strokes, 2 classical overlapping boxes, 16 classical fragments, 4 plural-
// stroke runs checked by the recogniser (mergePlural). top(box) -> [class, prob].
export const PLURAL_CLASSES = Object.freeze(new Set([997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1019, 1020, 1021,
  1022, 1023, 1024, 1025, 1026, 1027, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036]));   // Z2*, Z3*, Z4*, Z15A-I, Z16A-H

const PLURAL_STROKES = new Map([997, 998, 999, 1000, 1001, 1002, 1003, 1004].map((c) => [c, 3]).concat([[1005, 2], [1006, 2]],
  [...Array(9).keys()].map((i) => [1019 + i, 2 + i]), [...Array(8).keys()].map((i) => [1029 + i, 2 + i])));   // strokes per class

function dropContained(lb, frac = 0.8) {
  const out = [];
  lb.forEach((a, i) => {
    const aa = bw(a) * bh(a);
    const inside = lb.some((b, k) => k !== i && bw(b) * bh(b) > aa &&
      Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])) * Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1])) >= frac * aa);
    if (!inside) out.push(a);
  });
  return out;
}

// runs of stroke-like detector boxes -> one box when the recogniser reads the union as a plural-stroke sign more surely
// than its pieces (= detect.merge_plural)
function mergePlural(lb, lh, p, top) {
  let B = lb.map((b) => b.slice());
  for (const vert of [false, true]) {
    const T = (b) => vert ? [b[1], b[0], b[3], b[2]] : [b[0], b[1], b[2], b[3]];
    const elig = (b) => { const t = T(b), w = t[2] - t[0], h = t[3] - t[1]; return h >= 3 && h <= p.stroke_hmax * lh && w <= p.pl_wide * h; };
    const stroke = (b) => { const t = T(b); return (t[3] - t[1]) >= p.stroke_aspect * Math.max(t[2] - t[0], 1); };
    const idx = [];
    B.forEach((b, i) => { if (elig(b)) idx.push(i); });
    idx.sort((i, k) => cmpTuple([T(B[i])[0], T(B[i])[1], i], [T(B[k])[0], T(B[k])[1], k]));
    const nxt = new Map();
    for (let ii = 0; ii < idx.length; ii++) {
      const i = idx[ii], a = T(B[i]), ha = a[3] - a[1];
      for (let jj = ii + 1; jj < idx.length; jj++) {
        const k = idx[jj], b = T(B[k]), hb = b[3] - b[1];
        if (b[0] - a[2] > p.stroke_gap * Math.max(ha, hb)) break;
        if ((b[0] + b[2]) <= (a[0] + a[2])) continue;
        if (Math.max(ha, hb) > p.stroke_ratio * Math.min(ha, hb)) continue;
        if (Math.min(a[3], b[3]) - Math.max(a[1], b[1]) < 0.7 * Math.min(ha, hb)) continue;
        const y0 = Math.min(a[1], b[1]), y1 = Math.max(a[3], b[3]);
        let blocked = false;
        for (let m = 0; m < B.length; m++) {
          if (m === i || m === k) continue;
          const c = T(B[m]), cx = (c[0] + c[2]) / 2.0;
          if ((a[0] + a[2]) / 2.0 < cx && cx < (b[0] + b[2]) / 2.0 && Math.min(c[3], y1) - Math.max(c[1], y0) > 0) { blocked = true; break; }
        }
        if (!blocked) nxt.set(i, k);
        break;
      }
    }
    const hasPrev = new Set(nxt.values()), used = new Set(), runs = [];
    for (const i of idx) {
      if (hasPrev.has(i) || !nxt.has(i) || used.has(i)) continue;
      const r = [i]; used.add(i);
      while (nxt.has(r[r.length - 1]) && !used.has(nxt.get(r[r.length - 1]))) { const q = nxt.get(r[r.length - 1]); r.push(q); used.add(q); }
      runs.push(r);
    }
    const drop = new Set(), add = [];
    for (const r of runs) {
      let s0 = 0;
      while (r.length - s0 >= 2) {
        let done = false;
        for (let L = r.length - s0; L >= 2 && !done; L--) {
          for (let st = s0; st <= r.length - L; st++) {
            const sub = r.slice(st, st + L);
            if (!sub.some((m) => stroke(B[m]))) continue;
            let u = B[sub[0]].slice();
            for (const m of sub.slice(1)) u = [Math.min(u[0], B[m][0]), Math.min(u[1], B[m][1]), Math.max(u[2], B[m][2]), Math.max(u[3], B[m][3]), u[4] + B[m][4]].concat(u.slice(5));
            const tu = T(u);
            if (tu[2] - tu[0] > 1.3 * lh) continue;
            const [c, pr] = top(u);
            let mean = 0; for (const m of sub) mean += top(B[m])[1];
            mean /= sub.length;
            if (PLURAL_CLASSES.has(c) && pr >= p.pl_min && (!p.pl_count || L <= PLURAL_STROKES.get(c)) && pr - mean >= p.pl_gain) {
              if (u.length > 5) u[5] = Math.max(...sub.map((m) => B[m][5]));
              sub.forEach((m) => drop.add(m)); add.push(u); s0 = st + L; done = true;
              break;
            }
          }
        }
        if (!done) break;
      }
    }
    if (drop.size) B = B.filter((_, i) => !drop.has(i)).concat(add);
  }
  return B;
}

function mergeDetLine(lb, lh, p, top) {
  const m = p.det_merge | 0;
  if (m <= 0 || lb.length < 2) return lb;
  let out = lb.map((b) => b.slice());
  if (m & 8) out = dropContained(out);
  if (m & 1) out = mergeStrokes(out.map((b) => b.slice(0, 5)), lh, p);
  if (m & 2) out = mergeOverlapping(out.map((b) => b.slice(0, 5)), p.overlap);
  if (m & 16) out = mergeFragments(out.map((b) => b.slice(0, 5)), lh, p);
  if ((m & 4) && top) out = mergePlural(out.map((b) => b.slice(0, 5).concat([0.0])), lh, p, top);
  return out.map((u) => {
    let s = 0.0, any = false;
    for (const b of lb) if (b[0] >= u[0] && b[1] >= u[1] && b[2] <= u[2] && b[3] <= u[3]) { s = any ? Math.max(s, b[5]) : b[5]; any = true; }
    return u.slice(0, 5).concat([s]);
  });
}

// detector boxes -> segment()-style result (layout, angle, lines in reading order) (= detect.order_boxes)
function orderDetBoxes(boxes, f, direction, layout, p, score, top = null) {
  const res = { layout: 'rows', polarity: 'detector', factor: f, scale: 0.0, angle: 0.0, lines: [] };
  if (!boxes.length) return res;
  const sides = boxes.map((b) => Math.max(bw(b), bh(b))).sort((a, b) => a - b), n = sides.length;
  const scale = n % 2 ? sides[n >> 1] : (sides[n / 2 - 1] + sides[n / 2]) / 2.0;
  res.scale = scale;
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
    if (p.det_merge) {                                     // P3 C1
      const rc = lb.map((b) => rotCoords(b, axis, ang));
      lb = mergeDetLine(lb, Math.max(...rc.map((r) => r[3])) - Math.min(...rc.map((r) => r[2])), p, top);
    }
    const seq = orderLine(lb, axis, ang, direction, p);
    res.lines.push(seq.map(([b, gi]) => ({ box: [b[0] * f, b[1] * f, bw(b) * f, bh(b) * f], wbox: b.slice(0, 4), group: gi,
                                           conf: score ? score(b) : null, k: lb.indexOf(b), det: b[5] })));
  }
  return res;
}

// segmentation with the detector (same result format as segment(); polarity 'detector')
export function segmentDet(img, opts = {}) {
  const p = Object.assign({}, PARAMS, DET_PARAMS, opts.params || {});
  const direction = opts.direction === 'rtl' ? 'rtl' : 'ltr';
  const [g0, f] = downscale(img.d ? img : toGrey(img), p.max_side);
  const td = now();
  const boxes = detectBoxes(g0, p, opts.detector || DETECTOR);
  const detMs = now() - td;
  let score = null;
  if (opts.classify) {
    const cache = new Map();
    score = (b) => {
      const k = b[0] + ',' + b[1] + ',' + b[2] + ',' + b[3];
      if (!cache.has(k)) cache.set(k, opts.classify(prepCrop(g0, [b[0], b[1], b[2], b[3]], p.crop_pad)));
      return cache.get(k);
    };
  }
  let top = null;
  if (opts.classifyTop) {                                   // P3 C1: [class, prob] of a box (plural-stroke merge)
    const tc = new Map();
    top = (b) => {
      const k = b[0] + ',' + b[1] + ',' + b[2] + ',' + b[3];
      if (!tc.has(k)) tc.set(k, opts.classifyTop(prepCrop(g0, [b[0], b[1], b[2], b[3]], p.crop_pad)));
      return tc.get(k);
    };
  }
  const kept = score && p.det_floor > 0 ? boxes.filter((b) => score(b) >= p.det_floor) : boxes;   // unreadable = texture
  const res = orderDetBoxes(kept, f, direction, opts.layout || 'auto', p, score, top);
  let sc = 0.0;
  if (score) for (const L of res.lines) for (const s of L) sc += s.conf - p.sel_floor;
  res.selection_score = sc; res.n_det = boxes.length; res.work = g0; res.det_ms = detMs;
  return res;
}

// ------------------------------------------------------------------ dictionary decoding of each line (P3 C2)
// The recogniser reads every sign on its own. With a sign-sequence model of the corpus (signlm.json + signlm.bin, built
// by signlm.py from the phone database: unigram/bigram/trigram counts of the signs in written order, interpolated
// Witten-Bell) each line is decoded jointly: for every sign its top-k candidates; the score of a reading of the line is
// sum_i [ log p_rec_i(c_i) - mu * log prior(c_i) + lam * log P(c_i | c_i-2, c_i-1) ]. The search is exact over the
// top-k lattice (states = the last two candidates: k^2 states and k^3 transitions per sign), with forward-backward for
// each sign's posterior over its candidates. A line starts without context (photos rarely start at a sentence start).
// mode 'mpm': each sign takes its largest posterior; 'viterbi': the best joint reading. A sign's conf becomes
// posterior x (the recogniser's probability mass on its top-k), its alts the top 5 by posterior, and uncertain =
// conf < unc. The recogniser's own candidates stay in sign.rec, so reorder() can decode again in the other order.
let SIGNLM = null;
export const LM_DEFAULTS = Object.freeze({ k: 10, lam: 0.35, mu: 0.25, wb: 1.0, order: 3, mode: 'mpm', unc: 0.56 });
// (= signlm.py LM_DEFAULTS / signlm.json "params", chosen on the dev sets only)

function readVarintPairs(u8, off, n) {
  const keys = new Float64Array(n), cnt = new Uint32Array(n);
  let k = 0;
  for (let i = 0; i < n; i++) {
    for (let h = 0; h < 2; h++) {
      let v = 0, mul = 1, b;
      do { b = u8[off++]; v += (b & 0x7F) * mul; mul *= 128; } while (b & 0x80);
      if (h === 0) { k += v; keys[i] = k; } else cnt[i] = v;
    }
  }
  return { keys, cnt, off };
}

function bsearch(keys, key) {
  let lo = 0, hi = keys.length - 1;
  while (lo <= hi) { const m = (lo + hi) >> 1, v = keys[m]; if (v === key) return m; if (v < key) lo = m + 1; else hi = m - 1; }
  return -1;
}

function parseSignLM(meta, buf) {
  const u8 = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  if (u8.byteLength !== meta.total_bytes) throw new Error(meta.weights_file + ' has ' + u8.byteLength + ' bytes, expected ' + meta.total_bytes);
  const V = meta.V, u = new Float64Array(V), dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  let N = 0;
  for (let i = 0; i < V; i++) { u[i] = dv.getUint32(4 * i, true); N += u[i]; }
  const bi = readVarintPairs(u8, meta.offsets.bi, meta.n_bi), tri = readVarintPairs(u8, meta.offsets.tri, meta.n_tri);
  const c1 = new Float64Array(V), t1 = new Float64Array(V);
  for (let i = 0; i < bi.keys.length; i++) { const a = Math.floor(bi.keys[i] / V); c1[a] += bi.cnt[i]; t1[a] += 1; }
  const c2 = new Float64Array(bi.keys.length), t2 = new Float64Array(bi.keys.length), noCtx = [];
  const ts = new Int32Array(bi.keys.length).fill(-1), te = new Int32Array(bi.keys.length);   // trigram range per context
  for (let i = 0; i < tri.keys.length; i++) {
    const ab = Math.floor(tri.keys[i] / V), j = bsearch(bi.keys, ab);
    if (j < 0) { noCtx.push(ab); continue; }
    c2[j] += tri.cnt[i]; t2[j] += 1;
    if (ts[j] < 0) ts[j] = i;
    te[j] = i + 1;
  }
  if (noCtx.length) throw new Error('signlm: trigram contexts without a bigram (' + noCtx.length + ')');
  return { meta, V, nCls: meta.n_cls, u, N, bi, tri, c1, t1, c2, t2, ts, te, params: Object.assign({}, LM_DEFAULTS, meta.params || {}),
           bytes: u8.byteLength };
}

// loadSignLM('signlm.json') or loadSignLM({ json, bin }); without it recognise() keeps the recogniser's own choices.
export async function loadSignLM(src) {
  const { meta, bin } = await readModelFiles(src);
  SIGNLM = parseSignLM(meta, bin);
  return SIGNLM;
}

// interpolated Witten-Bell (= signlm.SignLM; the same operations in the same order)
function lmP1(lm, c) { return (lm.u[c] + 1.0) / (lm.N + lm.V); }
function lmP2(lm, b, c, wb) {
  const n = lm.c1[b];
  if (n === 0) return lmP1(lm, c);
  const t = wb * lm.t1[b], j = bsearch(lm.bi.keys, b * lm.V + c);
  return ((j >= 0 ? lm.bi.cnt[j] : 0) + t * lmP1(lm, c)) / (n + t);
}
function lmP3(lm, a, b, c, wb) {
  const j = bsearch(lm.bi.keys, a * lm.V + b), n = j >= 0 ? lm.c2[j] : 0;
  if (n === 0) return lmP2(lm, b, c, wb);
  const t = wb * lm.t2[j], q = bsearch(lm.tri.keys, (a * lm.V + b) * lm.V + c);
  return ((q >= 0 ? lm.tri.cnt[q] : 0) + t * lmP2(lm, b, c, wb)) / (n + t);
}
export function signLMLogP(ctx, c, lm = SIGNLM, params = {}) {      // tests: log P(c | ctx) with ctx = [] | [b] | [a, b]
  const P = Object.assign({}, lm.params, params);
  if (P.order >= 3 && ctx.length >= 2) return Math.log(lmP3(lm, ctx[ctx.length - 2], ctx[ctx.length - 1], c, P.wb));
  if (P.order >= 2 && ctx.length >= 1) return Math.log(lmP2(lm, ctx[ctx.length - 1], c, P.wb));
  return Math.log(lmP1(lm, c));
}

function lse(xs) {
  let m = -Infinity;
  const n = xs.length;
  for (let i = 0; i < n; i++) if (xs[i] > m) m = xs[i];
  if (m === -Infinity) return m;
  let s = 0;
  for (let i = 0; i < n; i++) s += Math.exp(xs[i] - m);
  return m + Math.log(s);
}

// one line: cands[i] = class ids (best first, <= k), probs[i] = their recogniser probabilities.
// -> { choice: [index into cands[i]], post: [[posterior per candidate]] }   (= signlm.decode_lattice)
export function decodeCandidates(cands, probs, lm = SIGNLM, params = {}, model = CURRENT) {
  const P = Object.assign({}, lm.params, params), n = cands.length, lam = P.lam, mu = P.mu, wb = P.wb, order = P.order;
  if (!n) return { choice: [], post: [] };
  const lp = (ctx, c) => (order >= 3 && ctx.length >= 2) ? Math.log(lmP3(lm, ctx[0], ctx[1], c, wb))
    : ((order >= 2 && ctx.length >= 1) ? Math.log(lmP2(lm, ctx[ctx.length - 1], c, wb)) : Math.log(lmP1(lm, c)));
  const E = cands.map((C, i) => C.map((c, r) => Math.log(Math.max(probs[i][r], 1e-30)) - (mu ? mu * (model.prior[c] / 0.5) : 0)));
  const K = cands.map((C) => C.length);
  // L[i][p][q][r] log P; i = 0: [r], i = 1: [q][r]. Fast path (same numbers as lp): P2 once per (q, r), the trigram
  // context's index and its contiguous range of trigrams once per (p, q).
  const L0 = cands[0].map((c) => lp([], c));
  const Lt = [];                                         // i >= 1: flat arrays
  const V = lm.V, bk = lm.bi.keys, tk = lm.tri.keys;
  const searchIn = (keys, lo, hi, key) => { hi--; while (lo <= hi) { const m = (lo + hi) >> 1, v = keys[m]; if (v === key) return m; if (v < key) lo = m + 1; else hi = m - 1; } return -1; };
  for (let i = 1; i < n; i++) {
    const kp = i >= 2 ? K[i - 2] : 1, kq = K[i - 1], kr = K[i], t = new Float64Array(kp * kq * kr);
    const p2 = new Float64Array(kq * kr);
    for (let q = 0; q < kq; q++) for (let r = 0; r < kr; r++) p2[q * kr + r] = order >= 2 ? lmP2(lm, cands[i - 1][q], cands[i][r], wb) : lmP1(lm, cands[i][r]);
    for (let p = 0; p < kp; p++) for (let q = 0; q < kq; q++) {
      const a = i >= 2 ? cands[i - 2][p] : -1, b = cands[i - 1][q];
      const jx = (i >= 2 && order >= 3) ? bsearch(bk, a * V + b) : -1, nn = jx >= 0 ? lm.c2[jx] : 0;
      for (let r = 0; r < kr; r++) {
        let pr = p2[q * kr + r];
        if (nn !== 0) {
          const tt = wb * lm.t2[jx], qx = searchIn(tk, lm.ts[jx], lm.te[jx], (a * V + b) * V + cands[i][r]);
          pr = ((qx >= 0 ? lm.tri.cnt[qx] : 0) + tt * pr) / (nn + tt);
        }
        t[(p * kq + q) * kr + r] = Math.log(pr);
      }
    }
    Lt.push(t);
  }
  // forward (alpha) and viterbi (v) over states (q, r) stored flat q * kr + r; i = 0: [r]
  const A = [], Vt = [], BP = [];
  A.push(L0.map((x, r) => lam * x + E[0][r])); Vt.push(A[0].slice()); BP.push(null);
  for (let i = 1; i < n; i++) {
    const kp = i >= 2 ? K[i - 2] : 1, kq = K[i - 1], kr = K[i], t = Lt[i - 1];
    const a = new Float64Array(kq * kr), v = new Float64Array(kq * kr), bp = new Int32Array(kq * kr);
    const xs = new Float64Array(kp);
    for (let q = 0; q < kq; q++) for (let r = 0; r < kr; r++) {
      let best = -Infinity, arg = 0;
      for (let p = 0; p < kp; p++) {
        const prevA = i >= 2 ? A[i - 1][p * kq + q] : A[0][q], prevV = i >= 2 ? Vt[i - 1][p * kq + q] : Vt[0][q];
        const tr = lam * t[(p * kq + q) * kr + r] + E[i][r];
        xs[p] = prevA + tr;
        const y = prevV + tr;
        if (y > best) { best = y; arg = p; }
      }
      a[q * kr + r] = lse(xs); v[q * kr + r] = best; bp[q * kr + r] = arg;
    }
    A.push(a); Vt.push(v); BP.push(bp);
  }
  // backward
  const B = new Array(n);
  B[n - 1] = new Float64Array(A[n - 1].length);
  for (let i = n - 1; i >= 2; i--) {
    const kp = K[i - 2], kq = K[i - 1], kr = K[i], t = Lt[i - 1], b = new Float64Array(kp * kq);
    const xs = new Float64Array(kr);
    for (let p = 0; p < kp; p++) for (let q = 0; q < kq; q++) {
      for (let r = 0; r < kr; r++) xs[r] = lam * t[(p * kq + q) * kr + r] + E[i][r] + B[i][q * kr + r];
      b[p * kq + q] = lse(xs);
    }
    B[i - 1] = b;
  }
  if (n >= 2) {
    const kq = K[0], kr = K[1], t = Lt[0], b = new Float64Array(kq);
    for (let q = 0; q < kq; q++) { const xs = []; for (let r = 0; r < kr; r++) xs.push(lam * t[q * kr + r] + E[1][r] + B[1][q * kr + r]); b[q] = lse(xs); }
    B[0] = b;
  }
  const Z = lse(Array.from(A[n - 1]));
  const post = [];
  for (let i = 0; i < n; i++) {
    if (i === 0) { post.push(Array.from(A[0], (x, r) => Math.exp(x + B[0][r] - Z))); continue; }
    const kq = K[i - 1], kr = K[i], m = [];
    for (let r = 0; r < kr; r++) { const xs = []; for (let q = 0; q < kq; q++) xs.push(A[i][q * kr + r] + B[i][q * kr + r] - Z); m.push(Math.exp(lse(xs))); }
    post.push(m);
  }
  let choice;
  if (P.mode === 'viterbi') {
    if (n === 1) choice = [argmax(Vt[0])];
    else {
      const kr = K[n - 1]; let s = argmax(Vt[n - 1]), q = Math.floor(s / kr), r = s % kr;
      choice = [r, q];
      for (let i = n - 1; i >= 2; i--) { const p = BP[i][q * K[i] + r]; choice.push(p); r = q; q = p; }
      choice.reverse();
    }
  } else choice = post.map(argmax);
  return { choice, post };
}
function argmax(a) { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; }

// decode recognise() lines in place (signs need .rec = { cls, prob }: the recogniser's top-k). -> { changed, signs }
export function decodeLines(lines, lm = SIGNLM, params = {}, model = CURRENT) {
  const P = Object.assign({}, lm.params, params), C = model.classes;
  let changed = 0, nSigns = 0;
  for (const L of lines) {
    const S = L.signs.filter((s) => s.rec && s.rec.cls.length);
    if (!S.length) continue;
    const cands = S.map((s) => s.rec.cls.slice(0, P.k)), probs = S.map((s) => s.rec.prob.slice(0, P.k));
    const { choice, post } = decodeCandidates(cands, probs, lm, P, model);
    S.forEach((s, i) => {
      let mass = 0; for (const x of probs[i]) mass += x;
      const idx = cands[i].map((_, j) => j).sort((x, y) => (post[i][y] - post[i][x]) || (x - y));
      if (idx[0] !== choice[i]) { idx.splice(idx.indexOf(choice[i]), 1); idx.unshift(choice[i]); }
      const alts = idx.slice(0, 5).map((j) => ({ char: C[cands[i][j]].char, codepoint: C[cands[i][j]].codepoint, conf: post[i][j] * mass, cls: cands[i][j] }));
      if (cands[i][choice[i]] !== s.rec.cls[0]) changed++;
      nSigns++;
      Object.assign(s, { char: alts[0].char, codepoint: alts[0].codepoint, conf: alts[0].conf, uncertain: alts[0].conf < P.unc, alts,
                         p_rec: probs[i][choice[i]] });
    });
  }
  return { changed, signs: nSigns };
}

// full pipeline: photo -> lines of recognised signs in reading order
//   opts.direction: 'ltr' | 'rtl' (given by the user) | 'auto' (decided from the signs' facing when loadFacing() was called)
//   result: { layout, direction, direction_source: 'user'|'auto'|'default (no facing model)', direction_certainty (auto),
//             direction_votes (auto: signs that voted), direction_flag (auto: null or 'too little evidence'),
//             direction_auto: { direction, certainty, level 'sure'|'likely'|'unsure', flagged, votes, facing_right,
//             facing_left, evidence, p_rtl }, polarity, scale, angle, lines: [{ signs: [{ char, codepoint, conf,
//             uncertain, alts, box, group, wbox, k, facing? }] }], stats }
export function recognise(imageData, opts = {}) {
  const model = opts.model || CURRENT;
  if (!model) throw new Error('call loadModel() first');
  const dirIn = opts.direction || 'ltr';
  const fm = opts.facing === undefined ? FACING : opts.facing;
  const auto = dirIn === 'auto' && !!fm;
  const direction = dirIn === 'rtl' ? 'rtl' : 'ltr';          // 'auto' segments in ltr order, then re-orders
  const p = Object.assign({}, PARAMS, opts.params || {});
  const cache = new Map(), maps = new WeakMap();
  const keepConf = auto ? Math.min(fm.vote.min_conf, opts.vote && opts.vote.min_conf != null ? opts.vote.min_conf : 1) : 1;
  const lm = opts.signlm === undefined ? SIGNLM : opts.signlm;                 // P3 C2: null = the recogniser's own choices
  const lmP = lm ? Object.assign({}, lm.params, opts.lm || {}) : null, kTop = lm ? Math.max(5, lmP.k) : 5;
  const t0 = now();
  let nCnn = 0, msCnn = 0;
  const cls = (crop) => {
    let h = 0; for (let i = 0; i < crop.length; i += 7) h = (h * 31 + crop[i]) | 0;
    const key = h + ':' + crop.length;
    let hit = cache.get(key);
    if (hit) for (const e of hit) if (e.crop.every((v, i) => v === crop[i])) return e.res;
    const tc = now();
    const tap = auto ? { after: fm.after } : null;
    const res = classify(crop, model, tap, kTop);
    if (tap && res.prob[0] >= keepConf && fm.voters.has(res.idx[0])) maps.set(res, tap.map);   // vote candidates only
    msCnn += now() - tc; nCnn++;
    if (!hit) cache.set(key, hit = []);
    hit.push({ crop, res });
    return res;
  };
  // P3 B: classical segmentation, the detector, or both ('auto', default with a detector): the larger quality score wins
  const det = opts.detector === undefined ? DETECTOR : opts.detector;
  const mode = det ? (opts.segmenter || 'auto') : 'classic';
  const grey = imageData.d ? imageData : toGrey(imageData);
  const sopts = { direction, layout: opts.layout, params: opts.params, classify: (crop) => cls(crop).prob[0], detector: det,
                  classifyTop: (crop) => { const r = cls(crop); return [r.idx[0], r.prob[0]]; } };
  let seg, scores = null;
  if (mode === 'classic') seg = segment(grey, sopts);
  else if (mode === 'detector') seg = segmentDet(grey, sopts);
  else {
    const b = segmentDet(grey, sopts), gp = Object.assign({}, DET_PARAMS, opts.params || {});
    if (b.lines.reduce((n, L) => n + L.length, 0) > gp.det_gate) { seg = b; scores = { classic: null, detector: b.selection_score, gate: true }; }
    else {
      const a = segment(grey, sopts);
      scores = { classic: a.selection_score, detector: b.selection_score, gate: false };
      seg = b.selection_score > a.selection_score ? b : a;
      seg.det_ms = b.det_ms;
    }
  }
  const C = model.classes;
  const lines = seg.lines.map((L) => ({
    signs: L.map((s) => {
      const r = cls(prepCrop(seg.work, s.wbox, p.crop_pad));
      const alts = r.idx.slice(0, 5).map((k, t) => ({ char: C[k].char, codepoint: C[k].codepoint, conf: r.prob[t], cls: k }));
      const o = { char: alts[0].char, codepoint: alts[0].codepoint, conf: r.prob[0], uncertain: r.prob[0] < model.threshold,
                  alts, box: s.box, group: s.group, wbox: s.wbox, k: s.k };
      if (lm) o.rec = { cls: r.idx.slice(0, lmP.k), prob: r.prob.slice(0, lmP.k) };     // the recogniser's own top-k (P3 C2)
      if (maps.has(r)) Object.defineProperty(o, 'fmap', { value: maps.get(r), enumerable: false });
      return o;
    }),
  }));
  const res = { layout: seg.layout, direction, direction_source: dirIn === 'auto' ? 'default (no facing model)' : 'user',
                polarity: seg.polarity, scale: seg.scale * seg.factor, angle: seg.angle || 0, params: opts.params || null, lines,
                segmenter: seg.polarity === 'detector' ? 'detector' : 'classic', segmenter_mode: mode, segmenter_scores: scores,
                stats: { cnn_crops: nCnn, cnn_ms: msCnn, det_ms: seg.det_ms != null ? seg.det_ms : null } };
  if (auto) {
    const tf = now();
    const v = voteDirection(lines.flatMap((L) => L.signs), fm, opts.vote);
    if (v.direction !== direction) res.lines = orderSigns(lines, seg.layout, res.angle, direction, v.direction, p);
    Object.assign(res, { direction: v.direction, direction_source: 'auto', direction_certainty: v.certainty, direction_votes: v.votes,
                         direction_flag: v.flagged ? 'too little evidence' : null, direction_auto: v });
    res.stats.facing_ms = now() - tf;
  }
  if (lm) {                                                    // P3 C2: decode each line in its final reading order
    const tl = now(), d = decodeLines(res.lines, lm, lmP, model);
    res.decode = { applied: true, params: lmP, changed: d.changed };
    res.stats.decode_ms = now() - tl;
  } else res.decode = { applied: false };
  res.stats.total_ms = now() - t0;
  return res;
}
