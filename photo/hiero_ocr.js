// hiero_ocr.js - offline hieroglyph recognition for photos and scans. Dependency-free ES module (browser, Node >= 20).
//
//   import { loadModel, loadFacing, recognise, reorder } from './hiero_ocr.js';
//   await loadModel('model.json');            // or loadModel({ json: <object|string>, bin: <ArrayBuffer|Uint8Array> })
//   await loadFacing('facing.json');          // optional (reading direction); or loadFacing({ json, bin })
//   await loadDetector('det.json');           // optional (P3 B sign detector for walls); or loadDetector({ json, bin })
//   await loadCloseDetector('det_close.json'); // optional (P3 F: the detector for the smaller pyramid levels = close-ups)
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
//   P3 G: at the close-up scale (few signs at the 640 pass, quadrat size U at that input >= 80 px) the quadrat path reads the
//   pyramid levels together (r.segmenter 'quad', r.segmenter_scores.quad = { U, U640, U_lines, levels, filled }); every
//   result line has quads: [{ at, n, tree }] = its quadrats of >= 2 signs (positions in reading order; tree { t: 'v' | 'h',
//   k: [position | subtree] } = the app's typed-input format: v = above each other, h = side by side). opts.quads: false
//   switches the quadrat path off.
//   P3 H: on EVERY path (wall, normal scale, close-up) the lines are cut into quadrats again after the signs are found:
//   fragments are dropped, neighbours that fit one square are joined, a quadrat's cell grows over the ink next to it, ink
//   left between quadrats is read as an empty quadrat, and a quadrat whose ink the signs do not cover is read again by
//   arrangements (1, 2/3 stacked, side by side, 2x2, tall + stack; see "P3 H"). r.quadrats = { cells, multi, examined,
//   changed, U }; every line's quads come from these quadrats. opts.quads: false switches G and H off.
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

// one XY cut along y (useY) or x (= segment.cut_parts): [parts in axis order, gap = smallest raw distance between parts]
function cutParts(idx, boxes, useY, tol) {
  const lo = useY ? (b) => b[1] : (b) => b[0];
  const hi = useY ? (b) => b[3] : (b) => b[2];
  const items = idx.slice().sort((i, j) => (lo(boxes[i]) + hi(boxes[i])) / 2.0 - (lo(boxes[j]) + hi(boxes[j])) / 2.0);
  const parts = []; let cur = [], end = null;
  for (const i of items) {
    const b = boxes[i], ext = hi(b) - lo(b), a0 = lo(b) + tol * ext;
    if (cur.length && a0 >= end) { parts.push(cur); cur = []; end = null; }
    cur.push(i);
    const e = hi(b) - tol * ext;
    end = end === null ? e : Math.max(end, e);
  }
  parts.push(cur);
  let gap = null;
  for (let k = 0; k + 1 < parts.length; k++) {
    const g = Math.min(...parts[k + 1].map((i) => lo(boxes[i]))) - Math.max(...parts[k].map((i) => hi(boxes[i])));
    gap = gap === null ? g : Math.min(gap, g);
  }
  return [parts, gap];
}
// P3 K3 (= segment.choose_cut): the preferred axis unless its parts overlap and the other axis cuts more cleanly
// (a tall sign beside a stack whose top sign ends a little below the tall sign's top is side by side: U6 beside r:r)
function chooseCut(idx, boxes, preferY, tol) {
  const first = !!preferY, [pa, ga] = cutParts(idx, boxes, first, tol);
  if (pa.length > 1 && ga >= 0) return [first, pa];
  const [pb, gb] = cutParts(idx, boxes, !first, tol);
  if (pa.length > 1) return pb.length > 1 && gb > ga ? [!first, pb] : [first, pa];
  return pb.length > 1 ? [!first, pb] : [null, null];
}
export function xyCut(idx, boxes, direction, preferY, tol, depth = 0) {
  if (idx.length <= 1 || depth > 8) return idx.slice();
  let [useY, parts] = chooseCut(idx, boxes, preferY, tol);
  if (parts) {
    if (!useY && direction === 'rtl') parts = parts.slice().reverse();
    const out = [];
    for (const q of parts) out.push(...xyCut(q, boxes, direction, !useY, tol, depth + 1));
    return out;
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
  // P3 K: for every 4 output channels of a conv layer, the range of input channels with a non-zero weight (the I+J reader's
  // conv5 is block-diagonal); the convolution skips the rest. Exact: a skipped term is 0 * pixel added to the sum.
  for (const L of layers) if (L.type === 'conv3x3' && L.W) {
    const C = L.in_ch, R = new Int32Array(2 * Math.ceil(L.out_ch / 4));
    for (let c0 = 0, g = 0; c0 < L.out_ch; c0 += 4, g++) {
      let lo = C, hi = 0;
      for (let o = c0; o < Math.min(L.out_ch, c0 + 4); o++) for (let ci = 0; ci < C; ci++) {
        const k = (o * C + ci) * 9; let nz = false; for (let t = 0; t < 9; t++) if (L.W[k + t] !== 0) { nz = true; break; }
        if (nz) { if (ci < lo) lo = ci; if (ci + 1 > hi) hi = ci + 1; }
      }
      R[2 * g] = lo < hi ? lo : 0; R[2 * g + 1] = lo < hi ? hi : 0;
    }
    L.ranges = R;
  }
  const prior = T.prior_logit || new Float32Array(meta.classes.length);
  return { meta, layers, prior, classes: meta.classes, threshold: meta.threshold != null ? meta.threshold : 0.56,
           norm: meta.normalisation, bytes: u8.byteLength };
}

export async function loadModel(src) {
  const { meta, bin } = await readModelFiles(src);
  CURRENT = parseModel(meta, bin);
  return CURRENT;
}

// P3 J: optional READER = a second recogniser (same conv1-3 as the model) that reads only the FINAL signs; the model
// loaded with loadModel() keeps making every segmentation decision. loadReader(null) removes it.
let READER = null;
export async function loadReader(src) {
  if (src === null) { READER = null; return null; }
  const { meta, bin } = await readModelFiles(src);
  READER = parseModel(meta, bin);
  return READER;
}
export function readerLoaded() { return READER; }

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

function conv3x3ReluPool(x, C, H, Wd, w, b, Cout, ranges = null) {
  // zero padding 1, stride 1, then ReLU and 2x2 max-pool stride 2. Four output channels per pass over the input
  // (each input pixel is loaded once for four channels); per output the arithmetic is the same as one channel at a time.
  const Hp = H + 2, Wp = Wd + 2, pad = new Float32Array(C * Hp * Wp);
  for (let c = 0; c < C; c++) for (let y = 0; y < H; y++) pad.set(x.subarray((c * H + y) * Wd, (c * H + y + 1) * Wd), (c * Hp + y + 1) * Wp + 1);
  const HW = H * Wd, Ho = H >> 1, Wo = Wd >> 1, out = new Float32Array(Cout * Ho * Wo);
  const conv = new Float32Array(4 * HW);
  for (let c0 = 0; c0 < Cout; c0 += 4) {
    const nb = Math.min(4, Cout - c0);
    for (let j = 0; j < 4; j++) conv.fill(j < nb ? b[c0 + j] : 0, j * HW, (j + 1) * HW);
    const ciLo = ranges ? ranges[c0 >> 1] : 0, ciHi = ranges ? ranges[(c0 >> 1) + 1] : C;      // P3 K: only the non-zero inputs
    for (let ci = ciLo; ci < ciHi; ci++) {
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
      x = conv3x3ReluPool(x, C, H, W, L.W, L.B, L.out_ch, L.ranges); C = L.out_ch; H >>= 1; W >>= 1;
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
function orderSigns(lines, axis, ang, from, to, p = PARAMS, quad = false) {
  if (from === to) return lines;
  if (quad) {                                    // P3 G: the quadrats stay; their order and the order inside each flips
    const out = lines.map((L) => {
      const runs = [];
      for (const s of L.signs) { if (!runs.length || runs[runs.length - 1][0].group !== s.group) runs.push([]); runs[runs.length - 1].push(s); }
      const G = axis === 'rows' ? runs.slice().reverse() : runs, signs = [];
      G.forEach((g, gi) => {
        const boxes = g.map((s) => [s.wbox[0], s.wbox[1], s.wbox[2], s.wbox[3], 0]);
        for (const i of xyCut(g.map((_, k) => k), boxes, to, true, p.cut_tol)) signs.push(Object.assign(g[i], { group: gi }));
      });
      return Object.assign({}, L, { signs });
    });
    return axis === 'columns' ? out.reverse() : out;
  }
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
  const pp = Object.assign({}, PARAMS, r.params || {});
  const lines = orderSigns(r.lines.map((L) => ({ signs: L.signs.map((s) => Object.assign({}, s)) })), r.layout, r.angle || 0,
                           r.direction, to, pp, r.segmenter === 'quad' || !!r.quadrats);
  for (const L of lines) L.quads = lineQuads(L.signs, to, pp.cut_tol);
  const out = Object.assign({}, r, { lines, direction: to, direction_source: source });
  if (source === 'auto' && r.direction_auto) out.direction_certainty = to === 'rtl' ? r.direction_auto.p_rtl : 1 - r.direction_auto.p_rtl;
  if (r.decode && r.decode.applied && to !== r.direction && SIGNLM) {      // P3 C2: the decoding depends on the order
    const d = decodeLines(out.lines, SIGNLM, r.decode.params, READER || CURRENT);   // P3 J
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
                                         det_merge: 6, pl_wide: 1.2, pl_min: 0.4, pl_gain: -9.0, pl_count: 1,   // P3 C1: det_merge, pl_*
                                         det_levels: [640, 320, 224, 160], det_pyr_gate: 30, det_lvl_floor: 0.3, det_cls_floor: 0.5, det_lvl_thr: 0.3 });  // P3 F
// P3 F (close-ups): det_levels = the detector pyramid (det_side values; the first pass is det_side), run when the first
// pass found <= det_pyr_gate signs: the detector was trained on signs of 12-140 px at its input, so large signs (a
// close-up) are only found whole at a smaller det_side. The smaller levels use the close-up detector (loadCloseDetector:
// the same net fine-tuned on close-ups, det_close.json/.bin; without it the main one) with the threshold det_lvl_thr.
// The level with the largest sum(conf - det_lvl_floor) is kept (the first pass wins ties); when the pyramid ran, the
// classical result is compared with it on sum(conf - det_cls_floor) instead of the usual selection score.
let DETECTOR_CLOSE = null;
export async function loadCloseDetector(src) {
  const { meta, bin } = await readModelFiles(src);
  const { T, bytes } = parseTensors(meta, bin);
  const ops = meta.ops.map((o) => Object.assign({}, o, o.op === 'conv' ? { W: T[o.name + '.weight'], B: T[o.name + '.bias'] } : {}));
  DETECTOR_CLOSE = { meta, ops, bytes, decode: meta.decode || {} };
  return DETECTOR_CLOSE;
}
// (= detect.DET_DEFAULTS, chosen on the dev sets only)

export async function loadDetector(src) {
  const { meta, bin } = await readModelFiles(src);
  const { T, bytes } = parseTensors(meta, bin);
  const ops = meta.ops.map((o) => Object.assign({}, o, o.op === 'conv' ? { W: T[o.name + '.weight'], B: T[o.name + '.bias'] } : {}));
  DETECTOR = { meta, ops, bytes, decode: meta.decode || {} };
  return DETECTOR;
}
// P3 K: an optional second main detector; the main detector's output maps become the mean of both (= detect.py 'a.json+b.json').
// loadDetector2(null) removes it. Call after loadDetector().
export async function loadDetector2(src) {
  if (!DETECTOR) throw new Error('call loadDetector() first');
  if (src === null) { delete DETECTOR.also; return null; }
  const { meta, bin } = await readModelFiles(src);
  const { T, bytes } = parseTensors(meta, bin);
  const ops = meta.ops.map((o) => Object.assign({}, o, o.op === 'conv' ? { W: T[o.name + '.weight'], B: T[o.name + '.bias'] } : {}));
  DETECTOR.also = { meta, ops, bytes, decode: meta.decode || {} };
  return DETECTOR.also;
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
// P3 K: the forward pass as a generator that yields after every layer (recogniseAsync shows real progress through it);
// detectorForward() runs it to the end in one go (the same computation).
export function detectorForward(work, side = DET_PARAMS.det_side, det = DETECTOR) {
  const it = forwardSteps(work, side, det);
  let r = it.next();
  while (!r.done) r = it.next();
  return r.value;
}
function* forwardSteps(work, side = DET_PARAMS.det_side, det = DETECTOR) {
  if (!det) throw new Error('call loadDetector() first');
  if (det.also) {                                          // P3 K: two detectors fused: mean of their output maps (float32 as numpy)
    const a = yield* forwardSteps(work, side, Object.assign({}, det, { also: null })), b = yield* forwardSteps(work, side, det.also), o = new Float32Array(a.out.length);
    for (let i = 0; i < o.length; i++) o[i] = Math.fround(Math.fround(a.out[i] + b.out[i]) / 2);
    return Object.assign({}, a, { out: o });
  }
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
    yield 1;
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
export function detectBoxes(work, p = Object.assign({}, PARAMS, DET_PARAMS), det = DETECTOR, fwd = null, fwd2 = null) {
  // P3 K: fwd = the forward pass already run (recogniseAsync); det.union = a second detector whose peaks join (one NMS)
  let keep;
  if (det.union) {
    const all = detPeaks(fwd || detectorForward(work, p.det_side, Object.assign({}, det, { union: null })), p, det)
      .concat(detPeaks(fwd2 || detectorForward(work, p.det_side, det.union), p, det.union));
    all.sort((a, b) => b[0] - a[0]); keep = [];
    for (const c of all) if (keep.every((k) => iouXYWH(c[1], k[1]) < p.det_nms)) keep.push(c);
  } else keep = detPeaks(fwd || detectorForward(work, p.det_side, det), p, det);
  const res = [];
  for (const [s, [bx, by, bwv, bhv]] of keep) {
    const x0 = Math.max(0, Math.floor(bx + 0.5)), y0 = Math.max(0, Math.floor(by + 0.5));
    const x1 = Math.min(work.w, Math.floor(bx + bwv + 0.5)), y1 = Math.min(work.h, Math.floor(by + bhv + 0.5));
    if (x1 - x0 >= p.det_min_px && y1 - y0 >= p.det_min_px) res.push([x0, y0, x1, y1, (x1 - x0) * (y1 - y0), s]);
  }
  return res;
}
// one detector's peaks [score, [x, y, w, h]] in work pixels after its own NMS (= det_common.decode)
function detPeaks(F, p, det) {
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
  return keep;
}
// P3 K: a second main detector whose peaks are united with the first one's (one NMS over both; = detect.py 'a.json|b.json')
export async function loadDetectorUnion(src) {
  if (!DETECTOR) throw new Error('call loadDetector() first');
  if (src === null) { delete DETECTOR.union; return null; }
  const { meta, bin } = await readModelFiles(src);
  const { T, bytes } = parseTensors(meta, bin);
  const ops = meta.ops.map((o) => Object.assign({}, o, o.op === 'conv' ? { W: T[o.name + '.weight'], B: T[o.name + '.bias'] } : {}));
  DETECTOR.union = { meta, ops, bytes, decode: meta.decode || {} };
  return DETECTOR.union;
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
  const boxes = detectBoxes(g0, p, opts.detector || DETECTOR, opts.fwd || null, opts.fwd2 || null);
  const detMs = now() - td + (opts.fwdMs || 0);
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
  res.selection_score = sc; res.n_det = boxes.length; res.work = g0; res.det_ms = detMs; res.det_side = p.det_side;
  return res;
}

const levelScore = (r, floor) => { let s = 0.0; for (const L of r.lines) for (const x of L) s += x.conf - floor; return s; };   // P3 F

// ------------------------------------------------------------------ P3 G: read the quadrats first, then the signs inside them
// (quads.py is the Python reference: same operations in the same order). At the close-up scale the boxes of all pyramid
// levels near the comfortable size are pooled: U (the quadrat size = line thickness) comes from the best level's confident
// signs, the signs are chosen ACROSS levels (most confident first, no two overlapping), the layout is the hypothesis with the
// fewest lines whose signs all cross one position (stabbing), lines are ordered, each line splits into quadrats (signs
// overlapping along the line, + short neighbours that fit one quadrat), inside a quadrat the reading order of segment()
// (top to bottom, then along the direction). Empty places inside a line long enough for a sign are read too (stage 3).
export const QUAD_PARAMS = Object.freeze({ q_on: 1, q_T: 56.0, q_R: 2.0, q_ufloor: 0.5, q_uq: 0.75, q_theta: 0.4, q_ov: 0.3, q_shrink: 0.15,
  q_lmerge: 0.5, q_qmax: 1.0, q_qgap: 0.08, q_qshort: 0.5, q_swap: 1, q_ucl: 100.0, q_lratio: 0.8,
  q_slot: 1, q_fu: 24.0, q_ek: 2, q_smin: 0.35, q_send: 1.2, q_sedge: 0.02, q_sprof: 0.3, q_stight: 0.15, q_ssplit: 1.3,
  q_fth: 0.5, q_fov: 0.15, q_gmean: 0.4, q_gshare: 0.28 });
// P3 J routing guard (= quads.py wall_guard): a first pass with more than det_gate signs that are mostly unreadable
// (mean top-1 < q_gmean and share of top-1 >= 0.5 below q_gshare) is texture on a close-up, not a wall -> pyramid.
const wallGuard = (r, p) => {
  if (!p.q_gmean) return false;
  const c = r.lines.flatMap((L) => L.map((s) => s.conf || 0));
  if (!c.length) return false;
  return c.reduce((a, v) => a + v, 0) / c.length < p.q_gmean && c.filter((v) => v >= 0.5).length / c.length < p.q_gshare;
};
const qArea = (b) => Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
const qConflict = (a, b, ov) => inter(a, b) >= ov * Math.min(qArea(a), qArea(b));

// levels: [[side, cands]] (cands {box: [x0,y0,x1,y1] work px, conf}) -> [U, side of the best level]
export function estimateU(levels, p = QUAD_PARAMS) {
  let best = null, bs = null;
  for (const lv of levels) {
    let s = 0; for (const c of lv[1]) s += c.conf - 0.3;
    if (best === null || s > bs) { best = lv; bs = s; }
  }
  const [side, cands] = best;
  let sel = cands.filter((c) => c.conf >= p.q_ufloor);
  if (!sel.length) sel = cands;
  if (!sel.length) return [0.0, side];
  const v = sel.map((c) => Math.max(c.box[2] - c.box[0], c.box[3] - c.box[1])).sort((a, b) => a - b);
  return [v[Math.floor(p.q_uq * (v.length - 1))], side];
}

export function selectCands(pool, p = QUAD_PARAMS) {
  const th = p.q_theta, ov = p.q_ov;
  const order = pool.map((_, i) => i).sort((i, j) => (pool[j].conf - pool[i].conf) || (qArea(pool[j].box) - qArea(pool[i].box)) || (i - j));
  let sel = [];
  for (const i of order) {
    if (pool[i].conf < th) continue;
    if (sel.every((j) => !qConflict(pool[i].box, pool[j].box, ov))) sel.push(i);
  }
  if (p.q_swap) {
    const snap = sel.slice().sort((i, j) => (pool[i].conf - pool[j].conf) || (i - j));
    for (const i of snap) {
      if (!sel.includes(i)) continue;
      const others = sel.filter((j) => j !== i);
      const S = order.filter((k) => !sel.includes(k) && pool[k].conf >= th && qConflict(pool[k].box, pool[i].box, ov) &&
                                    others.every((j) => !qConflict(pool[k].box, pool[j].box, ov)));
      const sub = [];
      for (const k of S) if (sub.every((m) => !qConflict(pool[k].box, pool[m].box, ov))) sub.push(k);
      let s = 0; for (const k of sub) s += pool[k].conf - th;
      if (sub.length >= 2 && s > pool[i].conf - th) sel = others.concat(sub);
    }
  }
  return sel.sort((a, b) => a - b);
}

function stabGroups(iv) {
  const order = iv.map((_, i) => i).sort((i, j) => (iv[i][1] - iv[j][1]) || (iv[i][0] - iv[j][0]) || (i - j));
  const pts = [];
  for (const i of order) if (!pts.length || iv[i][0] > pts[pts.length - 1]) pts.push(iv[i][1]);
  const groups = pts.map(() => []);
  for (let i = 0; i < iv.length; i++) {
    const [a, b] = iv[i], c = (a + b) / 2.0;
    let best = 0, bd = null;
    pts.forEach((t, k) => { if (a <= t && t <= b) { const d = Math.abs(t - c); if (bd === null || d < bd) { best = k; bd = d; } } });
    groups[best].push(i);
  }
  return groups.filter((g) => g.length);
}

function lineExtent(G, B, axis, ang) {
  let a0 = Infinity, a1 = -Infinity, u0 = Infinity, u1 = -Infinity;
  for (const i of G) { const r = rotCoords(B[i], axis, ang); a0 = Math.min(a0, r[2]); a1 = Math.max(a1, r[3]); u0 = Math.min(u0, r[0]); u1 = Math.max(u1, r[1]); }
  return [a0, a1, u0, u1];
}

function mergeLines(groups, B, axis, ang, p) {
  const gs = groups.map((G, i) => { const e = lineExtent(G, B, axis, ang); return [e[0] + e[1], i, G]; })
    .sort((a, b) => (a[0] - b[0]) || (a[1] - b[1])).map((x) => x[2]);
  const out = [];
  for (const G of gs) {
    if (out.length) {
      const [a0, a1] = lineExtent(out[out.length - 1], B, axis, ang), [b0, b1] = lineExtent(G, B, axis, ang);
      const ov = Math.min(a1, b1) - Math.max(a0, b0);
      if (ov >= p.q_lmerge * Math.min(a1 - a0, b1 - b0)) { out[out.length - 1] = out[out.length - 1].concat(G); continue; }
    }
    out.push(G.slice());
  }
  return out;
}

function quadratsOfLine(L, B, axis, ang, U, p, sp) {
  const Ls = L.slice().sort((a, b) => a - b), rc = new Map(Ls.map((i) => [i, rotCoords(B[i], axis, ang)]));
  const parent = new Map(Ls.map((i) => [i, i]));
  const find = (x) => { while (parent.get(x) !== x) { parent.set(x, parent.get(parent.get(x))); x = parent.get(x); } return x; };
  for (let ii = 0; ii < Ls.length; ii++) for (let jj = ii + 1; jj < Ls.length; jj++) {
    const i = Ls[ii], j = Ls[jj], a = rc.get(i), b = rc.get(j);
    const ov = Math.min(a[1], b[1]) - Math.max(a[0], b[0]);
    if (ov >= sp.group_overlap * Math.min(a[1] - a[0], b[1] - b[0])) {
      const x = find(i), y = find(j);
      if (x !== y) parent.set(Math.max(x, y), Math.min(x, y));
    }
  }
  const gm = new Map();
  for (const i of Ls) { const r = find(i); if (!gm.has(r)) gm.set(r, []); gm.get(r).push(i); }
  const start = (g) => { let m = Infinity; for (const i of g) m = Math.min(m, rc.get(i)[0]); return m; };
  const G = [...gm.values()].map((g) => [start(g), Math.min(...g), g]).sort((a, b) => (a[0] - b[0]) || (a[1] - b[1])).map((x) => x[2]);
  const ext = (g) => { let a = Infinity, b = -Infinity; for (const i of g) { a = Math.min(a, rc.get(i)[0]); b = Math.max(b, rc.get(i)[1]); } return [a, b]; };
  const out = [];
  for (const g of G) {
    if (out.length && U > 0) {
      const [a0, a1] = ext(out[out.length - 1]), [b0, b1] = ext(g);
      if (a1 - a0 <= p.q_qshort * U && b1 - b0 <= p.q_qshort * U && Math.max(a1, b1) - Math.min(a0, b0) <= p.q_qmax * U && b0 - a1 <= p.q_qgap * U) {
        out[out.length - 1] = out[out.length - 1].concat(g); continue;
      }
    }
    out.push(g.slice());
  }
  return out;
}

// = quads.xy_struct: the cut structure of xyCut (leaf = box index, node = ['v'|'h', children])
export function xyStruct(idx, boxes, direction, preferY, tol, depth = 0) {
  if (idx.length === 1) return idx[0];
  if (depth > 8) return ['v', idx.slice()];
  let [useY, parts] = chooseCut(idx, boxes, preferY, tol);
  if (parts) {
    if (!useY && direction === 'rtl') parts = parts.slice().reverse();
    const kind = useY ? 'v' : 'h', ch = [];
    for (const q of parts) {
      const t = xyStruct(q, boxes, direction, !useY, tol, depth + 1);
      if (typeof t !== 'number' && t[0] === kind) ch.push(...t[1]); else ch.push(t);
    }
    return [kind, ch];
  }
  const sgn = direction === 'rtl' ? -1 : 1;
  return ['v', idx.slice().sort((i, j) => cmpTuple([(boxes[i][1] + boxes[i][3]) / 2.0, sgn * (boxes[i][0] + boxes[i][2]) / 2.0],
                                                   [(boxes[j][1] + boxes[j][3]) / 2.0, sgn * (boxes[j][0] + boxes[j][2]) / 2.0]))];
}
const treePositions = (t, pos) => typeof t === 'number' ? pos.get(t) : { t: t[0], k: t[1].map((c) => treePositions(c, pos)) };

// the quadrats of a recognised line: runs of signs with the same group (>= 2 signs) -> [{ at, n, tree }] with the app's
// tree format ({ t: 'v' | 'h', k: [position | subtree] }, positions in reading order)  (= quads.line_quads)
export function lineQuads(signs, direction, tol = PARAMS.cut_tol) {
  const out = [];
  let i = 0;
  while (i < signs.length) {
    let j = i + 1;
    while (j < signs.length && signs[j].group === signs[i].group) j++;
    if (j - i >= 2) {
      const boxes = signs.slice(i, j).map((s) => s.wbox), idx = boxes.map((_, k) => k);
      const t = xyStruct(idx, boxes, direction, true, tol);
      out.push({ at: i, n: j - i, tree: treePositions(t, new Map(idx.map((k) => [k, k]))) });
    }
    i = j;
  }
  return out;
}

function assembleQuads(cands, direction, layout, U, p, sp, f = 1) {
  const B = cands.map((c) => [c.box[0], c.box[1], c.box[2], c.box[3], qArea(c.box)]);
  const res = { layout: 'rows', angle: 0.0, lines: [], U, factor: f, scale: 0.0, polarity: 'detector' };
  if (!B.length) return res;
  const sides = B.map((b) => Math.max(b[2] - b[0], b[3] - b[1])).sort((a, b) => a - b), n = sides.length;
  const scale = n % 2 ? sides[n >> 1] : (sides[n / 2 - 1] + sides[n / 2]) / 2.0;
  res.scale = scale;
  const hyp = {};
  for (const ax of ['rows', 'columns']) {
    const ang = skewAngle(B, ax, scale, sp);
    const iv = B.map((b) => { const r = rotCoords(b, ax, ang), d = (r[3] - r[2]) * p.q_shrink; return [r[2] + d, r[3] - d]; });
    const G = mergeLines(stabGroups(iv), B, ax, ang, p);
    hyp[ax] = [G.length, elongation(B, G, ax, ang), ang, G];
  }
  let axis;
  if (layout === 'rows' || layout === 'columns') axis = layout;
  else {
    const r = hyp.rows, c = hyp.columns;
    if (Math.min(r[0], c[0]) >= p.q_lratio * Math.max(r[0], c[0])) axis = r[1] >= c[1] ? 'rows' : 'columns';   // similar counts: elongation
    else axis = r[0] < c[0] ? 'rows' : 'columns';                                                                // clearly fewer lines
  }
  const ang = hyp[axis][2];
  let G = hyp[axis][3];
  res.layout = axis; res.angle = ang; res.hyp = { rows: hyp.rows.slice(0, 2), columns: hyp.columns.slice(0, 2) };
  const th = G.map((L) => { const e = lineExtent(L, B, axis, ang); return e[1] - e[0]; }).sort((a, b) => a - b);
  res.U_lines = th[(th.length - 1) >> 1];
  const Uq = U > 0 ? U : res.U_lines;
  G = G.map((L, i) => { const e = lineExtent(L, B, axis, ang); return [e[0] + e[1], i, L]; }).sort((a, b) => (a[0] - b[0]) || (a[1] - b[1])).map((x) => x[2]);
  if (axis === 'columns' && direction === 'rtl') G = G.reverse();
  for (const L of G) {
    let Q = quadratsOfLine(L, B, axis, ang, Uq, p, sp);
    if (axis === 'rows' && direction === 'rtl') Q = Q.reverse();
    const line = [];
    Q.forEach((g, gi) => {
      for (const i of xyCut(g, B, direction, true, sp.cut_tol)) {
        const b = B[i];
        line.push(Object.assign({}, cands[i], { box: [b[0] * f, b[1] * f, (b[2] - b[0]) * f, (b[3] - b[1]) * f], wbox: b.slice(0, 4), group: gi }));
      }
    });
    res.lines.push(line);
  }
  return res;
}

// stage 3 helpers (= quads.downscale_by, edge_energy, box_smooth, tight)
function downscaleBy(g, f) {
  const { w: W, h: H, d } = g;
  if (f <= 1) return { w: W, h: H, d: d.slice() };
  const h = Math.floor(H / f), w = Math.floor(W / f), out = new Uint8Array(w * h), ff = f * f, half = Math.floor(ff / 2);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let s = 0;
    for (let dy = 0; dy < f; dy++) { const row = (y * f + dy) * W + x * f; for (let dx = 0; dx < f; dx++) s += d[row + dx]; }
    out[y * w + x] = Math.floor((s + half) / ff);
  }
  return { w, h, d: out };
}
function lowerMedian(v) {                       // lower median of non-negative ints (sorted[(n-1)//2])
  const c = new Map(); let mx = 0;
  for (const x of v) { c.set(x, (c.get(x) || 0) + 1); if (x > mx) mx = x; }
  const r = Math.floor((v.length - 1) / 2) + 1;
  let acc = 0;
  for (let x = 0; x <= mx; x++) { acc += c.get(x) || 0; if (acc >= r) return x; }
  return mx;
}
function edgeEnergy(a, k) {
  const b = box3(a), { w: W, h: H } = b, gx = new Int32Array(W * H), gy = new Int32Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 1; x < W - 1; x++) gx[y * W + x] = Math.abs(b.d[y * W + x + 1] - b.d[y * W + x - 1]);
  for (let y = 1; y < H - 1; y++) for (let x = 0; x < W; x++) gy[y * W + x] = Math.abs(b.d[(y + 1) * W + x] - b.d[(y - 1) * W + x]);
  const mx = k * lowerMedian(gx), my = k * lowerMedian(gy), e = new Float64Array(W * H);
  for (let i = 0; i < W * H; i++) e[i] = Math.max(0, gx[i] - mx) + Math.max(0, gy[i] - my);
  return { w: W, h: H, e };
}
function boxSmooth(v, r) {
  const n = v.length, c = new Float64Array(n + 1), out = new Float64Array(n);
  for (let i = 0; i < n; i++) c[i + 1] = c[i] + v[i];
  for (let i = 0; i < n; i++) { const lo = Math.max(0, i - r), hi = Math.min(n, i + r + 1); out[i] = (c[hi] - c[lo]) / (hi - lo); }
  return out;
}
function tightRange(prof, frac) {
  if (!prof.length) return null;
  const s = boxSmooth(prof, 1);
  let mx = -Infinity; for (const x of s) if (x > mx) mx = x;
  if (mx <= 0) return null;
  let a = -1, b = -1;
  for (let i = 0; i < s.length; i++) if (s[i] >= frac * mx) { if (a < 0) a = i; b = i; }
  return [a, b + 1];
}

// stage 3: empty places along a line (= quads.fill_lines). top(box) -> [class, prob]
function fillLines(g0, res, U, p, top) {
  const fa = Math.max(1, Math.floor(U / p.q_fu));
  const E = edgeEnergy(downscaleBy(g0, fa), p.q_ek), Wa = E.w, Ha = E.h, H = g0.h, W = g0.w;
  const axis = res.layout, ang = res.angle;
  const chosen = res.lines.flatMap((L) => L.map((s) => s.wbox));
  const out = [];
  const colSum = (X0, X1, Y0, Y1) => { const v = new Float64Array(X1 - X0); for (let y = Y0; y < Y1; y++) for (let x = X0; x < X1; x++) v[x - X0] += E.e[y * Wa + x]; return v; };
  const rowSum = (X0, X1, Y0, Y1) => { const v = new Float64Array(Y1 - Y0); for (let y = Y0; y < Y1; y++) { let s = 0; for (let x = X0; x < X1; x++) s += E.e[y * Wa + x]; v[y - Y0] = s; } return v; };
  for (const L of res.lines) {
    if (!L.length) continue;
    const rc = L.map((s) => rotCoords([s.wbox[0], s.wbox[1], s.wbox[2], s.wbox[3], 0], axis, ang));
    let a0 = Infinity, a1 = -Infinity; for (const r of rc) { a0 = Math.min(a0, r[2]); a1 = Math.max(a1, r[3]); }
    const iv = rc.map((r) => [r[0], r[1]]).sort(cmpTuple);
    const gaps = [[iv[0][0] - p.q_send * U, iv[0][0]]];
    let end = iv[0][1];
    for (const [s0, s1] of iv.slice(1)) { if (s0 > end) gaps.push([end, s0]); end = Math.max(end, s1); }
    gaps.push([end, end + p.q_send * U]);
    for (const [g0_, g1_] of gaps) {
      if (g1_ - g0_ < p.q_smin * U) continue;
      let x0, x1, y0, y1;
      if (axis === 'rows') { x0 = g0_; x1 = g1_; y0 = a0; y1 = a1; } else { x0 = a0; x1 = a1; y0 = g0_; y1 = g1_; }
      const m = p.q_sedge * Math.max(H, W);
      if (x0 < m || y0 < m || x1 > W - m || y1 > H - m) {
        if (g0_ < 0 || g1_ > (axis === 'rows' ? W : H)) continue;
        x0 = Math.max(0, x0); y0 = Math.max(0, y0); x1 = Math.min(W, x1); y1 = Math.min(H, y1);
      }
      const X0 = Math.floor(x0 / fa), X1 = Math.min(Wa, Math.ceil(x1 / fa)), Y0 = Math.floor(y0 / fa), Y1 = Math.min(Ha, Math.ceil(y1 / fa));
      if (X1 - X0 < 3 || Y1 - Y0 < 3) continue;
      const tx = tightRange(colSum(X0, X1, Y0, Y1), p.q_sprof), ty = tightRange(rowSum(X0, X1, Y0, Y1), p.q_sprof);
      if (!tx || !ty) continue;
      const bx = [(X0 + tx[0]) * fa, (Y0 + ty[0]) * fa, Math.min(W, (X0 + tx[1]) * fa), Math.min(H, (Y0 + ty[1]) * fa)];
      if (Math.max(bx[2] - bx[0], bx[3] - bx[1]) < p.q_stight * U) continue;
      const opts = [[bx]];
      const along = axis === 'rows' ? bx[2] - bx[0] : bx[3] - bx[1];
      if (along > p.q_ssplit * U) {
        const sx0 = Math.floor(bx[0] / fa), sx1 = Math.ceil(bx[2] / fa), sy0 = Math.floor(bx[1] / fa), sy1 = Math.ceil(bx[3] / fa);
        const prof = axis === 'rows' ? colSum(sx0, Math.min(Wa, sx1), sy0, Math.min(Ha, sy1)) : rowSum(sx0, Math.min(Wa, sx1), sy0, Math.min(Ha, sy1));
        const nn = prof.length;
        if (nn >= 6) {
          let k = Math.floor(nn / 4);
          for (let t = Math.floor(nn / 4); t < nn - Math.floor(nn / 4); t++) if (prof[t] < prof[k]) k = t;
          const cut = (axis === 'rows' ? bx[0] : bx[1]) + k * fa;
          opts.push(axis === 'rows' ? [[bx[0], bx[1], cut, bx[3]], [cut, bx[1], bx[2], bx[3]]] : [[bx[0], bx[1], bx[2], cut], [bx[0], cut, bx[2], bx[3]]]);
        }
      }
      let best = null, bs = 0.0;
      for (const o of opts) {
        if (o.some((b) => chosen.some((c) => qConflict(b, c, p.q_fov)))) continue;
        const rd = o.map((b) => top(b));
        if (rd.every((r) => r[1] >= p.q_fth)) {
          let sc = 0; for (const r of rd) sc += r[1] - p.q_fth;
          if (best === null || sc > bs) { best = o.map((b, t) => [b, rd[t][1]]); bs = sc; }
        }
      }
      if (best) for (const [b, pr] of best) out.push({ box: b, conf: pr, level: -2 });
    }
  }
  return out;
}

// the quadrat path on the pyramid runs ([[side, segmentDet result]], the first = the 640 pass); null = not the close-up
// scale (U at the 640 input < q_ucl): the caller keeps the F rule.
function quadPath(runs, direction, layout, p, top) {
  const b = runs[0][1], g0 = b.work, M = Math.max(g0.w, g0.h);
  const levels = runs.map(([s, r]) => [s, r.lines.flatMap((L) => L.map((x) => ({ box: x.wbox.slice(0, 4), conf: x.conf })))]);
  const [U, bside] = estimateU(levels, p);
  const U640 = U * Math.min(1.0, 640.0 / M);
  if (U640 < p.q_ucl) return { seg: null, U, U640 };
  const pool = [];
  for (const [s, cs] of levels) {
    const Ui = U * Math.min(1.0, s / M);
    if (Math.abs(Math.log(Math.max(Ui, 1e-6) / p.q_T)) <= Math.log(p.q_R) || s === bside) for (const c of cs) pool.push(Object.assign({}, c, { level: s }));
  }
  const chosen = selectCands(pool, p).map((i) => pool[i]);
  let nfill = 0;
  let res = assembleQuads(chosen, direction, layout, U, p, p, b.factor);
  if (p.q_slot && top && U > 0 && !(p.q_h && p.h_noslot)) {
    const add = fillLines(g0, res, U, p, top);
    if (add.length) { chosen.push(...add); nfill = add.length; res = assembleQuads(chosen, direction, layout, U, p, p, b.factor); }
  }
  res.work = g0; res.segmenter = 'quad';
  let sc = 0.0; for (const L of res.lines) for (const s of L) sc += s.conf - p.sel_floor;
  res.selection_score = sc;
  res.quad = { U: U * b.factor, U640, U_lines: (res.U_lines || 0) * b.factor, levels: [...new Set(pool.map((c) => c.level))].sort((x, y) => x - y), filled: nfill };
  return { seg: res, U, U640 };
}

// ------------------------------------------------------------------ P3 H: quadrats on every photo, arrangement inside each
// (quads.py "P3 H" is the Python reference: same operations in the same order.) After the signs are found (wall path,
// normal path or the G quadrat path) every line is cut into quadrats again: (1) fragments go (a box below h_tiny U, or far
// below the natural size of the sign it reads as and inside another sign or read unsurely); (2) signs overlapping along the
// line are merged with neighbours that fit one square; every quadrat's cell grows over the contiguous ink next to it, ink
// left between quadrats is an empty cell; (3) in a cell the arrangements 1, 2/3 stacked, 2/3 side by side, 2x2, tall + a
// stack of 2/3 (either side), 2 side by side over/under 1 are cut at the valleys of the cell's ink profiles, every slot
// tightened to its ink and read; the detector's boxes are one more candidate; score = sum over slots [log(p / theta) +
// aspect and size of the read class's natural glyph vs the slot] + LM association of neighbours + ink coverage.
// GLYPH_WH = natural [w, h] of every class (1/100 of the height of A1; js/glyphm.json, built by h_work/make_glyphm.py).
const GLYPH_WH_B64 = 'TGRHZEpkRmRRZEpkSmRPZE9kYWRKZD9kW2REZExkT2RkWVRkZTlCZEpkUGQ+ZEhkTWRDZEtkSWQ9ZFBkSmRFZEtkP2Q3ZDZkPWRBZENkP2RPZFJkYWRjZHRkP2RCZD9kVmRWZDhkOGRNZDxkQGRPZFRkRGRDZEpkSmRZZCBkZSJlQk1kU2RQZFtkUmUxZDFkVGRfZDFkTmQ3ZE1kRGRVZDtkRGRCZEFkRGRJZEdkVGRQZEZkOGQ6ZDJkM2Q6ZDpkS2RUZDpkOmRAZDVkMWQ9ZFRkN2Q5ZEFkQWRFZDlkOmQmZC9kO2Q5ZDtkQGQ6ODk+YzhkHmIgYSliJmQiYydkO2RUNS8nJ2QTXixlLRhWYk80QkZEQkJkHWQrZDpkEWMpTUA3Jzc0SkpJY0ReOmQ6ZD1iZGNkW2RWZBdkIGQiZCBkIWUtZCRlIGUqZDBlSGUlZDplHWUeZDVVMBZkNGRSZGRkKGQoZDZkNmQeZIBkZRhkJWRkZSVGOkdJRzpCXWRdOmRkZElkZVZlVVRRZCktWGQ8Hh4cQEA+QEBkPmRAfTx9PWRkZFFkRWRJZWRkR2RkZF5kV1RkZFljQWRIZEhkPlZkY1hlXmRkZGRrX1xjZGJXZGRkY2RvPmQ7ZTFlMGQ8ZVQ7ZGNkXmRgY1xkYWRjT2NLZT9pNUlkZGRkWDo5TDNjWGJOZEJQZFdkZFBkZDdLJmQvYydkWEFYL2FkYWRkO2RkZR5kN2RKRzlJO008ZSVlJSBkY0IsZClkWWRkSDBkJGRkGmM6MTMiZCRjZD5dMFA4Sy9SUGRQTEFjHzw8Yy83ZDVkZR1lIWQdZR5lIGUgZWQnLWQpIGQnLy46PmReZGxjZGRkZGNkY2RkZFBkZWNSZExkT2RlZGQ0Y1lkP2RaX2RfZG1kX2RzZGRkZGRiZFtkZFNlW2RDZF1fZGRkZUtpPGRkhWRkZGRkZGRNZGRSZUZkO2RKZDtkZGVkZGRkZG1JQ2RTZFlkZGRkZFBkVGR0R2Q9ZExlVGQ6XGRlWDE4ZEY6SSU3ZR0eZC1kY2QpOGRkZEBlIGVSZDNkH0wwZFBMZGQjZC5kZGRkZGRkZEJkPGRkKmM+ZDRlL2QpbzJkKDBGYFteKUJkY2RkZEtkazdkIi8vLzBCZBpbKGRXZGRkT09kF2JkJGQqZC1kJGRjNWRVTE1VMGQlIGRFZGRkg2SSZD9kUmRSZDVkJmNVZEJkQmQ6ZDlkGWQ3ZCBkUGRjZFtkM2RkZDVkNWQgZEFkQmRjZEJkOWQSZBlkMFAZSyBkZRYfZEA/LGQlMjw3QjJjQjc4GWQhZEIURUJ0ZCdkZR1bZFZkZUYvL2Q5N2QtQjY2OztkImQiV1lPTFRVZRllD2UXZRhlGGU9ZBhCIVQePCBiJWUrZSFSMVI7UCkuMGQqZDA1ThkZWBhJTU1LYhBjZGQaZRplGmUaZRplQTo3OjdkZGRVZGRkZGNkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGNWZGNkZGRkZGRkZGRkZGRkZGRjY2RkZGRkZGRkZGRkZGRkYGRkZGRkZGRkZFxkY2NfZGFbZGRkZGRkZGRhZF9SKlRkVGRkZD0vRzpLPSdkKG0pbSltKW0pbSltKGRkZGRkZWRlZGVkUmQeZGRkZGRXUmVkZUNjT1dkZGRYZD9kPWQ6ZGRPZWQ/ZFlZIGRMZE1kZFsYZGQWFmQdZH9kZBhDZFRkZF1kEGRHJ2Q8azxrKmsqa2lLTVA9GUpEZDZjNmQeNmRCPjJEUB43Njg6Nzc1NDU0ZFJvOm86ZVVkQ2RFZDVjZCBkZGQUZGRkZWQUZC9kY0QpMFI4ZSdkQTJkZGRkZE9kZGRkMmRCZSplImU6Lk4qZDBkTWM5ZChkZEVPZBxkPWQlZCRkMWQ7ZC1kQWQtZGUPZBxkKFNkZGQuZDVkM2RCZGRkOGRPZGRkT2RkZDxkZWRkZERkR0ZANWQxZGRkZGRkVmRlZEljOWRkQlJjWFEmRjk5ZENkQmQVZD9GVWUdZBlPVGVkFmRkZGRkZBopZDVkTmRPZDtkIWQaZBhkHmQdZCJkEmQtZFRZPz9kGWM+GWQ6ZCdkUGRbZGQdM2QYZB9kZBBjEmQQYxVPTkdPJ2QZZBRkZCMkZGRFG2QdZBNkZC4fZB9kTUI1Y2REZCk1NWRPZC9kMWRQZGRlNGRkHWQTZDRkX2RkRWRkZGRkRT5jKmQqZGUxYzBcQ1xTUGRQZGRLZSdkMWRHWERkVWM9YxtjJiJAGmQYZCJkLWQgZBZkJWQlZEdkLD5kMylkHGQhZCVkW2QVZFREZGQiZDRkO2QeZB47Qj5kPnM5WGVYZGRkZWR6ZIQ0ZDtOVmUwW2Q1ZCNGI0YoZC9kI0o7P2Q4MjgzZTVkNWRkPGQ+Y0lkG2QbZEJkKypkImRLUSAgIUZMREtGZDxkPoQ9hDxkZFAfZByEHZAbY2RkQGRCYDoUZBpkZBhkExtkZGQvZGRkYyBjGWMdZB1lIDFALzw2STlGMWRDRWRCImQrZCAjRiM3ZD9kZCljHmNkZFlELEIxZCAqPy49NyYsLTYvOjA2MiBkZGQyZERkXWRWZGlkZGQrZC1kNDEhNTExKy9wJjZkMhsmPiY+ZRpkJGUaZCE7Mi8wIjAyZDRkZBobZGQYUWRQZGQjID4xYzZkBiljKEEoYxhBVEFUBWQoZBhjLyIyKDY9GiJkNyA1WR4uM1g4PWQvKjEzICkEKyArOCtYKztbO11YW1hdNGRkKysEKyArOytYWztcO1tYXFhkNTM0Oz5UPk40W1BkMmMpZClkU2QXZRViGmQSZA9jGGMiZBo8HTw7RlBJVB5kJ2RlZGRCZC9RZCdkLGQYZBRkGWQtZBlkNWQ=';
let GLYPH_WH = null;
function glyphWH() {
  if (!GLYPH_WH) {
    const s = GLYPH_WH_B64, tab = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/', out = [];
    let buf = 0, bits = 0;
    for (const ch of s) { if (ch === '=') break; buf = (buf << 6) | tab.indexOf(ch); bits += 6; if (bits >= 8) { bits -= 8; out.push((buf >> bits) & 255); } }
    GLYPH_WH = [];
    for (let i = 0; i + 1 < out.length; i += 2) GLYPH_WH.push([out[i] / 100.0, out[i + 1] / 100.0]);
  }
  return GLYPH_WH;
}
export const H_PARAMS = Object.freeze({ q_h: 1, h_noslot: 1, h_fu: 24.0, h_ek: 2, h_eq: 0.95, h_et: 0.25, h_en: 2, h_cmin: 1.0,
  h_tiny: 0.15, h_frs: 0.5, h_fin: 0.3, h_fconf: 0.7, h_qshort: 0.5, h_qlong: 1.0, h_qmax: 1.1, h_qgap: 0.12,
  h_egap: 0.08, h_pfrac: 0.05, h_xmax: 1.25, h_aext: 0.5, h_send: 1.2, h_smin: 0.3,
  h_cov: 0.8, h_low: 0.0, h_maxq: 60, h_all: 0, h_cutlo: 0.2, h_cuthi: 0.8, h_cutmax: 0.35, h_sprof: 0.3, h_slotmin: 0.12,
  h_theta: 0.3, h_wa: 0.5, h_sa: 0.4, h_wz: 0.5, h_sz: 0.5, h_wlm: 0.3, h_wcov: 2.0, h_margin: 0.0, h_fth: 0.9, h_rth: 0.9, h_same: 1, h_emin: 0.0 });

function hInkmap(g0, U, p) {
  const fa = Math.max(1, Math.floor(U / p.h_fu));
  const E = edgeEnergy(downscaleBy(g0, fa), p.h_ek), W = E.w, H = E.h, n = W * H;
  const v = Float64Array.from(E.e).sort();
  const hi = v[Math.floor(p.h_eq * (n - 1))], mid = v[Math.floor(0.75 * (n - 1))];
  const tau = Math.max(1, Math.floor(p.h_et * hi), Math.floor(p.h_en * mid));
  const m = new Uint8Array(n);
  for (let i = 0; i < n; i++) m[i] = E.e[i] >= tau ? 1 : 0;
  if (p.h_cmin > 0) {
    const [comps, lab] = components(m, W, H), lim = p.h_cmin * U / fa;
    for (let i = 0; i < n; i++) if (m[i] && comps[lab[i] - 1][4] < lim) m[i] = 0;
  }
  return { fa, m, W, H };
}
function hMrect(ink, b) {
  const fa = ink.fa;
  const X0 = Math.max(0, Math.min(ink.W, Math.floor(b[0] / fa))), Y0 = Math.max(0, Math.min(ink.H, Math.floor(b[1] / fa)));
  const X1 = Math.max(X0, Math.min(ink.W, Math.ceil(b[2] / fa))), Y1 = Math.max(Y0, Math.min(ink.H, Math.ceil(b[3] / fa)));
  return [X0, Y0, X1, Y1];
}
const hWrect = (ink, r, Wg, Hg) => [r[0] * ink.fa, r[1] * ink.fa, Math.min(Wg, r[2] * ink.fa), Math.min(Hg, r[3] * ink.fa)];
function hUnrot(al, ac, axis, ang) {
  const t = ang * (Math.PI / 180.0), c = Math.cos(t), s = Math.sin(t);
  if (axis === 'rows') return [al * c - ac * s, al * s + ac * c];
  return [ac * c - al * s, ac * s + al * c];
}
function hProf(ink, r, alongX) {                 // mask sums per column (alongX) or per row of map rect r
  const out = new Float64Array(alongX ? r[2] - r[0] : r[3] - r[1]);
  for (let y = r[1]; y < r[3]; y++) for (let x = r[0]; x < r[2]; x++) if (ink.m[y * ink.W + x]) out[alongX ? x - r[0] : y - r[1]] += 1;
  return out;
}
function hFragments(L, U, p, gwh, top) {
  const out = [];
  L.forEach((s, i) => {
    const b = s.wbox, l = Math.max(b[2] - b[0], b[3] - b[1]);
    if (l < p.h_tiny * U) { out.push(i); return; }
    const [c, pr] = top(b), nat = Math.max(gwh[c][0], gwh[c][1]);
    if (l < p.h_frs * nat * U) {
      const a = qArea(b);
      let ins = false;
      for (let j = 0; j < L.length; j++) if (j !== i && inter(b, L[j].wbox) >= p.h_fin * a) { ins = true; break; }
      if (ins || pr < p.h_fconf) out.push(i);
    }
  });
  return out;
}
function hQuadrats(L, axis, ang, U, p, sp) {
  const rc = L.map((s) => rotCoords(s.wbox.slice(0, 4).concat([0]), axis, ang));
  const parent = L.map((_, i) => i);
  const find = (x) => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
  for (let i = 0; i < L.length; i++) for (let j = i + 1; j < L.length; j++) {
    const ov = Math.min(rc[i][1], rc[j][1]) - Math.max(rc[i][0], rc[j][0]);
    if (ov >= sp.group_overlap * Math.min(rc[i][1] - rc[i][0], rc[j][1] - rc[j][0])) {
      const a = find(i), b = find(j);
      if (a !== b) parent[Math.max(a, b)] = Math.min(a, b);
    }
  }
  const gm = new Map();
  for (let i = 0; i < L.length; i++) { const r = find(i); if (!gm.has(r)) gm.set(r, []); gm.get(r).push(i); }
  const A = [...gm.values()].map((g) => [Math.min(...g.map((i) => rc[i][0])), Math.min(...g), g]).sort((a, b) => (a[0] - b[0]) || (a[1] - b[1])).map((x) => x[2]);
  const ext = (g) => { let a = Infinity, b = -Infinity; for (const i of g) { a = Math.min(a, rc[i][0]); b = Math.max(b, rc[i][1]); } return [a, b]; };
  const out = [];
  for (const g of A) {
    if (out.length && U > 0) {
      const [a0, a1] = ext(out[out.length - 1]), [b0, b1] = ext(g), la = a1 - a0, lb = b1 - b0;
      if (Math.max(a1, b1) - Math.min(a0, b0) <= p.h_qmax * U && b0 - a1 <= p.h_qgap * U &&
          Math.min(la, lb) <= p.h_qshort * U && Math.max(la, lb) <= p.h_qlong * U) { out[out.length - 1] = out[out.length - 1].concat(g); continue; }
    }
    out.push(g.slice());
  }
  return [out, rc];
}
function hRuns(P, thr, gapb) {
  const out = [];
  for (let j = 0; j < P.length; j++) if (P[j] >= thr) {
    if (out.length && j - out[out.length - 1][1] <= gapb) out[out.length - 1][1] = j + 1; else out.push([j, j + 1]);
  }
  return out;
}
function hWalk(P, thr, gapb, maxb) {
  let last = -1, empty = 0;
  for (let j = 0; j < Math.min(P.length, maxb); j++) {
    if (P[j] >= thr) { last = j; empty = 0; } else { empty++; if (empty > gapb) break; }
  }
  return last + 1;
}
const hRev = (P) => Float64Array.from(P).reverse();
function hCells(L, Q, rc, axis, ang, U, ink, p, lim, Wg, Hg) {
  const k = axis === 'rows' ? 0 : 1, fa = ink.fa;
  let a0 = Infinity, a1 = -Infinity; for (const r of rc) { a0 = Math.min(a0, r[2]); a1 = Math.max(a1, r[3]); }
  const [loLim, hiLim] = lim, nb = k === 0 ? Wg : Hg, j1 = 1 - k;
  const band = (al) => {
    const p0 = hUnrot(al, a0, axis, ang), p1 = hUnrot(al, a1, axis, ang);
    const q0 = hUnrot(al, Math.max(loLim, a0 - p.h_aext * U), axis, ang), q1 = hUnrot(al, Math.min(hiLim, a1 + p.h_aext * U), axis, ang);
    return [Math.min(p0[j1], p1[j1]), Math.max(p0[j1], p1[j1]), Math.min(q0[j1], q1[j1]), Math.max(q0[j1], q1[j1])];
  };
  const box = (l0, l1, c0, c1) => (k === 0 ? [l0, c0, l1, c1] : [c0, l0, c1, l1]);
  const ext = Q.map((g) => [Math.min(...g.map((i) => L[i].wbox[k])), Math.max(...g.map((i) => L[i].wbox[k + 2]))]);
  const cen = Q.map((g) => { let s = 0; for (const i of g) s += rc[i][0] + rc[i][1]; return s / (2.0 * g.length); });
  const gapb = Math.max(0, Math.floor(p.h_egap * U / fa));
  const newx = ext.map((e) => e.slice()), empties = [];
  for (let gi = 0; gi <= Q.length; gi++) {
    const g0 = gi > 0 ? ext[gi - 1][1] : Math.max(0.0, ext[0][0] - p.h_send * U);
    const g1 = gi < Q.length ? ext[gi][0] : Math.min(nb, ext[ext.length - 1][1] + p.h_send * U);
    if (g1 - g0 < fa) continue;
    let al = gi > 0 ? cen[gi - 1] : cen[0];
    if (gi > 0 && gi < Q.length) al = (cen[gi - 1] + cen[gi]) / 2.0;
    const [c0, c1] = band(al);
    const r = hMrect(ink, box(g0, g1, c0, c1)), Pg = hProf(ink, r, k === 0), n = Pg.length;
    if (n === 0) continue;
    const thr = Math.max(1.0, p.h_pfrac * (k === 0 ? r[3] - r[1] : r[2] - r[0]));
    let el = 0, er = 0;
    if (gi > 0) el = hWalk(Pg, thr, gapb, Math.max(0, Math.floor((p.h_xmax * U - (ext[gi - 1][1] - ext[gi - 1][0])) / fa)));
    if (gi < Q.length) er = hWalk(hRev(Pg), thr, gapb, Math.max(0, Math.floor((p.h_xmax * U - (ext[gi][1] - ext[gi][0])) / fa)));
    if (el + er > n) {
      const s0 = n - er, s1 = el;
      let j = s0;
      if (s1 > s0) { let bv = Infinity; for (let t = s0; t < s1; t++) if (Pg[t] < bv) { bv = Pg[t]; j = t; } }
      el = j; er = n - j;
    }
    const start = k === 0 ? r[0] : r[1];
    if (gi > 0 && el > 0) newx[gi - 1][1] = Math.max(newx[gi - 1][1], (start + el) * fa);
    if (gi < Q.length && er > 0) newx[gi][0] = Math.min(newx[gi][0], (start + n - er) * fa);
    const mid = Pg.subarray(el, n - er);
    for (const [u0, u1] of hRuns(mid, thr, gapb)) if ((u1 - u0) * fa >= p.h_smin * U) empties.push([gi, (start + el + u0) * fa, (start + el + u1) * fa, al]);
  }
  const out = [];
  let ei = 0;
  for (let gi = 0; gi <= Q.length; gi++) {
    while (ei < empties.length && empties[ei][0] === gi) { const [, l0, l1, al] = empties[ei++]; out.push({ kind: 'e', idx: [], l0, l1, al }); }
    if (gi < Q.length) out.push({ kind: 'q', idx: Q[gi].slice(), l0: newx[gi][0], l1: newx[gi][1], al: cen[gi] });
  }
  for (const c of out) {
    let [b0, b1, x0, x1] = band(c.al);
    if (c.kind === 'q') { for (const i of c.idx) { b0 = Math.min(b0, L[i].wbox[j1]); b1 = Math.max(b1, L[i].wbox[3 - k]); } }
    const ra = hMrect(ink, box(c.l0, c.l1, x0, b0)), Pa = hProf(ink, ra, k !== 0);
    const thr = Math.max(1.0, p.h_pfrac * (k === 0 ? ra[2] - ra[0] : ra[3] - ra[1]));
    const lo = hWalk(hRev(Pa), thr, gapb, Pa.length);
    const rb = hMrect(ink, box(c.l0, c.l1, b1, x1)), Pb = hProf(ink, rb, k !== 0);
    const hi = hWalk(Pb, thr, gapb, Pb.length);
    const c0 = lo > 0 ? ((k === 0 ? ra[1] : ra[0]) + Pa.length - lo) * fa : b0;
    const cc1 = hi > 0 ? ((k === 0 ? rb[1] : rb[0]) + hi) * fa : b1;
    const bx = box(c.l0, c.l1, Math.min(c0, b0), Math.max(cc1, b1)).map((v) => Math.max(0.0, v));
    bx[2] = Math.min(Wg, bx[2]); bx[3] = Math.min(Hg, bx[3]);
    c.box = bx;
  }
  return out;
}
const H_SPECS = [['1', 1], ['v2', ['v', [1, 1]]], ['v3', ['v', [1, 1, 1]]], ['h2', ['h', [1, 1]]], ['h3', ['h', [1, 1, 1]]],
  ['g4', ['v', [['h', [1, 1]], ['h', [1, 1]]]]], ['h1v2', ['h', [1, ['v', [1, 1]]]]], ['hv21', ['h', [['v', [1, 1]], 1]]],
  ['h1v3', ['h', [1, ['v', [1, 1, 1]]]]], ['hv31', ['h', [['v', [1, 1, 1]], 1]]], ['vh21', ['v', [['h', [1, 1]], 1]]], ['v1h2', ['v', [1, ['h', [1, 1]]]]]];
function hExtent(prof, frac, gap) {
  if (!prof.length) return null;
  const sm = boxSmooth(prof, 1);
  let mx = -Infinity; for (const x of sm) if (x > mx) mx = x;
  if (mx <= 0) return null;
  const runs = hRuns(sm, frac * mx, gap);
  const mass = runs.map(([a, b]) => { let s = 0; for (let i = a; i < b; i++) s += prof[i]; return s; });
  const big = Math.max(...mass), keep = runs.filter((r, i) => mass[i] >= 0.15 * big);
  return [keep[0][0], keep[keep.length - 1][1]];
}
function hTight(ink, r, p) {
  if (r[2] - r[0] <= 0 || r[3] - r[1] <= 0) return null;
  const tx = hExtent(hProf(ink, r, true), p.h_sprof, 2), ty = hExtent(hProf(ink, r, false), p.h_sprof, 2);
  if (!tx || !ty) return null;
  return [r[0] + tx[0], r[1] + ty[0], r[0] + tx[1], r[1] + ty[1]];
}
function hSplit(ink, r, nParts, vert, p) {
  const prof = hProf(ink, r, !vert), n = prof.length;
  if (n < 3 * nParts) return null;
  let tot = 0; for (const x of prof) tot += x;
  const mean = tot / n;
  if (mean <= 0) return null;
  const sm = boxSmooth(prof, 1);
  let js, cost;
  if (nParts === 2) {
    const j0 = Math.ceil(p.h_cutlo * n), j1 = Math.floor(p.h_cuthi * n);
    if (j1 < j0) return null;
    let j = j0; for (let t = j0; t <= j1; t++) if (sm[t] < sm[j]) j = t;
    js = [j]; cost = sm[j] / mean;
  } else {
    const m = Math.ceil(0.15 * n);
    let best = null;
    for (let a = m; a <= n - 2 * m; a++) for (let b = a + m; b <= n - m; b++) { const v = sm[a] + sm[b]; if (best === null || v < best[0]) best = [v, a, b]; }
    if (best === null) return null;
    js = [best[1], best[2]]; cost = Math.max(sm[best[1]], sm[best[2]]) / mean;
  }
  const edges = [0].concat(js, [n]), out = [];
  for (let u = 0; u < nParts; u++) out.push(vert ? [r[0], r[1] + edges[u], r[2], r[1] + edges[u + 1]] : [r[0] + edges[u], r[1], r[0] + edges[u + 1], r[3]]);
  return [out, cost];
}
function hFill(ink, r, spec, U, p) {
  if (spec === 1) {
    const t = hTight(ink, r, p);
    if (!t || Math.max(t[2] - t[0], t[3] - t[1]) * ink.fa < p.h_slotmin * U) return null;
    return [[t], 0.0];
  }
  const [kind, ch] = spec, sp = hSplit(ink, r, ch.length, kind === 'v', p);
  if (!sp) return null;
  let [rects, cost] = sp;
  const out = [];
  for (let i = 0; i < ch.length; i++) {
    const f = hFill(ink, rects[i], ch[i], U, p);
    if (!f) return null;
    out.push(...f[0]); cost = Math.max(cost, f[1]);
  }
  return [out, cost];
}
function hCover(ink, r, boxes) {
  const w = r[2] - r[0], h = r[3] - r[1];
  let tot = 0; for (let y = r[1]; y < r[3]; y++) for (let x = r[0]; x < r[2]; x++) tot += ink.m[y * ink.W + x];
  if (tot <= 0) return 1.0;
  const cov = new Uint8Array(Math.max(0, w * h));
  for (const b of boxes) {
    const x0 = Math.max(r[0], b[0] - 1) - r[0], y0 = Math.max(r[1], b[1] - 1) - r[1], x1 = Math.min(r[2], b[2] + 1) - r[0], y1 = Math.min(r[3], b[3] + 1) - r[1];
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) cov[y * w + x] = 1;
  }
  let s = 0; for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (cov[y * w + x]) s += ink.m[(y + r[1]) * ink.W + x + r[0]];
  return s / tot;
}
function hEnum(ink, cbox, U, p, Wg, Hg) {
  const r = hMrect(ink, cbox), out = [], seen = new Set();
  for (const [name, spec] of H_SPECS) {
    const f = hFill(ink, r, spec, U, p);
    if (!f || f[1] > p.h_cutmax) continue;
    const key = f[0].map((t) => t.join(',')).join(';');
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({ name, boxes: f[0].map((t) => hWrect(ink, t, Wg, Hg)), cost: f[1], cov: hCover(ink, r, f[0]) });
  }
  return out;
}
function hScore(boxes, reads, cov, U, p, gwh, lm, direction, sp) {
  let s = 0.0;
  boxes.forEach((b, i) => {
    const [c, pr] = reads[i], w = Math.max(1.0, b[2] - b[0]), h = Math.max(1.0, b[3] - b[1]), [gw, gh] = gwh[c];
    const da = Math.log(w / h) - Math.log(gw / gh), dz = Math.log(Math.max(w, h) / (Math.max(gw, gh) * U));
    s += Math.log(Math.max(pr, 1e-6) / p.h_theta) - p.h_wa * da * da / (2 * p.h_sa ** 2) - p.h_wz * dz * dz / (2 * p.h_sz ** 2);
  });
  if (lm && p.h_wlm && boxes.length > 1) {
    const order = xyCut(boxes.map((_, i) => i), boxes.map((b) => b.concat([0])), direction, true, sp.cut_tol);
    for (let t = 0; t + 1 < order.length; t++) {
      const ca = reads[order[t]][0], cb = reads[order[t + 1]][0];
      const v = Math.log(lmP2(lm, ca, cb, lm.params.wb)) - Math.log(lmP1(lm, cb));
      s += p.h_wlm * Math.max(-3.0, Math.min(3.0, v));
    }
  }
  return s + p.h_wcov * Math.log(Math.max(cov, 0.05));
}
function hLineLimits(lines, axis, ang) {
  const bands = lines.map((L) => {
    if (!L.length) return null;
    let a0 = Infinity, a1 = -Infinity;
    for (const s of L) { const r = rotCoords(s.wbox.slice(0, 4).concat([0]), axis, ang); a0 = Math.min(a0, r[2]); a1 = Math.max(a1, r[3]); }
    return [a0, a1];
  });
  const idx = lines.map((_, i) => i).filter((i) => bands[i]).sort((i, j) => (bands[i][0] + bands[i][1] - bands[j][0] - bands[j][1]) || (i - j));
  const lims = new Map();
  idx.forEach((li, t) => {
    const lo = t > 0 ? (bands[idx[t - 1]][1] + bands[li][0]) / 2.0 : -1e18, hi = t + 1 < idx.length ? (bands[li][1] + bands[idx[t + 1]][0]) / 2.0 : 1e18;
    lims.set(li, [lo, hi]);
  });
  return lims;
}
function hDecide(cells, p, U, gwh, lm, direction, sp, read) {
  for (const c of cells) c.flag = !!p.h_all || c.cov_det < p.h_cov || c.minconf < p.h_low || c.frag;
  const ex = new Set(cells.map((c, i) => i).filter((i) => cells[i].kind === 'e' || cells[i].flag)
    .sort((i, j) => (cells[i].cov_det - cells[j].cov_det) || (i - j)).slice(0, p.h_maxq));
  cells.forEach((c, i) => {
    const detb = c.det.map((s) => s.wbox);
    if (!ex.has(i) || !c.cands) { c.out = detb.length ? [detb, 'det'] : null; return; }
    let best = null, bs = null;
    if (detb.length) { best = [detb, 'det']; bs = hScore(detb, detb.map(read), c.cov_det, U, p, gwh, lm, direction, sp) + p.h_margin; }
    const need = detb.length ? p.h_rth : p.h_fth;
    for (const cd of c.cands) {
      if (detb.length && !p.h_same && cd.boxes.length === detb.length) continue;     // same number of signs: the detector's boxes stay
      const rd = [];
      for (const b of cd.boxes) { rd.push(read(b)); if (rd[rd.length - 1][1] < need) break; }   // a slot below `need` ends the candidate
      if (rd.length < cd.boxes.length || rd[rd.length - 1][1] < need) continue;
      const s = hScore(cd.boxes, rd, cd.cov, U, p, gwh, lm, direction, sp);
      if (!detb.length && s < p.h_emin) continue;
      if (bs === null || s > bs) { best = [cd.boxes, cd.name]; bs = s; }
    }
    c.out = best;
  });
}
function hLines(cells, nLines, axis, direction, sp, read) {
  const out = [];
  for (let li = 0; li < nLines; li++) {
    let cs = cells.filter((c) => c.line === li && c.out);
    if (axis === 'rows' && direction === 'rtl') cs = cs.reverse();
    const line = [];
    cs.forEach((c, gi) => {
      const [boxes, name] = c.out, conf = new Map(c.det.map((s) => [s.wbox.join(','), s.conf]));
      for (const j of xyCut(boxes.map((_, i) => i), boxes.map((b) => b.concat([0])), direction, true, sp.cut_tol)) {
        const b = boxes[j].map((v) => pyRound(v)), cf = conf.get(boxes[j].join(','));
        line.push({ wbox: b, conf: cf !== undefined ? cf : read(boxes[j])[1], group: gi, src: name });
      }
    });
    if (line.length) out.push(line);
  }
  return out;
}
// seg (lines of { wbox, conf }) -> seg with H's lines, U = quadrat size (work px), top(box) -> [class, prob]
function hApply(seg, g0, U, direction, p, top, lm, f) {
  const gwh = glyphWH(), Wg = g0.w, Hg = g0.h, axis = seg.layout, ang = seg.angle || 0.0, sp = p;
  const lines = seg.lines.map((L) => L.map((s) => ({ wbox: s.wbox.slice(0, 4).map((v) => Math.trunc(v)), conf: s.conf != null ? s.conf : 0.0 })));
  const ink = hInkmap(g0, U, p), lims = hLineLimits(lines, axis, ang), cells = [];
  lines.forEach((L0, li) => {
    if (!L0.length) return;
    const fr = new Set(hFragments(L0, U, p, gwh, top));
    const L = L0.filter((_, i) => !fr.has(i)), frags = L0.filter((_, i) => fr.has(i)).map((s) => s.wbox);
    if (!L.length) return;
    const [Q, rc] = hQuadrats(L, axis, ang, U, p, sp);
    for (const c of hCells(L, Q, rc, axis, ang, U, ink, p, lims.get(li), Wg, Hg)) {
      const r = hMrect(ink, c.box), det = c.idx.map((i) => L[i]);
      c.line = li;
      c.det = det.map((s) => ({ wbox: s.wbox.slice(), conf: s.conf }));
      c.cov_det = det.length ? hCover(ink, r, det.map((s) => hMrect(ink, s.wbox))) : 0.0;
      c.minconf = det.length ? Math.min(...det.map((s) => s.conf)) : 0.0;
      c.frag = frags.some((b) => c.box[0] <= (b[0] + b[2]) / 2.0 && (b[0] + b[2]) / 2.0 <= c.box[2] && c.box[1] <= (b[1] + b[3]) / 2.0 && (b[1] + b[3]) / 2.0 <= c.box[3]);
      cells.push(c);
    }
  });
  for (const c of cells) c.flag = !!p.h_all || c.cov_det < p.h_cov || c.minconf < p.h_low || c.frag;
  let ex = cells.map((c, i) => i).filter((i) => cells[i].kind === 'e' || cells[i].flag).sort((i, j) => (cells[i].cov_det - cells[j].cov_det) || (i - j));
  if (!p.h_all) ex = ex.slice(0, p.h_maxq);
  for (const i of ex) cells[i].cands = hEnum(ink, cells[i].box, U, p, Wg, Hg);
  hDecide(cells, p, U, gwh, lm, direction, sp, top);
  const nl = hLines(cells, lines.length, axis, direction, sp, top);
  seg.lines = nl.map((L) => L.map((s) => Object.assign(s, { box: [s.wbox[0] * f, s.wbox[1] * f, (s.wbox[2] - s.wbox[0]) * f, (s.wbox[3] - s.wbox[1]) * f] })));
  seg.h = { cells: cells.length, multi: cells.filter((c) => c.out && c.out[0].length > 1).length, examined: cells.filter((c) => c.cands).length,
            changed: cells.filter((c) => c.out && c.out[1] !== 'det').length, U: U * f };
  return seg;
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
// P3 K: the pipeline as a generator of progress events (stage, fraction, partial boxes). recognise() runs it to the end in one
// go (identical results); recogniseAsync() gives the browser a turn between the steps (the animation repaints) and can be
// cancelled (opts.signal). opts.onProgress(ev) receives every event in both: ev = { stage, fraction, ... } with the stages
// in order: start, detect (boxes of the first detector pass), level (pyramid levels of close-ups), lines (line bands),
// quadrats (quadrat boxes), read (signs as they are read, in reading order), direction, decode, done (fraction 1).
// Boxes are [x, y, w, h] in the pixels of imageData.
export function recognise(imageData, opts = {}) {
  const it = recogniseSteps(imageData, opts);
  let r = it.next();
  while (!r.done) r = it.next();
  return r.value;
}
export async function recogniseAsync(imageData, opts = {}) {
  const it = recogniseSteps(imageData, opts);
  const pause = opts.pause || (() => new Promise((res) => setTimeout(res, 0)));
  let r = it.next();
  while (!r.done) {
    if (opts.signal && opts.signal.aborted) { const e = new Error('cancelled'); e.name = 'AbortError'; throw e; }
    await pause(r.value);
    if (opts.signal && opts.signal.aborted) { const e = new Error('cancelled'); e.name = 'AbortError'; throw e; }
    r = it.next();
  }
  return r.value;
}
const unionBox = (bs) => { let x0 = 1e18, y0 = 1e18, x1 = -1e18, y1 = -1e18;
  for (const b of bs) { x0 = Math.min(x0, b[0]); y0 = Math.min(y0, b[1]); x1 = Math.max(x1, b[0] + b[2]); y1 = Math.max(y1, b[1] + b[3]); }
  return [x0, y0, x1 - x0, y1 - y0]; };
function progressLines(lines) {                       // line bands and quadrat boxes of a segmentation (lines of {box, group})
  const bands = [], quads = [];
  for (const L of lines) {
    if (!L.length) continue;
    bands.push(unionBox(L.map((x) => x.box)));
    const g = new Map(); for (const x of L) { if (!g.has(x.group)) g.set(x.group, []); g.get(x.group).push(x.box); }
    for (const bs of g.values()) quads.push(unionBox(bs));
  }
  return { bands, quads };
}
// P3 K2: LARGE signs (taller than the line's quadrat) that no pass boxed = bigsigns.py (JS = Python, parity in j_work/par/parity_K2.txt).
// After the pipeline, with U = the quadrat size (work pixels): the detectors (b_dets 0 det_close, 1 main, 2 both) also run at the
// coarse levels b_sides (threshold b_thr); every box and (b_clu) the union of each group of touching raw boxes outside the boxes
// read are candidates; BIG = cross size >= b_u U, long side <= b_umax U, long/short <= b_asp, not at the image border (b_edge),
// share inside the boxes read < b_cov, segmentation recogniser top-1 >= b_conf; taken by that probability (IoU < b_iou with one
// taken); each joins the line it overlaps most as its own quadrat, in reading order (never inside a quadrat).
export const BIG_PARAMS = Object.freeze({ b_on: 1, b_sides: [320, 224], b_dets: 0, b_thr: 0.25, b_clu: 1, b_gap: 0.1, b_cov: 0.35,
                                          b_u: 1.3, b_umax: 4.0, b_conf: 0.6, b_iou: 0.3, b_edge: 1, b_asp: 2.5 });
function bigCover(b, boxes) {
  const x0 = Math.trunc(b[0]), y0 = Math.trunc(b[1]), x1 = Math.trunc(b[2]), y1 = Math.trunc(b[3]);
  const w = Math.max(1, x1 - x0), h = Math.max(1, y1 - y0), m = new Uint8Array(w * h);
  for (const o of boxes) {
    const a0 = Math.max(x0, Math.trunc(o[0])), b0 = Math.max(y0, Math.trunc(o[1])), a1 = Math.min(x1, Math.trunc(o[2])), b1 = Math.min(y1, Math.trunc(o[3]));
    if (a1 > a0 && b1 > b0) for (let y = b0 - y0; y < b1 - y0; y++) m.fill(1, y * w + a0 - x0, y * w + a1 - x0);
  }
  let n = 0; for (let i = 0; i < m.length; i++) n += m[i];
  return n / m.length;
}
function bigIou(a, b) {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])), iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1])), i = ix * iy;
  const u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i;
  return u > 0 ? i / u : 0;
}
function bigClusters(boxes, gap) {
  const n = boxes.length, par = Array.from({ length: n }, (_, i) => i);
  const find = (i) => { while (par[i] !== i) { par[i] = par[par[i]]; i = par[i]; } return i; };
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    const a = boxes[i], b = boxes[j];
    if (a[0] - gap <= b[2] && b[0] - gap <= a[2] && a[1] - gap <= b[3] && b[1] - gap <= a[3]) {
      const ri = find(i), rj = find(j); if (ri !== rj) par[Math.max(ri, rj)] = Math.min(ri, rj);
    }
  }
  const g = new Map();
  for (let i = 0; i < n; i++) { const r = find(i); if (!g.has(r)) g.set(r, []); g.get(r).push(boxes[i]); }
  return [...g.keys()].sort((a, b) => a - b).map((k) => g.get(k)).filter((v) => v.length > 1)
    .map((v) => [Math.min(...v.map((b) => b[0])), Math.min(...v.map((b) => b[1])), Math.max(...v.map((b) => b[2])), Math.max(...v.map((b) => b[3]))]);
}
function bigCandidates(raw, have, U, cols, q, size) {
  const rb = raw.filter(([d, side, sc]) => sc >= q.b_thr && q.b_sides.includes(side) && (q.b_dets === 2 || d === q.b_dets)).map((r) => r[3].slice());
  const out = rb.concat(q.b_clu ? bigClusters(rb.filter((bx) => bigCover(bx, have) < 0.5), q.b_gap * U) : []);
  const res = [], seen = new Set();
  for (const bx of out) {
    const w = bx[2] - bx[0], h = bx[3] - bx[1];
    if ((cols ? w : h) < q.b_u * U || Math.max(w, h) > q.b_umax * U || Math.max(w, h) > q.b_asp * Math.max(1, Math.min(w, h))) continue;
    if (q.b_edge && size && (bx[0] <= 1 || bx[1] <= 1 || bx[2] >= size[0] - 1 || bx[3] >= size[1] - 1)) continue;
    const key = bx.join(',');
    if (seen.has(key) || bigCover(bx, have) >= q.b_cov) continue;
    seen.add(key); res.push(bx);
  }
  return res;
}
function bigSelect(cands, top, q) {
  const sc = cands.map((bx) => { const [c, p] = top(bx); return [p, c, bx]; }).filter((t) => t[0] >= q.b_conf);
  sc.sort((a, b) => b[0] - a[0]);
  const added = [];
  for (const [p, c, bx] of sc) if (added.every((a) => bigIou(bx, a[0]) < q.b_iou)) added.push([bx, c, p]);
  return added;
}
function bigPlace(lines, bx, cols, direction) {
  let best = 0, bo = null;
  lines.forEach((L, li) => {
    const lo = Math.min(...L.map((s) => cols ? s.wbox[0] : s.wbox[1])), hi = Math.max(...L.map((s) => cols ? s.wbox[2] : s.wbox[3]));
    const o = Math.min(hi, cols ? bx[2] : bx[3]) - Math.max(lo, cols ? bx[0] : bx[1]);
    if (bo === null || o > bo) { bo = o; best = li; }
  });
  const L = lines[best], c = cols ? (bx[1] + bx[3]) / 2 : (bx[0] + bx[2]) / 2, rev = !cols && direction === 'rtl';
  let at = L.length;
  for (let i = 0; i < L.length; i++) {
    const s = L[i], cs = cols ? (s.wbox[1] + s.wbox[3]) / 2 : (s.wbox[0] + s.wbox[2]) / 2;
    if (rev ? cs < c : cs > c) { at = i; break; }
  }
  while (at > 0 && at < L.length && L[at].group === L[at - 1].group) at++;
  return [best, at];
}
// the detector boxes at the coarse levels: [det index, side, score, [x0, y0, x1, y1]] (threshold 0.15; filtered later)
function* bigRaw(work, dets, sides, p, tick) {
  const out = [];
  for (let d = 0; d < dets.length; d++) {
    if (!dets[d]) continue;
    for (const side of sides) {
      const pp = Object.assign({}, p, { det_side: side, det_thr: 0.15 });
      for (const r of detectBoxes(work, pp, dets[d])) out.push([d, side, r[5], r.slice(0, 4)]);
      if (tick) yield tick();
    }
  }
  return out;
}
// seg: the pipeline result (lines of {box, wbox, group}); adds the big signs in place; returns their count (generator: progress)
function* bigSigns(seg, work, U, direction, p, dets, top, tick) {
  const q = Object.assign({}, BIG_PARAMS);
  for (const k of Object.keys(p || {})) if (k.startsWith('b_')) q[k] = p[k];
  if (!q.b_on || !(U > 0) || !seg.lines.length) return 0;
  const cols = seg.layout === 'columns', f = seg.factor || 1;
  const have = seg.lines.flatMap((L) => L.map((s) => s.wbox.slice(0, 4)));
  const raw = yield* bigRaw(work, dets.map((d, i) => (q.b_dets === 2 || q.b_dets === i) ? d : null), q.b_sides, p, tick);
  const added = bigSelect(bigCandidates(raw, have, U, cols, q, [work.w, work.h]), top, q);
  for (const [bx, c, pr] of added) {
    const [li, at] = bigPlace(seg.lines, bx, cols, direction), L = seg.lines[li];
    const g = Math.max(-1, ...L.map((s) => s.group)) + 1;
    L.splice(at, 0, { box: [bx[0] * f, bx[1] * f, (bx[2] - bx[0]) * f, (bx[3] - bx[1]) * f], wbox: bx.slice(), group: g, conf: pr, src: 'big', k: -1 });
  }
  if (added.length) seg.big = added.length;
  return added.length;
}
function* recogniseSteps(imageData, opts = {}) {
  const emit = (stage, fraction, data) => { const ev = Object.assign({ stage, fraction }, data || {}); if (opts.onProgress) opts.onProgress(ev); return ev; };
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
  // P3 J: the reader (if loaded) reads the final signs; same caching and facing taps as cls
  const reader = opts.reader === undefined ? READER : opts.reader;
  const rcache = new Map();
  const clsRead = !reader ? cls : (crop) => {
    let h = 0; for (let i = 0; i < crop.length; i += 7) h = (h * 31 + crop[i]) | 0;
    const key = h + ':' + crop.length;
    let hit = rcache.get(key);
    if (hit) for (const e of hit) if (e.crop.every((v, i) => v === crop[i])) return e.res;
    const tc = now();
    const tap = auto ? { after: fm.after } : null;
    const res = classify(crop, reader, tap, kTop);
    if (tap && res.prob[0] >= keepConf && fm.voters.has(res.idx[0])) maps.set(res, tap.map);
    msCnn += now() - tc; nCnn++;
    if (!hit) rcache.set(key, hit = []);
    hit.push({ crop, res });
    return res;
  };
  // P3 B: classical segmentation, the detector, or both ('auto', default with a detector): the larger quality score wins
  const det = opts.detector === undefined ? DETECTOR : opts.detector;
  const closeDet = opts.detectorClose === undefined ? DETECTOR_CLOSE : opts.detectorClose;      // P3 F
  const mode = det ? (opts.segmenter || 'auto') : 'classic';
  const grey = imageData.d ? imageData : toGrey(imageData);
  yield emit('start', 0.02, { width: grey.w, height: grey.h });
  const sopts = { direction, layout: opts.layout, params: opts.params, classify: (crop) => cls(crop).prob[0], detector: det,
                  classifyTop: (crop) => { const r = cls(crop); return [r.idx[0], r.prob[0]]; } };
  let seg, scores = null, bigU = 0, bigWork = null;     // P3 K2: U and work image of the reading, for the big-sign step
  if (mode === 'classic') seg = segment(grey, sopts);
  else if (mode === 'detector') seg = segmentDet(grey, sopts);
  else {
    // P3 K: real progress through the blocking first pass: the detector layer by layer ('scan'), then its peaks, top to bottom,
    // as they are checked by the recogniser ('peaks'); segmentDet then reuses both (same pass, cached reads = same result)
    const gp0 = Object.assign({}, PARAMS, DET_PARAMS, opts.params || {}), [g00, f00] = downscale(grey, gp0.max_side);
    const nOps = det.ops.length * (det.also ? 2 : 1) * (det.union ? 2 : 1), tf0 = now();
    let kOp = 0;
    const runF = function* (d) { const it = forwardSteps(g00, gp0.det_side, d); let r = it.next(); while (!r.done) { kOp++; if (kOp % 2 === 0) yield emit('scan', 0.02 + 0.1 * kOp / nOps); r = it.next(); } return r.value; };
    const fwd = yield* runF(det.union ? Object.assign({}, det, { union: null }) : det), fwd2 = det.union ? yield* runF(det.union) : null;
    const fwdMs = now() - tf0, boxes0 = detectBoxes(g00, gp0, det, fwd, fwd2);
    const scaled = (bb) => [bb[0] * f00, bb[1] * f00, (bb[2] - bb[0]) * f00, (bb[3] - bb[1]) * f00];
    if (gp0.det_floor > 0) for (let i = 0; i < boxes0.length; i++) {
      const bb = boxes0[i]; sopts.classify(prepCrop(g00, [bb[0], bb[1], bb[2], bb[3]], gp0.crop_pad));
      if (i % 6 === 5 || i === boxes0.length - 1) yield emit('peaks', 0.12 + 0.17 * (i + 1) / boxes0.length, { boxes: boxes0.slice(0, i + 1).map(scaled), y: (bb[3]) * f00 });
    }
    let b = segmentDet(grey, Object.assign({}, sopts, { fwd, fwd2, fwdMs }));
    const gp = Object.assign({}, DET_PARAMS, opts.params || {}), nb = b.lines.reduce((n, L) => n + L.length, 0);
    yield emit('detect', 0.3, { boxes: b.lines.flatMap((L) => L.map((x) => x.box)) });
    const hp = Object.assign({}, PARAMS, DET_PARAMS, QUAD_PARAMS, H_PARAMS, opts.params || {});
    const hTop = (work) => (bx) => sopts.classifyTop(prepCrop(work, bx, hp.crop_pad));
    const guard = nb > gp.det_gate && wallGuard(b, hp);              // P3 J
    if (nb > gp.det_gate && !guard) {
      seg = b; scores = { classic: null, detector: b.selection_score, gate: true };
      if (opts.quads !== false && hp.q_h) {                          // P3 H: quadrats on walls too
        const U = estimateU([[gp.det_side, b.lines.flatMap((L) => L.map((x) => ({ box: x.wbox.slice(0, 4), conf: x.conf })))]], hp)[0];
        if (U > 0) { yield emit('lines', 0.5, progressLines(seg.lines)); seg = hApply(seg, b.work, U, direction, hp, hTop(b.work), lm, b.factor); scores.h = seg.h; bigU = U; bigWork = b.work; }
      }
    } else {
      let levels = null, detMs = b.det_ms, quad = null;
      const runs = [[gp.det_side, b]];
      if ((nb <= gp.det_pyr_gate || guard) && gp.det_levels.length > 1) {   // P3 F: the detector pyramid (close-ups); P3 J guard
        let best = b, bs = levelScore(b, gp.det_lvl_floor);
        levels = [[gp.det_side, Math.round(bs * 1e4) / 1e4, nb]];
        for (const side of gp.det_levels) {
          if (side === gp.det_side) continue;
          const r = segmentDet(grey, Object.assign({}, sopts, { detector: closeDet || det,
            params: Object.assign({}, opts.params || {}, { det_side: side, det_thr: gp.det_lvl_thr }) }));
          const rs = levelScore(r, gp.det_lvl_floor);
          detMs += r.det_ms;
          yield emit('level', 0.3 + 0.25 * runs.length / gp.det_levels.length, { side, boxes: r.lines.flatMap((L) => L.map((x) => x.box)) });
          runs.push([side, r]);
          levels.push([side, Math.round(rs * 1e4) / 1e4, r.lines.reduce((n, L) => n + L.length, 0)]);
          if (rs > bs) { best = r; bs = rs; }
        }
        b = best;
        // P3 G: at the close-up scale the quadrat path reads the pooled levels (U, signs across levels, layout, quadrats)
        const qp = hp;
        if (opts.quads !== false && qp.q_on !== 0) {
          const work = runs[0][1].work;
          quad = quadPath(runs, direction, opts.layout || 'auto', qp, (bx) => sopts.classifyTop(prepCrop(work, bx, qp.crop_pad)));
        }
      }
      if (quad && quad.seg) {
        seg = quad.seg;
        scores = { classic: null, detector: b.selection_score, gate: false, levels, det_side: null, levels_detector: closeDet ? 'close' : 'main', quad: seg.quad };
      } else {
        const a = segment(grey, sopts);
        const sb = levels ? levelScore(b, gp.det_cls_floor) : b.selection_score, sa = levels ? levelScore(a, gp.det_cls_floor) : a.selection_score;
        scores = { classic: a.selection_score, detector: b.selection_score, gate: false, levels, det_side: b.det_side, levels_detector: levels ? (closeDet ? 'close' : 'main') : null };
        if (quad) scores.quad = { path: 'normal', U: quad.U * runs[0][1].factor, U640: quad.U640 };
        seg = sb > sa ? b : a;
      }
      seg.det_ms = detMs;
      if (quad && quad.U > 0) { bigU = quad.U; bigWork = runs[0][1].work; }     // P3 K2
      if (guard) scores.guard = nb;                                   // P3 J
      if (quad && quad.U > 0 && opts.quads !== false && hp.q_h) {    // P3 H: every path (quad and normal scale)
        yield emit('lines', 0.58, progressLines(seg.lines));
        const work = runs[0][1].work;
        seg = hApply(seg, work, quad.U, direction, hp, hTop(work), lm, runs[0][1].factor);
        if (scores.quad) scores.quad.h = seg.h; else scores.h = seg.h;
      }
    }
    // P3 K2: large signs (taller than the line's quadrat) that no pass boxed; the coarse detector passes report progress
    if (bigU > 0 && (seg.work || bigWork)) {
      const work = seg.work || bigWork, nT = BIG_PARAMS.b_sides.length * (BIG_PARAMS.b_dets === 2 ? 2 : 1);
      let kT = 0;
      yield emit('large', 0.6);
      const nb2 = yield* bigSigns(seg, work, bigU, direction, hp, [closeDet, det],
        (bx) => sopts.classifyTop(prepCrop(work, bx, hp.crop_pad)), () => emit('large', 0.6 + 0.05 * (++kT) / nT));
      if (nb2) scores.big = nb2;
    }
  }
  const RM = reader || model, C = RM.classes;                       // P3 J: the final signs are read by the reader
  { const pl = progressLines(seg.lines); yield emit('quadrats', 0.66, { bands: pl.bands, quads: pl.quads, layout: seg.layout, total: seg.lines.reduce((n, L) => n + L.length, 0) }); }
  const lines = [], nAll = seg.lines.reduce((n, L) => n + L.length, 0);
  let nRead = 0, fresh = [];
  for (const L of seg.lines) {
    const signs = [];
    for (const s of L) {
      const r = clsRead(prepCrop(seg.work, s.wbox, p.crop_pad));
      const alts = r.idx.slice(0, 5).map((k, t) => ({ char: C[k].char, codepoint: C[k].codepoint, conf: r.prob[t], cls: k }));
      const o = { char: alts[0].char, codepoint: alts[0].codepoint, conf: r.prob[0], uncertain: r.prob[0] < RM.threshold,
                  alts, box: s.box, group: s.group, wbox: s.wbox, k: s.k };
      if (lm) o.rec = { cls: r.idx.slice(0, lmP.k), prob: r.prob.slice(0, lmP.k) };     // the recogniser's own top-k (P3 C2)
      if (maps.has(r)) Object.defineProperty(o, 'fmap', { value: maps.get(r), enumerable: false });
      signs.push(o); nRead++; fresh.push(s.box);
      if (fresh.length >= 4) { yield emit('read', 0.66 + 0.28 * nRead / Math.max(1, nAll), { boxes: fresh, found: nRead, total: nAll }); fresh = []; }
    }
    lines.push({ signs });
  }
  if (fresh.length) yield emit('read', 0.94, { boxes: fresh, found: nRead, total: nAll });
  const res = { layout: seg.layout, direction, direction_source: dirIn === 'auto' ? 'default (no facing model)' : 'user',
                polarity: seg.polarity, scale: seg.scale * seg.factor, angle: seg.angle || 0, params: opts.params || null, lines,
                segmenter: seg.segmenter === 'quad' ? 'quad' : (seg.polarity === 'detector' ? 'detector' : 'classic'), segmenter_mode: mode, segmenter_scores: scores,
                stats: { cnn_crops: nCnn, cnn_ms: msCnn, det_ms: seg.det_ms != null ? seg.det_ms : null } };
  if (seg.h) res.quadrats = seg.h;                             // P3 H: the lines' quadrats come from the quadrat stage
  if (auto) {
    yield emit('direction', 0.96, { found: nRead });
    const tf = now();
    const v = voteDirection(lines.flatMap((L) => L.signs), fm, opts.vote);
    if (v.direction !== direction) res.lines = orderSigns(lines, seg.layout, res.angle, direction, v.direction, p, res.segmenter === 'quad' || !!res.quadrats);
    Object.assign(res, { direction: v.direction, direction_source: 'auto', direction_certainty: v.certainty, direction_votes: v.votes,
                         direction_flag: v.flagged ? 'too little evidence' : null, direction_auto: v });
    res.stats.facing_ms = now() - tf;
  }
  if (lm) {                                                    // P3 C2: decode each line in its final reading order
    yield emit('decode', 0.98, { found: nRead });
    const tl = now(), d = decodeLines(res.lines, lm, lmP, RM);       // P3 J: the prior of the recogniser that read the signs
    res.decode = { applied: true, params: lmP, changed: d.changed };
    res.stats.decode_ms = now() - tl;
  } else res.decode = { applied: false };
  for (const L of res.lines) L.quads = lineQuads(L.signs, res.direction, p.cut_tol);     // P3 G: the quadrats of each line
  res.quality = readingQuality(res, grey);
  res.stats.total_ms = now() - t0;
  emit('done', 1, { found: nRead });
  return res;
}

// P3 F: is the reading poor (few signs or low confidence)? Then one practical tip, by the most likely cause:
// 'step_back' (the signs are large in the picture: a close-up), 'light' (pale, low-contrast photo), 'closer' (tiny signs),
// otherwise 'steady'. Sizes relative to the photo; contrast = 99 % - 1 % grey level.
// min_signs / min_conf chosen on the DEV close-ups + pages_dev (JS outputs, NOTES_P3 F): flagged readings have SER > 50 % in
// 22 of 23 dev images; the tip thresholds are set a priori (large = median sign >= 15 % of the shorter side, pale = grey
// range < 60 levels, small = median sign < 2 % of the longer side).
export const QUALITY = Object.freeze({ min_signs: 2, min_conf: 0.6, large: 0.15, pale: 60, small: 0.02 });
export function readingQuality(res, grey, q = QUALITY) {
  const signs = res.lines.flatMap((L) => L.signs), n = signs.length;
  const mean = n ? signs.reduce((a, s) => a + s.conf, 0) / n : 0;
  const sz = signs.map((s) => Math.max(s.box[2], s.box[3])).sort((a, b) => a - b), med = n ? sz[n >> 1] : 0;
  const contrast = rankValue(grey.d, 0.99) - rankValue(grey.d, 0.01);
  const shortSide = Math.min(grey.w, grey.h), longSide = Math.max(grey.w, grey.h);
  const ds = res.segmenter_scores && res.segmenter_scores.det_side;
  const large = (n && med >= q.large * shortSide) || (res.segmenter === 'detector' && ds != null && ds < DET_PARAMS.det_side) || res.segmenter === 'quad';
  const poor = n < q.min_signs || mean < q.min_conf;
  const tip = !poor ? null : (large ? 'step_back' : (contrast < q.pale ? 'light' : (n && med < q.small * longSide ? 'closer' : 'steady')));
  return { poor, tip, signs: n, mean_conf: mean, sign_px: med, contrast, large: !!large };
}
