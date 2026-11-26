// Hình minh họa cho trang tổng quan: cùng nét vẽ với vòng người ở trang chủ
// (nhìn từ trên xuống, áo họa tiết tông xanh, màu bột, hạt giấy). Vẽ bằng SVG sinh từ code.
import { SHIRTS, HAIR, SKIN } from './landing.js';

const NS = 'http://www.w3.org/2000/svg';
const LEAVES = ['#24552c', '#2f6b38', '#4a8233', '#5f9a3f', '#7aa955', '#3c7232', '#548f3d'];

function el(name, attrs = {}, parent) {
  const node = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.append(node);
  return node;
}
function rng(seed) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32);
}

// Bộ lọc và họa tiết áo, đặt tên riêng theo từng hình để không đụng nhau.
function defsFor(svg, id, hr) {
  const defs = el('defs', {}, svg);
  SHIRTS.forEach(([bg, fg, kind], i) => {
    const t = Math.max(6, hr * 0.32);
    const p = el('pattern', { id: `${id}-s${i}`, width: t, height: t, patternUnits: 'userSpaceOnUse' }, defs);
    el('rect', { width: t, height: t, fill: bg }, p);
    if (kind === 'dots') el('circle', { cx: t / 2, cy: t / 2, r: t * 0.16, fill: fg, opacity: 0.75 }, p);
    if (kind === 'stripes') el('rect', { width: t, height: t * 0.38, fill: fg, opacity: 0.6 }, p);
    if (kind === 'check') {
      el('rect', { width: t, height: t * 0.22, fill: fg, opacity: 0.45 }, p);
      el('rect', { width: t * 0.22, height: t, fill: fg, opacity: 0.45 }, p);
    }
    if (kind === 'knit') el('path', { d: `M0,${t * 0.3} L${t / 2},${t * 0.7} L${t},${t * 0.3}`, stroke: fg, 'stroke-width': t * 0.12, fill: 'none' }, p);
  });
  const brush = el('filter', { id: `${id}-brush`, x: '-10%', y: '-10%', width: '120%', height: '120%' }, defs);
  el('feTurbulence', { type: 'fractalNoise', baseFrequency: 0.04, numOctaves: 3, seed: 3, result: 'n' }, brush);
  el('feDisplacementMap', { in: 'SourceGraphic', in2: 'n', scale: hr * 0.45, xChannelSelector: 'R', yChannelSelector: 'G' }, brush);
  const shade = el('filter', { id: `${id}-shade`, x: '-30%', y: '-30%', width: '160%', height: '160%' }, defs);
  el('feDropShadow', { dx: 0, dy: hr * 0.08, stdDeviation: hr * 0.14, 'flood-color': '#1d3b22', 'flood-opacity': 0.32 }, shade);
  const grain = el('filter', { id: `${id}-grain` }, defs);
  el('feTurbulence', { type: 'fractalNoise', baseFrequency: 0.85, numOctaves: 2, seed: 9 }, grain);
  el('feColorMatrix', { values: '0 0 0 0 0.1  0 0 0 0 0.15  0 0 0 0 0.1  0 0 0 -1.4 0.9' }, grain);
  const glow = el('radialGradient', { id: `${id}-glow`, cx: '50%', cy: '50%', r: '50%' }, defs);
  el('stop', { offset: '0%', 'stop-color': '#ffffff' }, glow);
  el('stop', { offset: '60%', 'stop-color': '#eef6e4' }, glow);
  el('stop', { offset: '100%', 'stop-color': '#cfe5b8' }, glow);
  return defs;
}

// Bãi cỏ tròn mép gồ ghề như vết cọ, rải lá.
function lawn(svg, id, cx, cy, r, rand, hr) {
  el('circle', { cx, cy, r, fill: `url(#${id}-glow)`, filter: `url(#${id}-brush)` }, svg);
  const g = el('g', {}, svg);
  for (let i = 0; i < 70; i++) {
    const a = rand() * Math.PI * 2;
    const d = r * (0.92 + rand() * 0.2);
    const len = hr * (0.18 + rand() * 0.22);
    el('path', {
      d: `M0,${-len} C${len * 0.6},${-len * 0.4} ${len * 0.6},${len * 0.4} 0,${len} C${-len * 0.6},${len * 0.4} ${-len * 0.6},${-len * 0.4} 0,${-len} Z`,
      transform: `translate(${cx + Math.cos(a) * d},${cy + Math.sin(a) * d}) rotate(${rand() * 360})`,
      fill: LEAVES[Math.floor(rand() * LEAVES.length)], opacity: 0.8,
    }, g);
  }
}

// Một bạn nhỏ nhìn từ trên xuống. Gốc tọa độ ở đầu, +y hướng về phía vật ở giữa.
// reach: tay vươn tới (x, y) theo hệ tọa độ riêng; null thì tay để xuôi.
function kid(parent, id, { x, y, deg, hr, shirt, hair, skin, style, reach }) {
  const g = el('g', { transform: `translate(${x},${y}) rotate(${deg})`, filter: `url(#${id}-shade)` }, parent);
  const fillShirt = `url(#${id}-s${shirt})`;
  const [bg] = SHIRTS[shirt];
  // Thân và chân phía sau đầu.
  el('path', { d: `M${-hr * 0.55},${-hr * 2.9} L${-hr * 0.5},${-hr * 4.4} M${hr * 0.55},${-hr * 2.9} L${hr * 0.5},${-hr * 4.4}`, stroke: '#3d4f6b', 'stroke-width': hr * 0.62, 'stroke-linecap': 'round' }, g);
  el('rect', { x: -hr * 1.15, y: -hr * 3.15, width: hr * 2.3, height: hr * 2.6, rx: hr * 0.75, fill: fillShirt }, g);
  // Tay.
  const hands = reach ? [[-hr * 0.35 + reach[0], reach[1]], [hr * 0.35 + reach[0], reach[1]]] : [[-hr * 1.55, -hr * 1.2], [hr * 1.55, -hr * 1.2]];
  [[-1, hands[0]], [1, hands[1]]].forEach(([s, [hx, hy]]) => {
    const sx = s * hr * 0.95, sy = -hr * 0.95;
    const mx = (sx + hx) / 2 + s * hr * 0.55, my = (sy + hy) / 2;
    el('path', { d: `M${sx},${sy} Q${mx},${my} ${hx},${hy}`, stroke: bg, 'stroke-width': hr * 0.5, 'stroke-linecap': 'round', fill: 'none' }, g);
    el('path', { d: `M${sx},${sy} Q${mx},${my} ${hx},${hy}`, stroke: fillShirt, 'stroke-width': hr * 0.4, 'stroke-linecap': 'round', fill: 'none' }, g);
    el('circle', { cx: hx, cy: hy, r: hr * 0.24, fill: skin }, g);
  });
  // Đầu: giống vòng người ở trang chủ.
  if (style === 1) for (const sd of [-1, 1]) el('path', { d: `M${sd * hr * 0.6},${-hr * 0.4} Q${sd * hr * 1.35},${hr * 0.2} ${sd * hr * 1.05},${hr * 0.95} Q${sd * hr * 0.85},${hr * 0.5} ${sd * hr * 0.7},${hr * 0.1} Z`, fill: hair }, g);
  if (style === 3) el('circle', { cx: 0, cy: -hr * 1.05, r: hr * 0.42, fill: hair }, g);
  el('circle', { cx: 0, cy: -hr * 0.35, r: hr * 0.98, fill: hair }, g);
  el('ellipse', { cx: 0, cy: hr * 0.18, rx: hr * 0.82, ry: hr * 0.84, fill: skin }, g);
  el('path', { d: `M${-hr * 0.9},${-hr * 0.05} Q${-hr * 0.4},${-hr * 0.35} 0,${-hr * 0.05} Q${hr * 0.45},${-hr * 0.4} ${hr * 0.9},${-hr * 0.05} L${hr * 0.9},${-hr * 0.6} L${-hr * 0.9},${-hr * 0.6} Z`, fill: hair }, g);
  for (const s of [-1, 1]) {
    el('path', { d: `M${s * hr * 0.42},${hr * 0.18} q${s * hr * 0.12},${hr * 0.1} ${s * hr * 0.24},0`, stroke: '#2a211d', 'stroke-width': hr * 0.07, 'stroke-linecap': 'round', fill: 'none' }, g);
    el('circle', { cx: s * hr * 0.48, cy: hr * 0.48, r: hr * 0.13, fill: '#f08f86', opacity: 0.45 }, g);
  }
  el('path', { d: `M${-hr * 0.17},${hr * 0.6} Q0,${hr * 0.75} ${hr * 0.17},${hr * 0.6}`, stroke: '#9a4a3a', 'stroke-width': hr * 0.07, 'stroke-linecap': 'round', fill: 'none' }, g);
  return g;
}

// Điện thoại nằm giữa, màn hình có vài bong bóng tin nhắn; warn: tô một bong bóng theo màu cảnh báo.
function setup(svg, W, H) {
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('aria-hidden', 'true');
  svg.replaceChildren();
}
// Hạt giấy chỉ phủ trong bãi cỏ tròn, để nền quanh hình vẫn trùng màu giấy của trang.
function grainOver(svg, id, W, H) {
  const clip = el('clipPath', { id: `${id}-clip` }, svg.querySelector('defs'));
  el('circle', { cx: W / 2, cy: H / 2, r: Math.min(W, H) * 0.47 }, clip);
  el('rect', { width: W, height: H, filter: `url(#${id}-grain)`, opacity: 0.3, 'pointer-events': 'none', 'clip-path': `url(#${id}-clip)` }, svg);
}
function person(i, rand) {
  return { shirt: i % SHIRTS.length, hair: HAIR[Math.floor(rand() * HAIR.length)], skin: SKIN[Math.floor(rand() * SKIN.length)], style: i % 5 === 4 ? 0 : i % 5 };
}

// Kết thúc: vòng người nhỏ quanh khoảng trời, gợi lại trang chủ.
export function drawClosing(svg) {
  const id = 'art-end', W = 460, H = 460, cx = W / 2, cy = H / 2, hr = 24;
  setup(svg, W, H);
  defsFor(svg, id, hr);
  const rand = rng(5);
  lawn(svg, id, cx, cy, 205, rand, hr);
  const kids = el('g', {}, svg);
  const N = 8;
  for (let i = 0; i < N; i++) {
    const deg = (360 / N) * i;
    const a = (deg * Math.PI) / 180, d = 118;
    kid(kids, id, { x: cx + Math.sin(a) * d, y: cy - Math.cos(a) * d, deg, hr, ...person(i + 1, rand), reach: null });
  }
  grainOver(svg, id, W, H);
}

