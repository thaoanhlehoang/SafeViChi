// Hình minh họa cho trang tổng quan: cùng nét vẽ với vòng người ở trang chủ
// (nhìn từ trên xuống, áo họa tiết tông xanh, màu bột, hạt giấy). Vẽ bằng SVG sinh từ code.
import { SHIRTS } from './landing.js';

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

function setup(svg, W, H) {
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('aria-hidden', 'true');
  svg.replaceChildren();
}
// Kết thúc: bãi cỏ phủ kín như trang chủ, rải hoa trắng, vài cánh chim mờ bay qua,
// nửa phải để khung chat mẫu, nửa trái để chữ.
export function drawClosing(svg) {
  const W = Math.max(320, svg.clientWidth || window.innerWidth);
  const H = Math.max(320, svg.clientHeight || window.innerHeight);
  const id = 'art-end', hr = Math.max(20, Math.min(44, Math.min(W, H) * 0.052));
  setup(svg, W, H);
  svg.setAttribute('preserveAspectRatio', 'xMidYMid slice');
  const defs = defsFor(svg, id, hr);
  const blur = el('filter', { id: `${id}-blur`, x: '-50%', y: '-50%', width: '200%', height: '200%' }, defs);
  el('feGaussianBlur', { stdDeviation: hr * 0.16 }, blur);
  const rand = rng(11);
  // Tán lá phủ kín.
  el('rect', { width: W, height: H, fill: '#2f5e2a' }, svg);
  const leaves = el('g', {}, svg);
  const nLeaf = Math.round((W * H) / 900);
  for (let i = 0; i < nLeaf; i++) {
    const x = rand() * (W + 60) - 30, y = rand() * (H + 60) - 30;
    const len = (10 + rand() * 16) * (hr / 50) * 1.6;
    el('path', {
      d: `M0,${-len} C${len * 0.6},${-len * 0.4} ${len * 0.6},${len * 0.4} 0,${len} C${-len * 0.6},${len * 0.4} ${-len * 0.6},${-len * 0.4} 0,${-len} Z`,
      transform: `translate(${x.toFixed(1)},${y.toFixed(1)}) rotate(${Math.round(rand() * 360)})`,
      fill: LEAVES[Math.floor(rand() * LEAVES.length)], opacity: (0.7 + rand() * 0.3).toFixed(2),
    }, leaves);
  }
  // Hoa trắng nhỏ.
  const flowers = el('g', { filter: `url(#${id}-shade)` }, svg);
  for (let i = 0; i < Math.round((W * H) / 26000); i++) {
    const x = rand() * W, y = rand() * H, r = hr * (0.14 + rand() * 0.08);
    if (W > 900 && x < W * 0.42 && y > H * 0.3 && y < H * 0.78) continue; // chừa chỗ cho chữ
    const g = el('g', { transform: `translate(${x.toFixed(1)},${y.toFixed(1)}) rotate(${Math.round(rand() * 72)})` }, flowers);
    for (let k = 0; k < 5; k++) {
      const a = (k * 72 * Math.PI) / 180;
      el('circle', { cx: Math.cos(a) * r, cy: Math.sin(a) * r, r: r * 0.75, fill: '#fbfaf2' }, g);
    }
    el('circle', { r: r * 0.5, fill: '#f2c94c' }, g);
  }
  // Chim trắng mờ bay ngang, như nhìn từ trên xuống.
  const doves = [[0.1, 0.17, -30, 1.15], [0.88, 0.15, 205, 1], [0.48, 0.93, 150, 1.25], [0.96, 0.84, -50, 0.85]];
  for (const [fx, fy, rot, sc] of doves) {
    const s = hr * 3 * sc;
    const g = el('g', { class: 'dove' }, svg);
    const b = el('g', { transform: `translate(${(W * fx).toFixed(1)},${(H * fy).toFixed(1)}) rotate(${rot}) scale(${(s / 110).toFixed(3)})`, filter: `url(#${id}-blur)`, opacity: 0.9 }, g);
    for (const sd of [-1, 1]) {
      // Cánh dang rộng, mép sau xẻ lông.
      el('path', { d: `M0,-4 C${sd * -22},-30 ${sd * -72},-40 ${sd * -112},-28 C${sd * -104},-20 ${sd * -96},-15 ${sd * -88},-11 C${sd * -96},-6 ${sd * -90},-1 ${sd * -78},1 C${sd * -82},6 ${sd * -72},10 ${sd * -60},9 C${sd * -46},13 ${sd * -20},12 ${sd * -4},10 Z`, fill: '#fbfbf6' }, b);
    }
    el('path', { d: 'M-11,24 L-16,50 L0,45 L16,50 L11,24 Z', fill: '#f1f1ea' }, b);
    el('ellipse', { cx: 0, cy: 2, rx: 11, ry: 30, fill: '#ffffff' }, b);
    el('circle', { cx: 0, cy: -27, r: 9.5, fill: '#ffffff' }, b);
    el('path', { d: 'M-3,-36 L0,-43 L3,-36 Z', fill: '#e8c46a' }, b);
  }
  // Hạt giấy phủ toàn bộ.
  el('rect', { width: W, height: H, filter: `url(#${id}-grain)`, opacity: 0.3, 'pointer-events': 'none' }, svg);
}
