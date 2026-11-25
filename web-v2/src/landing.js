// Trang đầu: nhìn từ dưới lên, một vòng người khoác vai nhau quanh khoảng trời tròn ở giữa,
// ngoài cùng là tán lá xanh. Vẽ bằng SVG sinh từ code, nét giả màu bột (gouache).

import { drawHands } from './hands.js';

const NS = 'http://www.w3.org/2000/svg';

// Áo theo tông xanh: [màu nền, màu họa tiết, kiểu họa tiết]
export const SHIRTS = [
  ['#2f6b3a', '#f3ecd6', 'dots'],
  ['#9cc56a', '#5f8f35', 'stripes'],
  ['#efe6cc', '#5f9a3c', 'check'],
  ['#3e7f5d', '#a8d4b4', 'stripes'],
  ['#c9dfa0', '#ffffff', 'dots'],
  ['#5f9a3c', '#d9ecb8', 'check'],
  ['#e9d77a', '#c9a93a', 'knit'],
  ['#8fc4a6', '#f6f2e4', 'dots'],
  ['#466b2e', '#7fae4a', 'knit'],
];
export const HAIR = ['#2a211d', '#5a3a26', '#1e1b1a', '#7a4a2a', '#3b2a22', '#a5672f'];
export const SKIN = ['#f6d3b5', '#eec29e', '#e0ad86', '#c98f69'];
// Lá trầm hơn một chút để chữ trắng ở màn đăng nhập đọc rõ mà không cần lớp phủ.
const LEAVES = ['#24552c', '#2f6b38', '#4a8233', '#5f9a3f', '#7aa955', '#3c7232', '#548f3d'];

function rng(seed) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32);
}

function el(name, attrs = {}, parent) {
  const node = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.append(node);
  return node;
}

function makePatterns(defs, hr) {
  SHIRTS.forEach(([bg, fg, kind], i) => {
    const t = Math.max(8, hr * 0.32);
    const p = el('pattern', { id: `shirt${i}`, width: t, height: t, patternUnits: 'userSpaceOnUse' }, defs);
    el('rect', { width: t, height: t, fill: bg }, p);
    if (kind === 'dots') el('circle', { cx: t / 2, cy: t / 2, r: t * 0.16, fill: fg, opacity: 0.75 }, p);
    if (kind === 'stripes') el('rect', { width: t, height: t * 0.38, fill: fg, opacity: 0.6 }, p);
    if (kind === 'check') {
      el('rect', { width: t, height: t * 0.22, fill: fg, opacity: 0.45 }, p);
      el('rect', { width: t * 0.22, height: t, fill: fg, opacity: 0.45 }, p);
    }
    if (kind === 'knit') el('path', { d: `M0,${t * 0.3} L${t / 2},${t * 0.7} L${t},${t * 0.3}`, stroke: fg, 'stroke-width': t * 0.12, fill: 'none' }, p);
  });
}

export function drawRing(svg) {
  const W = svg.clientWidth || window.innerWidth;
  const H = svg.clientHeight || window.innerHeight;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.replaceChildren();
  const rand = rng(7);

  const cx = W / 2;
  const cy = H / 2;
  const R = Math.min(H * 0.34, W * 0.3); // bán kính khoảng trời
  const hr = R * 0.2; // bán kính đầu
  const N = 9;
  const far = Math.hypot(W, H); // thân áo kéo ra tới mép màn hình

  const defs = el('defs', {}, svg);
  makePatterns(defs, hr);
  const sky = el('radialGradient', { id: 'sky', cx: '50%', cy: '50%', r: '50%' }, defs);
  el('stop', { offset: '0%', 'stop-color': '#ffffff' }, sky);
  el('stop', { offset: '55%', 'stop-color': '#e6f3fa' }, sky);
  el('stop', { offset: '100%', 'stop-color': '#a9d2ea' }, sky);
  // Mép trời gồ ghề như vết cọ.
  const brush = el('filter', { id: 'brush', x: '-10%', y: '-10%', width: '120%', height: '120%' }, defs);
  el('feTurbulence', { type: 'fractalNoise', baseFrequency: 0.035, numOctaves: 3, seed: 3, result: 'n' }, brush);
  el('feDisplacementMap', { in: 'SourceGraphic', in2: 'n', scale: hr * 0.5, xChannelSelector: 'R', yChannelSelector: 'G' }, brush);
  const blur = el('filter', { id: 'blur', x: '-50%', y: '-50%', width: '200%', height: '200%' }, defs);
  el('feGaussianBlur', { stdDeviation: hr * 0.35 }, blur);
  const shade = el('filter', { id: 'shade', x: '-20%', y: '-20%', width: '140%', height: '140%' }, defs);
  el('feDropShadow', { dx: 0, dy: hr * 0.06, stdDeviation: hr * 0.12, 'flood-color': '#1d3b22', 'flood-opacity': 0.35 }, shade);
  // Hạt giấy phủ toàn bộ cho cảm giác tranh vẽ tay.
  const grain = el('filter', { id: 'grain' }, defs);
  el('feTurbulence', { type: 'fractalNoise', baseFrequency: 0.85, numOctaves: 2, seed: 9 }, grain);
  el('feColorMatrix', { values: '0 0 0 0 0.1  0 0 0 0 0.15  0 0 0 0 0.1  0 0 0 -1.4 0.9' }, grain);

  // 1. Tán lá phủ kín màn hình.
  el('rect', { width: W, height: H, fill: '#2f5e2a' }, svg);
  const leaves = el('g', { class: 'leaves' }, svg);
  for (let i = 0; i < 1600; i++) {
    const x = rand() * (W + 80) - 40;
    const y = rand() * (H + 80) - 40;
    const len = (10 + rand() * 16) * (hr / 50);
    el('path', {
      d: `M0,${-len} C${len * 0.6},${-len * 0.4} ${len * 0.6},${len * 0.4} 0,${len} C${-len * 0.6},${len * 0.4} ${-len * 0.6},${-len * 0.4} 0,${-len} Z`,
      transform: `translate(${x},${y}) rotate(${rand() * 360})`,
      fill: LEAVES[Math.floor(rand() * LEAVES.length)],
      opacity: 0.7 + rand() * 0.3,
    }, leaves);
  }

  // 2. Khoảng trời tròn và mây.
  const skyG = el('g', { class: 'sky' }, svg);
  el('circle', { cx, cy, r: R + hr * 0.9, fill: 'url(#sky)', filter: 'url(#brush)' }, skyG);
  const clip = el('clipPath', { id: 'skyclip' }, defs);
  el('circle', { cx, cy, r: R + hr * 0.6 }, clip);
  const clouds = el('g', { class: 'clouds', 'clip-path': 'url(#skyclip)' }, skyG);
  const cloudInner = el('g', { filter: 'url(#blur)', opacity: 0.85 }, clouds);
  for (let i = 0; i < 6; i++) {
    const a = rand() * Math.PI * 2;
    const d = R * (0.3 + rand() * 0.6);
    el('ellipse', { cx: cx + Math.cos(a) * d, cy: cy + Math.sin(a) * d, rx: hr * (1.2 + rand() * 1.4), ry: hr * (0.45 + rand() * 0.3), fill: '#fff' }, cloudInner);
  }

  // 3. Vòng người. Hệ tọa độ riêng của mỗi người: gốc ở tâm, -y hướng ra ngoài.
  const ring = el('g', { class: 'ring' }, svg);
  const bodies = el('g', { filter: 'url(#shade)' }, ring);
  const arms = el('g', { filter: 'url(#shade)' }, ring);
  const heads = el('g', { filter: 'url(#shade)' }, ring);
  const headY = -(R + hr * 0.35);
  const y0 = headY - hr * 0.55; // vai
  const ws = hr * 1.85; // nửa bề rộng vai
  const half = Math.PI / N;
  const w1 = Math.tan(half) * far * 1.02;

  const people = Array.from({ length: N }, (_, i) => ({
    deg: (360 / N) * i + rand() * 6 - 3,
    shirt: i % SHIRTS.length,
    hair: HAIR[Math.floor(rand() * HAIR.length)],
    skin: SKIN[Math.floor(rand() * SKIN.length)],
    style: i % 5,
    tilt: rand() * 16 - 8,
  }));
  const frame = (p, parent) => el('g', { transform: `translate(${cx},${cy}) rotate(${p.deg})` }, parent);

  for (const p of people) {
    const g = frame(p, bodies);
    el('path', {
      d: `M${-ws},${y0 + hr * 0.25} Q${-ws * 1.05},${y0 - hr * 0.35} ${-ws * 0.75},${y0 - hr * 0.5} L${-w1},${-far} L${w1},${-far} L${ws * 0.75},${y0 - hr * 0.5} Q${ws * 1.05},${y0 - hr * 0.35} ${ws},${y0 + hr * 0.25} Z`,
      fill: `url(#shirt${p.shirt})`,
    }, g);
    // Cổ áo.
    el('path', { d: `M${-hr * 0.55},${y0 + hr * 0.1} Q0,${y0 - hr * 0.45} ${hr * 0.55},${y0 + hr * 0.1}`, stroke: '#00000022', 'stroke-width': hr * 0.12, fill: 'none' }, g);
  }

  // Tay phải khoác lên vai người bên cạnh (theo chiều kim đồng hồ).
  people.forEach((p, i) => {
    const q = people[(i + 1) % N];
    let dq = q.deg - p.deg;
    if (dq < 0) dq += 360;
    const g = frame(p, arms);
    const rad = (dq * Math.PI) / 180;
    // Vai trái người bên cạnh, quy về hệ tọa độ của p.
    const qx = -ws * 0.6;
    const qy = y0 - hr * 0.1;
    const hx = qx * Math.cos(rad) - qy * Math.sin(rad);
    const hy = qx * Math.sin(rad) + qy * Math.cos(rad);
    const sx = ws * 0.8;
    const sy = y0 - hr * 0.2;
    const ex = (sx + hx) / 2;
    const ey = (sy + hy) / 2 - hr * 0.35;
    const [bg] = SHIRTS[p.shirt];
    el('path', { d: `M${sx},${sy} Q${ex},${ey} ${hx},${hy}`, stroke: bg, 'stroke-width': hr * 0.62, 'stroke-linecap': 'round', fill: 'none' }, g);
    el('path', { d: `M${sx},${sy} Q${ex},${ey} ${hx},${hy}`, stroke: `url(#shirt${p.shirt})`, 'stroke-width': hr * 0.5, 'stroke-linecap': 'round', fill: 'none' }, g);
    el('circle', { cx: hx, cy: hy, r: hr * 0.3, fill: p.skin }, g);
  });

  // Đầu: tóc phía ngoài, cằm hướng vào tâm, mặt nhìn xuống người xem.
  for (const p of people) {
    const g = el('g', { transform: `translate(${cx},${cy}) rotate(${p.deg}) translate(0,${headY}) rotate(${p.tilt})` }, heads);
    const hair = p.hair;
    if (p.style === 1) { // tóc dài xõa hai bên
      for (const sd of [-1, 1]) el('path', { d: `M${sd * hr * 0.6},${-hr * 0.4} Q${sd * hr * 1.35},${hr * 0.2} ${sd * hr * 1.05},${hr * 0.95} Q${sd * hr * 0.85},${hr * 0.5} ${sd * hr * 0.7},${hr * 0.1} Z`, fill: hair }, g);
    }
    if (p.style === 3) el('circle', { cx: 0, cy: -hr * 1.05, r: hr * 0.42, fill: hair }, g); // búi tóc
    el('circle', { cx: 0, cy: -hr * 0.35, r: hr * 0.98, fill: hair }, g);
    el('ellipse', { cx: 0, cy: hr * 0.18, rx: hr * 0.82, ry: hr * 0.84, fill: p.skin }, g);
    // Mái tóc che trán.
    if (p.style === 4) { // mũ len
      el('path', { d: `M${-hr * 1.04},${-hr * 0.4} Q0,${-hr * 2.1} ${hr * 1.04},${-hr * 0.4} Z`, fill: '#e9c74f' }, g);
      el('rect', { x: -hr * 1.06, y: -hr * 0.62, width: hr * 2.12, height: hr * 0.3, rx: hr * 0.15, fill: '#d8b03c' }, g);
      el('circle', { cx: 0, cy: -hr * 1.3, r: hr * 0.22, fill: '#f3dc86' }, g);
    } else {
      el('path', { d: `M${-hr * 0.9},${-hr * 0.05} Q${-hr * 0.4},${-hr * 0.35} 0,${-hr * 0.05} Q${hr * 0.45},${-hr * 0.4} ${hr * 0.9},${-hr * 0.05} L${hr * 0.9},${-hr * 0.6} L${-hr * 0.9},${-hr * 0.6} Z`, fill: hair }, g);
    }
    // Mắt, má hồng, miệng (hướng vào tâm).
    for (const s of [-1, 1]) {
      el('path', { d: `M${s * hr * 0.42},${hr * 0.18} q${s * hr * 0.12},${hr * 0.1} ${s * hr * 0.24},0`, stroke: '#2a211d', 'stroke-width': hr * 0.07, 'stroke-linecap': 'round', fill: 'none' }, g);
      el('circle', { cx: s * hr * 0.48, cy: hr * 0.48, r: hr * 0.13, fill: '#f08f86', opacity: 0.45 }, g);
    }
    el('path', { d: `M${-hr * 0.17},${hr * 0.6} Q0,${hr * 0.75} ${hr * 0.17},${hr * 0.6}`, stroke: '#9a4a3a', 'stroke-width': hr * 0.07, 'stroke-linecap': 'round', fill: 'none' }, g);
  }

  // 4. Hạt giấy.
  el('rect', { class: 'grain', width: W, height: H, filter: 'url(#grain)', opacity: 0.35, 'pointer-events': 'none' }, svg);

  // Chuyển trời, vòng người và lớp hạt thành ảnh riêng: trình duyệt chỉ phải dịch/xoay ảnh
  // có sẵn trên GPU khi chuyển cảnh, không vẽ lại bóng đổ từng khung hình, nên mượt hơn nhiều.
  rasterizeLayers(svg, W, H, [['sky', 'art-sky'], ['ring', 'art-ring'], ['grain', 'art-grain']]);
}

function rasterizeLayers(svg, W, H, layers) {
  const host = document.getElementById('art-layers');
  if (!host) return;
  const defs = svg.querySelector('defs');
  const serializer = new XMLSerializer();
  const imgs = layers.map(([cls, imgClass]) => {
    const node = svg.querySelector(`:scope > .${cls}`);
    const doc = document.createElementNS(NS, 'svg');
    doc.setAttribute('xmlns', NS);
    doc.setAttribute('viewBox', `0 0 ${W} ${H}`);
    doc.setAttribute('width', W);
    doc.setAttribute('height', H);
    doc.append(defs.cloneNode(true), node.cloneNode(true));
    node.remove();
    const img = new Image(W, H);
    img.className = `art-layer ${imgClass}`;
    img.alt = '';
    img.decoding = 'sync';
    img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(serializer.serializeToString(doc));
    return img;
  });
  host.replaceChildren(...imgs);
}

// Điều hướng: trang chủ (cuộn dọc xem thông tin) → màn chat thử nghiệm, chuyển cảnh theo chiều sâu.
export function setupLanding({ onStart }) {
  const svg = document.getElementById('ring');
  const v = new URLSearchParams(location.search).get('v') || 'a';
  document.body.classList.add(`v-${v}`);
  if (v === 'b') document.body.classList.add('v-hands');
  const draw = { a: drawRing, b: drawHands }[v];
  let raf = 0;
  const redraw = () => { if (!draw) return; cancelAnimationFrame(raf); raf = requestAnimationFrame(() => draw(svg)); };
  redraw();
  window.addEventListener('resize', redraw);

  const track = document.getElementById('track');
  const panels = [...track.children];
  const home = panels[0];

  function go(step) {
    // Rời trang chủ: vòng người xoay văng ra, còn nền lá. Về trang chủ: vòng người xoay về.
    if (step > 0) home.scrollTo({ top: 0 });
    document.body.classList.add('animating');
    document.body.classList.toggle('opening', step > 0);
    track.dataset.step = String(step);
    // Đi "sâu vào trong": màn trước phóng to rồi mờ đi, màn sau hiện ra từ giữa.
    panels.forEach((p, i) => {
      p.inert = i !== step;
      p.dataset.pos = i < step ? 'before' : i > step ? 'after' : 'current';
    });
    clearTimeout(go.t);
    go.t = setTimeout(() => {
      document.body.classList.remove('animating');
      panels[step].querySelector('input, textarea, h2')?.focus({ preventScroll: true });
    }, 1150);
    if (step === 1) onStart();
  }

  document.getElementById('start').addEventListener('click', () => go(1));
  for (const b of document.querySelectorAll('[data-go]')) b.addEventListener('click', () => go(Number(b.dataset.go)));

  // Trang chủ: khi cuộn xuống phần thông tin thì làm mờ vòng người cho dễ đọc.
  home.addEventListener('scroll', () => {
    document.body.classList.toggle('reading', home.scrollTop > window.innerHeight * 0.35);
  }, { passive: true });
}
