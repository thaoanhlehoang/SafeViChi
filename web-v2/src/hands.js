// Phương án B: nhiều bàn tay chồng lên nhau ở bên phải, chữ ở bên trái. Tông xanh, vẽ bằng SVG.
const NS = 'http://www.w3.org/2000/svg';
const SLEEVES = [
  ['#2f6b3a', '#f3ecd6', 'stripes'], ['#9cc56a', '#5f8f35', 'plain'], ['#efe6cc', '#5f9a3c', 'check'],
  ['#3e7f5d', '#a8d4b4', 'dots'], ['#c9dfa0', '#ffffff', 'stripes'], ['#5f9a3c', '#d9ecb8', 'plain'],
  ['#e9d77a', '#c9a93a', 'check'], ['#8fc4a6', '#f6f2e4', 'dots'], ['#466b2e', '#7fae4a', 'stripes'],
  ['#b7d98b', '#2f6b3a', 'plain'], ['#1f4d2b', '#5f9a3c', 'dots'],
];
const SKIN = ['#f6d3b5', '#eec29e', '#e0ad86', '#c98f69', '#a86e4c', '#7d4f36'];

function rng(seed) { let s = seed >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32); }
function el(name, attrs = {}, parent) {
  const n = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.append(n);
  return n;
}

export function drawHands(svg) {
  const W = svg.clientWidth || window.innerWidth;
  const H = svg.clientHeight || window.innerHeight;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.replaceChildren();
  const rand = rng(11);
  const hs = Math.min(W, H) * 0.145; // cỡ bàn tay
  const C = [W * 0.69, H * 0.5];
  const far = Math.hypot(W, H);

  const defs = el('defs', {}, svg);
  SLEEVES.forEach(([bg, fg, kind], i) => {
    const t = hs * 0.28;
    const p = el('pattern', { id: `sl${i}`, width: t, height: t, patternUnits: 'userSpaceOnUse' }, defs);
    el('rect', { width: t, height: t, fill: bg }, p);
    if (kind === 'dots') el('circle', { cx: t / 2, cy: t / 2, r: t * 0.15, fill: fg, opacity: 0.8 }, p);
    if (kind === 'stripes') el('rect', { width: t * 0.4, height: t, fill: fg, opacity: 0.7 }, p);
    if (kind === 'check') { el('rect', { width: t, height: t * 0.2, fill: fg, opacity: 0.5 }, p); el('rect', { width: t * 0.2, height: t, fill: fg, opacity: 0.5 }, p); }
  });
  const shade = el('filter', { id: 'hshade', x: '-30%', y: '-30%', width: '160%', height: '160%' }, defs);
  el('feDropShadow', { dx: 0, dy: hs * 0.03, stdDeviation: hs * 0.06, 'flood-color': '#1d3b22', 'flood-opacity': 0.3 }, shade);
  const grain = el('filter', { id: 'hgrain' }, defs);
  el('feTurbulence', { type: 'fractalNoise', baseFrequency: 0.85, numOctaves: 2, seed: 4 }, grain);
  el('feColorMatrix', { values: '0 0 0 0 0.1  0 0 0 0 0.15  0 0 0 0 0.1  0 0 0 -1.4 0.9' }, grain);

  // Nền kem với các mảng xanh mềm, như giấy cắt.
  el('rect', { width: W, height: H, fill: '#f3f6ec' }, svg);
  el('path', { d: `M0,${H * 0.55} C${W * 0.18},${H * 0.42} ${W * 0.3},${H * 0.78} ${W * 0.46},${H * 0.7} S${W * 0.6},${H} ${W * 0.62},${H} L0,${H} Z`, fill: '#e2eed3' }, svg);
  el('path', { d: `M0,${H * 0.8} C${W * 0.1},${H * 0.7} ${W * 0.22},${H * 0.92} ${W * 0.3},${H} L0,${H} Z`, fill: '#cfe3b8' }, svg);
  el('path', { d: `M0,0 L${W * 0.34},0 C${W * 0.3},${H * 0.12} ${W * 0.12},${H * 0.2} 0,${H * 0.16} Z`, fill: '#e7f0dc' }, svg);
  el('circle', { cx: C[0], cy: C[1], r: hs * 3.2, fill: '#dcebc9', opacity: 0.6 }, svg);

  const N = 11;
  const g = el('g', { class: 'hands', filter: 'url(#hshade)' }, svg);
  const order = Array.from({ length: N }, (_, i) => i).sort(() => rand() - 0.5);
  for (const i of order) {
    // Tay đến từ phải, trên và dưới; chừa phía trái cho chữ.
    const deg = -112 + (224 * i) / (N - 1) + (rand() * 8 - 4);
    const a = (deg * Math.PI) / 180;
    const reach = hs * (0.45 + rand() * 0.65);
    const tx = C[0] + Math.cos(a) * reach;
    const ty = C[1] + Math.sin(a) * reach;
    const skin = SKIN[Math.floor(rand() * SKIN.length)];
    const line = { stroke: '#3b2418', 'stroke-opacity': 0.22, 'stroke-width': hs * 0.025 };
    const flip = rand() < 0.5 ? 1 : -1;
    const h = el('g', { transform: `translate(${tx},${ty}) rotate(${deg}) scale(1,${flip})` }, g);
    const wA = hs * 0.46;
    const L1 = hs * (2.1 + rand() * 0.6);
    el('rect', { x: L1, y: -wA * 0.68, width: far, height: wA * 1.36, rx: wA * 0.2, fill: `url(#sl${i % SLEEVES.length})` }, h); // tay áo
    el('rect', { x: L1 - hs * 0.08, y: -wA * 0.72, width: hs * 0.3, height: wA * 1.44, rx: wA * 0.12, fill: SLEEVES[i % SLEEVES.length][0] }, h); // cổ tay áo
    el('rect', { x: hs * 0.6, y: -wA * 0.5, width: L1 - hs * 0.5, height: wA, rx: wA * 0.45, fill: skin, ...line }, h); // cẳng tay
    el('ellipse', { cx: hs * 0.45, cy: 0, rx: hs * 0.58, ry: hs * 0.46, fill: skin, ...line }, h); // mu bàn tay
    const fingers = [[-0.3, 0.72], [-0.1, 0.82], [0.1, 0.78], [0.29, 0.62]];
    for (const [y, len] of fingers) {
      el('rect', { x: hs * (0.2 - len), y: hs * (y - 0.095), width: hs * (len + 0.2), height: hs * 0.19, rx: hs * 0.095, fill: skin, ...line }, h);
      el('path', { d: `M${hs * (0.2 - len * 0.45)},${hs * (y - 0.06)} l0,${hs * 0.12}`, stroke: '#00000018', 'stroke-width': hs * 0.02 }, h);
    }
    el('rect', { x: hs * -0.05, y: hs * 0.3, width: hs * 0.6, height: hs * 0.22, rx: hs * 0.11, fill: skin, ...line, transform: `rotate(-32 ${hs * 0.5} ${hs * 0.4})` }, h); // ngón cái
    const extra = rand();
    if (extra < 0.3) { // đồng hồ
      el('rect', { x: hs * 1.25, y: -wA * 0.55, width: hs * 0.16, height: wA * 1.1, fill: '#1f3324' }, h);
      el('circle', { cx: hs * 1.33, cy: 0, r: hs * 0.17, fill: '#f6f2e4', stroke: '#1f3324', 'stroke-width': hs * 0.04 }, h);
    } else if (extra < 0.5) { // vòng tay
      el('rect', { x: hs * 1.3, y: -wA * 0.54, width: hs * 0.07, height: wA * 1.08, fill: '#d8b03c', rx: hs * 0.03 }, h);
    }
  }
  el('rect', { width: W, height: H, filter: 'url(#hgrain)', opacity: 0.3, 'pointer-events': 'none' }, svg);
}
