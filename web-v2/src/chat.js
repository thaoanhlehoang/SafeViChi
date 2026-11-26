// Màn trò chuyện 3 cột: danh sách | khung chat | AI phân tích.
// - Luồng phân tích 2 tầm nhìn (cửa sổ 1–8 lượt + tách riêng tin mới nhất chống pha loãng lượt gộp cho cả 2 phía).
// - Luồng nhận tin (Người kia): chỉ tô bằng chứng verbatim và hiện dải cảnh báo (chưa thực thi hành động bảo vệ).
// - Mọi chữ người dùng gõ đều đi qua textContent / text node, không dùng innerHTML.

import { LABELS, MAX_LINES } from './labels.js';
import { SHIRTS, HAIR, SKIN } from './landing.js';
import { stripInvisibles } from './engine/dualorc.js';

// Hội thoại mẫu để thử nghiệm trực tiếp với mô hình thật (không lấy từ tập dữ liệu).
const CONVERSATIONS = [
  {
    id: 'toan12a', name: 'Nhóm học Toán 12A', group: true, sub: '5 thành viên',
    messages: [
      { from: 'them', author: 'Lan', text: 'Mai kiểm tra 15 phút chương 3 nha mọi người', time: '09:12' },
      { from: 'them', author: 'Huy', text: 'ok, ai có đề cương k gửi mình vs', time: '09:14' },
      { from: 'me', text: 'Để tối mình chụp gửi nhóm nhé', time: '09:20' },
      { from: 'them', author: 'Lan', text: 'cảm ơn bà nhiều 😊', time: '09:21' },
    ],
  },
  {
    id: 'tuan', name: 'Tuấn (lớp bên)', sub: 'Hoạt động 5 phút trước',
    messages: [
      { from: 'them', text: 'chào e, a là Tuấn lớp 11A2', time: '20:15' },
      { from: 'me', text: 'dạ chào anh', time: '20:20' },
      { from: 'them', text: 'a đọc bài văn của e trên trang trường, hay ghê', time: '20:22' },
      { from: 'me', text: 'dạ em cảm ơn anh ạ', time: '20:30' },
      { from: 'them', text: 'hôm nào a mời e trà sữa nhé', time: '20:31' },
      { from: 'them', text: 'e ơi tối nay rảnh k', time: '21:02' },
      { from: 'me', text: 'em bận học rồi ạ', time: '21:05' },
      { from: 'them', text: 'sao ko rep anh nhanh thế', time: '21:30' },
      { from: 'them', text: 'gửi ảnh cho a xem đi, a k nói ai đâu', time: '21:31' },
      { from: 'them', text: 'đi chơi với anh 1 hôm thôi mà', time: '21:40' },
    ],
  },
  {
    id: 'hoamai', name: 'Shop Hoa Mai', sub: 'Trang bán hàng',
    messages: [
      { from: 'them', text: 'Chúc mừng quý khách đã trúng thưởng điện thoại từ chương trình tri ân', time: '14:03' },
      { from: 'them', text: 'Vui lòng bấm vào link http://hoamai-tri-an.example để nhận quà', time: '14:03' },
      { from: 'me', text: 'thật ạ? em phải làm gì', time: '14:10' },
      { from: 'them', text: 'Bạn đọc giúp shop mã OTP vừa gửi về điện thoại để xác nhận nhé', time: '14:11' },
    ],
  },
  {
    id: 'nguoila', name: 'Người lạ', sub: 'Không có trong danh bạ',
    messages: [
      { from: 'them', text: 'm là đứa đăng bài hôm qua đúng k', time: '22:47' },
      { from: 'me', text: 'bạn là ai vậy', time: '22:50' },
      { from: 'them', text: 'm liệu hồn, k trả tiền thì t tung ảnh của m lên nhóm trường', time: '22:52' },
    ],
  },
  {
    id: 'minh', name: 'Minh', sub: 'Đang hoạt động',
    messages: [
      { from: 'me', text: 'hôm nay t quên mang vở, cho t mượn chép bài nhé', time: '16:20' },
      { from: 'them', text: 'lại quên à, đồ n.g.u', time: '16:24' },
      { from: 'them', text: 'óc chó thật sự 🙄', time: '16:24' },
    ],
  },
  {
    id: 'me', name: 'Mẹ', sub: 'Đang hoạt động',
    messages: [
      { from: 'them', text: 'Con nhớ ăn sáng rồi hẵng đi học nhé', time: '06:40' },
      { from: 'me', text: 'dạ con biết rồi ạ', time: '06:42' },
    ],
  },
];

// Mốc thời gian để xếp danh sách chat: tin mới nhất lên đầu.
// Hội thoại mẫu rải ở mấy ngày trước (giờ theo tin cuối); tin gửi trong buổi demo mang giờ thật.
const SEED_DAYS_AGO = { nguoila: 1, tuan: 1, minh: 2, hoamai: 3, toan12a: 4, me: 5 };
for (const conv of CONVERSATIONS) {
  const d = new Date();
  d.setDate(d.getDate() - (SEED_DAYS_AGO[conv.id] ?? 1));
  const [hh, mm] = (conv.messages.at(-1)?.time || '00:00').split(':').map(Number);
  conv.lastAt = new Date(d.getFullYear(), d.getMonth(), d.getDate(), hh, mm).getTime();
}
const byRecent = () => CONVERSATIONS.sort((a, b) => b.lastAt - a.lastAt);
const WEEKDAY = ['CN', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7'];
// Hôm nay: giờ; trong 6 ngày gần nhất: thứ (T2…CN); cũ hơn: ngày/tháng.
function contactTime(ts) {
  const d = new Date(ts);
  const now = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  if (d.toDateString() === now.toDateString()) return `${pad(d.getHours())}:${pad(d.getMinutes())}`;
  const startToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  if (startToday - ts < 6 * 864e5) return WEEKDAY[d.getDay()];
  return `${pad(d.getDate())}/${pad(d.getMonth() + 1)}`;
}
byRecent();

const ORDER = ['AN_TOAN', 'XUC_PHAM', 'QUAY_ROI', 'DE_DOA', 'LUA_DAO'];
const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

export function setupChat({ classify, onEngineStatus, retryLoad }) {
  let current = CONVERSATIONS[0];
  let view = null; // { conv, msg } tin đang xem ở cột AI
  let sender = 'me';
  let draft = null; // { conv, text, from, ctxAnalysis, warnAnalysis, source: 'context'|'standalone' }
  let ready = false;
  let busy = false;
  let engineError = null;
  let customCount = 0;
  const inflightMsg = new WeakMap();

  const shell = $('chat-shell');
  const list = $('contact-list');
  const box = $('messages');
  const input = $('composer-input');
  const sendBtn = $('composer-send');
  const composer = $('composer');
  const warn = $('send-warn');
  const safetyBar = $('safety-bar');
  const aiOpen = () => !shell.classList.contains('ai-closed');

  const who = (conv, m) => (m.from === 'me' ? 'Bạn' : m.author || conv.name);

  // Đúng quy ước dữ liệu SafeViChi v2 (CLAUDE.md):
  // gộp các tin liên tiếp theo phe ('me' vs 'them') để luôn luân phiên 2 phía (word_turn % 2),
  // giữ tối đa 8 lượt gần nhất và lưu vị trí từng bóng tin gốc để tô bằng chứng chính xác.
  // Nếu trong cùng một lượt gộp đã có tin bị đánh dấu độc hại đơn lẻ (soloFlag), cắt bỏ các tin an toàn
  // đứng trước nó trong cùng lượt đó để tránh tiếp tục pha loãng lượt ở các bước suy luận sau.
  function buildTurnWindow(conv, idx, draftText) {
    const rawItems = conv.messages.slice(0, idx).map((m, i) => ({ i, m }));
    if (draftText != null) {
      rawItems.push({
        i: -1,
        m: {
          from: sender,
          author: sender === 'them' && conv.group ? 'Thành viên' : undefined,
          time: 'chưa gửi',
          text: draftText,
        },
      });
    } else if (conv.messages[idx]) {
      rawItems.push({ i: idx, m: conv.messages[idx] });
    }

    const turns = [];
    for (const item of rawItems) {
      const party = item.m.from === 'me' ? 'me' : 'them';
      const authorLabel = who(conv, item.m);
      const prev = turns[turns.length - 1];
      if (prev && prev.party === party) {
        const start = prev.text.length + 1;
        prev.text += ` ${item.m.text}`;
        prev.time = item.m.time;
        if (!prev.authors.includes(authorLabel)) prev.authors.push(authorLabel);
        prev.who = prev.authors.join(', ');
        prev.parts.push({ i: item.i, m: item.m, start, end: prev.text.length });
      } else {
        turns.push({
          party,
          authors: [authorLabel],
          who: authorLabel,
          time: item.m.time,
          text: item.m.text,
          parts: [{ i: item.i, m: item.m, start: 0, end: item.m.text.length }],
        });
      }
    }

    const win = turns.slice(-MAX_LINES);
    for (const t of win) {
      let cutAt = -1;
      for (let k = t.parts.length - 1; k >= 0; k--) {
        if (t.parts[k].m.soloFlag) cutAt = k;
      }
      if (cutAt > 0) {
        const kept = t.parts.slice(cutAt);
        let text = '';
        const authors = [];
        const newParts = [];
        for (const p of kept) {
          const start = text ? text.length + 1 : 0;
          text = text ? `${text} ${p.m.text}` : p.m.text;
          const label = who(conv, p.m);
          if (!authors.includes(label)) authors.push(label);
          newParts.push({ ...p, start, end: text.length });
        }
        t.text = text;
        t.authors = authors;
        t.who = authors.join(', ');
        t.parts = newParts;
      }
    }
    return win;
  }

  function packAnalysis(turns, out, extras = {}) {
    const bubbleSpans = new Map();
    for (const t of turns) {
      for (const p of t.parts) bubbleSpans.set(p.i, []);
    }

    const rows = turns.map((t, dong) => {
      const spans = out.bang_chung.filter((b) => b.dong === dong);
      for (const sp of spans) {
        for (const p of t.parts) {
          const ovStart = Math.max(sp.start, p.start);
          const ovEnd = Math.min(sp.end, p.end);
          if (ovEnd > ovStart) {
            const localStart = ovStart - p.start;
            const localEnd = ovEnd - p.start;
            bubbleSpans.get(p.i).push({
              start: localStart,
              end: localEnd,
              text: p.m.text.slice(localStart, localEnd),
              nhan: out.nhan,
            });
          }
        }
      }
      return {
        who: t.who,
        time: t.time,
        text: t.text,
        msgIndices: t.parts.map((p) => p.i),
        // Vị trí từng tin gốc trong lượt gộp, để cột phân tích tách mỗi tin một dòng (chỉ hiển thị).
        partRanges: t.parts.map((p) => [p.start, p.end]),
        spans,
      };
    });

    return {
      nhan: out.nhan,
      xac_suat: out.xac_suat,
      rui_ro: out.rui_ro,
      ms: out.ms,
      scope: 'context',
      rows,
      bubbleSpans,
      ...extras,
    };
  }

  // Kết hợp kết quả kiểm tra đơn câu vào cấu trúc cửa sổ ngữ cảnh khi tin mới nhất bị pha loãng bởi
  // các tin an toàn trước đó (cả khi đứng sau nhiều lượt an toàn lẫn khi bị gộp chung vào 1 lượt).
  function elevateWithStandalone(ctxRes, soloOut, targetPart) {
    const soloSpansInMsg = soloOut.bang_chung
      .filter((b) => b.dong === 0)
      .map((b) => ({
        start: b.start,
        end: b.end,
        text: targetPart.m.text.slice(b.start, b.end),
      }));

    const rows = ctxRes.rows.map((r, i) => {
      if (i !== ctxRes.rows.length - 1) return r;
      const shifted = soloSpansInMsg.map((s) => ({
        dong: i,
        start: targetPart.start + s.start,
        end: targetPart.start + s.end,
        text: r.text.slice(targetPart.start + s.start, targetPart.start + s.end),
      }));
      return { ...r, spans: shifted };
    });

    const bubbleSpans = new Map(ctxRes.bubbleSpans);
    bubbleSpans.set(targetPart.i, soloSpansInMsg.map((sp) => ({ ...sp, nhan: soloOut.nhan })));

    return {
      nhan: soloOut.nhan,
      xac_suat: soloOut.xac_suat,
      rui_ro: soloOut.rui_ro,
      ms: ctxRes.ms + soloOut.ms,
      scope: 'message',
      rows,
      bubbleSpans,
    };
  }

  async function analyze(conv, idx, draftText = null, { priority = false } = {}) {
    const turns = buildTurnWindow(conv, idx, draftText);
    const ctxOut = await classify(turns.map((t) => t.text), { priority });
    const ctxRes = packAnalysis(turns, ctxOut);

    // Chống pha loãng cho CẢ 2 phía (Tôi & Người kia):
    // Nếu ngữ cảnh gộp trả AN_TOAN nhưng cửa sổ có > 1 tin nhắn (nhiều lượt HOẶC nhiều tin cùng phía
    // bị gộp thành 1 lượt), kiểm tra thêm riêng tin mới nhất (~20–45 ms).
    const windowMsgCount = turns.reduce((sum, t) => sum + t.parts.length, 0);
    if (ctxRes.nhan === 'AN_TOAN' && windowMsgCount > 1) {
      const lastTurn = turns[turns.length - 1];
      const targetPart = lastTurn.parts[lastTurn.parts.length - 1];
      if (targetPart?.m?.text && stripInvisibles(targetPart.m.text).trim()) {
        const soloOut = await classify([targetPart.m.text], { priority });
        if (soloOut.nhan !== 'AN_TOAN') {
          return elevateWithStandalone(ctxRes, soloOut, targetPart);
        }
      }
    }

    return ctxRes;
  }

  function applySpans(conv, res) {
    for (const [msgIdx, spans] of res.bubbleSpans.entries()) {
      if (msgIdx >= 0 && conv.messages[msgIdx]) {
        const msg = conv.messages[msgIdx];
        msg.spans = spans.length > 0 ? spans : (msg.soloSpans || []);
      }
    }
  }

  async function ensureAnalyzed(conv, idx, { priority = false, updateSpans = false } = {}) {
    const m = conv.messages[idx];
    if (!m) return null;
    if (m.analysis) return m.analysis;
    if (inflightMsg.has(m)) return inflightMsg.get(m);

    const task = analyze(conv, idx, null, { priority })
      .then((res) => {
        m.analysis = res;
        if (res.scope === 'message') {
          m.soloFlag = true;
          m.soloSpans = res.bubbleSpans.get(idx) || [];
        }
        if (updateSpans || idx === conv.messages.length - 1) applySpans(conv, res);
        return res;
      })
      .finally(() => {
        inflightMsg.delete(m);
      });

    inflightMsg.set(m, task);
    return task;
  }

  async function analyzeLatestConversations() {
    const order = [current, ...CONVERSATIONS.filter((c) => c !== current)];
    for (let k = 0; k < order.length; k++) {
      const conv = order[k];
      const lastIdx = conv.messages.length - 1;
      if (lastIdx < 0) continue;
      try {
        await ensureAnalyzed(conv, lastIdx, { priority: k === 0, updateSpans: true });
      } catch {
        continue;
      }
      renderContacts();
      if (conv === current) {
        if (!view || view.conv !== current) view = { conv, msg: conv.messages[lastIdx] };
        renderMessages();
        renderSafetyBar();
        renderView();
      }
    }
  }

  function syncComposerState() {
    composer.classList.toggle('busy', busy);
    input.readOnly = busy;
    const disabled = !ready || busy;
    sendBtn.setAttribute('aria-disabled', String(disabled));

    if (!ready) {
      // Lúc đang tải, hộp tải mô hình đã che màn chat, ô nhập để nguyên như bình thường.
      input.placeholder = 'Nhập tin nhắn…';
      sendBtn.replaceChildren('Gửi ', el('kbd', '', '↵'));
    } else if (busy) {
      sendBtn.textContent = 'Đang kiểm tra…';
    } else {
      input.placeholder = sender === 'me' ? 'Nhập tin nhắn với tư cách Tôi…' : `Nhập tin nhắn giả lập từ ${current.name}…`;
      sendBtn.replaceChildren('Gửi ', el('kbd', '', '↵'));
    }
  }

  // Hộp tải mô hình: khóa cả màn chat (inert), chỉ còn nút về Trang chủ, cho tới khi mô hình sẵn sàng.
  const loader = $('model-loader');
  const lockedCols = [...shell.querySelectorAll('.col-contacts, .col-chat, .col-ai')];
  const MB = (b) => (b / 1048576).toFixed(0);
  function updateLoader(s) {
    const locked = s.kind === 'loading' || s.kind === 'error';
    lockedCols.forEach((c) => { c.inert = locked; });
    if (!locked) {
      loader.classList.add('done');
      setTimeout(() => { if (loader.classList.contains('done')) loader.hidden = true; }, 350);
      return;
    }
    loader.hidden = false;
    loader.classList.remove('done');
    const failed = s.kind === 'error';
    loader.classList.toggle('failed', failed);
    $('loader-title').textContent = failed ? 'Không tải được mô hình' : 'Đang tải mô hình';
    $('loader-retry').hidden = !failed;
    if (failed) {
      $('loader-stage').textContent = s.message || 'Đã có lỗi khi tải mô hình.';
      return;
    }
    const total = s.total > 1 ? s.total : 0;
    const pct = total ? Math.min(100, Math.round((100 * s.loaded) / total)) : 0;
    $('loader-stage').textContent = s.stage || 'Đang tải…';
    $('loader-bar').style.setProperty('--p', pct / 100);
    $('loader-bar').setAttribute('aria-valuenow', String(pct));
    $('loader-pct').textContent = `${pct}%`;
    $('loader-bytes').textContent = total ? `${MB(s.loaded)} / ${MB(total)} MB` : '';
  }
  $('loader-retry').addEventListener('click', () => { if (typeof retryLoad === 'function') retryLoad(); });

  onEngineStatus((s) => {
    updateLoader(s);
    const badge = $('ai-engine');
    badge.textContent = s.text;
    badge.dataset.kind = s.kind;
    if (s.kind === 'error') {
      engineError = s.message || 'Không thể nạp mô hình ONNX.';
      syncComposerState();
      renderView();
      return;
    }
    engineError = null;
    if (s.kind === 'real' && !ready) {
      ready = true;
      syncComposerState();
      analyzeLatestConversations();
    } else if (s.kind === 'loading' && !ready) {
      syncComposerState();
    }
  });

  // Trả về nhãn nguy hại hiện hành dựa trên kết quả phân tích mới nhất của hội thoại (tránh kẹt chấm cũ).
  function avatar(name) {
    let h = 0;
    for (const ch of name) h = (h * 31 + ch.codePointAt(0)) >>> 0;
    const shirt = SHIRTS[h % SHIRTS.length][0];
    const hair = HAIR[(h >>> 3) % HAIR.length];
    const skin = SKIN[(h >>> 6) % SKIN.length];
    const NS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', '0 0 40 40');
    svg.setAttribute('aria-hidden', 'true');
    const add = (tag, attrs) => {
      const n = document.createElementNS(NS, tag);
      for (const k in attrs) n.setAttribute(k, attrs[k]);
      svg.append(n);
    };
    add('rect', { width: 40, height: 40, fill: '#cddbbd' });
    add('path', { d: 'M5 42c1-9 7-13 15-13s14 4 15 13z', fill: shirt });
    add('rect', { x: 17, y: 23, width: 6, height: 7, rx: 2, fill: skin });
    add('circle', { cx: 20, cy: 17, r: 8, fill: skin });
    if (h % 3 === 0) add('path', { d: 'M11.5 18c-1.5-8 4-11.5 8.5-11.5s10 3 8.5 11.5c-1-4-3-6-8.5-6.5-5 .5-7.5 2.5-8.5 6.5z', fill: hair });
    else if (h % 3 === 1) add('path', { d: 'M11 22c-2-10 3-14 9-14s11 4 9 14c-.5-5-2.5-8.5-9-9-6.5.5-8.5 4-9 9z', fill: hair });
    else add('path', { d: 'M12 15.5c0-5 3.5-8 8-8s8 3 8 8c-2.5-2.5-5-3.5-8-3.5s-5.5 1-8 3.5z', fill: hair });
    const a = el('span', 'avatar');
    a.append(svg);
    return a;
  }

  function renderContacts() {
    const q = $('contact-search').value.trim().toLowerCase();
    list.replaceChildren(...byRecent()
      .filter((conv) => !q || conv.name.toLowerCase().includes(q))
      .map((conv) => {
        const li = el('li');
        const b = el('button', 'contact');
        b.type = 'button';
        b.setAttribute('aria-current', conv === current ? 'true' : 'false');
        const last = conv.messages.at(-1);
        const body = el('span', 'contact-body');
        const top = el('span', 'contact-top');
        top.append(el('span', 'contact-name', conv.name), el('span', 'contact-time', !last ? 'Mới' : contactTime(conv.lastAt)));
        const prevText = last
          ? (last.from === 'me' ? 'Bạn: ' : last.author ? `${last.author}: ` : '') + last.text
          : 'Chưa có tin nhắn nào';
        const prev = el('span', 'contact-preview', prevText);
        body.append(top, prev);
        b.append(avatar(conv.name), body);

        // Chấm màu theo nhãn của tin cuối (xanh lá = an toàn), chỉ khi đã phân tích xong.
        const nhan = last?.analysis?.nhan;
        if (LABELS[nhan]) {
          const info = LABELS[nhan];
          const dot = el('span', 'contact-flag');
          dot.style.setProperty('--c', info.mau);
          dot.setAttribute('aria-hidden', 'true');
          b.append(dot, el('span', 'visually-hidden', nhan === 'AN_TOAN' ? 'An toàn' : `Cảnh báo: ${info.ten}`));
        }
        b.onclick = () => openConversation(conv);
        li.append(b);
        return li;
      }));
  }

  function createNewConversation() {
    customCount++;
    const conv = {
      id: `custom-${customCount}`,
      name: customCount === 1 ? 'Hội thoại mới' : `Hội thoại mới ${customCount}`,
      custom: true,
      messages: [],
      lastAt: Date.now(),
    };
    CONVERSATIONS.unshift(conv);
    openConversation(conv);
    input.focus();
  }

  function selectMessage(conv, idx) {
    const m = conv.messages[idx];
    if (!m) return;
    // Bấm xem tin cũ sẽ ẩn cảnh báo nháp để cột AI phản hồi ngay đúng tin vừa chọn.
    if (draft) clearDraft();
    view = { conv, msg: m };
    document.querySelector('.ai-body').scrollTop = 0;
    openAI(true);
    renderSafetyBar();
    renderView();
    if (ready && !m.analysis) {
      ensureAnalyzed(conv, idx, { priority: true, updateSpans: idx === conv.messages.length - 1 })
        .then(() => {
          if (conv === current) {
            renderMessages();
            renderSafetyBar();
            renderView();
          }
          renderContacts();
        })
        .catch(() => {});
    }
  }

  function openConversation(conv) {
    current = conv;
    input.value = '';
    clearDraft();
    const lastIdx = conv.messages.length - 1;
    view = lastIdx >= 0 ? { conv, msg: conv.messages[lastIdx] } : null;
    document.querySelector('.ai-body').scrollTop = 0;
    renderContacts();
    renderHeader();
    syncComposerState();
    renderMessages();
    renderSafetyBar();
    renderView();
    if (ready && lastIdx >= 0 && !conv.messages[lastIdx].analysis) {
      ensureAnalyzed(conv, lastIdx, { priority: true, updateSpans: true })
        .then(() => {
          if (conv === current) {
            renderMessages();
            renderSafetyBar();
            renderView();
          }
          renderContacts();
        })
        .catch(() => {});
    }
  }

  function renderHeader() {
    $('chat-avatar').replaceWith(Object.assign(avatar(current.name), { id: 'chat-avatar' }));
    $('chat-name').textContent = current.name;
    $('chat-sub').textContent = current.sub;
  }

  // <mark> bằng chứng, tô nền theo màu của nhãn độc hại.
  function markEl(text, nhan) {
    const mk = el('mark', 'evidence-mark', text);
    if (LABELS[nhan]) mk.style.setProperty('--c', LABELS[nhan].mau);
    return mk;
  }

  // Chèn chữ thường và <mark> bằng chứng xen kẽ, theo vị trí trên câu gốc (100% text node, chống XSS).
  function highlighted(text, spans) {
    const frag = document.createDocumentFragment();
    let pos = 0;
    for (const s of [...spans].sort((a, b) => a.start - b.start)) {
      if (s.start < pos) continue;
      frag.append(text.slice(pos, s.start));
      frag.append(markEl(text.slice(s.start, s.end), s.nhan));
      pos = s.end;
    }
    frag.append(text.slice(pos));
    return frag;
  }

  function windowSet() {
    const res = draft ? draft.analysis : view?.msg?.analysis;
    if (!res) return null;
    const set = new Set();
    for (const r of res.rows) {
      for (const idx of r.msgIndices) if (idx >= 0) set.add(idx);
    }
    return set;
  }

  function renderMessages() {
    if (current.messages.length === 0) {
      const empty = el('li', 'messages-empty');
      empty.append(
        el('p', 'messages-empty-title', 'Hội thoại trống'),
        el('p', 'messages-empty-desc', 'Nhập tin nhắn bên dưới để kiểm tra trực tiếp bằng SafeViChi v2. Bạn có thể chuyển đổi giữa "Tôi" và "Người kia" để thử hội thoại nhiều lượt.'),
      );
      box.replaceChildren(empty);
      box.dataset.conv = current.id;
      return;
    }

    const win = aiOpen() ? windowSet() : null;
    const items = [];
    let prevAuthor = null;
    current.messages.forEach((m, i) => {
      const li = el('li', `msg ${m.from}`);
      if (win && !win.has(i)) li.classList.add('out-ctx');
      if (m.from === 'them' && current.group && m.author !== prevAuthor) li.append(el('span', 'msg-author', m.author));
      prevAuthor = m.from === 'them' ? m.author : null;
      const bubble = el('button', 'bubble');
      bubble.type = 'button';
      bubble.append(highlighted(m.text, m.spans || []));
      bubble.setAttribute('aria-pressed', !draft && view?.msg === m ? 'true' : 'false');
      bubble.onclick = () => selectMessage(current, i);
      li.append(bubble, el('span', 'msg-time', m.time));
      items.push(li);
    });
    const keep = box.scrollHeight - box.scrollTop - box.clientHeight < 40 || box.dataset.conv !== current.id;
    box.replaceChildren(...items);
    box.dataset.conv = current.id;
    if (keep) box.scrollTop = box.scrollHeight;
  }

  // Chỉ hiện dải cảnh báo gọn khi hội thoại có dấu hiệu nguy hại (chưa thực thi hành động bảo vệ).
  function renderSafetyBar() {
    const last = current.messages.at(-1);
    const analysis = last?.analysis;
    const nhan = analysis?.nhan;
    const dismissedAt = current.dismissedSafetyAt ?? -1;

    if (draft || !analysis || !nhan || nhan === 'AN_TOAN' || dismissedAt === current.messages.length) {
      safetyBar.hidden = true;
      safetyBar.replaceChildren();
      return;
    }

    const info = LABELS[nhan];
    safetyBar.hidden = false;
    safetyBar.dataset.label = nhan;
    safetyBar.style.setProperty('--c', info.mau);

    const warnCopy = {
      XUC_PHAM: 'Phát hiện nội dung có dấu hiệu xúc phạm trong hội thoại.',
      QUAY_ROI: 'Phát hiện nội dung có dấu hiệu quấy rối trong hội thoại.',
      DE_DOA: 'Phát hiện nội dung có dấu hiệu đe dọa / tống tiền trong hội thoại.',
      LUA_DAO: 'Phát hiện nội dung có dấu hiệu lừa đảo trong hội thoại.',
    };

    const body = el('div', 'safety-body');
    const title = el('p', 'safety-title');
    title.append(el('strong', 'safety-label', info.ten), ` · ${warnCopy[nhan]}`);
    body.append(title);

    const closeBtn = el('button', 'safety-close', '×');
    closeBtn.type = 'button';
    closeBtn.setAttribute('aria-label', 'Ẩn cảnh báo');
    closeBtn.onclick = () => {
      current.dismissedSafetyAt = current.messages.length;
      renderSafetyBar();
    };

    safetyBar.replaceChildren(body, closeBtn);
  }

  function renderView() {
    if (draft) {
      renderAI(draft.analysis);
    } else if (view?.msg) {
      renderAI(view.msg.analysis);
    } else {
      renderAI(null);
    }
  }

  function renderAI(res, kicker) {
    $('ai-kicker').textContent = kicker;
    const ctx = $('ai-context');
    const verdict = $('ai-verdict');
    const probs = $('ai-probs');
    const ev = $('ai-evidence');
    if (!res) {
      $('ai-summary').replaceChildren();
      const emptyMsg = !ready
        ? ''
        : current.messages.length === 0
          ? ''
          : 'Đang phân tích…';
      ctx.replaceChildren(el('li', 'ai-empty', emptyMsg));
      verdict.replaceChildren();
      probs.replaceChildren();
      ev.replaceChildren();
      return;
    }

    const n = res.rows.length;
    const top = LABELS[res.nhan];
    const sChip = el('span', 'label-chip', top.ten);
    sChip.style.setProperty('--c', top.mau);
    $('ai-summary').style.setProperty('--c', top.mau);
    $('ai-summary').replaceChildren(sChip, el('span', 'summary-pct', `${(100 * res.xac_suat[res.nhan]).toFixed(1)}%`));

    ctx.replaceChildren(...res.rows.map((r, k) => {
      const li = el('li', 'ctx-row');
      if (k === n - 1) li.classList.add('is-new');
      if (r.spans.length) li.classList.add('has-ev');
      const head = el('div', 'ctx-head');
      head.append(el('span', 'ctx-who', r.who), el('span', 'ctx-time', r.time));
      li.append(head);
      // Mô hình nhận cả lượt nối bằng dấu cách (đúng quy ước dữ liệu huấn luyện v2);
      // ở đây tách lại mỗi tin một dòng cho dễ đọc.
      const ranges = r.partRanges?.length ? r.partRanges : [[0, r.text.length]];
      for (const [a, b] of ranges) {
        li.append(el('p', 'ctx-text', r.text.slice(a, b)));
      }
      return li;
    }));

    verdict.replaceChildren();

    probs.replaceChildren(...ORDER.map((k) => {
      const li = el('li');
      li.style.setProperty('--c', LABELS[k].mau);
      li.style.setProperty('--p', res.xac_suat[k]);
      if (k === res.nhan) li.className = 'top';
      const bar = el('span', 'bar');
      bar.append(el('span'));
      li.append(el('span', 'prob-name', LABELS[k].ten), bar, el('span', 'prob-val', `${(100 * res.xac_suat[k]).toFixed(1)}`));
      return li;
    }));

    // Cụm bằng chứng trải qua nhiều tin trong cùng lượt gộp: tách theo từng tin, mỗi tin một dòng.
    const evs = res.rows.flatMap((r) => r.spans.map((sp) => {
      const ranges = r.partRanges?.length ? r.partRanges : [[0, r.text.length]];
      const pieces = ranges
        .map(([a, b]) => r.text.slice(Math.max(sp.start, a), Math.min(sp.end, b)).trim())
        .filter(Boolean);
      return pieces.length ? pieces : [sp.text];
    }));
    ev.style.setProperty('--c', LABELS[res.nhan].mau);
    ev.replaceChildren(...(evs.length
      ? evs.map((pieces) => {
        const li = el('li');
        for (const t of pieces) li.append(el('span', 'evidence-text', t));
        return li;
      })
      : [el('li', 'ai-empty', res.nhan === 'AN_TOAN'
        ? 'Không thấy dấu hiệu độc hại trong đoạn này.'
        : 'Mô hình cảnh báo theo ngữ cảnh chung, không khoanh cụm từ rời rạc.')]));
  }

  const nowTime = () => {
    const d = new Date();
    return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
  };

  function clearDraft() {
    draft = null;
    warn.hidden = true;
    $('warn-send').hidden = false;
    $('warn-edit').textContent = 'Chỉnh sửa';
  }

  function commit(conv, text, from, analysis) {
    const m = { from, text, time: nowTime() };
    if (from === 'them' && conv.group) m.author = 'Thành viên';
    conv.messages.push(m);
    conv.lastAt = Date.now();
    const newIdx = conv.messages.length - 1;

    const rows = analysis.rows.map((r, i) => (
      i === analysis.rows.length - 1
        ? { ...r, time: m.time, msgIndices: r.msgIndices.map((idx) => (idx === -1 ? newIdx : idx)) }
        : r
    ));
    const bubbleSpans = new Map();
    for (const [k, v] of analysis.bubbleSpans.entries()) {
      bubbleSpans.set(k === -1 ? newIdx : k, v);
    }

    const finalAnalysis = { ...analysis, rows, bubbleSpans };
    m.analysis = finalAnalysis;
    if (finalAnalysis.scope === 'message') {
      m.soloFlag = true;
      m.soloSpans = bubbleSpans.get(newIdx) || [];
    }
    applySpans(conv, finalAnalysis);
    view = { conv, msg: m };
    if (conv === current) {
      renderMessages();
      renderSafetyBar();
      renderView();
      box.scrollTop = box.scrollHeight;
    }
    renderContacts();
  }

  function showWarning(res) {
    const info = LABELS[res.nhan];
    const label = el('strong', 'warn-label', info.ten);
    label.style.setProperty('--c', info.mau);
    $('warn-title').replaceChildren('Tin nhắn này có dấu hiệu ', label);

    const desc = $('warn-desc');
    const draftSpans = res.bubbleSpans.get(-1) || [];
    if (draftSpans.length > 0) {
      desc.replaceChildren('Cụm từ bị đánh dấu: ');
      draftSpans.slice(0, 3).forEach((sp, i) => {
        if (i > 0) desc.append(', ');
        desc.append(markEl(sp.text, sp.nhan || res.nhan));
      });
      desc.append('. Bạn có muốn sửa trước khi gửi không?');
    } else {
      desc.textContent = 'Bạn có muốn sửa trước khi gửi không?';
    }
    $('warn-send').hidden = false;
    $('warn-edit').textContent = 'Chỉnh sửa';
    warn.hidden = false;
    renderSafetyBar();
    $('warn-edit').focus();
  }

  function showSubmitError(errMsg) {
    $('warn-title').textContent = 'Không thể phân tích tin nhắn';
    $('warn-desc').textContent = errMsg;
    $('warn-send').hidden = true;
    $('warn-edit').textContent = 'Đóng';
    warn.hidden = false;
    renderSafetyBar();
    $('warn-edit').focus();
  }

  input.addEventListener('input', () => {
    if (draft || !warn.hidden) {
      clearDraft();
      renderMessages();
      renderSafetyBar();
      renderView();
    }
  });

  composer.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!ready || busy) return;

    const text = input.value.trim().slice(0, 300);
    if (!text || !stripInvisibles(text).trim()) return;

    // Nếu người dùng nhấn Enter lại trên chính bản nháp đang bị cảnh báo, không chạy lại ONNX thừa.
    if (draft && draft.conv === current && draft.text === text && draft.from === sender) {
      $('warn-edit').focus();
      return;
    }

    const conv = current;
    const from = sender;
    const prevNhan = conv.messages.at(-1)?.analysis?.nhan || 'AN_TOAN';

    busy = true;
    syncComposerState();

    let res;
    try {
      res = await analyze(conv, conv.messages.length, text, { priority: true });
    } catch (err) {
      busy = false;
      syncComposerState();
      if (conv === current) showSubmitError(String(err?.message || err));
      return;
    }

    busy = false;
    syncComposerState();
    if (conv !== current) return;

    const draftHasSpan = (res.bubbleSpans.get(-1)?.length || 0) > 0;
    const harmful = from === 'me' && res.nhan !== 'AN_TOAN'
      && (res.scope === 'message' || draftHasSpan || prevNhan === 'AN_TOAN');

    if (!harmful) {
      input.value = '';
      clearDraft();
      commit(conv, text, from, res);
      return;
    }

    draft = { conv, text, from, analysis: res };
    renderMessages();
    renderView();
    showWarning(res);
  });

  $('warn-edit').onclick = () => {
    clearDraft();
    renderMessages();
    renderSafetyBar();
    renderView();
    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);
  };

  $('warn-send').onclick = () => {
    const d = draft;
    if (!d) return;
    clearDraft();
    input.value = '';
    commit(d.conv, d.text, d.from, d.analysis);
    input.focus();
  };

  warn.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') $('warn-edit').click();
  });

  for (const b of document.querySelectorAll('.sender-switch button')) {
    b.onclick = () => {
      if (busy) return;
      sender = b.dataset.from;
      for (const o of document.querySelectorAll('.sender-switch button')) {
        o.setAttribute('aria-pressed', String(o === b));
      }
      if (draft) {
        clearDraft();
        renderMessages();
        renderSafetyBar();
        renderView();
      }
      syncComposerState();
      input.focus();
    };
  }

  function openAI(open) {
    shell.classList.toggle('ai-closed', !open);
    $('ai-toggle').setAttribute('aria-pressed', String(open));
    renderMessages();
  }
  $('ai-toggle').onclick = () => openAI(!aiOpen());
  $('contact-search').addEventListener('input', renderContacts);
  $('new-chat').onclick = createNewConversation;

  // Trên màn hình hẹp (< 1024px), tự thu cột AI để ưu tiên khung chat rộng rãi; người dùng vẫn bấm mở được.
  if (window.innerWidth < 1024) {
    shell.classList.add('ai-closed');
    $('ai-toggle').setAttribute('aria-pressed', 'false');
  }

  view = current.messages.length ? { conv: current, msg: current.messages.at(-1) } : null;
  renderContacts();
  renderHeader();
  syncComposerState();
  renderMessages();
  renderSafetyBar();
  renderAI(null, 'Đang khởi động mô hình ONNX…');
}
