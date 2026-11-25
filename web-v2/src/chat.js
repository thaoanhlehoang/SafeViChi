// Màn trò chuyện 3 cột: danh sách | khung chat | AI phân tích.
// Khung chat chỉ tô đỏ bằng chứng; toàn bộ kết quả pipeline nằm ở cột AI.
// Mọi chữ người dùng gõ đều đi qua textContent / text node, không dùng innerHTML.
import { LABELS, MAX_LINES } from './labels.js';
import { SHIRTS, HAIR, SKIN } from './landing.js';

// Hội thoại mẫu để thiết kế, tự viết mới (không lấy từ tập dữ liệu).
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

const AVATAR_TONES = ['#2f6b38', '#4a8233', '#5f7f3a', '#3c7262', '#6a8f4e', '#2d5a4a'];
const ORDER = ['AN_TOAN', 'XUC_PHAM', 'QUAY_ROI', 'DE_DOA', 'LUA_DAO'];
const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

export function setupChat({ classify, onEngineStatus }) {
  let current = CONVERSATIONS[0];
  let view = null; // {conv, msg} tin đang xem ở cột AI
  let sender = 'me';
  let draft = null; // tin đang bị giữ lại chờ bạn chọn: {conv, text, from, analysis}
  let ready = false;

  const shell = $('chat-shell');
  const list = $('contact-list');
  const box = $('messages');
  const input = $('composer-input');
  const aiOpen = () => !shell.classList.contains('ai-closed');

  onEngineStatus((s) => {
    $('ai-engine').textContent = s.text;
    $('ai-engine').dataset.kind = s.kind;
    if (s.kind !== 'loading' && !ready) { ready = true; analyzeAll(); }
  });

  const who = (conv, m) => (m.from === 'me' ? 'Bạn' : m.author || conv.name);

  // Mô hình không xét một tin riêng lẻ: mỗi lần gửi vào tối đa 8 tin gần nhất (cả hai phía),
  // trả về MỘT nhãn cho cả đoạn và bằng chứng có thể nằm ở bất kỳ tin nào trong đoạn.
  async function analyze(conv, idx, draftText) {
    const rows = conv.messages.slice(0, idx).map((m, i) => ({ i, m, text: m.text }));
    if (draftText != null) rows.push({ i: -1, m: { from: sender, author: sender === 'them' && conv.group ? 'Thành viên' : undefined, time: 'chưa gửi' }, text: draftText });
    else rows.push({ i: idx, m: conv.messages[idx], text: conv.messages[idx].text });
    const win = rows.slice(-MAX_LINES);
    const out = await classify(win.map((r) => r.text));
    return {
      nhan: out.nhan,
      xac_suat: out.xac_suat,
      ms: out.ms,
      rows: win.map((r, dong) => ({
        i: r.i,
        who: who(conv, r.m),
        time: r.m.time,
        text: r.text,
        norm: out.chuan_hoa?.[dong] ?? r.text,
        spans: out.bang_chung.filter((b) => b.dong === dong),
      })),
    };
  }

  // Lần phân tích mới nhất có chứa một tin sẽ quyết định chỗ tô đỏ trong tin đó,
  // vì một câu có thể chỉ độc hại khi đặt cạnh các câu trước.
  function applySpans(conv, res) {
    for (const r of res.rows) if (r.i >= 0) conv.messages[r.i].spans = r.spans;
  }

  async function analyzeAll() {
    for (const conv of [current, ...CONVERSATIONS.filter((c) => c !== current)]) {
      for (let i = 0; i < conv.messages.length; i++) {
        if (!conv.messages[i].analysis) {
          conv.messages[i].analysis = await analyze(conv, i);
          applySpans(conv, conv.messages[i].analysis);
        }
      }
      renderContacts();
      if (conv === current) { if (!view) view = { conv, msg: latestFlagged(conv) }; renderMessages(); renderView(); }
    }
  }

  const latestFlagged = (conv) => [...conv.messages].reverse().find((m) => m.analysis && m.analysis.nhan !== 'AN_TOAN') || conv.messages.at(-1);
  const flagged = (conv) => conv.messages.some((m) => m.spans?.length);

  // Chân dung nhỏ vẽ cùng kiểu người ở trang đầu: áo, mặt, tóc theo tên.
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
    const add = (tag, attrs) => { const n = document.createElementNS(NS, tag); for (const k in attrs) n.setAttribute(k, attrs[k]); svg.append(n); };
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
    list.replaceChildren(...CONVERSATIONS.map((conv, i) => [conv, i])
      .filter(([conv]) => !q || conv.name.toLowerCase().includes(q))
      .map(([conv, i]) => {
        const li = el('li');
        const b = el('button', 'contact');
        b.type = 'button';
        b.setAttribute('aria-current', conv === current ? 'true' : 'false');
        const last = conv.messages.at(-1);
        const body = el('span', 'contact-body');
        const top = el('span', 'contact-top');
        top.append(el('span', 'contact-name', conv.name), el('span', 'contact-time', last.time));
        const prev = el('span', 'contact-preview', (last.from === 'me' ? 'Bạn: ' : last.author ? last.author + ': ' : '') + last.text);
        body.append(top, prev);
        b.append(avatar(conv.name), body);
        if (flagged(conv)) b.append(el('span', 'contact-flag'));
        b.onclick = () => openConversation(conv);
        li.append(b);
        return li;
      }));
  }

  function openConversation(conv) {
    current = conv;
    input.value = '';
    clearDraft();
    view = { conv, msg: latestFlagged(conv) };
    document.querySelector('.ai-inner').scrollTop = 0;
    renderContacts();
    renderHeader();
    renderMessages();
    renderView();
  }

  function renderHeader() {
    $('chat-avatar').replaceWith(Object.assign(avatar(current.name), { id: 'chat-avatar' }));
    $('chat-name').textContent = current.name;
    $('chat-sub').textContent = current.sub;
  }

  // Chèn chữ thường và <mark> bằng chứng xen kẽ, theo vị trí trên câu gốc.
  function highlighted(text, spans) {
    const frag = document.createDocumentFragment();
    let pos = 0;
    for (const s of [...spans].sort((a, b) => a.start - b.start)) {
      if (s.start < pos) continue;
      frag.append(text.slice(pos, s.start));
      frag.append(el('mark', 'evidence-mark', text.slice(s.start, s.end)));
      pos = s.end;
    }
    frag.append(text.slice(pos));
    return frag;
  }

  // Các tin nằm trong đoạn mà cột AI đang hiển thị.
  function windowSet() {
    const res = draft ? draft.analysis : view?.msg.analysis;
    if (!res) return null;
    return new Set(res.rows.filter((r) => r.i >= 0).map((r) => r.i));
  }

  function renderMessages() {
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
      bubble.onclick = () => { view = { conv: current, msg: m }; document.querySelector('.ai-inner').scrollTop = 0; openAI(true); renderView(); };
      li.append(bubble, el('span', 'msg-time', m.time));
      items.push(li);
    });
    const keep = box.scrollHeight - box.scrollTop - box.clientHeight < 40 || box.dataset.conv !== current.id;
    box.replaceChildren(...items);
    box.dataset.conv = current.id;
    if (keep) box.scrollTop = box.scrollHeight;
  }

  function renderView() {
    if (draft) renderAI(draft.analysis, 'Tin chưa gửi');
    else if (view) renderAI(view.msg.analysis, `Khi ${view.msg.from === 'me' ? 'bạn gửi' : 'nhận'} tin lúc ${view.msg.time}`);
  }

  // So từng từ câu gốc với câu chuẩn hóa (LCS) để đánh dấu chỗ đã đổi.
  function wordDiff(a, b) {
    const A = a.split(/\s+/).filter(Boolean);
    const B = b.split(/\s+/).filter(Boolean);
    const dp = Array.from({ length: A.length + 1 }, () => new Array(B.length + 1).fill(0));
    for (let i = A.length - 1; i >= 0; i--) for (let j = B.length - 1; j >= 0; j--) dp[i][j] = A[i] === B[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    const keepB = new Set();
    let changedA = 0;
    for (let i = 0, j = 0; i < A.length || j < B.length;) {
      if (i < A.length && j < B.length && A[i] === B[j]) { keepB.add(j); i++; j++; }
      else if (j >= B.length || (i < A.length && dp[i + 1][j] >= dp[i][j + 1])) { i++; changedA++; }
      else j++;
    }
    return { changedA, B: B.map((w, i) => [w, !keepB.has(i)]) };
  }

  // Câu chuẩn hóa, gộp các từ đổi liền nhau thành một khối ("trả lời").
  function normLine(pairs) {
    const p = el('p', 'ctx-norm');
    let run = null;
    pairs.forEach(([w, changed], i) => {
      if (changed && run) { run.textContent += ' ' + w; return; }
      if (i) p.append(' ');
      if (changed) { run = el('span', 'w-new', w); p.append(run); } else { run = null; p.append(w); }
    });
    return p;
  }

  function renderAI(res, kicker) {
    $('ai-kicker').textContent = kicker;
    const ctx = $('ai-context');
    const verdict = $('ai-verdict');
    const probs = $('ai-probs');
    const ev = $('ai-evidence');
    if (!res) {
      $('ai-summary').replaceChildren();
      ctx.replaceChildren(el('li', 'ai-empty', 'Đang phân tích…'));
      verdict.replaceChildren(); probs.replaceChildren(); ev.replaceChildren();
      return;
    }
    const n = res.rows.length;
    // Tóm tắt luôn nằm trên cùng: nhãn của cả đoạn.
    const top = LABELS[res.nhan];
    const sChip = el('span', 'label-chip', top.ten);
    sChip.style.setProperty('--c', top.mau);
    $('ai-summary').replaceChildren(sChip, el('span', 'summary-pct', `${(100 * res.xac_suat[res.nhan]).toFixed(1)}%`));

    // 1. Đoạn hội thoại gửi vào mô hình, mỗi tin kèm bản chuẩn hóa nếu có sửa.
    ctx.replaceChildren(...res.rows.map((r, k) => {
      const li = el('li', 'ctx-row');
      if (k === n - 1) li.classList.add('is-new');
      if (r.spans.length) li.classList.add('has-ev');
      const head = el('div', 'ctx-head');
      head.append(el('span', 'ctx-who', r.who), el('span', 'ctx-time', r.time));
      const orig = el('p', 'ctx-text');
      orig.append(highlighted(r.text, r.spans));
      li.append(head, orig);
      const d = wordDiff(r.text, r.norm);
      if (d.changedA || d.B.some(([, c]) => c)) {
        li.append(normLine(d.B));
      }
      return li;
    }));

    // 2. Một nhãn cho cả đoạn.
    const info = LABELS[res.nhan];
    verdict.replaceChildren(el('p', 'verdict-action', `Gợi ý xử lý: ${info.hanh_dong}`));
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

    // 3. Bằng chứng, ghi rõ nằm ở tin nào.
    const evs = res.rows.flatMap((r, k) => r.spans.map((s) => ({ s, r, k })));
    ev.replaceChildren(...(evs.length
      ? evs.map(({ s, r, k }) => {
        const li = el('li');
        li.append(el('mark', 'evidence-mark', s.text), el('span', 'ev-where', `${String(k + 1).padStart(2, '0')} · ${r.who}`));
        return li;
      })
      : [el('li', 'ai-empty', 'Không thấy dấu hiệu độc hại trong đoạn này.')]));

  }

  // Chỉ phân tích khi bấm gửi. Tin của bạn có dấu hiệu độc hại thì dừng lại hỏi trước.
  const warn = $('send-warn');
  const nowTime = () => { const d = new Date(); return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`; };

  function clearDraft() { draft = null; warn.hidden = true; }

  function commit(conv, text, from, analysis) {
    const m = { from, text, time: nowTime() };
    if (from === 'them' && conv.group) m.author = 'Thành viên';
    conv.messages.push(m);
    // Dòng cuối của lần phân tích là tin vừa gửi: gắn nó vào đúng tin trong cuộc trò chuyện.
    const rows = analysis.rows.slice();
    rows[rows.length - 1] = { ...rows.at(-1), i: conv.messages.length - 1, time: m.time, who: who(conv, m) };
    analysis = { ...analysis, rows };
    m.analysis = analysis;
    applySpans(conv, analysis);
    view = { conv, msg: m };
    if (conv === current) { renderMessages(); renderView(); box.scrollTop = box.scrollHeight; }
    renderContacts();
  }

  function showWarning(res) {
    const info = LABELS[res.nhan];
    const label = el('strong', 'warn-label', info.ten);
    label.style.setProperty('--c', info.mau);
    $('warn-title').replaceChildren('Tin nhắn này có dấu hiệu ', label);
    warn.hidden = false;
    $('warn-edit').focus();
  }

  input.addEventListener('input', () => {
    // Sửa lại tin thì bỏ cảnh báo cũ, lần gửi sau sẽ phân tích lại.
    if (draft) { clearDraft(); renderMessages(); renderView(); }
  });

  $('composer').addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = input.value.trim().slice(0, 300);
    if (!text || !ready || $('composer').classList.contains('busy')) return;
    const conv = current;
    const from = sender;
    $('composer').classList.add('busy');
    const res = await analyze(conv, conv.messages.length, text);
    $('composer').classList.remove('busy');
    if (conv !== current) return;
    const harmful = from === 'me' && res.nhan !== 'AN_TOAN' && res.rows.at(-1).spans.length > 0;
    if (!harmful) {
      input.value = '';
      commit(conv, text, from, res);
      return;
    }
    draft = { conv, text, from, analysis: res };
    renderMessages();
    renderView();
    openAI(aiOpen());
    showWarning(res);
  });

  $('warn-edit').onclick = () => {
    clearDraft(); renderMessages(); renderView();
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
  warn.addEventListener('keydown', (e) => { if (e.key === 'Escape') $('warn-edit').click(); });

  for (const b of document.querySelectorAll('.sender-switch button')) {
    b.onclick = () => {
      sender = b.dataset.from;
      for (const o of document.querySelectorAll('.sender-switch button')) o.setAttribute('aria-pressed', String(o === b));
      if (draft) { clearDraft(); renderMessages(); renderView(); }
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

  renderContacts();
  renderHeader();
  renderMessages();
  renderAI(null, 'Đang tải mô hình');
}
