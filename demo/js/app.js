import '@fontsource/be-vietnam-pro/400.css';
import '@fontsource/be-vietnam-pro/500.css';
import '@fontsource/be-vietnam-pro/600.css';
import '@fontsource/be-vietnam-pro/700.css';
import 'material-symbols/rounded.css';
import { initModel, isReady, scanMessage } from './ai_scanner.js';
import { createModelLoadController } from './model_loader.js';

const QUERY = new URLSearchParams(window.location.search);
const QA_MODE = QUERY.has('qa');
const QA_STATE = QUERY.get('state');
const PRELOAD_MODEL = QUERY.get('preload') === '1';

const AVATARS = {
  hoangNam: new URL('../assets/avatars/hoang-nam.png', import.meta.url).href,
  khanhLinh: new URL('../assets/avatars/khanh-linh.png', import.meta.url).href,
  minhAnh: new URL('../assets/avatars/minh-anh.png', import.meta.url).href,
};

const conversations = [
  {
    id: 'minh-anh', name: 'Minh Anh', avatar: AVATARS.minhAnh, online: true,
    time: '10:24', unread: 2, preview: 'm co bị n.gu l k z ma',
    messages: [
      { id: 'ma-1', side: 'received', text: 'Ê, cậu làm xong bài tập chưa?', time: '10:20' },
      { id: 'ma-2', side: 'sent', text: 'Tớ đang làm nè, chắc xong sớm thôi', time: '10:21', delivered: true },
      { id: 'ma-3', side: 'received', text: 'Sao dạo này cậu chậm vậy?', time: '10:22' },
      { id: 'ma-4', side: 'sent', text: 'Tớ cũng đang cố gắng mà', time: '10:23', delivered: true },
      { id: 'ma-5', side: 'received', text: 'm co bị n.gu l k z ma', time: '10:24', harmful: true },
      { id: 'ma-6', side: 'sent', text: '…', time: '10:24', delivered: true },
    ],
  },
  {
    id: 'gia-dinh', name: 'Nhóm Gia đình', icon: 'family_restroom', online: true,
    time: '09:12', unread: 3, preview: 'Ba: Tối nay cả nhà ăn cơm nhé!',
    messages: [
      { id: 'gd-1', side: 'received', text: 'Mọi người về nhà lúc mấy giờ?', time: '08:55' },
      { id: 'gd-2', side: 'sent', text: 'Con về khoảng 6 giờ ạ.', time: '09:03', delivered: true },
      { id: 'gd-3', side: 'received', text: 'Tối nay cả nhà ăn cơm nhé!', time: '09:12' },
    ],
  },
  {
    id: 'khanh-linh', name: 'Khánh Linh', avatar: AVATARS.khanhLinh, online: true,
    time: 'Hôm qua', unread: 0, preview: 'Mai nhớ gửi bài cho cô nha',
    messages: [
      { id: 'kl-1', side: 'received', text: 'Mai nhớ gửi bài cho cô nha.', time: '21:08' },
      { id: 'kl-2', side: 'sent', text: 'Ừ, cảm ơn cậu đã nhắc!', time: '21:10', delivered: true },
    ],
  },
  {
    id: 'truyen-thong', name: 'CLB Truyền thông', icon: 'campaign', online: false,
    time: '08/09', unread: 5, preview: 'Lan: Mai họp lúc 3h ở phòng C',
    messages: [
      { id: 'tt-1', side: 'received', text: 'Mai họp lúc 3h ở phòng C nhé mọi người.', time: '16:42' },
      { id: 'tt-2', side: 'sent', text: 'Mình sẽ có mặt đúng giờ.', time: '16:48', delivered: true },
    ],
  },
  {
    id: 'hoang-nam', name: 'Hoàng Nam', avatar: AVATARS.hoangNam, online: false,
    time: '07/09', unread: 0, preview: 'Ok, mình gửi file rồi nhé',
    messages: [
      { id: 'hn-1', side: 'received', text: 'Cậu xem giúp mình bản cuối được không?', time: '19:18' },
      { id: 'hn-2', side: 'sent', text: 'Được, mình xem ngay đây.', time: '19:22', delivered: true },
      { id: 'hn-3', side: 'received', text: 'Ok, mình gửi file rồi nhé.', time: '19:23' },
    ],
  },
];

const state = {
  conversationId: 'minh-anh',
  pendingSend: QA_STATE === 'warning'
    ? { text: 'Đồ ngu, biến đi', status: 'warning', conversationId: 'minh-anh' }
    : null,
  modelState: QA_MODE ? 'ready' : 'idle',
  modelDetails: QA_MODE ? { threads: 1 } : null,
  search: '',
};

const el = {
  modelLoadButton: document.getElementById('modelLoadButton'),
  modelStatusText: document.getElementById('modelStatusText'),
  modelStatusAnnouncement: document.getElementById('modelStatusAnnouncement'),
  list: document.getElementById('conversationList'),
  search: document.getElementById('conversationSearch'),
  conversationPanel: document.getElementById('conversationPanel'),
  activeAvatarWrap: document.getElementById('activeAvatarWrap'),
  activeName: document.getElementById('activeConversationName'),
  activePresence: document.getElementById('activeConversationPresence'),
  messages: document.getElementById('messageList'),
  composer: document.getElementById('messageComposer'),
  input: document.getElementById('messageInput'),
  sendButton: document.getElementById('sendButton'),
  openConversations: document.getElementById('openConversationsButton'),
  toasts: document.getElementById('toastRegion'),
};

const escapeHtml = (value) => String(value)
  .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
  .replaceAll('"', '&quot;').replaceAll("'", '&#039;');

const activeConversation = () => conversations.find((item) => item.id === state.conversationId);

function avatarContent(conversation) {
  return conversation.avatar
    ? `<img class="avatar" src="${conversation.avatar}" alt="" />`
    : `<span class="avatar avatar-icon"><span class="material-symbols-rounded" aria-hidden="true">${conversation.icon}</span></span>`;
}

function avatarMarkup(conversation) {
  return `<span class="avatar-wrap ${conversation.online ? 'is-online' : ''}">
    ${avatarContent(conversation)}<span class="online-dot" aria-hidden="true"></span>
  </span>`;
}

function renderConversationList() {
  const query = state.search.trim().toLocaleLowerCase('vi');
  const filtered = conversations.filter((item) => `${item.name} ${item.preview}`.toLocaleLowerCase('vi').includes(query));
  if (!filtered.length) {
    el.list.innerHTML = '<div class="conversation-empty">Không tìm thấy cuộc trò chuyện phù hợp.</div>';
    return;
  }
  el.list.innerHTML = filtered.map((item) => `
    <button class="conversation-item ${item.id === state.conversationId ? 'is-active' : ''}"
      type="button" data-conversation-id="${item.id}" aria-pressed="${item.id === state.conversationId}">
      ${avatarMarkup(item)}
      <span class="conversation-copy">
        <span class="conversation-name">${escapeHtml(item.name)}</span>
        <span class="conversation-preview">${escapeHtml(item.preview)}</span>
      </span>
      <span class="conversation-meta">
        <span class="conversation-time">${escapeHtml(item.time)}</span>
        ${item.unread ? `<span class="unread-count">${item.unread}</span>` : ''}
      </span>
    </button>`).join('');
}

function renderHeader() {
  const conversation = activeConversation();
  el.activeName.textContent = conversation.name;
  el.activeAvatarWrap.className = `avatar-wrap ${conversation.online ? 'is-online' : ''}`;
  el.activeAvatarWrap.innerHTML = `${avatarContent(conversation)}<span class="online-dot" aria-hidden="true"></span>`;
  el.activePresence.innerHTML = conversation.online
    ? '<span class="presence-dot" aria-hidden="true"></span><span>Đang hoạt động</span>'
    : '<span>Hoạt động gần đây</span>';
}

function messageWarningMarkup(sent = false) {
  return `<div class="message-warning ${sent ? 'is-sent-warning' : ''}" role="note">
    <span class="material-symbols-rounded" aria-hidden="true">warning</span>
    <span><strong>Nội dung có thể gây hại</strong><small>SafeViChi phát hiện ngôn từ có thể làm tổn thương người nhận.</small></span>
  </div>`;
}

function pendingSendMarkup() {
  const pending = state.pendingSend;
  if (!pending || pending.conversationId !== state.conversationId) return '';
  const warning = pending.status === 'warning';
  return `<article class="message-row is-sent is-pending-send">
    <div class="message-cluster">
      <div class="bubble-line"><div class="message-bubble">${escapeHtml(pending.text)}</div></div>
      ${warning ? `<div class="send-warning" role="alert">
        <div class="send-warning-copy">
          <span class="warning-icon"><span class="material-symbols-rounded" aria-hidden="true">warning</span></span>
          <span><strong>Hãy dừng lại một chút</strong><small>Tin nhắn này có thể chứa nội dung gây hại. Bạn vẫn muốn gửi?</small></span>
        </div>
        <div class="send-warning-actions">
          <button class="warning-secondary" type="button" data-pending-action="edit">Chỉnh sửa</button>
          <button class="warning-primary" type="button" data-pending-action="send-anyway">Vẫn gửi</button>
        </div>
      </div>` : `<div class="safety-check" role="status">
        <span class="spinner" aria-hidden="true"></span><span>SafeViChi đang kiểm tra trước khi gửi…</span>
      </div>`}
    </div>
  </article>`;
}

function renderMessages(scrollToBottom = false) {
  const conversation = activeConversation();
  el.messages.innerHTML = `<div class="date-separator"><span>Hôm nay</span></div>${conversation.messages.map((message) => {
    const sent = message.side === 'sent';
    return `<article class="message-row ${sent ? 'is-sent' : ''}" data-message-id="${message.id}">
      ${sent ? '' : avatarMarkup(conversation)}
      <div class="message-cluster">
        <div class="bubble-line">
          <div class="message-bubble">${escapeHtml(message.text)}</div>
          <button class="message-menu" type="button" aria-label="Tùy chọn cho tin nhắn"><span class="material-symbols-rounded" aria-hidden="true">more_horiz</span></button>
        </div>
        <div class="message-time"><span>${escapeHtml(message.time)}</span>
          ${message.delivered ? '<span class="material-symbols-rounded" aria-label="Đã gửi">done_all</span>' : ''}</div>
        ${message.harmful ? messageWarningMarkup(sent) : ''}
      </div>
    </article>`;
  }).join('')}${pendingSendMarkup()}`;
  if (scrollToBottom) requestAnimationFrame(() => { el.messages.scrollTop = el.messages.scrollHeight; });
}

function renderModelStatus() {
  const status = state.modelState;
  const text = status === 'ready'
    ? state.modelDetails?.threads > 1
      ? `Mô hình sẵn sàng · ${state.modelDetails.threads} luồng cục bộ`
      : 'Mô hình sẵn sàng · Chế độ tương thích'
    : status === 'loading_encoder' ? 'Đang tải bộ mã hoá · 418 MiB'
      : status === 'loading_decoder' ? 'Đang tải bộ giải mã · 621 MiB'
        : status === 'error' ? 'Tải lỗi · Thử lại'
          : 'Tải mô hình · khoảng 1,09 GB';
  el.modelLoadButton.dataset.state = status;
  el.modelLoadButton.disabled = status === 'ready' || status.startsWith('loading_');
  el.modelLoadButton.setAttribute('aria-label', status === 'error' ? 'Thử tải lại mô hình' : text);
  el.modelStatusText.textContent = text;
  el.modelStatusAnnouncement.textContent = text;
}

function renderComposerState() {
  const locked = Boolean(state.pendingSend);
  el.input.disabled = locked;
  el.sendButton.disabled = locked;
  el.composer.classList.toggle('is-locked', locked);
}

function toast(message, icon = 'info') {
  const node = document.createElement('div');
  node.className = 'toast';
  node.innerHTML = `<span class="material-symbols-rounded" aria-hidden="true">${icon}</span><span>${escapeHtml(message)}</span>`;
  el.toasts.append(node);
  setTimeout(() => node.remove(), 3000);
}

function switchConversation(id) {
  if (state.pendingSend) {
    toast('Hãy xử lý cảnh báo trước khi chuyển cuộc trò chuyện.', 'warning');
    return;
  }
  state.conversationId = id;
  renderConversationList(); renderHeader(); renderMessages(true);
  el.conversationPanel.classList.remove('is-open');
}

function qaResult(text) {
  const risky = /n[.\-_]?gu|ngu|m4y|đần|óc chó/i.test(text);
  return { original: text, normalized: text.replace(/n[.\-_]gu/gi, 'ngu'), label: risky ? 'hate' : 'clean' };
}

async function classifyBeforeSend(text) {
  if (QA_MODE) {
    await new Promise((resolve) => setTimeout(resolve, 420));
    return qaResult(text);
  }
  if (!isReady()) throw new Error('Mô hình chưa sẵn sàng.');
  return scanMessage(text, { topK: 3, threshold: 0.15 });
}

function commitSend(text, harmful = false) {
  const conversation = activeConversation();
  const time = new Intl.DateTimeFormat('vi-VN', { hour: '2-digit', minute: '2-digit', hour12: false }).format(new Date());
  conversation.messages.push({
    id: `${conversation.id}-${Date.now()}`, side: 'sent', text, time, delivered: true, harmful,
  });
  conversation.preview = text; conversation.time = time;
  state.pendingSend = null;
  renderComposerState(); renderConversationList(); renderMessages(true);
  toast(harmful ? 'Tin nhắn đã được gửi theo lựa chọn của bạn.' : 'Tin nhắn đã gửi an toàn.', harmful ? 'warning' : 'done');
}

async function attemptSendMessage() {
  const text = el.input.value.trim();
  if (!text || state.pendingSend) return;
  if (!QA_MODE && !isReady()) {
    startModelLoad();
    toast('Tin nhắn vẫn được giữ lại. Hãy nhấn Gửi lần nữa khi mô hình sẵn sàng.', 'hourglass_top');
    return;
  }
  state.pendingSend = { text, status: 'checking', conversationId: state.conversationId };
  el.input.value = '';
  renderComposerState(); renderMessages(true);
  try {
    const result = await classifyBeforeSend(text);
    if (result.label === 'hate') {
      state.pendingSend.status = 'warning';
      state.pendingSend.result = result;
      renderMessages(true);
    } else {
      commitSend(text, false);
    }
  } catch (error) {
    console.error(error);
    state.pendingSend = null;
    el.input.value = text;
    renderComposerState(); renderMessages(true);
    toast('Không thể kiểm tra tin nhắn. Nội dung chưa được gửi.', 'error');
    el.input.focus();
  }
}

function editPendingMessage() {
  if (!state.pendingSend) return;
  const text = state.pendingSend.text;
  state.pendingSend = null;
  el.input.value = text;
  renderComposerState(); renderMessages(true);
  el.input.dispatchEvent(new Event('input'));
  el.input.focus();
}

function sendPendingAnyway() {
  if (!state.pendingSend || state.pendingSend.status !== 'warning') return;
  commitSend(state.pendingSend.text, true);
}

const modelLoader = createModelLoadController({
  load: (onStage) => initModel({ onStage }),
  onChange: ({ status, detail, error }) => {
    state.modelState = status;
    state.modelDetails = detail;
    renderModelStatus();
    if (status === 'ready') {
      toast('Mô hình đã sẵn sàng. Tin nhắn được xử lý trên thiết bị.', 'verified_user');
      if (detail?.threads === 1 && !globalThis.crossOriginIsolated) {
        toast('Trình duyệt đang dùng chế độ tương thích một luồng.', 'info');
      }
    }
    if (status === 'error') {
      console.error(error);
      toast('Không thể tải mô hình. Tin nhắn chưa được gửi.', 'error');
    }
  },
});

function startModelLoad() {
  if (QA_MODE) return Promise.resolve(state.modelDetails);
  return modelLoader.start().catch(() => null);
}

function bootModel() {
  renderModelStatus();
  if (!QA_MODE && PRELOAD_MODEL) startModelLoad();
}

el.list.addEventListener('click', (event) => {
  const item = event.target.closest('[data-conversation-id]');
  if (item) switchConversation(item.dataset.conversationId);
});
el.search.addEventListener('input', (event) => { state.search = event.target.value; renderConversationList(); });
el.messages.addEventListener('click', (event) => {
  const action = event.target.closest('[data-pending-action]')?.dataset.pendingAction;
  if (action === 'edit') editPendingMessage();
  if (action === 'send-anyway') sendPendingAnyway();
});
el.composer.addEventListener('submit', (event) => { event.preventDefault(); attemptSendMessage(); });
el.input.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); attemptSendMessage(); }
});
el.openConversations.addEventListener('click', () => el.conversationPanel.classList.toggle('is-open'));
el.modelLoadButton.addEventListener('click', startModelLoad);
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') el.conversationPanel.classList.remove('is-open');
});

renderConversationList(); renderHeader(); renderMessages(true); renderModelStatus(); renderComposerState(); bootModel();
