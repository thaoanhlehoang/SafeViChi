// Thanh tab "Nhắn tin & cảnh báo | Kiểm duyệt & quản trị" trong màn chính.
// Đổi tab tại chỗ: không mở tab trình duyệt mới, không rời trang, giữ nguyên trạng thái chat.
// Khu vực quản trị nằm trong ShadowRoot để CSS của nó và CSS màn chat không lẫn vào nhau;
// CSS nạp bằng <link> cùng origin nên vẫn giữ CSP style-src 'self'.
import cssChung from './quan_tri.css?url';
import cssNhung from './quan_tri_nhung.css?url';
import { dungKhuVucQuanTri } from './khu_vuc.js';

export function setupAppTabs() {
  const tabs = [...document.querySelectorAll('.app-tab[role="tab"]')];
  if (!tabs.length) return;
  const panelOf = (tab) => document.getElementById(tab.getAttribute('aria-controls'));
  let khuVuc;

  function dungQuanTri(host) {
    const shadow = host.attachShadow({ mode: 'open' });
    for (const href of [cssChung, cssNhung]) {
      const link = document.createElement('link');
      link.rel = 'stylesheet';
      link.href = href;
      shadow.append(link);
    }
    khuVuc = dungKhuVucQuanTri(shadow);
    khuVuc.batDau();
  }

  function chon(tab, { focus = false } = {}) {
    for (const other of tabs) {
      const selected = other === tab;
      other.setAttribute('aria-selected', String(selected));
      other.tabIndex = selected ? 0 : -1;
      panelOf(other).hidden = !selected;
    }
    const panel = panelOf(tab);
    if (panel.dataset.khuVuc === 'quan-tri') {
      if (!khuVuc) dungQuanTri(panel);
      if (focus) khuVuc.focus();
    }
  }

  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => chon(tab, { focus: true }));
    tab.addEventListener('keydown', (event) => {
      const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
      let next;
      if (step) next = tabs[(index + step + tabs.length) % tabs.length];
      else if (event.key === 'Home') next = tabs[0];
      else if (event.key === 'End') next = tabs[tabs.length - 1];
      if (!next) return;
      event.preventDefault();
      next.focus();
      chon(next);
    });
  });
}
