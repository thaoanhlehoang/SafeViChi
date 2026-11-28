import { isLevel2Enabled, redactPreviewSelection, setLevel2Enabled } from '../feedback/client.js';

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text) node.textContent = text;
  return node;
}

export function oDanhDau(text, checked = false) {
  const label = el('label', 'ph-check');
  const box = el('input');
  box.type = 'checkbox';
  box.checked = checked;
  label.append(box, el('span', null, text));
  return { label, box };
}

export function taoManXemTruoc(ban, moTa, { onCheThem, onDongY }) {
  const panel = el('section', 'ph-preview');
  const title = el('h4', null, 'Đoạn hội thoại sẽ gửi');
  title.tabIndex = -1;
  const huongDan = el('p', 'ph-preview-help',
    'Thông tin nhận diện đã được che tự động trên thiết bị. Nếu còn chi tiết cần che, hãy chọn văn bản trong một lượt rồi bấm “Che phần đã chọn”.');
  const ketQuaChe = el('p', 'ph-preview-help');
  ketQuaChe.setAttribute('role', 'status');
  const luot = el('div', 'ph-preview-turns');
  const { label, box } = oDanhDau('Tôi đã xem và đồng ý gửi bản đã che này để người kiểm duyệt xem xét.');
  box.onchange = () => onDongY(box.checked);

  function hienLuot(focusIndex) {
    const rows = ban.text.split('\n').map((line, index) => {
      const row = el('div', 'ph-preview-turn');
      const head = el('div', 'ph-preview-turn-head');
      const lineLabel = el('label', null, 'Lượt ' + (index + 1));
      const text = el('textarea');
      text.id = 'ph-luot-' + index;
      lineLabel.htmlFor = text.id;
      text.value = line;
      text.readOnly = true;
      text.rows = Math.max(2, Math.min(5, Math.ceil(line.length / 55)));
      const mask = el('button', 'ph-preview-mask', 'Che phần đã chọn');
      mask.type = 'button';
      mask.setAttribute('aria-label', 'Che phần đã chọn ở lượt ' + (index + 1));
      mask.onclick = () => {
        const { selectionStart: start, selectionEnd: end } = text;
        if (start === end) {
          ketQuaChe.textContent = 'Hãy chọn phần văn bản cần che trong lượt ' + (index + 1) + '.';
          text.focus();
          return;
        }
        try {
          const banMoi = redactPreviewSelection(ban, index, start, end);
          if (banMoi.text === ban.text) {
            ketQuaChe.textContent = 'Phần bạn chọn đã được che.';
            return;
          }
          ban = banMoi;
          box.checked = false;
          onDongY(false);
          onCheThem(ban);
          hienLuot(index);
          ketQuaChe.textContent = 'Đã che thêm ở lượt ' + (index + 1) + '. Hãy xác nhận lại bản mới.';
        } catch {
          ketQuaChe.textContent = 'Vùng chọn không hợp lệ. Hãy chọn lại phần cần che.';
        }
      };
      head.append(lineLabel, mask);
      row.append(head, text);
      return row;
    });
    luot.replaceChildren(...rows);
    if (focusIndex !== undefined) rows[focusIndex].querySelector('textarea').focus();
  }

  hienLuot();
  panel.append(title, huongDan, luot, ketQuaChe);
  if (moTa) {
    const moTaBox = el('div', 'ph-preview-turn');
    moTaBox.append(el('strong', null, 'Mô tả kèm theo'), el('p', 'ph-preview-help', moTa));
    panel.append(moTaBox);
  }
  panel.append(label);
  return { panel, title };
}

export function moCaiDatPhanHoi(host) {
  const dialog = el('dialog', 'ph-dialog');
  dialog.setAttribute('aria-labelledby', 'ph-cai-dat-title');
  const head = el('header', 'ph-head');
  const title = el('h3', null, 'Cài đặt phản hồi');
  title.id = 'ph-cai-dat-title';
  head.append(title);
  const body = el('div', 'ph-body ph-settings');
  const { label, box } = oDanhDau('Cho phép tùy chọn phản hồi mức 2', isLevel2Enabled());
  box.onchange = () => setLevel2Enabled(box.checked);
  const link = el('a', 'ph-settings-link', 'Kiểm duyệt và quản trị');
  link.href = '/quan-tri.html';
  body.append(label, el('p', 'ph-preview-help',
    'Mỗi lần gửi kèm đoạn hội thoại, bạn vẫn cần xem trước và đồng ý.'), link);
  const foot = el('footer', 'ph-foot ph-actions');
  const close = el('button', 'ph-gui', 'Đóng');
  close.type = 'button';
  close.onclick = () => dialog.close();
  foot.append(close);
  dialog.append(head, body, foot);
  dialog.addEventListener('close', () => dialog.remove());
  host.append(dialog);
  dialog.showModal();
}
