// Bộ nhãn v2 và hành động của ứng dụng (theo CLAUDE.md). UI đọc màu/hành động từ đây.
export const LABELS = {
  AN_TOAN:  { ten: 'An toàn',  mau: '#16a34a', hanh_dong: 'Hiện bình thường' },
  XUC_PHAM: { ten: 'Xúc phạm', mau: '#d97706', hanh_dong: 'Làm mờ, bấm mới xem' },
  QUAY_ROI: { ten: 'Quấy rối', mau: '#9333ea', hanh_dong: 'Gợi ý chặn, báo cáo' },
  DE_DOA:   { ten: 'Đe dọa',   mau: '#dc2626', hanh_dong: 'Cảnh báo mạnh, gợi ý lưu bằng chứng' },
  LUA_DAO:  { ten: 'Lừa đảo',  mau: '#2563eb', hanh_dong: 'Cảnh báo, vô hiệu link' },
};

// Mô tả ngắn cho phần thông tin (theo docs/v2/HUONG_DAN_GAN_NHAN.md, rút gọn).
export const LABEL_INFO = {
  AN_TOAN: 'Tin nhắn bình thường, không thuộc 4 loại dưới.',
  XUC_PHAM: 'Lăng mạ, miệt thị một người hay một nhóm, kể cả chê bai vùng miền.',
  QUAY_ROI: 'Gạ gẫm không mong muốn, nhắn bám riết, bêu riếu người khác.',
  DE_DOA: 'Dọa gây hại thân thể, tài sản, danh dự, hoặc tống tiền.',
  LUA_DAO: 'Thông tin giả để dụ chuyển tiền, đưa mã OTP hay bấm vào link.',
};

// Ứng dụng gửi tối đa 8 lượt gần nhất (mỗi lượt gộp tin liên tiếp cùng người, xem CLAUDE.md).
export const MAX_LINES = 8;
