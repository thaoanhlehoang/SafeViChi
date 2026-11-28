// Câu chữ và mã cho biểu mẫu phản hồi mức 1.
// Nguồn câu chữ: docs/v3/DE_XUAT_TRIEN_KHAI_CHANG_2.md, Mục 1.
// Mã đáp án khớp hợp đồng API trong src/feedback/contract.js.

export { HARMFUL_LABEL_CODES as NHAN_NGUY_HAI } from '../feedback/contract.js';

export const TEN_NHAN = {
  AN_TOAN: 'An toàn',
  XUC_PHAM: 'Xúc phạm',
  QUAY_ROI: 'Quấy rối',
  DE_DOA: 'Đe dọa',
  LUA_DAO: 'Lừa đảo',
  CHUA_CHAC: 'Chưa chắc',
};

// Bảng dấu hiệu theo nhãn, dùng chung cho nhánh "sai loại" và "bỏ sót".
export const DAU_HIEU = {
  XUC_PHAM: [
    { ma: 'lang_ma_ca_nhan', ten: 'Lăng mạ cá nhân' },
    { ma: 'miet_thi_nhom', ten: 'Miệt thị một nhóm hoặc vùng miền' },
  ],
  QUAY_ROI: [
    { ma: 'ga_gam', ten: 'Gạ gẫm không mong muốn' },
    { ma: 'binh_pham_co_the', ten: 'Bình phẩm cơ thể' },
    { ma: 'bam_riet_kiem_soat', ten: 'Nhắn bám riết hoặc kiểm soát' },
    { ma: 'beu_rieu', ten: 'Bêu riếu' },
  ],
  DE_DOA: [
    { ma: 'than_the_tai_san', ten: 'Dọa gây hại thân thể hoặc tài sản' },
    { ma: 'danh_du', ten: 'Dọa làm hại danh dự' },
    { ma: 'cuong_ep_tong_tien', ten: 'Cưỡng ép hoặc tống tiền' },
  ],
  LUA_DAO: [
    { ma: 'mao_danh', ten: 'Mạo danh' },
    { ma: 'chuyen_tien', ten: 'Dụ chuyển tiền' },
    { ma: 'otp_mat_khau', ten: 'Dụ cung cấp OTP hoặc mật khẩu' },
    { ma: 'lien_ket_ung_dung', ten: 'Dụ bấm liên kết hoặc cài ứng dụng' },
  ],
};

const KHAC = { ma: 'khac', ten: 'Khác', khac: true };

// Loại báo cáo. `hien(nhanMoHinh)` là điều kiện hiển thị theo bảng ở Mục 1.
export const LOAI_BAO_CAO = [
  { ma: 'tin_nhan_an_toan', ten: 'Tin nhắn này an toàn', hien: (n) => n !== 'AN_TOAN' },
  { ma: 'sai_loai', ten: 'Có vấn đề, nhưng AI chọn sai loại', hien: (n) => n !== 'AN_TOAN' },
  { ma: 'bo_sot', ten: 'AI bỏ sót nội dung nguy hại', hien: (n) => n === 'AN_TOAN' },
  { ma: 'loi_to_sang', ten: 'Nhãn đúng, phần tô sáng có vấn đề', hien: (n) => n !== 'AN_TOAN' },
  { ma: 'khac_chua_chac', ten: 'Vấn đề khác hoặc tôi chưa chắc', hien: () => true },
];

// Nhánh "Tin nhắn này an toàn"
export const LY_DO_AN_TOAN = [
  {
    ma: 'dua_noi_than_mat', ten: 'Đùa hoặc nói thân mật',
    phu: {
      hoi: 'Điều gì giúp bạn nhận ra đây là lời đùa hoặc cách nói thân mật?',
      options: [
        { ma: 'cach_noi_chuyen', ten: 'Cách hai bên nói chuyện với nhau' },
        { ma: 'tra_loi_nguoi_nhan', ten: 'Câu trả lời của người nhận' },
        { ma: 'luot_truoc', ten: 'Các lượt trước' },
      ],
    },
  },
  {
    ma: 'tu_long_dia_phuong', ten: 'Từ lóng hoặc cách nói địa phương',
    phu: {
      hoi: 'Điều gì giúp bạn hiểu đúng cách nói này?',
      options: [
        { ma: 'nghia_quen_thuoc', ten: 'Nghĩa quen thuộc của từ' },
        { ma: 'cach_dung', ten: 'Cách dùng từ trong câu' },
        { ma: 'luot_truoc', ten: 'Các lượt trước' },
      ],
    },
  },
  {
    ma: 'trich_dan_ke_lai', ten: 'Trích dẫn hoặc kể lại',
    phu: {
      hoi: 'Bạn nhận ra đây là lời trích dẫn hoặc lời kể bằng cách nào?',
      options: [
        { ma: 'dau_hieu_trich_dan', ten: 'Có dấu hiệu trích dẫn' },
        { ma: 'noi_ro_dang_ke', ten: 'Người gửi nói rõ họ đang kể lại' },
        { ma: 'luot_truoc', ten: 'Các lượt trước cho biết đó là lời người khác' },
      ],
    },
  },
  {
    ma: 'ngu_canh_binh_thuong', ten: 'Ngữ cảnh cho thấy tin nhắn bình thường',
    phu: {
      hoi: 'Tin nhắn thuộc tình huống nào?',
      options: [
        { ma: 'cong_viec_hoc_tap', ten: 'Công việc hoặc học tập' },
        { ma: 'giao_dich_hop_le', ten: 'Giao dịch hoặc xác minh hợp lệ' },
        { ma: 'tro_chuyen_hang_ngay', ten: 'Trò chuyện hằng ngày' },
      ],
    },
  },
  KHAC,
];

// Nhánh "Có vấn đề, nhưng AI chọn sai loại"
export const LY_DO_SAI_LOAI = [
  { ma: 'sai_y_ngu_canh', ten: 'AI hiểu sai ý hoặc ngữ cảnh' },
  { ma: 'uu_tien_dau_hieu_khac', ten: 'Có nhiều dấu hiệu và tôi ưu tiên dấu hiệu khác' },
  KHAC,
];

// Nhánh "AI bỏ sót nội dung nguy hại". Dấu hiệu trực tiếp lấy từ DAU_HIEU theo nhãn đã chọn.
export const LY_DO_BO_SOT = [
  { ma: 'dau_hieu_truc_tiep', ten: 'Dấu hiệu thể hiện trực tiếp', phuTheoNhan: true },
  {
    ma: 'can_luot_truoc', ten: 'Phải đọc các lượt trước mới thấy',
    phu: {
      hoi: 'Các lượt trước cho thấy điều gì?',
      options: [
        { ma: 'ham_y_nguy_hai', ten: 'Câu này mang hàm ý nguy hại' },
        { ma: 'hanh_vi_lap_lai', ten: 'Hành vi đã lặp lại' },
        { ma: 'nguoi_nhan_tu_choi', ten: 'Người nhận đã từ chối' },
      ],
    },
  },
  {
    ma: 'viet_bien_dang', ten: 'Nội dung được viết biến dạng',
    phu: {
      hoi: 'Bạn thấy kiểu viết nào?',
      options: [
        { ma: 'bo_dau', ten: 'Bỏ dấu' },
        { ma: 'viet_tat_teencode', ten: 'Viết tắt hoặc teencode' },
        { ma: 'chen_ky_tu', ten: 'Chèn dấu cách hoặc ký tự' },
        { ma: 'viet_dinh', ten: 'Viết dính từ' },
        { ma: 'ky_tu_dong_dang', ten: 'Dùng ký tự trông giống chữ' },
      ],
    },
  },
  KHAC,
];

// Nhánh "Nhãn đúng, phần tô sáng có vấn đề"
const VI_TRI_THIEU = {
  hoi: 'Dấu hiệu còn thiếu nằm ở đâu?',
  options: [
    { ma: 'cung_luot', ten: 'Cùng lượt tin' },
    { ma: 'luot_truoc', ten: 'Ở lượt trước' },
    { ma: 'nhieu_luot', ten: 'Nằm rải qua nhiều lượt' },
  ],
};
export const LY_DO_TO_SANG = [
  {
    ma: 'to_nham', ten: 'Tô nhầm đoạn', canToSang: true,
    phu: {
      hoi: 'Đoạn được tô có vấn đề gì?',
      options: [
        { ma: 'doan_vo_hai', ten: 'Đoạn đó vô hại' },
        { ma: 'lien_quan_mot_phan', ten: 'Chỉ một phần đoạn được tô có liên quan' },
        { ma: 'lien_quan_cho_khac', ten: 'Đoạn liên quan nằm ở chỗ khác' },
      ],
    },
  },
  { ma: 'to_thieu', ten: 'Tô thiếu đoạn', canToSang: true, phu: VI_TRI_THIEU },
  { ma: 'chua_to', ten: 'Thiếu đoạn cần tô sáng', canToSang: false, phu: VI_TRI_THIEU },
  KHAC,
];

// Nhánh "Vấn đề khác hoặc tôi chưa chắc"
export const LY_DO_KHAC = [
  {
    ma: 'chua_chac', ten: 'Chưa chắc về nhãn',
    phu: {
      hoi: 'Bạn đang phân vân điều gì?',
      options: [
        { ma: 'chua_chac_nguy_hai', ten: 'Chưa chắc tin có nguy hại' },
        { ma: 'phan_van_giua_nhan', ten: 'Phân vân giữa các nhãn nguy hại' },
        { ma: 'can_ngu_canh', ten: 'Cần thêm ngữ cảnh' },
      ],
    },
  },
  {
    ma: 'van_de_khac', ten: 'Vấn đề khác',
    phu: {
      hoi: 'Vấn đề nằm ở đâu?',
      batBuoc: true,
      options: [
        { ma: 'canh_bao_kho_hieu', ten: 'Thông báo cảnh báo khó hiểu' },
        { ma: 'canh_bao_qua_thuong_xuyen', ten: 'Cảnh báo xuất hiện quá thường xuyên' },
        { ma: 'muc_canh_bao', ten: 'Mức cảnh báo chưa phù hợp' },
        KHAC,
      ],
    },
  },
];

export const CHU = {
  tieuDe: 'Phản hồi kết quả AI',
  moDau: (tenNhan) => ['AI gợi ý nhãn ', tenNhan, '. Bạn thấy kết quả này có vấn đề gì?'],
  nhacMoTa: 'Đừng chép lại tin nhắn hoặc nhập thông tin cá nhân.',
  thongBaoGui: 'Phản hồi mức 1 không đính kèm tin nhắn, đoạn được tô sáng hoặc mã tài khoản.',
  huy: 'Hủy',
  gui: 'Gửi phản hồi',
  dangGui: 'Đang gửi…',
  daGui: 'Đã gửi phản hồi. Cảm ơn bạn đã góp ý.',
  loiGui: 'Chưa gửi được phản hồi. Câu trả lời của bạn vẫn được giữ, hãy thử lại.',
};

export const MO_TA_TOI_DA = 200;

const TEN_LUA_CHON = new Map();
for (const item of [...LOAI_BAO_CAO, ...LY_DO_AN_TOAN, ...LY_DO_SAI_LOAI, ...LY_DO_BO_SOT,
  ...LY_DO_TO_SANG, ...LY_DO_KHAC, ...Object.values(DAU_HIEU).flat()]) {
  TEN_LUA_CHON.set(item.ma, item.ten);
  for (const opt of item.phu?.options ?? []) if (!TEN_LUA_CHON.has(opt.ma)) TEN_LUA_CHON.set(opt.ma, opt.ten);
}

export function tenLuaChon(ma) {
  return TEN_LUA_CHON.get(ma) ?? ma;
}
