// Câu chữ và mã cho biểu mẫu phản hồi mức 1.
// Nguồn câu chữ: docs/v3/DE_XUAT_TRIEN_KHAI_CHANG_2.md, Mục 1.
// MÃ TẠM: phần mã (chuỗi snake_case) do giao diện đặt trước để làm song song. Khi bảng mã chính thức
// (Mục 2, ưu tiên 1) chốt xong, chỉ cần sửa các chuỗi `ma` trong file này, không phải sửa giao diện.

export const NHAN_NGUY_HAI = ['XUC_PHAM', 'QUAY_ROI', 'DE_DOA', 'LUA_DAO'];

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
    { ma: 'doa_than_the_tai_san', ten: 'Dọa gây hại thân thể hoặc tài sản' },
    { ma: 'doa_danh_du', ten: 'Dọa làm hại danh dự' },
    { ma: 'cuong_ep_tong_tien', ten: 'Cưỡng ép hoặc tống tiền' },
  ],
  LUA_DAO: [
    { ma: 'mao_danh', ten: 'Mạo danh' },
    { ma: 'du_chuyen_tien', ten: 'Dụ chuyển tiền' },
    { ma: 'du_otp_mat_khau', ten: 'Dụ cung cấp OTP hoặc mật khẩu' },
    { ma: 'du_bam_link_cai_app', ten: 'Dụ bấm liên kết hoặc cài ứng dụng' },
  ],
};

const KHAC = { ma: 'khac', ten: 'Khác', khac: true };

// Loại báo cáo. `hien(nhanMoHinh)` là điều kiện hiển thị theo bảng ở Mục 1.
export const LOAI_BAO_CAO = [
  { ma: 'tin_an_toan', ten: 'Tin nhắn này an toàn', hien: (n) => n !== 'AN_TOAN' },
  { ma: 'sai_loai', ten: 'Có vấn đề, nhưng AI chọn sai loại', hien: (n) => n !== 'AN_TOAN' },
  { ma: 'bo_sot', ten: 'AI bỏ sót nội dung nguy hại', hien: (n) => n === 'AN_TOAN' },
  { ma: 'to_sang', ten: 'Nhãn đúng, phần tô sáng có vấn đề', hien: (n) => n !== 'AN_TOAN' },
  { ma: 'khac_chua_chac', ten: 'Vấn đề khác hoặc tôi chưa chắc', hien: () => true },
];

// Nhánh "Tin nhắn này an toàn"
export const LY_DO_AN_TOAN = [
  {
    ma: 'dua_noi_than_mat', ten: 'Đùa hoặc nói thân mật',
    phu: {
      hoi: 'Điều gì giúp bạn nhận ra đây là lời đùa hoặc cách nói thân mật?',
      options: [
        { ma: 'cach_hai_ben_noi_chuyen', ten: 'Cách hai bên nói chuyện với nhau' },
        { ma: 'cau_tra_loi_nguoi_nhan', ten: 'Câu trả lời của người nhận' },
        { ma: 'cac_luot_truoc', ten: 'Các lượt trước' },
      ],
    },
  },
  {
    ma: 'tu_long_dia_phuong', ten: 'Từ lóng hoặc cách nói địa phương',
    phu: {
      hoi: 'Điều gì giúp bạn hiểu đúng cách nói này?',
      options: [
        { ma: 'nghia_quen_thuoc', ten: 'Nghĩa quen thuộc của từ' },
        { ma: 'cach_dung_tu_trong_cau', ten: 'Cách dùng từ trong câu' },
        { ma: 'cac_luot_truoc', ten: 'Các lượt trước' },
      ],
    },
  },
  {
    ma: 'trich_dan_ke_lai', ten: 'Trích dẫn hoặc kể lại',
    phu: {
      hoi: 'Bạn nhận ra đây là lời trích dẫn hoặc lời kể bằng cách nào?',
      options: [
        { ma: 'dau_hieu_trich_dan', ten: 'Có dấu hiệu trích dẫn' },
        { ma: 'nguoi_gui_noi_ro_ke_lai', ten: 'Người gửi nói rõ họ đang kể lại' },
        { ma: 'luot_truoc_loi_nguoi_khac', ten: 'Các lượt trước cho biết đó là lời người khác' },
      ],
    },
  },
  {
    ma: 'ngu_canh_binh_thuong', ten: 'Ngữ cảnh cho thấy tin nhắn bình thường',
    phu: {
      hoi: 'Tin nhắn thuộc tình huống nào?',
      options: [
        { ma: 'cong_viec_hoc_tap', ten: 'Công việc hoặc học tập' },
        { ma: 'giao_dich_xac_minh_hop_le', ten: 'Giao dịch hoặc xác minh hợp lệ' },
        { ma: 'tro_chuyen_hang_ngay', ten: 'Trò chuyện hằng ngày' },
      ],
    },
  },
  KHAC,
];

// Nhánh "Có vấn đề, nhưng AI chọn sai loại"
export const LY_DO_SAI_LOAI = [
  { ma: 'hieu_sai_ngu_canh', ten: 'AI hiểu sai ý hoặc ngữ cảnh' },
  { ma: 'uu_tien_dau_hieu_khac', ten: 'Có nhiều dấu hiệu và tôi ưu tiên dấu hiệu khác' },
  KHAC,
];

// Nhánh "AI bỏ sót nội dung nguy hại". Phần phụ của `truc_tiep` lấy từ DAU_HIEU theo nhãn đã chọn.
export const LY_DO_BO_SOT = [
  { ma: 'dau_hieu_truc_tiep', ten: 'Dấu hiệu thể hiện trực tiếp', phuTheoNhan: true },
  {
    ma: 'phai_doc_luot_truoc', ten: 'Phải đọc các lượt trước mới thấy',
    phu: {
      hoi: 'Các lượt trước cho thấy điều gì?',
      options: [
        { ma: 'ham_y_nguy_hai', ten: 'Câu này mang hàm ý nguy hại' },
        { ma: 'hanh_vi_lap_lai', ten: 'Hành vi đã lặp lại' },
        { ma: 'nguoi_nhan_da_tu_choi', ten: 'Người nhận đã từ chối' },
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
        { ma: 'chen_cach_ky_tu', ten: 'Chèn dấu cách hoặc ký tự' },
        { ma: 'viet_dinh_tu', ten: 'Viết dính từ' },
        { ma: 'ky_tu_giong_chu', ten: 'Dùng ký tự trông giống chữ' },
      ],
    },
  },
  KHAC,
];

// Nhánh "Nhãn đúng, phần tô sáng có vấn đề"
const VI_TRI_THIEU = {
  hoi: 'Dấu hiệu còn thiếu nằm ở đâu?',
  options: [
    { ma: 'cung_luot_tin', ten: 'Cùng lượt tin' },
    { ma: 'o_luot_truoc', ten: 'Ở lượt trước' },
    { ma: 'rai_nhieu_luot', ten: 'Nằm rải qua nhiều lượt' },
  ],
};
export const LY_DO_TO_SANG = [
  {
    ma: 'to_nham_doan', ten: 'Tô nhầm đoạn', canToSang: true,
    phu: {
      hoi: 'Đoạn được tô có vấn đề gì?',
      options: [
        { ma: 'doan_vo_hai', ten: 'Đoạn đó vô hại' },
        { ma: 'chi_mot_phan_lien_quan', ten: 'Chỉ một phần đoạn được tô có liên quan' },
        { ma: 'doan_lien_quan_o_cho_khac', ten: 'Đoạn liên quan nằm ở chỗ khác' },
      ],
    },
  },
  { ma: 'to_thieu_doan', ten: 'Tô thiếu đoạn', canToSang: true, phu: VI_TRI_THIEU },
  { ma: 'thieu_doan_can_to', ten: 'Thiếu đoạn cần tô sáng', canToSang: false, phu: VI_TRI_THIEU },
  KHAC,
];

// Nhánh "Vấn đề khác hoặc tôi chưa chắc"
export const LY_DO_KHAC = [
  {
    ma: 'chua_chac_nhan', ten: 'Chưa chắc về nhãn',
    phu: {
      hoi: 'Bạn đang phân vân điều gì?',
      options: [
        { ma: 'chua_chac_nguy_hai', ten: 'Chưa chắc tin có nguy hại' },
        { ma: 'phan_van_giua_nhan', ten: 'Phân vân giữa các nhãn nguy hại' },
        { ma: 'can_them_ngu_canh', ten: 'Cần thêm ngữ cảnh' },
      ],
    },
  },
  {
    ma: 'van_de_khac', ten: 'Vấn đề khác',
    phu: {
      hoi: 'Vấn đề nằm ở đâu?',
      batBuoc: true,
      options: [
        { ma: 'thong_bao_kho_hieu', ten: 'Thông báo cảnh báo khó hiểu' },
        { ma: 'canh_bao_qua_thuong_xuyen', ten: 'Cảnh báo xuất hiện quá thường xuyên' },
        { ma: 'muc_canh_bao_chua_phu_hop', ten: 'Mức cảnh báo chưa phù hợp' },
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
