// 30 ca kiểm thử che / ẩn danh hóa PII (SafeViChi v3). DỮ LIỆU GIẢ LẬP: mọi số điện thoại, số tài khoản,
// CCCD, email, tên người, địa chỉ đều do nhóm tự bịa, không phải thông tin thật.
//
// Mỗi ca:
//   ky_vong : văn bản đã che kỳ vọng (so khớp tuyệt đối)
//   pii     : các chuỗi PII gốc KHÔNG được còn trong kết quả (còn → "sót")
//   giu     : các chuỗi KHÔNG phải PII phải còn nguyên trong kết quả (mất → "che nhầm")
//   cheThem : (tùy chọn) các đoạn người dùng tự che thêm ở bước xem trước, vị trí theo văn bản gốc
// File này dùng chung cho tests/che_thong_tin.test.mjs và scripts/do_che_thong_tin.mjs.

function doan(text, sub, lan = 0) {
  let i = -1;
  for (let k = 0; k <= lan; k++) i = text.indexOf(sub, i + 1);
  if (i < 0) throw new Error(`Không thấy "${sub}" trong ca kiểm thử`);
  return { start: i, end: i + sub.length };
}

const T28 = 'Gặp nhau ở quán Gió Chiều lúc 7h nhé';
const T29 = 'sdt 0912345678 của anh Bình';

export const CA_CHE_THONG_TIN = [
  // --- Số điện thoại ---
  {
    id: 1, nhom: 'sdt', mo_ta: 'SĐT 10 số viết liền',
    text: 'Gọi cho mình số 0912345678 nhé',
    ky_vong: 'Gọi cho mình số [SĐT] nhé',
    pii: ['0912345678'], giu: ['Gọi cho mình số', 'nhé'],
  },
  {
    id: 2, nhom: 'sdt', mo_ta: 'SĐT có dấu cách, teencode "sdt", không dấu',
    text: 'sdt cua e la 0912 345 678',
    ky_vong: 'sdt cua e la [SĐT]',
    pii: ['0912 345 678', '345 678'], giu: ['sdt cua e la'],
  },
  {
    id: 3, nhom: 'sdt', mo_ta: 'SĐT có dấu chấm và dấu gạch',
    text: 'Liên hệ 0987.654.321 hoặc 0365-123-456',
    ky_vong: 'Liên hệ [SĐT] hoặc [SĐT]',
    pii: ['0987.654.321', '0365-123-456'], giu: ['Liên hệ', 'hoặc'],
  },
  {
    id: 4, nhom: 'sdt', mo_ta: 'SĐT dạng +84',
    text: 'Hotline hỗ trợ: +84 912 345 678',
    ky_vong: 'Hotline hỗ trợ: [SĐT]',
    pii: ['+84 912 345 678', '912 345 678'], giu: ['Hotline hỗ trợ:'],
  },
  // --- Email ---
  {
    id: 5, nhom: 'email', mo_ta: 'Email giữa câu',
    text: 'Gửi CV qua email tran.thi.mai99@gmail.com giúp chị',
    ky_vong: 'Gửi CV qua email [EMAIL] giúp chị',
    pii: ['tran.thi.mai99', '@gmail.com'], giu: ['Gửi CV qua email', 'giúp chị'],
  },
  {
    id: 6, nhom: 'email', mo_ta: 'Email cuối câu, có dấu chấm',
    text: 'Liên hệ hotro.khachhang@ngan-hang-gia.vn.',
    ky_vong: 'Liên hệ [EMAIL].',
    pii: ['hotro.khachhang', 'ngan-hang-gia.vn'], giu: ['Liên hệ'],
  },
  // --- Số tài khoản / thẻ ---
  {
    id: 7, nhom: 'stk', mo_ta: 'STK có dấu cách, sau "stk", trước tên ngân hàng',
    text: 'Chuyển vào stk 1903 5678 9012 Techcombank nhé',
    ky_vong: 'Chuyển vào stk [STK] Techcombank nhé',
    pii: ['1903 5678 9012', '5678 9012'], giu: ['Chuyển vào stk', 'Techcombank nhé'],
  },
  {
    id: 8, nhom: 'stk', mo_ta: 'STK không dấu, sau "so tai khoan vietcombank"',
    text: 'so tai khoan vietcombank 0071000123456, ck ngay giup minh',
    ky_vong: 'so tai khoan vietcombank [STK], ck ngay giup minh',
    pii: ['0071000123456'], giu: ['so tai khoan vietcombank', 'ck ngay giup minh'],
  },
  {
    id: 9, nhom: 'stk', mo_ta: 'Số thẻ 16 số chia nhóm 4',
    text: 'Thẻ 4111 1111 1111 1111 hết hạn 12/28',
    ky_vong: 'Thẻ [STK] hết hạn 12/28',
    pii: ['4111 1111 1111 1111', '1111 1111'], giu: ['Thẻ', 'hết hạn 12/28'],
  },
  // --- OTP / mã xác minh ---
  {
    id: 10, nhom: 'otp', mo_ta: 'Mã OTP 6 số',
    text: 'Mã OTP của bạn là 482913, không chia sẻ cho ai',
    ky_vong: 'Mã OTP của bạn là [MÃ], không chia sẻ cho ai',
    pii: ['482913'], giu: ['Mã OTP của bạn là', 'không chia sẻ cho ai'],
  },
  {
    id: 11, nhom: 'otp', mo_ta: 'Teencode "code", 4 số',
    text: 'đọc giúp e cái code 5521 vừa gửi về đt với',
    ky_vong: 'đọc giúp e cái code [MÃ] vừa gửi về đt với',
    pii: ['5521'], giu: ['đọc giúp e cái code', 'vừa gửi về đt với'],
  },
  {
    id: 12, nhom: 'otp', mo_ta: 'Mã xác minh không dấu, các số cách nhau',
    text: 'ma xac minh la 7 3 9 1 0 2',
    ky_vong: 'ma xac minh la [MÃ]',
    pii: ['7 3 9 1 0 2', '9 1 0 2'], giu: ['ma xac minh la'],
  },
  // --- CCCD / CMND ---
  {
    id: 13, nhom: 'giay_to', mo_ta: 'CCCD 12 số',
    text: 'Số CCCD của tôi là 001203004567',
    ky_vong: 'Số CCCD của tôi là [GIẤY_TỜ]',
    pii: ['001203004567'], giu: ['Số CCCD của tôi là'],
  },
  {
    id: 14, nhom: 'giay_to', mo_ta: 'CMND 9 số, không dấu, có năm phải giữ',
    text: 'cmnd 125678901 cap nam 2015',
    ky_vong: 'cmnd [GIẤY_TỜ] cap nam 2015',
    pii: ['125678901'], giu: ['cmnd', 'cap nam 2015'],
  },
  // --- Địa chỉ ---
  {
    id: 15, nhom: 'dia_chi', mo_ta: 'Số nhà + tên đường + phường + quận',
    text: 'Ship tới 25 Lý Thường Kiệt, phường Trần Hưng Đạo, quận Hoàn Kiếm nha',
    ky_vong: 'Ship tới [ĐỊA_CHỈ] nha',
    pii: ['25 Lý Thường Kiệt', 'Trần Hưng Đạo', 'Hoàn Kiếm'], giu: ['Ship tới', 'nha'],
  },
  {
    id: 16, nhom: 'dia_chi', mo_ta: 'Số nhà + ngõ + phố + thành phố',
    text: 'nhà em ở số 12 ngõ 45 phố Huế, Hà Nội',
    ky_vong: 'nhà em ở [ĐỊA_CHỈ]',
    pii: ['số 12', 'ngõ 45', 'phố Huế'], giu: ['nhà em ở'],
  },
  {
    id: 17, nhom: 'dia_chi', mo_ta: 'Địa chỉ không dấu, số nhà có dấu "/"',
    text: 'dia chi: 88/3 duong Nguyen Trai, quan 5',
    ky_vong: 'dia chi: [ĐỊA_CHỈ]',
    pii: ['88/3', 'Nguyen Trai', 'quan 5'], giu: ['dia chi:'],
  },
  // --- Tên người ---
  {
    id: 18, nhom: 'ten', mo_ta: 'Họ tên đầy đủ sau "tên là"',
    text: 'Chào anh, em tên là Phạm Thu Hà, bên ngân hàng gọi',
    ky_vong: 'Chào anh, em tên là [TÊN_1], bên ngân hàng gọi',
    pii: ['Phạm Thu Hà', 'Phạm'], giu: ['Chào anh, em tên là', 'bên ngân hàng gọi'],
  },
  {
    id: 19, nhom: 'ten', mo_ta: 'Hai tên, cùng tên lặp lại → cùng số',
    text: 'anh Tuấn ơi, chị Lan nhờ em hỏi anh Tuấn chuyện hôm qua',
    ky_vong: 'anh [TÊN_1] ơi, chị [TÊN_2] nhờ em hỏi anh [TÊN_1] chuyện hôm qua',
    pii: ['Tuấn', 'Lan'], giu: ['ơi, chị', 'nhờ em hỏi anh', 'chuyện hôm qua'],
  },
  {
    id: 20, nhom: 'ten', mo_ta: 'Đa dòng: @nhắc tên và "mình là", giữ "\\n"',
    text: '@Minh mai đi học không\nkhông, mình là Khoa mà',
    ky_vong: '@[TÊN_1] mai đi học không\nkhông, mình là [TÊN_2] mà',
    pii: ['Minh', 'Khoa'], giu: ['mai đi học không\nkhông, mình là', ' mà'],
  },
  // --- Link ---
  {
    id: 21, nhom: 'link', mo_ta: 'Link lừa đảo: giữ tên miền, che đường dẫn và tham số',
    text: 'Quý khách vui lòng xác minh tại https://vietcombank-xacminh.com/login?id=123 để tránh bị khóa',
    ky_vong: 'Quý khách vui lòng xác minh tại [LINK:vietcombank-xacminh.com] để tránh bị khóa',
    pii: ['/login', 'id=123'], giu: ['vietcombank-xacminh.com', 'xác minh tại', 'để tránh bị khóa'],
  },
  {
    id: 22, nhom: 'link', mo_ta: 'Link rút gọn không có "https://"',
    text: 'vào bit.ly/3abcXYZ nhận quà',
    ky_vong: 'vào [LINK:bit.ly] nhận quà',
    pii: ['3abcXYZ'], giu: ['bit.ly', 'vào', 'nhận quà'],
  },
  // --- Số tiền phải giữ ---
  {
    id: 23, nhom: 'tien', mo_ta: 'Nhiều dạng số tiền, không có PII',
    text: 'Chuyển trước 500k, còn lại 2 triệu, tổng 1.500.000đ hoặc 3tr cũng được',
    ky_vong: 'Chuyển trước 500k, còn lại 2 triệu, tổng 1.500.000đ hoặc 3tr cũng được',
    pii: [], giu: ['500k', '2 triệu', '1.500.000đ', '3tr'],
  },
  {
    id: 24, nhom: 'tien', mo_ta: 'Số tiền 8 chữ số cạnh STK: chỉ che STK',
    text: 'ck 15000000 vnd vào tk 0123456789 MB nhé',
    ky_vong: 'ck 15000000 vnd vào tk [STK] MB nhé',
    pii: ['0123456789'], giu: ['15000000 vnd', 'vào tk', 'MB nhé'],
  },
  // --- Không có PII ---
  {
    id: 25, nhom: 'khong_pii', mo_ta: 'Câu thường, không có PII',
    text: 'Hôm nay trời đẹp quá, đi cà phê không mày',
    ky_vong: 'Hôm nay trời đẹp quá, đi cà phê không mày',
    pii: [], giu: ['Hôm nay trời đẹp quá, đi cà phê không mày'],
  },
  {
    id: 26, nhom: 'khong_pii', mo_ta: 'Câu có số nhưng không phải PII',
    text: 'Lớp mình có 35 bạn, năm 2024 thi được 8.5 điểm',
    ky_vong: 'Lớp mình có 35 bạn, năm 2024 thi được 8.5 điểm',
    pii: [], giu: ['Lớp mình có 35 bạn, năm 2024 thi được 8.5 điểm'],
  },
  {
    id: 27, nhom: 'khong_pii', mo_ta: 'Từ chửi viết hoa sau đại từ không phải tên (giữ bằng chứng XUC_PHAM)',
    text: 'Mày là thằng Ngu à',
    ky_vong: 'Mày là thằng Ngu à',
    pii: [], giu: ['thằng Ngu'],
  },
  // --- Che thêm thủ công ---
  {
    id: 28, nhom: 'che_them', mo_ta: 'Người dùng bôi đen che thêm tên quán',
    text: T28,
    cheThem: [doan(T28, 'Gió Chiều')],
    ky_vong: 'Gặp nhau ở quán [ĐÃ_CHE] lúc 7h nhé',
    pii: ['Gió Chiều'], giu: ['Gặp nhau ở quán', 'lúc 7h nhé'],
  },
  {
    id: 29, nhom: 'che_them', mo_ta: 'Đoạn che thêm rỗng / ngược / vượt biên không bỏ được che tự động',
    text: T29,
    cheThem: [{ start: 0, end: 0 }, { start: 50, end: 10 }, { start: 4, end: 4 }, { start: -5, end: -1 }],
    ky_vong: 'sdt [SĐT] của anh [TÊN_1]',
    pii: ['0912345678', 'Bình'], giu: ['sdt', 'của anh'],
  },
  // --- Đa dòng, kịch bản demo lừa đảo ---
  {
    id: 30, nhom: 'da_dong', mo_ta: 'Kịch bản demo mức 2, 2 dòng: tên + OTP + STK + SĐT, giữ "\\n"',
    text: 'Chị Lan ơi em bên Vietcombank\nchị gửi mã OTP 482913 và stk 0123456789 cho em số 0912 345 678',
    ky_vong: 'Chị [TÊN_1] ơi em bên Vietcombank\nchị gửi mã OTP [MÃ] và stk [STK] cho em số [SĐT]',
    pii: ['Lan', '482913', '0123456789', '0912 345 678'], giu: ['Vietcombank\nchị gửi mã OTP', 'cho em số'],
  },
];
