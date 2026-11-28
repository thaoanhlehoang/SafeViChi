# Kế hoạch SafeViChi – chặng 2: phản hồi, kháng nghị và sửa nhãn

Thời lượng: **8 giờ**. Nhóm: 2 người (gọi là **A** và **B**). Nền làm việc: `web-v2/` (demo chạy mô hình ONNX trên trình duyệt).
Đây **không** phải bài học tăng cường (RL): không có vòng lặp tự học, mọi thay đổi mô hình đi qua con người.

## 1. Đề bài và cách hiểu

Ban tổ chức nêu 3 vấn đề:

1. Gán nhãn độc hại tự động có thể ảnh hưởng tự do ngôn luận và gây khiếu nại.
2. Phải có cơ chế **phản hồi, kháng nghị, sửa nhãn**; **giới hạn quyền quyết định cuối cùng của AI**.
3. **Giảm lưu trữ thông tin nhạy cảm.**

Hiện trạng SafeViChi v2: mô hình chạy trên thiết bị, chỉ **gắn nhãn và cảnh báo**, không bao giờ chặn
(nút "Vẫn gửi" luôn có trong `web-v2/index.html`, `XUC_PHAM` chỉ làm mờ, bấm là xem). Không có backend,
không gửi tin nhắn ra ngoài. Vì vậy điểm xuất phát đã tốt cho yêu cầu 2 và 3; việc của chặng 2 là:

- Thêm đường để người dùng nói "AI sai rồi", ở hai mức, mức mặc định **không gửi nội dung tin nhắn**.
- Thêm màn hình cho người duyệt và một quy trình để phản hồi trở thành kịch bản huấn luyện lại hoặc chỉnh ngưỡng,
  **luôn do người quyết và có nhật ký**.
- Viết thành văn bản các giới hạn quyền của AI và chứng minh được chúng trong demo.

## 2. Nguyên tắc (đặt lên slide đầu)

1. **AI chỉ cảnh báo, không chặn, không xóa.** Người dùng luôn có quyền "Vẫn gửi", "Vẫn xem".
2. **Con người quyết định.** Người dùng quyết với tin của mình; người duyệt quyết nhãn đúng; quản trị viên quyết thay đổi mô hình.
3. **Mọi thay đổi mô hình hoặc ngưỡng phải được người phê duyệt và ghi nhật ký** (ai, khi nào, vì sao, số đo trước/sau).
4. **Tối thiểu hóa dữ liệu.** Mặc định không gửi nội dung. Muốn gửi nội dung thì người dùng phải chủ động đồng ý, xem trước, và
   thông tin cá nhân đã được che ngay trên máy.
5. **Phản hồi không đi thẳng vào huấn luyện.** Phản hồi mức 1 chỉ là tín hiệu thống kê; chỉ mẫu mức 2 do người duyệt xác nhận mới được dùng để huấn luyện lại.

## 3. Thiết kế

### 3.1. Mức 1 – báo nhãn sai, không gửi nội dung (mặc định)

**Điểm vào trên giao diện:**

- Trên cột AI (`renderAI` trong `web-v2/src/chat.js`) và trên bong bóng tin đã gắn nhãn: nút **"Nhãn này sai?"**.
- Trên tin được gắn `AN_TOAN`: nút nhỏ **"Báo bỏ sót"** (để bắt false negative, không chỉ false positive).
- Trên hộp cảnh báo trước khi gửi (`showWarning`): thêm liên kết "AI hiểu sai?" bên cạnh "Chỉnh sửa" / "Vẫn gửi".

**Biểu mẫu (một màn, 3 bước, tối đa 10 giây):**

1. Chọn nhãn đúng (5 nhãn, nhãn mô hình đoán được bỏ chọn sẵn).
2. Chọn lý do từ danh sách:
   - Đùa giữa bạn bè, người nhận không thấy bị xúc phạm
   - Tiếng lóng, từ địa phương, không có ý xấu
   - Trích dẫn hoặc kể lại lời người khác
   - Có nguy hại nhưng sai loại (ví dụ là lừa đảo chứ không phải quấy rối)
   - Tin này nguy hại nhưng AI không cảnh báo (cho "Báo bỏ sót")
   - Cụm từ bị tô sáng không phải chỗ có vấn đề
   - **Lý do khác** → ô chữ tự do, **tối đa 200 ký tự**, chạy qua bộ che thông tin (mục 3.3) trước khi gửi,
     hiện dòng nhắc "Đừng chép lại nội dung tin nhắn vào đây".
3. Bấm "Gửi phản hồi". Hiện thông báo ngắn: "Đã gửi. Phản hồi không kèm nội dung tin nhắn."

**Dữ liệu gửi lên máy chủ (toàn bộ gói, không có gì khác):**

```json
{
  "loai": "muc_1",
  "nhan_mo_hinh": "XUC_PHAM",
  "nhan_dung": "AN_TOAN",
  "ly_do": "dua_ban_be",
  "ly_do_khac": null,
  "khoang_diem": "0.6-0.7",
  "phien_ban_mo_hinh": "dualorc-mmbert-int8@v2.0",
  "phien_ban_nguong": "nguong@2026-11-26",
  "so_luot_ngu_canh": 3,
  "do_dai": "ngan",
  "bam_cum": "sha256:9f2c…",
  "ma_thiet_bi": "c3b1…(ngẫu nhiên, ẩn danh)",
  "ngay": "2026-11-28"
}
```

Lưu ý thiết kế:

- Điểm làm tròn về khoảng 0,1 và độ dài chỉ ghi `ngan/vua/dai` để gói tin không thành dấu vân tay.
- Chỉ ghi **ngày**, không ghi giờ phút.
- `bam_cum` (tùy chọn): băm SHA-256 của cụm bị tô sáng, giúp đếm "cùng một cụm bị báo sai nhiều lần".
  Phải nói thật: cụm ngắn có thể đoán ngược bằng từ điển, nên đây là **giảm lộ**, không phải ẩn danh tuyệt đối.
  Mặc định bật cho `XUC_PHAM` (cụm thường là từ lóng, ít thông tin cá nhân), tắt cho các nhãn khác.
- **Không gửi** mã tài khoản đăng nhập (Firebase trong `web-v2/src/auth.js`), số điện thoại, IP hay tên liên hệ.
  `ma_thiet_bi` là UUID ngẫu nhiên tạo trong `localStorage`, chỉ dùng để giới hạn tần suất (mục 3.5).

**Trên thiết bị – làm dịu cảnh báo khi người dùng báo sai lặp lại:**

- Đếm số lần báo sai theo nhãn trong `localStorage` (chỉ đếm, không lưu tin).
- Báo sai cùng một nhãn **3 lần trong 7 ngày** → cảnh báo nhãn đó hạ một bậc trình bày:
  - `XUC_PHAM`: không làm mờ nữa, chỉ hiện chip nhãn nhỏ.
  - `QUAY_ROI`: bỏ gợi ý chặn, giữ chip.
  - `DE_DOA`, `LUA_DAO`: **có mức sàn**: vẫn hiện chip và vẫn vô hiệu link lừa đảo, chỉ bỏ hộp cảnh báo lớn.
    Lý do: làm dịu hoàn toàn nhãn lừa đảo có thể khiến người dùng mất tiền thật.
- Có nút "Khôi phục cảnh báo mặc định" trong phần cài đặt. Việc làm dịu **không đổi mô hình**, chỉ đổi cách hiện.

**Trên máy chủ – báo cáo tổng hợp hằng ngày và hằng tuần:**

- Bảng: nhãn nào bị báo sai nhiều nhất, theo lý do, theo khoảng điểm, theo phiên bản mô hình.
- Ví dụ một dòng đọc được: "Tuần này `XUC_PHAM` bị báo sai 41 lần, 70% lý do *tiếng lóng*, phần lớn ở khoảng điểm 0,5–0,7."
- Từ đó đội hệ thống **tự tay** viết kịch bản huấn luyện lại (ví dụ sinh thêm câu tiếng lóng vô hại theo
  `docs/v2/KICH_BAN.md`) hoặc đề xuất chỉnh ngưỡng. Mức 1 **không bao giờ** được chuyển thẳng thành mẫu huấn luyện.

### 3.2. Mức 2 – gửi kèm tin nhắn (người dùng chủ động chọn)

**Đồng ý:**

- Trong biểu mẫu mức 1 có ô **"Gửi kèm đoạn tin nhắn để giúp cải thiện SafeViChi"**, **không đánh dấu sẵn**.
- Bấm vào thì hiện thông báo ngắn, rõ:
  > Đoạn tin nhắn sẽ được che số điện thoại, email, tên, số giấy tờ, địa chỉ, số tài khoản ngay trên máy bạn
  > trước khi gửi. Dữ liệu chỉ dùng để người duyệt của SafeViChi kiểm tra nhãn và cải thiện sản phẩm,
  > không gắn với tài khoản hay số điện thoại của bạn. Bạn có thể tắt lựa chọn này bất cứ lúc nào trong Cài đặt.
- Đồng ý được hỏi **mỗi lần** gửi mức 2 (hoặc người dùng chọn "nhớ lựa chọn"); tắt trong Cài đặt thì từ lúc đó không gửi nữa.
- Đồng ý theo từng lần, không phải một lần cho mọi thứ: ghi trên máy thời điểm đồng ý và phiên bản thông báo.

**Che thông tin trên thiết bị:** xem mục 3.3. Gọi đúng tên là **che / ẩn danh hóa**, không gọi là mã hóa.

**Xem trước:** người dùng thấy chính xác đoạn sẽ gửi (đã che), có thể bôi đen để **che thêm** (thành `[ẨN]`),
không thể bỏ che. Nút "Gửi" chỉ bật sau khi đã hiện xem trước.

**Lưu trữ phía máy chủ – cơ sở dữ liệu cho người duyệt (không dùng Google Sheet):**

- Demo: **SQLite** sau API của `web-v2/serve.py` (cùng origin nên không phải nới CSP `connect-src`).
  Bản thật: Supabase/Postgres có Row Level Security và mã hóa lưu trữ.
- Phân quyền theo vai trò: `nguoi_duyet` (xem hàng đợi, gán nhãn), `quan_tri` (phê duyệt thay đổi mô hình/ngưỡng).
- **Nhật ký truy cập và sửa**: mỗi lần mở một mẫu, mỗi lần gán nhãn đều ghi `ai, lúc nào, mẫu nào, hành động`.
- **Không có chức năng xuất hàng loạt**: API trả từng mẫu một, không có endpoint xuất CSV. Nói thật với giám khảo:
  việc này giảm rủi ro chứ không loại bỏ được (người duyệt vẫn chụp màn hình được), nên có thêm nhật ký truy cập.

**Duyệt và dùng để huấn luyện:**

- Người duyệt gán lại nhãn và cụm bằng chứng bằng tay, theo `docs/v2/HUONG_DAN_GAN_NHAN.md`.
- Nhãn nhạy cảm (`DE_DOA`, và `XUC_PHAM` nhắm vào nhóm người hoặc vùng miền) cần **2 người duyệt** cùng xác nhận
  một nhãn thì mẫu mới đủ điều kiện; khác nhau thì mẫu ở lại hàng đợi.
- Chỉ mẫu mức 2 đã xác nhận mới vào kho huấn luyện lại, với `cach_gan="nguoi"`, `nguon="phan_hoi"`.
  Mẫu phản hồi **chỉ vào train**, không vào test đã đóng băng (giữ quy tắc trong `CLAUDE.md`).

### 3.3. Che thông tin cá nhân trên thiết bị

Chạy trong trình duyệt, trước khi bất kỳ byte nào rời máy. Thay bằng nhãn giữ chỗ:

| Loại | Cách nhận ra (regex + từ khóa) | Thay bằng |
| --- | --- | --- |
| Số điện thoại VN | `0` hoặc `+84` + 9 số, cho phép dấu cách, chấm, gạch | `[SĐT]` |
| Email | mẫu email chuẩn | `[EMAIL]` |
| CCCD / CMND | 12 số hoặc 9 số đứng riêng | `[GIẤY_TỜ]` |
| Số tài khoản / thẻ | 6–19 số gần `stk`, `tk`, `số tài khoản`, `ngân hàng`, tên ngân hàng; 16 số dạng thẻ | `[STK]` |
| Mã OTP | 4–8 số gần `otp`, `mã`, `code` | `[MÃ]` |
| Địa chỉ | số nhà + `đường/phố/ngõ/hẻm/phường/xã/quận/huyện` | `[ĐỊA_CHỈ]` |
| Tên người | sau đại từ xưng hô (`anh/chị/em/bạn/cô/chú/thằng/con`) + từ viết hoa; tên liên hệ trong cuộc trò chuyện; `@nhắc tên` | `[TÊN_1]`, `[TÊN_2]`… |
| Đường link | giữ tên miền (cần cho `LUA_DAO`), bỏ đường dẫn và tham số | `[LINK:ten-mien.vn]` |

- Tên được đánh số **nhất quán trong một lần gửi**: cùng một người luôn là `[TÊN_1]`, để người duyệt vẫn hiểu ai nói với ai.
- **Không lưu bảng ánh xạ** tên thật ↔ `[TÊN_1]`, kể cả trên máy. Không có cách khôi phục.
- Giữ nguyên số tiền (cần để nhận ra lừa đảo).
- Regex không bao giờ hoàn hảo (teencode, số viết bằng chữ "không chín…", tên viết thường). Vì vậy **xem trước là lớp
  phòng thủ cuối cùng**, và nói rõ điều này với giám khảo.
- Đo: bộ 30 câu tự viết có thông tin cá nhân, báo "che đúng x/30, sót các ca …". Số này sinh từ test, không gõ tay.

### 3.4. Giới hạn quyền quyết định của AI (viết thành điều khoản, hiện được trong demo)

| Giới hạn | Hiện trong sản phẩm thế nào |
| --- | --- |
| AI không chặn gửi | Nút "Vẫn gửi" luôn có; không có nhánh code nào chặn |
| AI không xóa, không ẩn vĩnh viễn | `XUC_PHAM` chỉ làm mờ, bấm là xem |
| AI nói rõ mình có thể sai | Dòng "AI có thể nhầm – báo cho chúng tôi" cạnh mỗi nhãn; hiện độ tin cậy |
| AI không tự học | Không có vòng lặp tự cập nhật; phản hồi chỉ vào báo cáo và hàng đợi người duyệt |
| Thay đổi mô hình/ngưỡng phải có người phê duyệt | Bảng `de_xuat_thay_doi`: người đề xuất, lý do, số đo trên validation trước/sau, người phê duyệt; ngưỡng mới chỉ có hiệu lực khi `trang_thai="da_duyet"` |
| Người dùng thấy phiên bản đang chạy | Cột AI ghi `phiên bản mô hình` và `phiên bản ngưỡng` |

Đề xuất chỉnh ngưỡng phải tuân thủ quy tắc cũ: dò trên validation, không nhìn test.

### 3.5. Bảo mật và chống đầu độc dữ liệu

- **Giới hạn tần suất theo thiết bị** bằng `ma_thiet_bi` ẩn danh: tối đa 20 phản hồi mức 1 và 5 phản hồi mức 2 mỗi ngày.
  Bộ đếm theo IP chỉ giữ trong bộ nhớ vài phút để chặn spam, **không ghi xuống đĩa, không gắn với phản hồi**.
- **Cảnh báo đột biến**: số báo sai của một nhãn trong 1 giờ vượt 3 lần trung bình 7 ngày → gắn cờ "nghi đầu độc",
  tách khỏi báo cáo cho tới khi quản trị viên xem.
- Mức 1 không vào huấn luyện, mức 2 phải qua người duyệt (2 người với nhãn nhạy cảm): đó chính là lớp chống đầu độc chính.
- Không liên kết mã tài khoản, số điện thoại, IP với phản hồi.
- HTTPS khi triển khai; lưu trữ mã hóa ở bản thật (Postgres/Supabase). Bản demo SQLite cục bộ thì nói thẳng là chưa mã hóa.
- Pháp lý: căn cứ **Nghị định 13/2023/NĐ-CP** về bảo vệ dữ liệu cá nhân (đồng ý rõ ràng, đúng mục đích, tối thiểu hóa,
  bên xử lý chịu trách nhiệm) và **Luật Bảo vệ dữ liệu cá nhân 2025** (hiệu lực từ 01/01/2026; kiểm tra lại số hiệu và
  điều khoản trước khi đưa lên slide).
- Cập nhật mục "Bảo mật & Quyền riêng tư" trong `web-v2/README.md`: phân tích vẫn 100% trên máy; chỉ khi người dùng bấm
  gửi phản hồi mới có request, và gói tin là gói ở mục 3.1/3.2.

### 3.6. Ngoài phạm vi (đã chốt bỏ)

- Mã theo dõi kháng nghị, yêu cầu xóa dữ liệu đã gửi.
- Cam kết thời hạn lưu trữ cụ thể.
- Biện pháp đo thiên kiến người duyệt (kiểm tra chéo, đo đồng thuận).

Nếu giám khảo hỏi, trả lời ở mục 7.

## 4. Mã cần thêm (dự kiến, chỉ thêm mới, không đụng mô hình)

```
web-v2/
  src/phan_hoi/
    bieu_mau.js        # biểu mẫu mức 1 + ô đồng ý mức 2 + xem trước
    che_thong_tin.js   # che PII trên thiết bị (mục 3.3)
    lam_diu.js         # đếm báo sai theo nhãn, hạ bậc cảnh báo
    gui.js             # gửi gói tin, ma_thiet_bi
  duyet.html           # màn người duyệt (đăng nhập vai trò demo)
  src/duyet/           # hàng đợi mức 2, báo cáo mức 1, đề xuất thay đổi, nhật ký
  server/
    api.py             # gắn vào serve.py: /api/phan-hoi, /api/duyet/*, /api/de-xuat/*
    schema.sql         # bảng bên dưới
  tests/
    che_thong_tin.test.mjs   # 30 câu PII tự viết
```

Bảng SQLite:

| Bảng | Cột chính |
| --- | --- |
| `phan_hoi_muc_1` | nhãn mô hình, nhãn đúng, lý do, lý do khác (đã che), khoảng điểm, phiên bản, băm cụm, ngày, cờ nghi đầu độc |
| `mau_muc_2` | đoạn đã che, nhãn mô hình, nhãn người dùng chọn, lý do, phiên bản thông báo đồng ý, trạng thái |
| `duyet` | mẫu, người duyệt, nhãn, cụm bằng chứng, thời điểm |
| `de_xuat_thay_doi` | nội dung (ngưỡng/kịch bản), lý do, số đo validation trước/sau, người đề xuất, người phê duyệt, trạng thái |
| `nhat_ky` | ai, hành động (xem/gán/duyệt/phê duyệt), đối tượng, thời điểm |
| `nguoi_dung_noi_bo` | tài khoản người duyệt, vai trò |

## 5. Lịch 8 giờ

Mốc tính từ lúc bắt đầu (0:00). Mỗi khối có một kết quả kiểm được.

| Giờ | A | B | Xong khi |
| --- | --- | --- | --- |
| **0:00–1:00 Thiết kế** | Chốt danh sách lý do, gói tin mức 1/2, bảng SQLite | Chốt luồng giao diện (vẽ tay 4 màn: nút báo sai, biểu mẫu, xem trước, màn duyệt), nội dung thông báo đồng ý | Hai người cùng đọc lại mục 3, không còn câu hỏi mở |
| **1:00–5:00 Xây dựng (4 giờ)** | `server/`: schema, API nhận phản hồi, hàng đợi duyệt, quy tắc 2 người duyệt, đề xuất thay đổi + phê duyệt, nhật ký, giới hạn tần suất, cảnh báo đột biến | `src/phan_hoi/`: nút "Nhãn này sai?" và "Báo bỏ sót", biểu mẫu mức 1, làm dịu cảnh báo; dòng "AI có thể nhầm" và phiên bản mô hình ở cột AI; `duyet.html` (hàng đợi, báo cáo mức 1 dạng bảng) | 3:00: gửi được mức 1 từ demo vào SQLite. 5:00: đi hết luồng mức 1 → báo cáo, mức 2 → duyệt (chưa che PII) |
| **5:00–6:30 Che thông tin + quyền riêng tư** | Ô đồng ý không tích sẵn, lưu trạng thái đồng ý, tắt trong Cài đặt; cập nhật README phần bảo mật | `che_thong_tin.js` + màn xem trước có che thêm; viết 30 câu PII và chạy test | Gửi mức 2 chỉ thấy bản đã che trong DB; test che có số x/30 |
| **6:30–8:00 Slide + kịch bản demo** | Slide (mục 6, 7), chụp màn hình | Diễn tập demo 2 lần, ghi video dự phòng, kiểm tra F12 → Network | Video dự phòng có sẵn; demo chạy dưới 4 phút |

**Cắt giảm nếu trễ** (theo thứ tự cắt): biểu đồ báo cáo → để bảng; cảnh báo đột biến → chỉ giữ giới hạn tần suất;
quy tắc 2 người duyệt → nói trên slide, demo 1 người; băm cụm → bỏ. **Không cắt**: biểu mẫu mức 1, che + xem trước
mức 2, nhật ký phê duyệt thay đổi. Đó là ba thứ trả lời trực tiếp đề bài.

## 6. Kịch bản demo (khoảng 4 phút)

1. **Báo nhãn sai mức 1 (45 giây).** Trong trò chuyện bạn bè, gửi "m ngu vl haha thua game r" → AI gắn `XUC_PHAM`,
   hỏi "Bạn có muốn sửa không?" → bấm **Vẫn gửi** (AI không chặn). Bấm "Nhãn này sai?" → chọn `AN_TOAN`, lý do
   "Đùa giữa bạn bè" → Gửi. Mở F12 → Network: gói tin chỉ có nhãn, lý do, khoảng điểm, **không có chữ nào của tin nhắn**.
2. **Làm dịu cảnh báo (20 giây).** Báo sai thêm 2 lần → tin `XUC_PHAM` tiếp theo không bị làm mờ, chỉ còn chip nhỏ.
   Cho thấy với `LUA_DAO` vẫn giữ mức sàn.
3. **Mức 2 có đồng ý (60 giây).** Tin lừa đảo "Chị Lan ơi em bên Vietcombank, chị gửi mã OTP 482913 và stk
   0123456789 cho em số 0912 345 678" bị AI gắn sai `QUAY_ROI`. Báo sai → tích ô gửi kèm (ô không tích sẵn) → đọc
   thông báo → xem trước: `[TÊN_1]`, `[MÃ]`, `[STK]`, `[SĐT]` → bôi đen che thêm một từ → Gửi.
4. **Màn người duyệt (60 giây).** Đăng nhập vai trò `nguoi_duyet` → mẫu vừa gửi nằm trong hàng đợi, chỉ thấy bản đã che
   → gán `LUA_DAO` + cụm bằng chứng. Mở một mẫu `DE_DOA`: hiện "cần người duyệt thứ 2". Mở **nhật ký**: thấy dòng
   "nguoi_duyet_1 xem mẫu #12, gán LUA_DAO". Không có nút xuất.
5. **Báo cáo và phê duyệt thay đổi (45 giây).** Báo cáo tuần: `XUC_PHAM` bị báo sai nhiều nhất, lý do tiếng lóng, khoảng
   điểm 0,5–0,7. Tạo đề xuất "nâng ngưỡng `XUC_PHAM`" kèm số đo validation → đăng nhập `quan_tri` → phê duyệt → nhật ký
   ghi lại → cột AI hiện phiên bản ngưỡng mới.
6. **Chốt (10 giây).** Slide nguyên tắc: AI cảnh báo, người quyết, mọi thay đổi có dấu vết.

Chuẩn bị sẵn: DB demo có khoảng 50 phản hồi mức 1 giả lập (ghi rõ là dữ liệu giả lập để demo) để báo cáo có hình dạng.

## 7. Nói gì với giám khảo

**Ba câu chính:**

1. "SafeViChi chưa bao giờ chặn hay xóa. AI chỉ cảnh báo; người dùng, người duyệt và quản trị viên là người quyết."
2. "Kháng nghị mặc định không gửi một chữ nào của tin nhắn. Muốn gửi nội dung thì người dùng phải tự tích, xem trước, và
   thông tin cá nhân đã được che ngay trên máy."
3. "Phản hồi không tự đi vào mô hình. Nó thành báo cáo, rồi thành kịch bản huấn luyện hoặc đề xuất ngưỡng do người phê duyệt, có nhật ký."

**Câu hỏi có thể gặp:**

| Câu hỏi | Trả lời |
| --- | --- |
| Kẻ xấu spam báo sai để làm mô hình yếu đi? | Mức 1 không vào huấn luyện; có giới hạn tần suất theo thiết bị và cảnh báo đột biến; mức 2 phải qua người duyệt, nhãn nhạy cảm cần 2 người. |
| Che thông tin bằng regex có sót không? | Có. Đo được x/30, các ca sót là … Vì vậy có bước xem trước và người dùng che thêm được. Đây là che, không phải mã hóa. |
| Làm dịu cảnh báo có nguy hiểm với lừa đảo không? | Có mức sàn cho `DE_DOA` và `LUA_DAO`: vẫn hiện nhãn, vẫn vô hiệu link. Làm dịu chỉ đổi cách hiện, không đổi mô hình. |
| Người dùng muốn xóa dữ liệu đã gửi? | Phản hồi không gắn tài khoản, số điện thoại hay IP, nên không thể truy về một người; nội dung mức 2 đã che trước khi gửi. Cơ chế yêu cầu xóa là việc tiếp theo, chưa làm trong 8 giờ. |
| Lưu bao lâu? | Chưa chốt thời hạn trong bản này; nguyên tắc là chỉ giữ mẫu còn trong hàng đợi hoặc đã dùng để huấn luyện. Sẽ chốt theo tư vấn pháp lý. |
| Người duyệt thiên kiến thì sao? | Người duyệt theo hướng dẫn gán nhãn chung (`HUONG_DAN_GAN_NHAN.md`), nhãn nhạy cảm cần 2 người. Đo đồng thuận và kiểm tra chéo là bước sau. |
| Tại sao không dùng Google Sheet cho nhanh? | Sheet không có nhật ký truy cập theo mẫu, dễ chia sẻ nhầm và xuất toàn bộ. DB có phân quyền và nhật ký. |
| Có tuân thủ pháp luật không? | Thiết kế theo các nguyên tắc của Nghị định 13/2023/NĐ-CP và Luật Bảo vệ dữ liệu cá nhân: đồng ý rõ ràng, đúng mục đích, tối thiểu hóa. Chưa qua thẩm định pháp lý chính thức. |

## 8. Danh sách kiểm tra trước khi trình bày

- [ ] Gửi mức 1: F12 → Network không có nội dung tin nhắn.
- [ ] Phân tích tin bình thường vẫn không phát sinh request nào (giữ cam kết cũ của README).
- [ ] Ô đồng ý mức 2 không tích sẵn; tắt trong Cài đặt thì không gửi được mức 2.
- [ ] DB chỉ chứa bản đã che; không có cột mã tài khoản, số điện thoại, IP.
- [ ] Không có endpoint xuất hàng loạt.
- [ ] Ngưỡng mới chỉ có hiệu lực sau khi `quan_tri` phê duyệt; nhật ký có dòng tương ứng.
- [ ] Test che thông tin chạy được, con số trên slide lấy từ kết quả test.
- [ ] `pytest -q` và `npm run verify:parity` vẫn pass (không làm hỏng mô hình đang chạy).
- [ ] Video demo dự phòng.
