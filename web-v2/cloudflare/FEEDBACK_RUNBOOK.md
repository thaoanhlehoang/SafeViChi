# Chạy và tích hợp luồng phản hồi SafeViChi trên Cloudflare Pages

API chạy cùng origin bằng [Pages Functions](https://developers.cloudflare.com/pages/functions/wrangler-configuration/) và dùng D1. Các tệp giao diện tiếp tục được Vite dựng từ `web-v2/`. `public/_routes.json` chỉ chuyển `/api/*` vào Functions. R2 tiếp tục phục vụ gói mô hình nếu cấu hình `VITE_MODEL_BASE_URL` trỏ tới URL gói mô hình.

## Chạy thử trên Windows

Từ thư mục `web-v2`, dùng hai cửa sổ PowerShell. Các lệnh dưới đây dùng phiên bản Wrangler đã kiểm thử và D1 cục bộ. Có thể dùng `wrangler` nếu lệnh đã có trên PATH.

```powershell
npm ci
npm run build
node scripts/smoke-test-feedback.mjs --init-demo
npx --yes wrangler@4.169.0 d1 migrations apply DEMO_FEEDBACK_DB --local
npx --yes wrangler@4.169.0 pages dev dist --ip 127.0.0.1 --port 8788
```

`--init-demo` thêm khóa giới hạn tần suất vào `.env` nếu còn thiếu. Lệnh migration áp dụng cả `0001_feedback.sql` và `0002_kiem_duyet.sql`. Wrangler dùng cơ sở dữ liệu demo riêng; chỉ dùng dữ liệu giả lập vì ai mở bản demo cũng vào được khu vực kiểm duyệt.

- Mở [trang chat](http://127.0.0.1:8788/). Phía trên khung có hai tab **Nhắn tin & cảnh báo** và **Kiểm duyệt & quản trị**, đổi tại chỗ. Trang riêng [/quan-tri](http://127.0.0.1:8788/quan-tri.html) vẫn dùng được làm dự phòng.
- Cổng 8788 chỉ phục vụ bản đã build trong `dist/`. Để sửa giao diện và thấy ngay, chạy thêm `npm run dev` rồi mở cổng 5173: Vite tự tải lại khi lưu tệp, phục vụ ONNX cục bộ nếu có và chuyển `/api/*` sang Wrangler ở 8788. Sửa code trong `functions/` hoặc `cloudflare/api/` thì khởi động lại Wrangler.

Trong cửa sổ còn lại, chạy kiểm thử che thông tin và các luồng HTTP.

```powershell
npm test
node scripts/smoke-test-feedback.mjs http://127.0.0.1:8788
```

Script HTTP chạy đủ năm nhánh biểu mẫu, gửi lại và gửi đồng thời, gửi kèm đoạn hội thoại, duyệt một lượt là chốt, tab hàng đợi, ghi chú, tài khoản kiểm duyệt duy nhất, nhật ký và chuyển phiên bản ngưỡng sau phê duyệt. Có thể kiểm tra bảng D1 bằng câu lệnh chỉ trả số lượng.

```powershell
npx --yes wrangler@4.169.0 d1 execute DEMO_FEEDBACK_DB --local --command "SELECT loai, trang_thai, count(*) AS so_luot FROM phan_hoi GROUP BY loai, trang_thai"
```

## Luồng trên giao diện

**Người dùng gửi phản hồi.** Mở biểu mẫu bằng nút **Phản hồi kết quả** trên dải trạng thái chat, hoặc **Báo AI hiểu sai** trên dải cảnh báo trước khi gửi. Nút thứ hai dùng kết quả của bản nháp, không gửi tin nháp đi; đóng biểu mẫu xong vẫn còn **Vẫn gửi** / **Chỉnh sửa**.

1. Trả lời các câu hỏi (loại báo cáo, lý do, chi tiết, mô tả nếu chọn **Khác**) rồi bấm **Tiếp tục**.
2. Bước **Chia sẻ đoạn hội thoại?** luôn được hỏi mỗi lần: khung cam kết quyền riêng tư và hai nút **Đồng ý gửi kèm đoạn hội thoại** / **Không, chỉ gửi phản hồi**. Không có ô tích sẵn, không nhớ lựa chọn, không có công tắc trong Cài đặt.
3. Đoạn hội thoại được che tự động trên thiết bị bằng `src/phan_hoi/che_thong_tin.js` (tên lặp lại dùng cùng `[TÊN_1]`, số tiền được giữ, liên kết thành `[LINK]`). Bản che không hiện cho người dùng; chính bản đó được gửi khi bấm Đồng ý.
4. Chỉ báo "Đã gửi phản hồi. Cảm ơn bạn đã góp ý." sau khi máy chủ xác nhận; lỗi thì giữ câu trả lời để thử lại.

**Người kiểm duyệt.** Bản demo chỉ có một tài khoản kiểm duyệt, không có chọn vai trò; tài khoản này làm được mọi việc. Menu trái:

- **Thống kê phản hồi** (mở mặc định): đếm mọi phản hồi, có hay không kèm đoạn. Lọc 7 ngày / 30 ngày / tùy chọn; ô số tổng, kèm đoạn, đã duyệt, chờ duyệt; thẻ theo loại báo cáo, cặp nhãn AI → người dùng, lý do phổ biến.
- **Hàng đợi kiểm duyệt**: tab **Chờ duyệt** / **Đã duyệt**, xếp theo ngày gửi, mẫu cũ nhất trước. Mở một mẫu: trái là đoạn hội thoại dạng bong bóng; phải là kết quả AI, lời người dùng báo và quyết định (5 nút nhãn, bằng chứng qua **Thêm bằng chứng**, ghi chú tối đa 200 ký tự, **Lưu và mở mẫu tiếp** / **Bỏ qua**). Một lượt duyệt là chốt mẫu. Người duyệt không che thêm được.
- **Ngưỡng mô hình**: tạo đề xuất kèm số đo validation, phê duyệt kèm lý do.
- **Nhật ký**: các lần mở mẫu, lưu lượt duyệt, tạo và phê duyệt đề xuất.

**Đưa mẫu đã duyệt vào huấn luyện** (chạy tay, không qua API):

```powershell
node scripts/xuat_mau_da_duyet.mjs --out ../data/v2/interim/phan_hoi_da_duyet_YYYY-MM-DD.jsonl
```

Script chỉ xuất mẫu kèm đoạn đã duyệt, đúng định dạng `data/v2` với `nguon="phan_hoi"`, `cach_gan="nguoi"`, `split="train"`; bỏ câu trùng nguyên văn với tập test và kho test; không ghi đè tệp có sẵn. Vẫn phải khử trùng gần đúng trước khi gộp vào train.

## Nối frontend

| Chức năng | Điểm nối |
| --- | --- |
| Phản hồi không kèm đoạn | `createLevel1Payload(formData)` rồi `submitFeedback(payload)`. Thử lại thì gửi lại chính `payload` để giữ mã lần gửi. |
| Phản hồi kèm đoạn | `createPreview(turns, { contactNames })` để che trên thiết bị, rồi `createLevel2Payload(formData, preview, { consent: true })` khi người dùng bấm Đồng ý. Gói kèm `dong_y.phien_ban = 'muc-2-v2'` và SHA-256 của bản che. |
| Nhãn và phiên bản | Lấy nhãn, xác suất và việc có đoạn tô từ kết quả đang được báo cáo. `getScoreBucket(probability)` trả số 0–9. `phien_ban_mo_hinh`, `phien_ban_nguong` lấy từ sự kiện `ready` của worker. |
| Cấu hình ngưỡng | Gửi `{ type: 'load', dungNguongDaDuyet: true }` để worker đọc phiên bản đã duyệt cho đúng mã mô hình. Chưa có cấu hình hoặc lỗi tải thì dùng `baseline`. Suy luận luôn chạy trên thiết bị. |

Ví dụ `formData` cho **Tin nhắn này an toàn**:

```js
const formData = {
  ma_bao_cao: 'tin_nhan_an_toan',
  nhan_mo_hinh: 'XUC_PHAM',
  nhan_de_xuat: 'AN_TOAN',
  ma_ly_do: 'dua_noi_than_mat',
  ma_chi_tiet: 'luot_truoc',
  co_to_sang: true,
  khoang_diem: 8,
  phien_ban_mo_hinh: ready.phien_ban_mo_hinh,
  phien_ban_nguong: ready.phien_ban_nguong,
};
```

`nhan_de_xuat` có thể để trống ở nhánh `khac_chua_chac`. Mô tả **Khác** được che bằng cùng bộ che trước khi gửi. Phép che theo mẫu có thể bỏ sót tên và cách viết lạ; vì không còn bước xem trước, người duyệt cần để ý các trường hợp đó.

## API

Request ghi dùng JSON. Bản demo nhận danh tính từ header `X-SafeViChi-Demo-Role` (giao diện dùng `kiem_duyet`, có đủ quyền). Môi trường dữ liệu thật xác thực bằng `Authorization: Bearer <token>` và vẫn phân quyền `nguoi_duyet` / `quan_tri`. Bằng chứng dùng `dong` bắt đầu từ 0 và khoảng `[start, end)` tính theo UTF-16 của bản đã che.

| Phương thức và đường dẫn | Quyền | Kết quả |
| --- | --- | --- |
| `POST /api/phan-hoi` | Người dùng | Có hoặc không kèm đoạn. 201 khi lưu, 200 khi gửi lại cùng gói, 409 khi mã đã dùng với dữ liệu khác, 429 khi vượt 20 lượt không kèm đoạn hoặc 5 lượt kèm đoạn trong ngày (giờ Việt Nam). |
| `GET /api/noi-bo/che-do` | Công khai | Chế độ hiện tại và tài khoản kiểm duyệt của bản demo. |
| `GET /api/noi-bo/phien` | Nội bộ | Danh tính và vai trò hiện tại. |
| `GET /api/duyet?trang_thai=cho_duyet` hoặc `da_duyet` | Người duyệt | Tối đa 25 mục metadata (mã lựa chọn, không có nội dung) xếp theo ngày, kèm số đếm hai tab. Phân trang bằng `sau_ngay` và `sau_id`. |
| `GET /api/duyet/tiep-theo` | Người duyệt | Mẫu cũ nhất chưa được người này duyệt hoặc `mau: null`. |
| `GET /api/duyet/{id}` | Người duyệt | Một mẫu đã che kèm quyết định nếu có. Ghi nhật ký khi mở. |
| `POST /api/duyet/{id}` | Người duyệt | Nhận `{ nhan, nham_nhom, bang_chung, ghi_chu }`. Một lượt là chốt mẫu (`da_duyet`). |
| `GET /api/thong-ke` | Nội bộ | Thống kê mọi phản hồi: tổng, kèm đoạn, đã duyệt/chờ duyệt, theo loại báo cáo, cặp nhãn AI → người dùng, lý do phổ biến. Tham số `tu`, `den` (`YYYY-MM-DD`), mặc định 7 ngày. |
| `GET /api/de-xuat`, `GET /api/de-xuat/{id}` | Nội bộ | Đề xuất ngưỡng còn chờ hoặc chi tiết một đề xuất. |
| `POST /api/de-xuat` | Nội bộ | Tạo đề xuất: phiên bản mô hình, phiên bản ngưỡng gốc, ngưỡng mới, lý do, số đo validation trước/sau. |
| `POST /api/de-xuat/{id}/phe-duyet` | Quản trị | Nhận `{ ly_do }`. 409 nếu cấu hình hiệu lực đã đổi so với phiên bản gốc của đề xuất. |
| `GET /api/nhat-ky`, `GET /api/nhat-ky/{id}` | Quản trị | Tối đa 100 sự kiện gần nhất, lọc được theo mẫu hoặc đề xuất. |
| `GET /api/nguong?phien_ban_mo_hinh=...` | Công khai | Cấu hình đã phê duyệt cho đúng mô hình hoặc `cau_hinh: null`. |

## Git và Cloudflare

- Frontend nối qua các mô-đun trong `src/feedback/` và `src/phan_hoi/`; khu vực kiểm duyệt nằm trong `src/quan_tri/`. Routes, truy vấn D1 và migration nằm trong `functions/` và `cloudflare/`.
- Với Pages Git, đặt thư mục dự án là `web-v2`, lệnh dựng `npm run build`, thư mục đầu ra `dist`. `wrangler.jsonc` dùng cho phát triển cục bộ; binding triển khai do Dashboard quản lý.
- Bật bản trình diễn trên Preview bằng `FEEDBACK_DEMO=1` và binding `DEMO_FEEDBACK_DB` trỏ đến D1 dành riêng cho dữ liệu giả lập. Đặt Secret `FEEDBACK_RATE_SECRET` là chuỗi ngẫu nhiên ít nhất 32 ký tự. [Bindings của Pages Functions](https://developers.cloudflare.com/pages/functions/bindings/)
- Môi trường dữ liệu thật dùng `FEEDBACK_DB`, `FEEDBACK_RATE_SECRET` và `INTERNAL_TOKEN_HASHES` với chế độ demo tắt. `INTERNAL_TOKEN_HASHES` ánh xạ SHA-256 của mã truy cập sang `{ id, vai_tro }`.
- Chạy migration trên D1 mục tiêu trước khi mở API, gồm cả `0002_kiem_duyet.sql`: `wrangler d1 migrations apply <ten-db> --remote --config <tep-config>`. [Hướng dẫn migration D1](https://developers.cloudflare.com/d1/reference/migrations/)
