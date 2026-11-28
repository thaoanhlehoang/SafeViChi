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

`--init-demo` thêm khóa giới hạn tần suất vào `.env` nếu còn thiếu. Wrangler dùng cơ sở dữ liệu demo riêng. Dùng dữ liệu giả lập vì người mở bản demo đều có thể chọn vai trò.

- Mở [trang chat](http://127.0.0.1:8788/) và [trang kiểm duyệt, quản trị](http://127.0.0.1:8788/quan-tri.html) trên cùng bản Wrangler. Có thể mở khu vực nội bộ từ **Cài đặt phản hồi** trong chat.
- Chọn **Người duyệt A**, **Người duyệt B** hoặc **Quản trị viên** ngay trên màn hình. Chuyển vai trò trong thanh trên để trình diễn ca cần hai lượt duyệt hoặc phê duyệt ngưỡng. Gửi một phản hồi mức 2 bằng ca mẫu trong chat để có mục mới trong hàng đợi.
- Nếu có gói ONNX trên máy, có thể chạy thêm `npm run dev` và mở cổng 5173. Vite phục vụ ONNX cục bộ và chuyển `/api/*` đến Wrangler trên cổng 8788.

Trong cửa sổ còn lại, chạy kiểm thử che thông tin và các luồng HTTP.

```powershell
npm test
node scripts/smoke-test-feedback.mjs http://127.0.0.1:8788
```

Script HTTP chạy đủ năm nhánh mức 1, thử gửi lại và gửi đồng thời, gửi bản xem trước mức 2, duyệt bằng hai vai trò, kiểm tra nhật ký và chuyển phiên bản ngưỡng sau phê duyệt. Có thể kiểm tra bảng D1 bằng câu lệnh chỉ trả số lượng.

```powershell
npx --yes wrangler@4.169.0 d1 execute DEMO_FEEDBACK_DB --local --command "SELECT loai, trang_thai, count(*) AS so_luot FROM phan_hoi GROUP BY loai, trang_thai"
```

## Nối frontend

Biểu mẫu mức 1 đã được nối qua `src/phan_hoi/api.js` với SDK trong `src/feedback/client.js`. Mở biểu mẫu bằng nút **Báo cáo/kháng cáo mô hình** trên dải trạng thái chat. Khi gửi, SDK che mô tả tự do, tạo gói theo hợp đồng API và chỉ hiện xác nhận sau khi D1 đã lưu. Các mã lựa chọn trong `src/phan_hoi/ma.js` được kiểm thử đối chiếu với `REPORT_OPTIONS` và `LABEL_INDICATORS` trong `src/feedback/contract.js`. Bản chữ của biểu mẫu nằm trong [Đề xuất Chặng 2](../../docs/v3/DE_XUAT_TRIEN_KHAI_CHANG_2.md).

Chọn **Gửi kèm đoạn hội thoại đã che (mức 2)** rồi bấm **Xem trước**. Bộ che trong `src/phan_hoi/che_thong_tin.js` tự nhận diện thông tin trên thiết bị. Tên lặp lại có thể dùng cùng nhãn `[TÊN_1]`, số tiền được giữ để người duyệt hiểu ngữ cảnh và liên kết nhận diện được đổi thành `[LINK]`. Nếu còn chi tiết cần che, chọn vùng văn bản và bấm **Che phần đã chọn**. Khi bản xem trước đổi, xác nhận đồng ý trở về trạng thái chưa chọn. Tắt lựa chọn gửi kèm trong **Cài đặt phản hồi** để chỉ dùng mức 1.

Trong khu vực nội bộ, người duyệt mở từng mẫu từ **Hàng đợi kiểm duyệt**. Chọn nhãn, chọn cụm nguyên văn trong một lượt và bấm **Thêm bằng chứng**. Hệ thống trả trạng thái sau khi lưu. Cả hai vai trò xem được thống kê và đề xuất ngưỡng. Quản trị viên xem thêm **Nhật ký** và phê duyệt đề xuất kèm lý do.

| Chức năng | Điểm nối |
| --- | --- |
| Biểu mẫu mức 1 | Gọi `createLevel1Payload(formData)`, sau đó `submitFeedback(payload)` khi người dùng bấm gửi. Nếu thử lại cùng phản hồi, gửi lại chính `payload` để giữ mã lần gửi. |
| Mức 2 | Dùng `isLevel2Enabled()` và `setLevel2Enabled(boolean)` cho Cài đặt. Gọi `createPreview(turns, { contactNames })`, hiển thị `preview.text` dưới dạng văn bản, và cập nhật sang bản do `redactPreviewSelection(preview, lineIndex, start, end)` trả về. Dùng chính bản đang hiển thị khi gọi `createLevel2Payload(formData, preview, { consent: true })` sau khi người dùng xác nhận. |
| Nhãn và phiên bản | Lấy nhãn, xác suất và việc có đoạn tô từ kết quả đang được báo cáo. Dùng `getScoreBucket(probability)` để lấy số 0–9. Nhận `phien_ban_mo_hinh` và `phien_ban_nguong` từ sự kiện `ready` của worker. |
| Cấu hình ngưỡng | Khi tích hợp, gửi `{ type: 'load', dungNguongDaDuyet: true }` để worker đọc phiên bản đã duyệt cho đúng mã mô hình khi khởi tạo. Tải lại để nhận cấu hình mới. Nếu chưa có cấu hình phù hợp hoặc không tải được, worker dùng `baseline`. Mỗi lần suy luận chạy trên thiết bị. |
| Làm dịu cảnh báo | SDK ghi nhận cục bộ sau khi máy chủ lưu phản hồi **Tin nhắn này an toàn** lần đầu. Gọi `getWarningPreference(label)` từ `src/feedback/warning-preferences.js` trước khi trình bày cảnh báo mới. Sau ba lượt trong bảy ngày, helper có thể trả `{ blur: false }` hoặc `{ suggestBlock: false }`. Cảnh báo Đe dọa và Lừa đảo giữ cách hiển thị mặc định. |

Ví dụ `formData` cho **Tin nhắn này an toàn** với câu trả lời bổ sung.

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

`nhan_de_xuat` có thể để trống ở nhánh `khac_chua_chac`. Mức 1 gửi dữ liệu theo danh sách trường của hợp đồng, gồm mô tả đã được xử lý bằng cùng bộ che nếu chọn **Khác**. Nội dung hội thoại và bằng chứng gốc ở lại trên máy. SDK mức 2 dùng SHA-256 để kiểm tra bản cuối trong request. D1 chỉ lưu bản đã gửi và phiên bản thông báo đồng ý. Phép che theo mẫu có thể bỏ sót tên và cách viết lạ. Bước xem trước cho phép phát hiện và che thêm các trường hợp đó.

## API

Tất cả request ghi dùng JSON. Bản demo nhận vai trò từ header `X-SafeViChi-Demo-Role`. Môi trường có dữ liệu thật xác thực API nội bộ bằng `Authorization: Bearer <token>` theo vai trò. Đáp án bằng chứng dùng `dong` bắt đầu từ 0 và khoảng `[start, end)` tính theo UTF-16 của bản đã che.

| Phương thức và đường dẫn | Quyền | Kết quả |
| --- | --- | --- |
| `POST /api/phan-hoi` | Người dùng | Mức 1 hoặc 2. Trả 201 khi lưu, 200 khi gửi lại cùng gói, 409 khi mã đã dùng với dữ liệu khác, 429 khi vượt 20 lượt mức 1 hoặc 5 lượt mức 2 trong ngày theo giờ Việt Nam. |
| `GET /api/noi-bo/che-do` | Công khai | Trả chế độ hiện tại và các vai trò có thể chọn trong bản demo. |
| `GET /api/noi-bo/phien` | Người duyệt, quản trị | Trả danh tính và vai trò hiện tại. |
| `GET /api/duyet` | Người duyệt | Tối đa 25 mục metadata chưa được người này xử lý. Phân trang bằng cặp `sau_ngay` và `sau_id`. Mở từng mẫu để đọc nội dung đã che. |
| `GET /api/duyet/tiep-theo` | Người duyệt | Một mẫu chưa được người này duyệt hoặc `mau: null`. |
| `GET /api/duyet/{id}` | Người duyệt | Một mẫu đã che kèm quyết định nếu có. Ghi nhật ký khi mở mẫu. |
| `POST /api/duyet/{id}` | Người duyệt | Nhận `{ nhan, nham_nhom, bang_chung }`. Cập nhật trạng thái và chờ lượt thứ hai nếu cần. |
| `GET /api/thong-ke?theo=nhan` | Người duyệt, quản trị | Thống kê mức 1 theo `nhan`, `ly_do`, `khoang_diem` hoặc `phien_ban`. Tham số `tu`, `den` theo ngày `YYYY-MM-DD`, mặc định bảy ngày gần nhất. |
| `GET /api/de-xuat` và `GET /api/de-xuat/{id}` | Người duyệt, quản trị | Xem các đề xuất còn chờ hoặc chi tiết một đề xuất. |
| `POST /api/de-xuat` | Người duyệt, quản trị | Tạo đề xuất có phiên bản mô hình, phiên bản ngưỡng gốc, ngưỡng mới, lý do và số đo validation trước/sau. Xem payload mẫu trong script thử luồng. |
| `POST /api/de-xuat/{id}/phe-duyet` | Quản trị | Nhận `{ ly_do }`. Trả 409 nếu cấu hình hiệu lực đã đổi kể từ phiên bản gốc được dùng trong đề xuất. |
| `GET /api/nhat-ky` hoặc `GET /api/nhat-ky/{id}` | Quản trị | Tối đa 100 sự kiện gần nhất, có thể lọc theo mẫu hoặc đề xuất. |
| `GET /api/nguong?phien_ban_mo_hinh=...` | Công khai | Cấu hình đã phê duyệt cho đúng mô hình hoặc `cau_hinh: null`. |

## Git và Cloudflare

- Frontend tích hợp bằng các mô-đun trong `src/feedback/`. Routes, truy vấn D1 và migration nằm trong `functions/` và `cloudflare/`. Chỉ gắn các lời gọi trên vào các điểm vào đã có của giao diện.
- Với Pages Git, đặt thư mục dự án là `web-v2`, lệnh dựng là `npm run build` và thư mục đầu ra là `dist`. `wrangler.jsonc` dùng cho phát triển cục bộ. Các binding triển khai do Dashboard quản lý.
- Bản build chứa cả trang chat và `quan-tri.html`. Trên Pages, có thể mở khu vực nội bộ tại `/quan-tri`. Sau khi triển khai Preview, kiểm tra cả hai trang, thử đăng nhập theo từng vai trò và dùng dữ liệu giả lập đi hết luồng mức 2. [Quy tắc phục vụ trang HTML của Cloudflare Pages](https://developers.cloudflare.com/pages/configuration/serving-pages/)
- Bật bản trình diễn trên Preview bằng biến `FEEDBACK_DEMO=1` và binding `DEMO_FEEDBACK_DB` trỏ đến D1 dành riêng cho dữ liệu giả lập. Đặt Secret `FEEDBACK_RATE_SECRET` là chuỗi ngẫu nhiên ít nhất 32 ký tự. [Bindings của Pages Functions](https://developers.cloudflare.com/pages/functions/bindings/)
- Môi trường có dữ liệu thật dùng `FEEDBACK_DB`, `FEEDBACK_RATE_SECRET` và `INTERNAL_TOKEN_HASHES` với chế độ demo tắt. `INTERNAL_TOKEN_HASHES` ánh xạ SHA-256 của mã truy cập sang `{ id, vai_tro }`.
- Chạy migration trên D1 mục tiêu trước khi mở API. Dùng bản sao cấu hình Wrangler với đúng tên và ID D1, rồi chạy `wrangler d1 migrations apply <ten-db> --remote --config <tep-config>`. [Hướng dẫn migration D1](https://developers.cloudflare.com/d1/reference/migrations/)
