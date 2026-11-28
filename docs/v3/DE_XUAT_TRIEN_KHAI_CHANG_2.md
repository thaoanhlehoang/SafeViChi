# Đề xuất triển khai SafeViChi Chặng 2

Trong phiên làm việc 8 giờ của hai người, ưu tiên nối biểu mẫu mức 1 với API và SQLite để đo được người dùng đang báo sai ở đâu. Sau đó hoàn thành luồng mức 2 từ đồng ý, che thông tin và xem trước đến sửa nhãn có nhật ký. Quản trị viên phê duyệt thay đổi ngưỡng dựa trên kết quả validation. Người dùng vẫn có quyền **Vẫn gửi** và **Vẫn xem**.

Hiện `web-v2/serve.py` phục vụ bản build và gói mô hình cục bộ. Việc đầu tiên ngoài giao diện là chốt hợp đồng phản hồi để hai người có thể làm song song. Phạm vi demo và mốc 8 giờ lấy từ [Kế hoạch Chặng 2](KE_HOACH_CHANG_2.md).

## 1. Nội dung chính thức đề xuất cho biểu mẫu mức 1

**Tiêu đề** · Phản hồi kết quả AI

**Dòng mở đầu** · AI gợi ý nhãn **[tên nhãn]**. Bạn thấy kết quả này có vấn đề gì?

Người dùng chọn một loại báo cáo. Điểm vào là **Nhãn này sai?** trên kết quả nguy hại, **Báo bỏ sót** trên kết quả An toàn và **AI hiểu sai?** ở cảnh báo trước khi gửi.

| Lựa chọn báo cáo | Điều kiện hiển thị | Nhãn đề xuất được ghi nhận |
| --- | --- | --- |
| **Tin nhắn này an toàn** | AI gắn nhãn nguy hại | Tự điền **An toàn**. |
| **Có vấn đề, nhưng AI chọn sai loại** | AI gắn nhãn nguy hại | Bắt buộc chọn một trong ba nhãn nguy hại còn lại. |
| **AI bỏ sót nội dung nguy hại** | AI gắn **An toàn** | Bắt buộc chọn **Xúc phạm**, **Quấy rối**, **Đe dọa** hoặc **Lừa đảo**. |
| **Nhãn đúng, phần tô sáng có vấn đề** | AI gắn nhãn nguy hại | Ghi nhận người dùng đồng ý với nhãn AI. |
| **Vấn đề khác hoặc tôi chưa chắc** | Luôn hiển thị | Có thể chọn một trong năm nhãn hoặc **Chưa chắc**. Nếu bỏ qua hoặc chọn **Chưa chắc**, nhãn đề xuất để trống. |

### Tin nhắn này an toàn

**Vì sao bạn cho rằng tin nhắn này an toàn?** · Bắt buộc, chọn một.

| Lý do | Câu hỏi bổ sung và đáp án |
| --- | --- |
| **Đùa hoặc nói thân mật** | **Điều gì giúp bạn nhận ra đây là lời đùa hoặc cách nói thân mật?** Cách hai bên nói chuyện với nhau / Câu trả lời của người nhận / Các lượt trước. |
| **Từ lóng hoặc cách nói địa phương** | **Điều gì giúp bạn hiểu đúng cách nói này?** Nghĩa quen thuộc của từ / Cách dùng từ trong câu / Các lượt trước. |
| **Trích dẫn hoặc kể lại** | **Bạn nhận ra đây là lời trích dẫn hoặc lời kể bằng cách nào?** Có dấu hiệu trích dẫn / Người gửi nói rõ họ đang kể lại / Các lượt trước cho biết đó là lời người khác. |
| **Ngữ cảnh cho thấy tin nhắn bình thường** | **Tin nhắn thuộc tình huống nào?** Công việc hoặc học tập / Giao dịch hoặc xác minh hợp lệ / Trò chuyện hằng ngày. |
| **Khác** | Hiện ô **Mô tả ngắn**. |

### Có vấn đề, nhưng AI chọn sai loại

1. **Theo bạn, nhãn nào phù hợp hơn?** · Bắt buộc.
2. **Vì sao bạn chọn nhãn khác?** · Không bắt buộc. **AI hiểu sai ý hoặc ngữ cảnh**, **Có nhiều dấu hiệu và tôi ưu tiên dấu hiệu khác**, **Khác**.
3. **Bạn nhận thấy dấu hiệu nào?** · Không bắt buộc. Hiện sau khi chọn nhãn. Khi chọn lý do **Khác**, hiện ô **Mô tả ngắn** thay câu này.

| Nhãn người dùng chọn | Đáp án cho câu hỏi về dấu hiệu |
| --- | --- |
| **Xúc phạm** | Lăng mạ cá nhân / Miệt thị một nhóm hoặc vùng miền. |
| **Quấy rối** | Gạ gẫm không mong muốn / Bình phẩm cơ thể / Nhắn bám riết hoặc kiểm soát / Bêu riếu. |
| **Đe dọa** | Dọa gây hại thân thể hoặc tài sản / Dọa làm hại danh dự / Cưỡng ép hoặc tống tiền. |
| **Lừa đảo** | Mạo danh / Dụ chuyển tiền / Dụ cung cấp OTP hoặc mật khẩu / Dụ bấm liên kết hoặc cài ứng dụng. |

### AI bỏ sót nội dung nguy hại

**Theo bạn, tin nhắn thuộc loại nào?** · Bắt buộc. Sau đó hỏi **Bạn nhận ra nội dung nguy hại bằng cách nào?** · Không bắt buộc.

| Lý do | Câu hỏi bổ sung và đáp án |
| --- | --- |
| **Dấu hiệu thể hiện trực tiếp** | **Bạn nhận thấy dấu hiệu nào?** Dùng bảng dấu hiệu theo nhãn ở trên. |
| **Phải đọc các lượt trước mới thấy** | **Các lượt trước cho thấy điều gì?** Câu này mang hàm ý nguy hại / Hành vi đã lặp lại / Người nhận đã từ chối. |
| **Nội dung được viết biến dạng** | **Bạn thấy kiểu viết nào?** Bỏ dấu / Viết tắt hoặc teencode / Chèn dấu cách hoặc ký tự / Viết dính từ / Dùng ký tự trông giống chữ. |
| **Khác** | Hiện ô **Mô tả ngắn**. |

### Nhãn đúng, phần tô sáng có vấn đề

**Phần tô sáng có vấn đề gì?** · Bắt buộc, chọn một.

| Vấn đề | Câu hỏi bổ sung và đáp án |
| --- | --- |
| **Tô nhầm đoạn** | **Đoạn được tô có vấn đề gì?** Đoạn đó vô hại / Chỉ một phần đoạn được tô có liên quan / Đoạn liên quan nằm ở chỗ khác. |
| **Tô thiếu đoạn** | **Dấu hiệu còn thiếu nằm ở đâu?** Cùng lượt tin / Ở lượt trước / Nằm rải qua nhiều lượt. |
| **Khác** | Hiện ô **Mô tả ngắn**. |

Nếu AI chưa tô đoạn nào, chỉ hiện **Thiếu đoạn cần tô sáng** và **Khác**. Khi chọn **Thiếu đoạn cần tô sáng**, hỏi **Dấu hiệu còn thiếu nằm ở đâu?** với ba đáp án như trên.

### Vấn đề khác hoặc tôi chưa chắc

**Bạn muốn báo điều gì?** · Bắt buộc, chọn một.

| Lựa chọn | Câu hỏi bổ sung và đáp án |
| --- | --- |
| **Chưa chắc về nhãn** | **Bạn đang phân vân điều gì?** Chưa chắc tin có nguy hại / Phân vân giữa các nhãn nguy hại / Cần thêm ngữ cảnh. Có thể bỏ qua câu hỏi này và gửi. |
| **Vấn đề khác** | **Vấn đề nằm ở đâu?** Thông báo cảnh báo khó hiểu / Cảnh báo xuất hiện quá thường xuyên / Mức cảnh báo chưa phù hợp / Khác. Bắt buộc chọn một. **Khác** mở ô **Mô tả ngắn**. |

**Nếu đã có nhãn đề xuất, bạn chọn nhãn nào?** · Không bắt buộc. Hiện **An toàn**, **Xúc phạm**, **Quấy rối**, **Đe dọa**, **Lừa đảo** và **Chưa chắc**.

### Quy tắc điền và nội dung khi gửi

- Mỗi câu hỏi chỉ chọn một đáp án. Câu hỏi bổ sung mặc định để trống và được phép bỏ qua, ngoại trừ câu **Vấn đề nằm ở đâu?**. Khi đổi nhánh, nhãn hoặc lý do, xóa các đáp án phụ đã mất điều kiện hiển thị.
- **Mô tả ngắn** chỉ hiện và trở thành bắt buộc khi chọn **Khác**. Giới hạn 200 ký tự sau khi bỏ khoảng trắng đầu và cuối. Dòng nhắc dưới ô nhập là **“Đừng chép lại tin nhắn hoặc nhập thông tin cá nhân.”** Trình duyệt che thông tin nhận diện được trong phần mô tả trước khi gửi.
- **Thông báo trước khi gửi** · “Phản hồi mức 1 không đính kèm tin nhắn, đoạn được tô sáng hoặc mã tài khoản.”
- **Nút** · **Hủy** và **Gửi phản hồi**. Chỉ báo **“Đã gửi phản hồi. Cảm ơn bạn đã góp ý.”** sau khi máy chủ xác nhận đã lưu. Nếu gửi lỗi, giữ câu trả lời trong biểu mẫu để người dùng thử lại.

## 2. Việc cần làm ngoài giao diện

| Ưu tiên | Đầu việc | Kết quả cần bàn giao |
| --- | --- | --- |
| **1** | **Chốt hợp đồng phản hồi mức 1.** Gán mã ổn định cho từng loại báo cáo, lý do và đáp án phụ trong Mục 1. Thống nhất trường bắt buộc, điều kiện theo nhánh và mẫu JSON hợp lệ/lỗi. | Bảng mã và quy tắc dùng chung cho giao diện, API và kiểm thử. Có ca `nhan_de_xuat = null` và ca nhãn đúng nhưng tô sai. |
| **2** | **API cùng origin và SQLite.** Gắn `POST /api/phan-hoi` vào `web-v2/serve.py`. Kiểm tra danh sách trường cho phép, kiểu dữ liệu, giới hạn kích thước và tổ hợp lựa chọn. Lưu phản hồi và kiểm soát gửi lặp trong giao dịch SQLite. | Phản hồi hợp lệ lưu đúng một lần khi gửi lại cùng mã lần gửi. Gói sai bị từ chối trước khi lưu. Dùng mã thiết bị ngẫu nhiên riêng cho bộ đếm tần suất, với mức 20 lần mức 1 và 5 lần mức 2 mỗi ngày theo kế hoạch demo. |
| **3** | **Che thông tin trên thiết bị.** Viết hàm dùng chung cho mô tả mức 1 và bản xem trước mức 2. Dùng dữ liệu giả lập để kiểm tra tên, số điện thoại, email, giấy tờ, số tài khoản, OTP, địa chỉ và liên kết. | Bộ 30 ca có đáp án kỳ vọng và kết quả chạy ghi số ca che đúng, ca sót và ca che nhầm. Mức 2 gửi đúng bản cuối người dùng đã xem, kể cả phần họ che thêm. |
| **4** | **Hàng đợi duyệt mức 2.** API trả từng mẫu đã che. Vai trò `nguoi_duyet` xác nhận hoặc sửa nhãn, chọn bằng chứng nguyên văn từ chính bản đã che và ghi nhật ký khi mở mẫu hoặc lưu quyết định. | Ca `DE_DOA` và `XUC_PHAM` nhắm vào nhóm hoặc vùng miền cần hai người duyệt khác nhau cùng xác nhận nhãn. Khi bất đồng, mẫu tiếp tục chờ duyệt. Mẫu mức 2 đã xác nhận mới đủ điều kiện vào train với `cach_gan="nguoi"`, `nguon="phan_hoi"`. |
| **5** | **Thống kê mức 1 và phê duyệt ngưỡng.** Tổng hợp lượt báo cáo theo nhánh, cặp nhãn, lý do, khoảng điểm và phiên bản. Chuẩn bị đề xuất ngưỡng có lý do, phiên bản, số đo trước/sau trên cùng tập validation và người đề xuất. | Báo cáo mức 1 là thống kê phản hồi. Vai trò `quan_tri` phê duyệt cấu hình, ghi nhật ký ai, lúc nào và vì sao. Phiên bản hiệu lực chỉ đổi sau phê duyệt và được hiển thị trên cột AI. |

### Hợp đồng dữ liệu để bắt đầu

Phản hồi mức 1 đề xuất có `loai`, `ma_lan_gui`, `ma_bao_cao`, `nhan_mo_hinh`, `nhan_de_xuat`, `ma_ly_do`, `ma_chi_tiet`, `ly_do_khac`, `co_to_sang`, `khoang_diem`, `phien_ban_mo_hinh` và `phien_ban_nguong`. `nhan_de_xuat` có thể nhận `null` theo nhánh. `ly_do_khac` chỉ có giá trị khi nhánh có lựa chọn **Khác** và đã qua bước che. Máy chủ ghi ngày tiếp nhận. `ma_thiet_bi` chỉ đi vào bộ đếm tần suất, tách khỏi bản ghi phản hồi.

API nhận các trường có trong hợp đồng. Tin nhắn, bằng chứng nguyên văn, băm cụm và mã tài khoản nằm ngoài gói mức 1. Mức 2 có hợp đồng riêng cho đoạn cuối đã che và phiên bản thông báo đồng ý. Người dùng chủ động đồng ý từng lần và có thể tắt gửi mức 2 trong Cài đặt. Nhật ký nội bộ ghi người thao tác, hành động, mã mẫu và thời điểm.

Trong giờ đầu, hai người cùng duyệt hai gói mẫu để kiểm tra hợp đồng. Ca **Tin nhắn này an toàn** có nhãn đề xuất `AN_TOAN`, lý do `dua_noi_than_mat` và có thể bỏ trống đáp án phụ. Ca **Chưa chắc về nhãn** giữ `nhan_de_xuat = null` và có thể bỏ trống cả đáp án phụ lẫn mô tả. Người A bàn giao hai gói mẫu và phản hồi lỗi cho giao diện trước khi hoàn thành API.

Ứng dụng lấy phiên bản ngưỡng đã được phê duyệt khi khởi tạo phiên demo. Giữ xử lý từng tin nhắn trên thiết bị. Kiểm tra cùng một ca trước và sau phê duyệt, tải lại ứng dụng rồi đối chiếu phiên bản hiển thị với nhật ký.

## 3. Chia việc cho hai người trong 8 giờ

| Mốc | Người A · API, dữ liệu và phân quyền | Người B · Quyền riêng tư, đánh giá và nghiệm thu | Kết quả cuối mốc |
| --- | --- | --- | --- |
| **0:00–1:00** | Chốt bảng mã, quy tắc kiểm tra và schema SQLite. Soạn mẫu request/response cho phía giao diện. | Chốt câu chữ, điều kiện từng nhánh và danh sách ca kiểm thử. Chuẩn bị tin nhắn và thông tin cá nhân giả lập. | Có hợp đồng và dữ liệu mẫu để hai người làm song song. |
| **1:00–3:00** | Làm API mức 1, ghi SQLite, xử lý gửi lặp và giới hạn tần suất. | Làm hàm che thông tin, kiểm thử 30 ca và đối chiếu các nhánh mức 1 với hợp đồng. | Gửi được mức 1 vào SQLite bằng dữ liệu giả lập. Có kết quả kiểm thử che lần đầu. |
| **3:00–5:30** | Nối API mức 2, hàng đợi, quyền người duyệt, quy tắc hai người và nhật ký. | Kiểm tra bản xem trước khớp gói gửi, thử che thêm và tắt mức 2. Chuẩn bị số đo validation trước/sau cho một đề xuất ngưỡng. | Hoàn thành một ca từ đồng ý chia sẻ đến sửa nhãn. Có số liệu cho đề xuất ngưỡng. |
| **5:30–7:00** | Làm truy vấn tổng hợp mức 1 và luồng quản trị phê duyệt phiên bản ngưỡng. | Kiểm tra chéo quyền, nhật ký, số đo validation và thời điểm phiên bản ngưỡng mới có hiệu lực. | Sẵn sàng demo bảng thống kê và phê duyệt có nhật ký. |
| **7:00–8:00** | Chạy kiểm thử API và các lệnh kiểm chứng repo. | Chạy demo đầu cuối, đối chiếu Network với SQLite và ghi lại kết quả cho phần trình bày. | Có bằng chứng nghiệm thu cho mức 1, mức 2 và phê duyệt ngưỡng. |

Nếu thiếu thời gian, dùng bảng thống kê đơn giản và dời làm dịu cảnh báo lặp lại cùng cảnh báo đột biến sang sau demo. Ưu tiên giữ luồng mức 1, luồng đồng ý và xem trước mức 2, sửa nhãn có nhật ký và kiểm tra quyền phê duyệt.

## 4. Nghiệm thu trước khi demo

- **Mức 1** · Thử đủ năm nhánh, đổi nhánh sau khi đã chọn, gửi với **Chưa chắc**, thử mô tả rỗng hoặc quá 200 ký tự và thử gửi lại cùng mã lần gửi. Đối chiếu Network và SQLite với danh sách trường ở Mục 2. Kiểm tra phân tích thường sau khi tài nguyên đã tải xong để xác nhận thao tác phân tích không phát sinh request.
- **Mức 2** · Xác nhận phải có đồng ý và xem trước, bản gửi khớp bản cuối đã che, tắt trong Cài đặt ngăn gửi mức 2, người thiếu quyền không mở được mẫu và mẫu cần hai người chưa được xác nhận sau lượt duyệt đầu.
- **Nhãn và ngưỡng** · Thử **Vẫn gửi** và **Vẫn xem**. Đối chiếu nhật ký mở mẫu, sửa nhãn và phê duyệt. Đo ngưỡng trên validation, khóa cấu hình trước khi mở test và xác nhận cấu hình mới chỉ có hiệu lực sau phê duyệt. Đọc báo cáo mức 1 theo số lượt gửi, vì dữ liệu tự báo cáo không đủ để ước lượng tỷ lệ lỗi của mô hình.
- **Kiểm chứng repo** · Chạy `pytest -q` ở gốc repo và `npm run build`, `npm run verify:parity` trong `web-v2/` sau khi triển khai. Lấy kết quả che thông tin từ lần chạy kiểm thử và phân biệt rõ dữ liệu giả lập với số đo validation trong demo.
