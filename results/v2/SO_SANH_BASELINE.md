# So sánh baseline trên bộ test v2: gpt-oss-120b (API) và SafeViChi v1

> Tệp này sinh tự động bằng `python -m src.v2.eval.bao_cao_baseline`. Không sửa tay.

Chi tiết: [`llm_api/cham_lai/README.md`](llm_api/cham_lai/README.md), [`v1_baseline/cham_lai/README.md`](v1_baseline/cham_lai/README.md). v1 chỉ có nhãn nhị phân nên bảng dưới so sánh nhị phân nguy hại/an toàn.

## Nhị phân nguy hại/an toàn

| Chỉ số | v1 C0 | v1 C1 | v1 C2 | LLM C0 | LLM C1 | LLM C2 |
| --- | --- | --- | --- | --- | --- | --- |
| Precision | 1,0000 | 0,9670 | 0,9370 | 0,9848 | 0,9335 | 0,9748 |
| Recall | 0,1849 | 0,1849 | 0,2500 | 0,8151 | 0,6197 | 0,6492 |
| Binary-F1 | 0,3121 | 0,3104 | 0,3947 | 0,8920 | 0,7449 | 0,7793 |
| FPR an toàn | 0,0% | 1,6% | 4,2% | 3,1% | 10,9% | 4,2% |

## Binary-F1 theo giọng vùng

| Hệ | Tập | Bắc | Trung | Nam |
| --- | --- | --- | --- | --- |
| v1 | C0 | 0,3191 | 0,2444 | 0,3673 |
| v1 | C1 | 0,3011 | 0,2222 | 0,3980 |
| v1 | C2 | 0,4348 | 0,3085 | 0,4327 |
| LLM | C0 | 0,8889 | 0,8811 | 0,9054 |
| LLM | C1 | 0,7176 | 0,7433 | 0,7732 |
| LLM | C2 | 0,7490 | 0,7597 | 0,8261 |

## Vì sao Binary-F1 ở C2 cao hơn C1

C1 và C2 làm hỏng câu theo hai cách khác nhau; "chưa thấy" (C2) nghĩa là nằm ngoài dữ liệu train của mô hình v2, không có nghĩa là khó hơn. Với hai baseline này, cả C1 và C2 đều là kiểu chưa từng huấn luyện.

- **C1 làm hỏng chính các từ** (bỏ dấu, teencode, gõ sai, số thay chữ...): thông tin mất thật, ví dụ "đánh" và "dành" bỏ dấu đều thành "danh".
- **C2 chỉ đổi lớp ký tự Unicode** (fullwidth, ký tự vô hình, chữ Cyrillic giống chữ Latin...): từ vẫn còn nguyên, LLM lớn đọc lại được (Binary-F1 của LLM với `fullwidth`: 0,8765). Riêng `space_injection` (chèn dấu cách giữa các chữ) làm LLM hụt rõ (Binary-F1 0,5862).
- **Với v1, mức tăng ở C2 là ảo:** đến chủ yếu từ `space_injection` (recall 0,4512, báo nhầm 7/32 câu an toàn) và `double_elongation` (recall 0,3433). Văn bản bị băm nát khiến v1 nghiêng về HATE kể cả với câu an toàn; năm kiểu C2 còn lại cộng lại chỉ báo nhầm 1 câu.

Vì vậy báo cáo nên trình bày theo từng kiểu biến dạng thay vì một con số cho mỗi tập.

### Binary-F1 theo kiểu biến dạng

| Tập | Kiểu | v1 | LLM | Báo nhầm an toàn v1 | FPR an toàn LLM |
| --- | --- | --- | --- | --- | --- |
| C1 | `bo_dau` | 0,3789 | 0,7107 | 1/26 | 7,7% |
| C1 | `teencode` | 0,3778 | 0,8033 | 0/30 | 0,0% |
| C1 | `go_sai` | 0,3023 | 0,7167 | 2/33 | 18,2% |
| C1 | `dinh_tu` | 0,2821 | 0,7387 | 0/29 | 10,3% |
| C1 | `so_thay_chu` | 0,3500 | 0,8136 | 0/27 | 14,8% |
| C1 | `keo_dai` | 0,1765 | 0,7143 | 0/27 | 3,7% |
| C1 | `chen_ky_tu` | 0,2571 | 0,7059 | 0/20 | 25,0% |
| C2 | `fullwidth` | 0,3333 | 0,8765 | 0/32 | 3,1% |
| C2 | `zero_width` | 0,3585 | 0,7917 | 0/29 | 0,0% |
| C2 | `space_injection` | 0,5873 | 0,5862 | 7/32 | 0,0% |
| C2 | `exotic_separator` | 0,2921 | 0,7969 | 0/32 | 3,1% |
| C2 | `cyrillic_homoglyph` | 0,2143 | 0,7937 | 1/36 | 5,6% |
| C2 | `double_elongation` | 0,5111 | 0,7863 | 0/31 | 12,9% |
