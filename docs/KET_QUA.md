<!-- Tệp sinh tự động bằng `python -m src.v2.eval.xuat_tai_lieu` từ docs/v2/bao_cao/so_lieu.tex. Không sửa tay. -->

# SafeViChi v2: kết quả chi tiết

Tài liệu gom toàn bộ số liệu đánh giá của SafeViChi v2 (mô hình SafeVi-DualORC-CRF, backbone `jhu-clsp/mmBERT-base`). Phần nhận xét và phân tích lỗi nằm ở [DANH_GIA.md](DANH_GIA.md); bố cục mã và dữ liệu ở [CAU_TRUC_THU_MUC.md](CAU_TRUC_THU_MUC.md).

**Quy ước.** F1, precision (P), recall (R) tính trên thang 0–100. *Macro-F1* là trung bình F1 của 5 nhãn; *Binary-F1* gộp 4 nhãn nguy hại thành một lớp; *FPR* là tỉ lệ đoạn `AN_TOAN` bị báo nguy hại; *BC F1* là F1 bằng chứng theo ý (mệnh đề), *BC F1 từ* là F1 bằng chứng mức từ. Ngưỡng và hệ số nhiệt độ được dò trên tập `cal`, đóng băng trước khi mở bộ test.

## 1. Dữ liệu

### 1.1 Nguồn và quy trình làm sạch

| Bước | Số lượng |
|---|---:|
| Câu gốc ViHSD / ViHOS / ViTHSD | 33.398 / 11.056 / 10.000 |
| Tổng câu gốc | 54.454 |
| Nhóm câu khác nhau sau khử trùng gần đúng | 30.580 |
| Kho dành cho test (không bao giờ gửi LLM) | 6.724 |
| Kho dành cho train | 23.856 |
| Câu thật đưa LLM gán lại 5 nhãn | 10.062 |
| Câu thật giữ lại / bỏ (nhãn LLM mâu thuẫn nhãn gốc) | 8.649 / 1.413 |
| Đoạn hội thoại do LLM sinh theo kịch bản | 10.557 |
| Đoạn sinh giữ lại sau lọc chéo (tỉ lệ) | 9.625 (91,2%) |

Đồng thuận người – LLM trên mẫu kiểm 200 câu: nhãn LLM đúng ở 86/100 câu được giữ (86,0%, khoảng tin cậy 95% 77,9%–91,5%). Nguồn: `results/v2/dong_thuan_nhan/metrics.json`.

Chi phí gọi LLM (deepseek/deepseek-v4.1-flash-1017) để tạo dữ liệu: 3.170 lượt gọi, 17,0 triệu token, 2,28 USD (~61.555 đồng), 47 phút. Nguồn: `results/v2/chi_phi_du_lieu/chi_phi.json`.

### 1.2 Tập huấn luyện (`data/v2/train.jsonl`)

| Nhãn | Câu thật | LLM sinh | Tổng |
|---|---:|---:|---:|
| `AN_TOAN` | 3.967 | 2.000 | 5.967 |
| `XUC_PHAM` | 2.527 | 470 | 2.997 |
| `QUAY_ROI` | 62 | 1.098 | 1.160 |
| `DE_DOA` | 156 | 1.361 | 1.517 |
| `LUA_DAO` | 0 | 1.751 | 1.751 |
| **Tổng** | 6.712 | 6.680 | **13.392** |

Trong đó 8.044 mẫu là hội thoại 1 dòng.

### 1.3 Bộ test khóa

Bộ test gồm 334 đoạn gốc C0 do người trong nhóm viết hoặc duyệt, cộng 668 bản biến dạng C1 (7 kiểu né lọc thường gặp) và 668 bản C2 (6 kiểu điểm mù chưa từng có trong train). Mỗi bản biến dạng đổi khoảng 60,0% số từ. Tổng 1.670 mẫu, khóa bằng SHA-256 trong `results/v2/official_eval/dualorc_mmbert_base/test/REAL_TEST_LOCK.json`.

| Phân bố C0 | Số đoạn |
|---|---:|
| `AN_TOAN` | 96 |
| `XUC_PHAM` | 53 |
| `QUAY_ROI` | 51 |
| `DE_DOA` | 62 |
| `LUA_DAO` | 72 |
| Giọng Bắc | 112 |
| Giọng Trung | 111 |
| Giọng Nam | 111 |
| Kịch bản trong phân phối train | 185 |
| Kịch bản chưa thấy khi train | 149 |

| Kiểu C1 | Số bản |
|---|---:|
| Bỏ dấu | 102 |
| Chèn ký tự | 81 |
| Dính từ | 96 |
| Gõ sai | 104 |
| Kéo dài ký tự | 89 |
| Số thay chữ | 93 |
| Teencode | 103 |

| Kiểu C2 (điểm mù) | Số bản |
|---|---:|
| Homoglyph Cyrillic | 110 |
| Kéo dài kép | 98 |
| Ký tự ngăn cách lạ | 108 |
| Ký tự fullwidth | 122 |
| Chèn dấu cách | 114 |
| Ký tự vô hình (zero-width) | 116 |

**Kiểm tra rò rỉ** (`results/v2/kiem_tra_ro_ri/bao_cao.json`): 0 cặp gần trùng giữa test và train ở ngưỡng 0,8; các kịch bản giữ ngoài train (DD6, LD12, LD7, QR6) có 0 mẫu trong train.

## 2. Chọn mô hình trên tập dev

Tập train được chia thành fit / dev / cal = 10.713 / 1.340 / 1.339 đoạn. Bốn ứng viên huấn luyện cùng dữ liệu và seed 42. Nguồn: `results/v2/dev_selection/dev_summary.json`.

| Ứng viên | Macro-F1 | Binary-F1 | FPR | BC F1 | Tô oan câu an toàn | Chênh vùng | Macro-F1 tiếng lóng (n=178) | Trọng số FP32 (MiB) | p50 (ms) | Train (phút) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **mmBERT-base (chốt)** | 92,3 | 92,6 | 8,4% | 69,7 | 7,2% | 1,5 | 74,8 | 1.177 | 39,4 | 44 |
| BamiBERT | 92,2 | 93,3 | 8,6% | 71,6 | 6,9% | 1,9 | 55,5 | 397 | 18,6 | 19 |
| ViSoBERT | 92,9 | 93,5 | 6,0% | 70,6 | 5,7% | 1,8 | 62,7 | 376 | 20,8 | 19 |
| GuardMerge (LLM 0,6B + LoRA) | 90,0 | 91,2 | 10,7% | 67,7 | 9,9% | 6,5 | 73,4 | 1.177 | 82,3 | 21 |

McNemar ghép cặp trên dev: mmBERT-base vượt GuardMerge có ý nghĩa (p = 0,023); với ViSoBERT (p = 0,29) và BamiBERT (p = 0,69) không khác biệt có ý nghĩa.

## 3. Cấu hình mô hình chốt

| Thành phần | Giá trị |
|---|---|
| Hệ số nhiệt độ T (hiệu chuẩn trên cal) | 1,22 |
| Ngưỡng bằng chứng / ngưỡng cụm một từ | 0,30 / 0,40 |
| Độ dài tối đa (token) | 512 |
| Số epoch | 5 |
| Tăng cường trực tuyến: bỏ dấu / biến dạng bề mặt | 30,0% / 20,0% |
| Hệ số FGM | 0,5 |
| Trọng số FP32 (MiB) | 1.177 |
| GPU huấn luyện | NVIDIA L4 |
| PyTorch / Transformers | 2.14.1+cu130 / 4.57.6 |
| Phiên bản dữ liệu (commit) | `546300b` |

## 4. Kết quả trên bộ test khóa

### 4.1 So sánh ba hệ

Nguồn: `results/v2/so_sanh_3_he/metrics.json`. v1 chỉ có nhãn nhị phân nên không có Macro-F1 và bằng chứng.

| Tập | Hệ | Macro-F1 | Binary-F1 | P nguy hại | R nguy hại | FPR | BC F1 |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| C0 | SafeViChi v1 | – | 31,2 | 100,0 | 18,5 | 0,0% | – |
| C0 | gpt-oss-120b (API) | 77,2 | 89,2 | 98,5 | 81,5 | 3,1% | 70,7 |
| C0 | **SafeViChi v2** | **86,5** [82,5–89,9] | **92,9** [90,2–95,1] | 95,6 | 90,3 | 10,4% | **83,0** |
| C1 | SafeViChi v1 | – | 31,0 | 96,7 | 18,5 | 1,6% | – |
| C1 | gpt-oss-120b (API) | 58,5 | 74,5 | 93,3 | 62,0 | 10,9% | 56,1 |
| C1 | **SafeViChi v2** | **78,6** [75,3–81,8] | **90,3** [88,1–92,4] | 95,1 | 85,9 | 10,9% | **76,5** |
| C2 | SafeViChi v1 | – | 39,5 | 93,7 | 25,0 | 4,2% | – |
| C2 | gpt-oss-120b (API) | 62,7 | 77,9 | 97,5 | 64,9 | 4,2% | 32,2 |
| C2 | **SafeViChi v2** | **77,1** [74,0–80,2] | **90,9** [88,8–92,7] | 95,4 | 86,8 | 10,4% | **75,3** |

Khoảng trong ngoặc vuông là khoảng tin cậy bootstrap 95%.

### 4.2 Theo từng nhãn (SafeViChi v2)

| Nhãn | n (C0) | C0: P / R / F1 | C1: P / R / F1 | C2: P / R / F1 | F1 LLM API (C0) |
|---|:---:|:---:|:---:|:---:|:---:|
| `AN_TOAN` | 96 | 78,9 / 89,6 / **83,9** | 71,9 / 89,1 / **79,5** | 73,2 / 89,6 / **80,6** | 79,8 |
| `XUC_PHAM` | 53 | 98,0 / 90,6 / **94,1** | 85,8 / 91,5 / **88,6** | 79,2 / 89,6 / **84,1** | 86,7 |
| `QUAY_ROI` | 51 | 87,8 / 70,6 / **78,3** | 83,9 / 51,0 / **63,4** | 85,1 / 55,9 / **67,5** | 59,7 |
| `DE_DOA` | 62 | 79,7 / 82,3 / **81,0** | 70,8 / 68,5 / **69,7** | 63,9 / 61,3 / **62,5** | 64,1 |
| `LUA_DAO` | 72 | 95,8 / 94,4 / **95,1** | 94,8 / 88,9 / **91,8** | 96,9 / 85,4 / **90,8** | 95,8 |

### 4.3 Ma trận nhầm lẫn trên C0 (SafeViChi v2)

Hàng là nhãn đúng, cột là nhãn dự đoán.

| Đúng \ Dự đoán | `AN_TOAN` | `XUC_PHAM` | `QUAY_ROI` | `DE_DOA` | `LUA_DAO` |
|---|:---:|:---:|:---:|:---:|:---:|
| `AN_TOAN` | 86 | 0 | 1 | 6 | 3 |
| `XUC_PHAM` | 2 | 48 | 3 | 0 | 0 |
| `QUAY_ROI` | 8 | 0 | 36 | 7 | 0 |
| `DE_DOA` | 9 | 1 | 1 | 51 | 0 |
| `LUA_DAO` | 4 | 0 | 0 | 0 | 68 |

Tổng cộng 10 đoạn an toàn bị báo nhầm và 23 đoạn nguy hại bị bỏ sót (dự đoán `AN_TOAN`).

### 4.4 Bằng chứng tô sáng

Đo tách riêng khỏi phân loại, ngưỡng bằng chứng đặt riêng.

| Độ đo | C0 | C1 | C2 |
|---|:---:|:---:|:---:|
| BC precision theo ý | 85,7 | 82,5 | 81,5 |
| BC recall theo ý | 84,6 | 76,9 | 75,6 |
| **BC F1 theo ý** | **83,0** | **76,5** | **75,3** |
| BC F1 mức từ | 79,3 | 72,2 | 70,2 |
| BC F1 theo ý của LLM API | 70,7 | 56,1 | 32,2 |
| Tỉ lệ tô oan đoạn an toàn | 9,4% | 8,8% | 8,8% |
| Tỉ lệ đoạn nguy hại có bằng chứng rỗng | 3,3% | 6,4% | 4,8% |
| Tỉ lệ cụm không chép nguyên văn | 0,0% | 0,0% | 0,0% |
| Số cụm bằng chứng dự đoán | 277 | 499 | 527 |

| BC F1 theo nhãn (v2) | C0 | C1 | C2 |
|---|:---:|:---:|:---:|
| `XUC_PHAM` | 91,8 | 85,2 | 84,2 |
| `QUAY_ROI` | 78,1 | 66,0 | 65,0 |
| `DE_DOA` | 76,4 | 73,4 | 74,6 |
| `LUA_DAO` | 84,7 | 78,6 | 75,8 |

## 5. Kết quả theo lát cắt

### 5.1 Theo kiểu biến dạng

| Lát cắt | n | v2 Macro-F1 | v2 Binary-F1 | v2 FPR | v2 BC F1 | LLM Macro-F1 | v1 Binary-F1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| C1 · Bỏ dấu | 102 | 80,1 | 91,4 | 0,0% | 71,8 | 58,1 | 37,9 |
| C1 · Chèn ký tự | 81 | 76,7 | 91,5 | 15,0% | 67,6 | 50,4 | 25,7 |
| C1 · Dính từ | 96 | 87,1 | 93,0 | 6,9% | 85,5 | 57,4 | 28,2 |
| C1 · Gõ sai | 104 | 72,8 | 88,6 | 21,2% | 75,5 | 53,4 | 30,2 |
| C1 · Kéo dài ký tự | 89 | 81,0 | 90,4 | 3,7% | 76,9 | 55,1 | 17,6 |
| C1 · Số thay chữ | 93 | 79,5 | 92,9 | 7,4% | 76,0 | 67,5 | 35,0 |
| C1 · Teencode | 103 | 70,9 | 84,7 | 20,0% | 81,7 | 56,0 | 37,8 |
| C2 · Homoglyph Cyrillic | 110 | 78,1 | 89,4 | 11,1% | 80,8 | 72,4 | 21,4 |
| C2 · Kéo dài kép | 98 | 82,4 | 92,9 | 3,2% | 74,8 | 67,0 | 51,1 |
| C2 · Ký tự ngăn cách lạ | 108 | 78,2 | 93,8 | 3,1% | 75,1 | 55,8 | 29,2 |
| C2 · Ký tự fullwidth | 122 | 82,9 | 94,8 | 3,1% | 74,0 | 73,7 | 33,3 |
| C2 · Chèn dấu cách | 114 | 67,1 | 85,9 | 21,9% | 70,5 | 45,1 | 58,7 |
| C2 · Ký tự vô hình (zero-width) | 116 | 72,4 | 88,6 | 20,7% | 77,0 | 57,9 | 35,9 |

C1 chia theo kiểu biến dạng đã có trong tăng cường lúc train và kiểu chưa có:

| Nhóm C1 | n | v2 Macro-F1 | v2 Binary-F1 | v2 FPR | v2 BC F1 | LLM Macro-F1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Kiểu đã thấy khi train | 398 | 76,1 | 88,7 | 12,1% | 76,3 | 56,4 |
| Kiểu chưa thấy khi train | 270 | 82,2 | 92,5 | 9,2% | 76,7 | 61,3 |

### 5.2 Kịch bản trong phân phối và kịch bản chưa thấy

| Lát cắt | n | v2 Macro-F1 | v2 Binary-F1 | v2 FPR | v2 BC F1 | LLM Macro-F1 | v1 Binary-F1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| C0 · trong phân phối | 185 | 93,0 | 97,4 | 2,0% | 86,8 | 84,8 | 46,6 |
| C0 · chưa thấy | 149 | 62,1 | 86,9 | 19,6% | 77,4 | 53,0 | 5,7 |
| C1 · trong phân phối | 370 | 86,7 | 95,9 | 6,0% | 79,3 | 63,3 | 45,0 |
| C1 · chưa thấy | 298 | 54,4 | 82,5 | 16,3% | 71,8 | 39,0 | 8,3 |
| C2 · trong phân phối | 370 | 86,0 | 96,4 | 6,0% | 76,3 | 67,6 | 49,6 |
| C2 · chưa thấy | 298 | 53,2 | 83,0 | 15,2% | 73,6 | 45,5 | 24,2 |

### 5.3 Theo giọng vùng miền

| Lát cắt | n | v2 Macro-F1 | v2 Binary-F1 | v2 FPR | v2 BC F1 | LLM Macro-F1 | v1 Binary-F1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| C0 · Bắc | 112 | 86,4 | 92,9 | 12,1% | 83,9 | 75,6 | 31,9 |
| C0 · Trung | 111 | 84,5 | 94,0 | 3,1% | 85,6 | 72,8 | 24,4 |
| C0 · Nam | 111 | 87,9 | 91,7 | 16,1% | 79,7 | 81,0 | 36,7 |
| C1 · Bắc | 224 | 81,6 | 92,5 | 9,1% | 76,9 | 55,5 | 30,1 |
| C1 · Trung | 222 | 73,0 | 89,7 | 4,7% | 78,1 | 57,8 | 22,2 |
| C1 · Nam | 222 | 80,2 | 88,7 | 19,4% | 74,6 | 60,2 | 39,8 |
| C2 · Bắc | 224 | 79,1 | 90,8 | 13,6% | 80,1 | 61,8 | 43,5 |
| C2 · Trung | 222 | 73,6 | 92,2 | 1,6% | 72,2 | 57,9 | 30,9 |
| C2 · Nam | 222 | 78,4 | 89,6 | 16,1% | 73,6 | 67,8 | 43,3 |

| Chênh lệch lớn nhất giữa 3 vùng (v2) | C0 | C1 | C2 |
|---|:---:|:---:|:---:|
| Macro-F1 | 3,5 | 8,6 | 5,6 |
| Binary-F1 | 2,3 | 3,8 | 2,6 |
| FPR | 13,0% | 14,7% | 14,6% |

Mỗi vùng chỉ có khoảng 31–33 đoạn `AN_TOAN`, nên một đoạn sai làm FPR vùng đổi khoảng 3 điểm.

### 5.4 Theo kênh giao tiếp

| Lát cắt | n | v2 Macro-F1 | v2 Binary-F1 | v2 FPR | v2 BC F1 | LLM Macro-F1 | v1 Binary-F1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| C0 · Messenger/Zalo | 127 | 93,4 | 97,0 | 10,7% | 82,8 | 78,2 | 37,7 |
| C0 · Nhóm lớp | 63 | 86,0 | 93,0 | 10,5% | 81,6 | 62,4 | 37,0 |
| C0 · Bình luận | 63 | 86,9 | 92,7 | 5,0% | 85,6 | 77,4 | 28,0 |
| C0 · Chat game | 35 | 84,1 | 90,5 | 0,0% | 81,4 | 73,4 | 29,6 |
| C0 · SMS | 46 | 67,1 | 80,0 | 23,5% | 83,6 | 87,3 | 0,0 |
| C1 · Messenger/Zalo | 254 | 86,4 | 95,7 | 12,5% | 78,0 | 52,7 | 38,4 |
| C1 · Nhóm lớp | 126 | 79,4 | 91,0 | 7,9% | 75,5 | 52,2 | 30,5 |
| C1 · Bình luận | 126 | 76,8 | 85,9 | 7,5% | 69,5 | 56,8 | 29,4 |
| C1 · Chat game | 70 | 74,6 | 89,7 | 8,3% | 81,1 | 63,5 | 25,9 |
| C1 · SMS | 92 | 57,6 | 75,7 | 17,6% | 78,8 | 76,4 | 9,8 |
| C2 · Messenger/Zalo | 254 | 84,5 | 96,2 | 10,7% | 76,7 | 62,2 | 41,3 |
| C2 · Nhóm lớp | 126 | 80,8 | 90,5 | 10,5% | 80,2 | 51,3 | 46,7 |
| C2 · Bình luận | 126 | 72,4 | 89,2 | 2,5% | 68,8 | 57,4 | 41,4 |
| C2 · Chat game | 70 | 72,6 | 90,7 | 4,2% | 74,1 | 69,3 | 35,7 |
| C2 · SMS | 92 | 57,5 | 74,3 | 23,5% | 71,9 | 76,7 | 18,8 |

### 5.5 Theo số lượt, độ lộ và người viết

| Lát cắt | n | v2 Macro-F1 | v2 Binary-F1 | v2 FPR | v2 BC F1 | LLM Macro-F1 | v1 Binary-F1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| C0 · hội thoại 1 dòng | 108 | 80,5 | 87,6 | 13,9% | 84,9 | 80,4 | 17,7 |
| C0 · hội thoại 2–8 dòng | 226 | 89,6 | 95,1 | 8,3% | 82,3 | 74,6 | 36,4 |
| C0 · nguy hại lộ liễu | 134 | 90,5 | 96,1 | – | 83,1 | 81,9 | 46,9 |
| C0 · nguy hại ẩn ý | 104 | 84,3 | 93,3 | – | 83,0 | 64,7 | 5,6 |
| C1 · hội thoại 1 dòng | 216 | 69,9 | 81,8 | 12,5% | 72,9 | 65,6 | 22,1 |
| C1 · hội thoại 2–8 dòng | 452 | 82,5 | 93,7 | 10,0% | 77,7 | 55,4 | 34,6 |
| C1 · nguy hại lộ liễu | 268 | 80,8 | 93,2 | – | 78,9 | 60,9 | 46,9 |
| C1 · nguy hại ẩn ý | 208 | 75,0 | 91,4 | – | 73,3 | 51,5 | 5,6 |
| C2 · hội thoại 1 dòng | 216 | 67,0 | 83,2 | 12,5% | 69,9 | 65,7 | 33,1 |
| C2 · hội thoại 2–8 dòng | 452 | 81,4 | 94,0 | 9,2% | 77,2 | 61,2 | 42,1 |
| C2 · nguy hại lộ liễu | 268 | 78,8 | 94,7 | – | 74,3 | 69,1 | 55,9 |
| C2 · nguy hại ẩn ý | 208 | 71,9 | 90,5 | – | 76,7 | 47,2 | 13,5 |

| Lát cắt | n | v2 Macro-F1 | v2 Binary-F1 | v2 FPR | v2 BC F1 | LLM Macro-F1 | v1 Binary-F1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| C0 · người viết A | 118 | 80,7 | 92,5 | 11,4% | 79,1 | 78,7 | 12,7 |
| C0 · người viết B | 132 | 80,0 | 89,3 | 11,4% | 80,2 | 65,0 | 0,0 |
| C0 · người viết C | 39 | 70,6 | 93,1 | 0,0% | 83,4 | 53,1 | 0,0 |
| C0 · người viết D | 45 | 100,0 | 100,0 | – | 93,5 | 98,9 | 92,9 |
| C1 · người viết A | 236 | 74,7 | 88,5 | 13,6% | 74,7 | 54,8 | 13,8 |
| C1 · người viết B | 264 | 67,9 | 87,5 | 10,2% | 71,2 | 52,2 | 3,3 |
| C1 · người viết C | 78 | 59,4 | 88,3 | 0,0% | 75,4 | 35,7 | 6,2 |
| C1 · người viết D | 90 | 99,4 | 99,4 | – | 88,2 | 85,4 | 88,9 |
| C2 · người viết A | 236 | 69,8 | 90,0 | 12,5% | 74,7 | 62,8 | 33,9 |
| C2 · người viết B | 264 | 63,4 | 87,5 | 10,2% | 68,7 | 47,8 | 13,6 |
| C2 · người viết C | 78 | 58,4 | 88,3 | 0,0% | 76,4 | 46,4 | 19,7 |
| C2 · người viết D | 90 | 99,4 | 100,0 | – | 86,2 | 88,2 | 86,1 |

Lát cắt độ lộ chỉ gồm đoạn nguy hại nên không có FPR. Người viết D không viết đoạn `AN_TOAN` nào.

## 6. Ablation

### 6.1 Đọc cả hội thoại hay chỉ dòng cuối

| Cấu hình (dev, đoạn nhiều dòng) | Macro-F1 |
|---|:---:|
| Đọc toàn bộ hội thoại | 98,0 |
| Chỉ đọc dòng cuối | 70,9 |
| Bộ phân loại chỉ dựa vào độ dài đoạn | 30,2 |

n = 587 đoạn nhiều dòng của tập dev; đọc cả hội thoại tăng +27,1 điểm.

### 6.2 Bật bộ chuẩn hóa v1 trước mô hình

Trên tập dev:

| Bộ thử dev | n | Văn bản gốc | Có chuẩn hóa | Chênh |
|---|:---:|:---:|:---:|:---:|
| Đoạn sạch C0 | 1.340 | 92,3 | 91,6 | −0,8 |
| Bỏ dấu | 1.216 | 87,5 | 87,2 | −0,3 |
| C1 trong phân phối | 1.255 | 88,8 | 90,4 | +1,6 |
| C1 giữ ngoài | 1.153 | 92,6 | 91,3 | −1,3 |
| Điểm mù homoglyph | 1.203 | 91,0 | 91,0 | 0,0 |
| Điểm mù zero-width | 1.330 | 90,1 | 89,3 | −0,7 |

Trên bộ test khóa (Macro-F1 và BC F1):

| Tập test | n | Macro-F1 gốc | Macro-F1 chuẩn hóa | Chênh | BC F1 gốc | BC F1 chuẩn hóa | Chênh BC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| C0 | 334 | 86,5 | 85,2 | −1,2 | 83,0 | 84,5 | +1,5 |
| C1 | 668 | 78,6 | 82,8 | +4,2 | 76,5 | 81,7 | +5,2 |
| C2 | 668 | 77,1 | 75,7 | −1,4 | 75,3 | 74,1 | −1,2 |
| Biến dạng chưa thấy khi train | 764 | 78,4 | 77,3 | −1,1 | 76,6 | 75,6 | −1,0 |

Chênh Macro-F1 theo từng kiểu biến dạng khi bật chuẩn hóa:

| Kiểu | Chênh Macro-F1 |
|---|:---:|
| C1 · Bỏ dấu | +0,6 |
| C1 · Chèn ký tự | +17,5 |
| C1 · Dính từ | +1,0 |
| C1 · Gõ sai | −1,2 |
| C1 · Kéo dài ký tự | +3,8 |
| C1 · Số thay chữ | +3,5 |
| C1 · Teencode | +4,2 |
| C2 · Homoglyph Cyrillic | −1,4 |
| C2 · Kéo dài kép | +2,1 |
| C2 · Ký tự ngăn cách lạ | +1,8 |
| C2 · Ký tự fullwidth | −0,7 |
| C2 · Chèn dấu cách | −3,6 |
| C2 · Ký tự vô hình (zero-width) | −4,3 |

## 7. Nén INT8 và chi phí trên thiết bị

### 7.1 FP32 so với ONNX Hybrid INT8

File mô hình INT8 301 MiB so với FP32 1.178 MiB (nhỏ hơn 3,9×), tokenizer 33 MiB. Nguồn: `results/v2/official_eval/dualorc_mmbert_base/test/int8_comparison.json`.

| Tập | n | Macro-F1 FP32 | Macro-F1 INT8 | Chênh | BC F1 FP32 | BC F1 INT8 | FPR FP32 | FPR INT8 | McNemar p |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| C0 | 334 | 86,5 | 86,0 | −0,5 | 83,0 | 84,0 | 10,4% | 10,4% | 1,00 |
| C1 | 668 | 78,6 | 77,7 | −0,9 | 76,5 | 76,7 | 10,9% | 10,9% | 0,58 |
| C2 | 668 | 77,1 | 75,7 | −1,4 | 75,3 | 76,3 | 10,4% | 13,0% | 0,12 |
| ALL | 1.670 | 79,6 | 78,6 | −1,0 | 77,4 | 78,1 | 10,6% | 11,7% | 0,07 |

Tỉ lệ khớp nhãn INT8 với FP32: C0 97,9%, C1 97,8%, C2 95,8%, toàn bộ 97,0%.

### 7.2 Laptop CPU AMD Ryzen AI 7 H 350 w/ Radeon 860M (onnxruntime 1.26.0)

| Cấu hình (batch 1) | n | p50 (ms) | p95 (ms) | RAM đỉnh (MiB) | Nạp mô hình (s) |
|---|:---:|:---:|:---:|:---:|:---:|
| ONNX INT8, 8 luồng | 100 | 24,7 | 31,5 | 986 | 1,5 |
| ONNX INT8, 4 luồng | 100 | 28,5 | 41,3 | 982 | 1,6 |
| ONNX INT8, 1 luồng | 50 | 61,5 | 92,5 | 982 | 1,4 |
| ONNX FP32, 8 luồng | 60 | 44,0 | 58,5 | 2.419 | 2,0 |
| PyTorch FP32, 8 luồng | 40 | 60,4 | 80,4 | 4.781 | 20,4 |

Nguồn: `results/v2/official_eval/dualorc_mmbert_base/test/bench_cpu.json`.

### 7.3 Laptop CPU Intel(R) Core(TM) Ultra 7 155H, chạy pin (onnxruntime 1.32.0)

| Số luồng | p50 (ms) | p95 (ms) | Thông lượng (đoạn/s) | RAM đỉnh (MiB) | Nạp (s) |
|---|:---:|:---:|:---:|:---:|:---:|
| 1 | 197,2 | 450,2 | 4,4 | 902 | 5,2 |
| 4 | 81,6 | 150,6 | 11,3 | 903 | 5,2 |
| 8 | 69,1 | 120,8 | 13,5 | 904 | 4,5 |

Chạy trọn 1.670 mẫu: Macro-F1 78,3, BC F1 77,7, khớp nhãn PyTorch 97,0%. Năng lượng đo từ pin (cửa sổ 300 s): công suất nghỉ 8,6 W, khi tải 14,3 W; tăng thêm 510 mJ/đoạn (1.285 mJ/đoạn tính cả máy), tức 0,14 Wh cho 1.000 đoạn. Nguồn: `results/v2/do_thiet_bi/cpu_laptop.json`.

### 7.4 Trình duyệt Edge 158.0.4370.33 (ONNX Runtime Web 1.31.1, WASM · 4 luồng)

| Chỉ số | Giá trị |
|---|---|
| Độ trễ p50 / p95 (ms) | 337 / 605 |
| Thông lượng (đoạn/s) | 2,8 |
| Dung lượng tải về lần đầu (MiB) | 348 |
| Thời gian nạp mô hình (s) | 7,2 |
| RAM tab trống / sau nạp / đỉnh (MiB) | 434 / 1.823 / 1.383 |
| RAM tăng thêm so với tab trống (MiB) | 949 |
| Năng lượng tăng thêm (mJ/đoạn) / cho 1.000 đoạn (Wh) | 966 / 0,27 |
| Macro-F1 / BC F1 trên 1.670 mẫu | 78,4 / 77,8 |
| Khớp nhãn với PyTorch / với CLI | 96,8% / 97,2% |

Nguồn: `results/v2/do_thiet_bi/trinh_duyet.json`.

### 7.5 Điện thoại Samsung S26 Ultra (ONNX INT8, 4 luồng)

p50 25,0 ms, p95 35,4 ms, RAM đỉnh 678 MiB (PSS 416 MiB), nạp mô hình 0,7 s; khớp nhãn với FP32 32/32 trên 32 đoạn C0.

### 7.6 Baseline LLM lớn qua API

gpt-oss-120b qua venice/int4, reasoning low, 4 luồng song song. Tổng chi phí chạy cả bộ test: 0,14 USD.

| Tập | p50 (s) | p95 (s) | USD / 1.000 đoạn |
|---|:---:|:---:|:---:|
| C0 | 3,46 | 9,55 | 0,085 |
| C1 | 3,16 | 8,22 | 0,085 |
| C2 | 2,83 | 4,85 | 0,085 |

## 8. SafeViChi v1 trên bộ test gốc của v1

| Tập v1 | F1 lớp HATE (ViHateT5 FGM + chuẩn hóa) |
|---|:---:|
| C0 | 85,3 |
| C1 | 82,7 |
| C2 | 66,5 |

n = 2.376 câu, nguồn `v1/results/step3/test/matrix.json`; bảng đầy đủ các baseline v1 (blacklist, ViHateT5 gốc, có/không chuẩn hóa) ở [v1/results/step3/test/matrix.md](../v1/results/step3/test/matrix.md). Recall theo nhãn v2 của v1 trên C0: `AN_TOAN` 100,0%, `XUC_PHAM` 73,6%, `QUAY_ROI` 0,0%, `DE_DOA` 8,1%, `LUA_DAO` 0,0%.
