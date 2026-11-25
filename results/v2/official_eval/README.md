# Phần 2 — Mô hình Chính thức (`SafeVi-DualORC-CRF / mmBERT-base`) & Đánh giá Bộ Test Khóa

Thư mục này tập trung toàn bộ hồ sơ của **mô hình chính thức đã chốt** từ giai đoạn lựa chọn trên `dev` ([`../dev_selection/`](../dev_selection/)) — **`SafeVi-DualORC-CRF (mmBERT-base)`** — bao gồm cấu hình, tham số hiệu chuẩn đóng băng, kết quả `dev` đối chiếu và kết quả đánh giá chính thức trên bộ test người viết đã khóa (`1.670` mẫu: `334 C0 + 668 C1 + 668 C2`).

---

## 1. Hồ sơ mô hình chốt & Cấu hình đóng băng

| Thuộc tính | Giá trị đóng băng |
| :--- | :--- |
| **Tên cấu hình** | `dualorc_mmbert_base` (`SafeVi-DualORC-CRF`, Schema v3) |
| **Backbone** | `jhu-clsp/mmBERT-base` (rev `c5955035435e2bf121cde7f3c8863ef52ff35d82`, 307M, 22 lớp, ModernBERT) |
| **Checkpoint & Hạt giống** | Epoch 5 / 5 (`best_select_score = 0,833690`), `seed = 42`, `dtype = torch.bfloat16` |
| **Dữ liệu khóa** | Commit `371a0f1caf3f7b62f62b87dbf557c66eda9c544a` (`so_sanh.jsonl` giữ nguyên SHA-256, `0` rò rỉ với train, xác minh tại [`../kiem_tra_ro_ri/bao_cao.json`](../kiem_tra_ro_ri/bao_cao.json); bộ đo bằng chứng `f1_theo_y_clause_he_so_phu_1.5`) |
| **Hiệu chuẩn đóng băng (`calibration.json`)** | `T = 1,2198`, `chosen_decoder = marginal`, `marginal_threshold = 0,30`, `single_word_threshold = 0,40`, `o_bias = -0,15`, `bridge_ratio = 0,0`, `fallback_min_prob = 0,20` |
| **Chi phí suy luận GPU (L4, `batch=1`)** | Trọng số FP32 `1.177,0 MiB` (đỉnh VRAM `1.218,2 MiB`), độ trễ `p50 = 39,43 ms` / `p95 = 42,73 ms` |

---

## 2. Ma trận chính thức trên Bộ Test Khóa: `C0 / C1 / C2` (Dòng `#5` thô)

Mô hình sử dụng nguyên vẹn checkpoint và `calibration.json` đã đóng băng (không hiệu chỉnh lại bất kỳ siêu tham số hay ngưỡng nào sau khi chấm test) trên toàn bộ 3 tập test người viết:
- **`C0` (`n = 334`):** Hội thoại sạch 1–8 lượt do người tự viết (`96 AN_TOAN`, `72 LUA_DAO`, `62 DE_DOA`, `53 XUC_PHAM`, `51 QUAY_ROI`), cân bằng 3 giọng vùng (`112` Bắc, `111` Trung, `111` Nam), gồm `185` mẫu trong phân phối và `149` mẫu kịch bản chưa thấy.
- **`C1` (`n = 668`):** `334 × 2` bản biến dạng né lọc có kiểm soát (`bo_dau`, `chen_ky_tu`, `dinh_tu`, `go_sai`, `keo_dai`, `so_thay_chu`, `teencode`; biến dạng ~60% số từ, luôn trúng cụm bằng chứng).
- **`C2` (`n = 668`):** `334 × 2` bản biến dạng điểm mù (`cyrillic_homoglyph`, `double_elongation`, `exotic_separator`, `fullwidth`, `space_injection`, `zero_width`) — **100% kiểu né lọc chưa từng xuất hiện khi huấn luyện**.

### 2.1. Bảng chỉ số chính thức (Dòng `#5` — Mô hình thô)

| Chỉ số đánh giá | `C0` — Sạch (`n = 334`) | `C1` — Né lọc (`n = 668`) | `C2` — Điểm mù (`n = 668`) |
| :--- | :---: | :---: | :---: |
| **Macro-F1 (5 lớp)** | **0,8647** | **0,7859** | **0,7708** |
| Bootstrap 95% CI (Macro-F1) | `[0,8255 – 0,8990]` | `[0,7532 – 0,8176]` | `[0,7399 – 0,8018]` |
| Macro-F1 (4 lớp nguy hại chính) | 0,8456 | 0,7609 | 0,7534 |
| **Binary-F1 (Nguy hại vs `AN_TOAN`)** | **0,9287** | **0,9029** | **0,9087** |
| Accuracy | 0,8653 | 0,7979 | 0,7829 |
| `FPR(AN_TOAN)` *(càng thấp càng tốt)* | **10,42%** | 10,94% | **10,42%** |
| **Evidence F1 (mức từ)** | **0,7931** | 0,7220 | 0,7017 |
| **Evidence F1 theo ý** *(hệ số phủ `1,5`, tính trên mẫu báo nguy hại)* | **0,8303** | 0,7649 | 0,7531 |
| `safe_mark_rate` *(tô oan `AN_TOAN`)* | 9,38% | **8,85%** | **8,85%** |
| `word_length_ratio` | 0,9758× | 0,8735× | 0,8546× |
| `harm_empty_evidence_rate` | **3,26%** | 6,36% | 4,84% |
| **Worst-group Region Macro-F1** | **0,8448** | 0,7301 | 0,7355 |
| **Region Gap (`max − min`)** | **0,0347** | 0,0857 | 0,0559 |

### 2.2. Độ suy giảm dưới tấn công né lọc (`Δ` so với `C0`) & Đối chiếu `dev → test C0`

| Phép so sánh | Δ Macro-F1 (5 lớp) | Δ Binary-F1 | Δ Evidence F1 theo ý | Nhận xét kỹ thuật |
| :--- | :---: | :---: | :---: | :--- |
| **`C1 − C0`** *(né lọc chuẩn)* | `-0,0788` | `-0,0258` | `-0,0654` | Biến dạng mạnh ~60% từ làm giảm phân tách mịn giữa 4 nhãn nguy hại, nhưng phát hiện nguy hại nhị phân vẫn đạt `0,9029` và `FPR` chỉ nhích nhẹ (`10,42% → 10,94%`). |
| **`C2 − C0`** *(100% điểm mù chưa thấy)* | `-0,0939` | **`-0,0200`** | `-0,0772` | Dưới 6 kiểu điểm mù chưa từng thấy khi train, **Binary-F1 chỉ tụt `-2,00 pp` (`0,9287 → 0,9087`)**, `FPR` giữ nguyên `10,42%` và `Evidence F1 theo ý` đạt `0,7531`, khẳng định sức chống chịu mạnh của biểu diễn đa ngôn ngữ `mmBERT-base` + `Byte-CNN`. |
| **`Test C0` vs `Dev`** *(chuyển miền sang người viết)* | `-0,0584` *(0,9231 → 0,8647)* | **`+0,0024`** *(0,9263 → 0,9287)* | *(bộ đo `f1_theo_y = 0,8303`)* | Chuyển sang hội thoại người viết hoàn toàn mới: **Binary-F1 tăng `+0,24 pp`** (`0,9263 → 0,9287`), `Evidence F1 mức từ = 0,7931`, `Evidence F1 theo ý = 0,8303` và `0,0%` non-verbatim. |

---

## 3. Phân tích lợi ích bộ chuẩn hóa: Dòng `#6` (`#5` + Chuẩn hóa v1) vs Dòng `#5` (Thô)

### 3.1. Tổng hợp hiệu ứng chuẩn hóa v1 (`#6 − #5`)

| Tập / Chỉ số | Dòng `#5` (Thô) | Dòng `#6` (+ Chuẩn hóa v1) | Lợi ích (`#6 − #5`) |
| :--- | :---: | :---: | :---: |
| `C0` — Macro-F1 (5 lớp) | **0,8647** | 0,8524 | `-0,0123` |
| **`C1` — Macro-F1 (5 lớp)** | 0,7859 | **0,8275** | **`+0,0416`** |
| **`C1` — Evidence F1 theo ý** | 0,7649 | **0,8165** | **`+0,0516`** |
| `C2` — Macro-F1 (5 lớp) | **0,7708** | 0,7568 | `-0,0140` |
| `C2` — Evidence F1 theo ý | **0,7531** | 0,7411 | `-0,0120` |
| `strictly_unseen` (`n = 764`) | **0,7836** | 0,7726 | `-0,0110` |

### 3.2. Chi tiết hiệu ứng chuẩn hóa v1 theo từng kiểu biến dạng (`by_kieu`, 13 kiểu)

| Nhóm | Kiểu biến dạng | `Δ Macro-F1 (#6 − #5)` | Đánh giá |
| :---: | :--- | :---: | :--- |
| **C1** | `chen_ky_tu` | **`+0,1751`** | Cải thiện đột biến (`+17,51 pp`, đạt `0,9423`; Ev-Y `+16,44 pp`) — chuẩn hóa khôi phục chính xác từ bị chèn ký tự xen kẽ. |
| **C1** | `teencode` | **`+0,0421`** | Từ điển teencode giúp chuẩn hóa cách viết tắt tự nhiên và phục hồi bằng chứng rõ rệt (`+5,11 pp` F1 theo ý). |
| **C1** | `keo_dai` | **`+0,0382`** | Rút gọn ký tự kéo dài giúp khôi phục biên từ và bằng chứng (`+4,62 pp` F1 theo ý). |
| **C1** | `so_thay_chu` | **`+0,0354`** | Phục hồi hiệu quả các biến thể leetspeak thay chữ bằng số (`+11,14 pp` F1 theo ý). |
| **C1** | `dinh_tu` | **`+0,0105`** | Cải thiện nhẹ biên tách từ trên nhóm dính từ (`0,8709 → 0,8814`). |
| **C1** | `bo_dau` | `+0,0062` | Gần trung tính (`0,8006 → 0,8068`); mô hình thô đã được tăng cường `accent_aug = 0,30` nên tự xử lý tiếng Việt không dấu ổn định. |
| **C1** | `go_sai` | `-0,0120` | Giảm nhẹ khi luật chuẩn hóa v1 sửa nhầm từ gõ sai ngữ cảnh. |
| **C2** | `double_elongation` | **`+0,0209`** | Hưởng lợi rõ từ bước rút gọn ký tự lặp (`0,8243 → 0,8452`). |
| **C2** | `exotic_separator` | **`+0,0179`** | Hưởng lợi từ bước loại bỏ ký tự phân cách lạ (`0,7816 → 0,7995`). |
| **C2** | `fullwidth` | `-0,0074` | Gần trung tính so với bản thô (`0,8288 → 0,8214`). |
| **C2** | `cyrillic_homoglyph` | `-0,0135` | Tokenizer đa ngôn ngữ 256k + `Byte-CNN` của `mmBERT-base` tự xử lý homoglyph ổn định hơn bộ quy tắc cứng v1. |
| **C2** | `space_injection` | `-0,0364` | Giảm do v1 không ghép khoảng trắng và từ điển v1 kích hoạt sai trên các mảnh rời (`0,6709 → 0,6345`). |
| **C2** | `zero_width` | `-0,0431` | Mô hình thô tự xử lý tốt ký tự ẩn hơn so với pipeline chuẩn hóa toàn cục v1 (`0,7239 → 0,6808`). |

**Chính sách triển khai khuyến nghị:** Giữ báo cáo song song cả Dòng `#5` (thô), Dòng `#6` (+ chuẩn hóa v1 đối chiếu) và **Bộ chuẩn hóa của SafeViChi v2 (`auto`, mặc định trên CLI `safevichi`)** tại [`src/shared/normalization/v2.py`](../../../src/shared/normalization/v2.py) — loại bỏ hoàn toàn tác hại lên `C0` (`0,00 pp` giảm) đồng thời khắc phục trọn vẹn cả `C1` lẫn 6 kiểu điểm mù `C2` và né lọc trộn đa tầng.

---

## 4. Phân tích chi tiết theo lát cắt trên tập test sạch `C0` (`n = 334`)

| Nhóm lát cắt | Lát cắt | `n` | `macro_present` | `binary_f1` | `accuracy` |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Tổng thể** | `that` / `bien_dang:C0` | 334 | **0,8647** | **0,9287** | **0,8653** |
| **Độ dài hội thoại** | `nhiều lượt` *(2–8 lượt)* | 226 | **0,8961** | **0,9509** | **0,8938** |
| | `1 lượt` | 108 | 0,8048 | 0,8759 | 0,8056 |
| **Khái quát kịch bản** | `phan:trong_phan_phoi` | 185 | **0,9300** | **0,9736** | **0,9405** |
| | `phan:chua_thay` *(LD7, LD12, DD6, QR6, mới)* | 149 | 0,6209 | **0,8687** | 0,7718 |
| **Giọng vùng miền** | `vung:bac` | 112 | 0,8642 | 0,9290 | **0,8661** |
| *(gap = `0,0346`)* | `vung:trung` | 111 | 0,8448 | **0,9404** | 0,8649 |
| | `vung:nam` | 111 | **0,8795** | 0,9172 | 0,8649 |
| **Độ lộ liễu** | `do_lo:lo_lieu` | 134 | **0,9048** | **0,9612** | **0,8806** |
| | `do_lo:an_y` | 104 | 0,8426 | **0,9333** | 0,8173 |
| **Kênh giao tiếp** | `kenh:messenger_zalo` | 127 | **0,9344** | **0,9697** | **0,9370** |
| | `kenh:binh_luan` | 63 | 0,8690 | 0,9268 | 0,8730 |
| | `kenh:nhom_lop` | 63 | 0,8598 | 0,9302 | 0,8413 |
| | `kenh:chat_game` | 35 | 0,8414 | 0,9048 | 0,8286 |
| | `kenh:sms` | 46 | 0,6712 | 0,8000 | 0,7174 |
| **Nguồn gán nhãn** | `cach_gan:nguoi` | 325 | **0,8833** | **0,9330** | **0,8800** |
| | `cach_gan:llm_nguoi_duyet` *(ca ranh giới nhỏ)* | 9 | 0,2727 | 0,8000 | 0,3333 |

### Điểm nhấn phân tích lát cắt
1. **Hội thoại đa lượt & kênh nhắn tin chủ lực rất mạnh:** Trên `nhiều lượt` (`n = 226`), mô hình đạt `Macro-F1 = 0,8961`, `Binary-F1 = 0,9509`; trên `kenh:messenger_zalo` (`n = 127`) đạt `Macro-F1 = 0,9344`, `Binary-F1 = 0,9697` — phản ánh đúng hiệu quả của việc chọn `mmBERT-base` nhờ lợi thế suy luận liên lượt trên `dev`.
2. **Ổn định trên cả 3 vùng miền:** Cả 3 miền Bắc (`0,8642`), Trung (`0,8448`) và Nam (`0,8795`) đều đạt độ chính xác đồng đều (`86,49%–86,61%`, đúng `15` lỗi ở mỗi vùng) với `region_gap_max_min = 0,0346`.
3. **Ngôn ngữ ẩn ý (`do_lo:an_y`, `n = 104`):** Giữ vững `Macro-F1 = 0,8426` và `Binary-F1 = 0,9333` trên hội thoại ẩn ý do người viết.
4. **Kịch bản chưa thấy (`phan:chua_thay`, `n = 149`) & kênh SMS:** Trong phân phối (`n = 185`) đạt `Macro-F1 = 0,9300`, `Accuracy = 94,05%`; sang kịch bản chưa thấy (`n = 149`), `Binary-F1` vẫn đạt **`0,8687`** và `Accuracy = 77,18%` (`34/45` lỗi C0 tập trung ở nhóm này). Lưu ý `macro_present = 0,6209` chịu độ nhạy lớn do lớp `XUC_PHAM` trong `phan:chua_thay` chỉ có đúng `n = 3` mẫu (xem `data/v2/test_stats.json`). Tương tự, `cach_gan:llm_nguoi_duyet` chỉ có `n = 9` ca ranh giới (`Binary-F1 = 0,8000`), còn `325` mẫu `cach_gan:nguoi` đạt `Macro-F1 = 0,8833`. Kênh `sms` (`n = 46`, `0,6712`, `13/45` lỗi) gồm nhiều tin nhắn ngắn, ít ngữ cảnh hội thoại và kịch bản lừa đảo/cảnh báo giao dịch ranh giới — là trọng tâm phân tích lỗi trong báo cáo.

---

## 5. Xuất ONNX, Lượng tử hóa Hybrid INT8 (Dòng `#7`) & Benchmark Laptop CPU

Để đáp ứng Yêu cầu #3 (triển khai cục bộ `< 2 GB`, đo chi phí độ trễ/RAM/năng lượng trên môi trường đích và định lượng đánh đổi chất lượng khi nén), mô hình chính thức được xuất sang **ONNX opset 17** và lượng tử hóa theo chiến lược **Outlier-Aware Hybrid INT8** (`python -m src.v2.bench.export_onnx` & `python -m src.v2.bench.benchmark_cpu`):

- **Chiến lược nén 3 thành phần:**
  1. **Bảng từ vựng 256k (`tok_embeddings`):** `PerRowInt8Embedding` (INT8 đối xứng theo từng hàng từ vựng + scale FP32) — nén `~750 MiB → ~188,5 MiB` (`4,0×`, sai số logit tối đa `0,034`).
  2. **22 tầng ModernBERT:** `quantize_dynamic` (`QInt8 per-channel` W8A8) cho các lớp Attention & `mlp.Wi`, kết hợp nén trọng số **W8A32 (`DequantizeLinear`, `axis=1`)** riêng cho **22 lớp chiếu xuống GeGLU `mlp.Wo`** — triệt tiêu hoàn toàn hiện tượng cắt ngọn ngoại lai của GeGLU (giảm sai số logit từ `4,60` xuống `0,52`, đạt `16/16` khớp nhãn với FP32).
  3. **Đầu `SafeVi-DualORC-CRF` (~6,5 MiB):** giữ nguyên FP32 để bảo toàn độ hiệu chuẩn xác suất và điểm bằng chứng; giải mã BIO-CRF Forward-Backward/Viterbi chạy thuần NumPy (không cần PyTorch lúc suy luận).
- **Toàn vẹn kiểm chứng số học:** ONNX FP32 khớp PyTorch FP32 tới `8e-6` (`16/16` nhãn), CRF thuần NumPy khớp PyTorch tới `1,45e-5`.

### 5.1. Đánh đổi chất lượng khi nén ONNX Hybrid INT8 trên trọn bộ test `1.670` mẫu (`int8_comparison.json`)

Đối chiếu trực tiếp **ONNX FP32** vs **ONNX Hybrid INT8** trên cùng bản bộ test hiện tại (`data/v2/test_day_du.jsonl`, `test_perturb_C1.jsonl`, `test_perturb_C2.jsonl` — `100%` cụm bằng chứng verbatim cho cả hai):

| Tầng đánh giá | `n` | Macro-F1 FP32 → INT8 (`Δ` / Giữ lại) | Binary-F1 FP32 → INT8 | Evidence F1 theo ý FP32 → INT8 | Khớp nhãn | McNemar 5 lớp / Nhị phân (`p`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`C0` (Sạch)** | 334 | `0,8647` → **`0,8598`** (`-0,0049` / **`99,43%`**) | `0,9287` → **`0,9333`** | `0,8303` → **`0,8401`** (`+0,98 pp`) | **97,90%** | `1,0000` / `0,6875` *(hòa)* |
| **`C1` (Né lọc)** | 668 | `0,7859` → **`0,7767`** (`-0,0092` / **`98,83%`**) | `0,9029` → **`0,9053`** | `0,7649` → **`0,7674`** (`+0,25 pp`) | **97,75%** | `0,5811` / `0,7539` *(hòa)* |
| **`C2` (Điểm mù)** | 668 | `0,7708` → **`0,7567`** (`-0,0141` / **`98,17%`**) | `0,9087` → `0,9061` | `0,7531` → **`0,7630`** (`+0,99 pp`) | **95,81%** | `0,1153` / `0,6291` *(hòa)* |
| **Toàn bộ (`ALL`)** | **1.670** | `0,7960` → **`0,7857`** (`-0,0103` / **`98,71%`**) | `0,9104` → **`0,9113`** | `0,7737` → **`0,7808`** (`+0,71 pp`) | **97,01%** | `0,0730` / `1,0000` *(hòa)* |

> Không có bất kỳ khác biệt có ý nghĩa thống kê nào giữa **ONNX Hybrid INT8** và **FP32** trên mọi tầng (`p ≥ 0,073` cho 5 lớp; `p ≥ 0,629` cho nhị phân). Mô hình INT8 giữ **`99,43%`** Macro-F1 trên `C0`, **`98,71%`** toàn bộ `1.670` mẫu, đồng thời **Binary-F1 tổng thể (`0,9113`) và Evidence F1 theo ý (`0,7808`) đều nhích nhẹ so với FP32**.

### 5.2. Hiệu năng, RAM đỉnh & Năng lượng ước tính trên Laptop CPU đích (`bench_cpu.json`)

- **Môi trường đích:** `AMD Ryzen AI 7 H 350` (8 nhân / 16 luồng), `31,29 GiB` RAM, Windows 11, Python `3.12.9`, PyTorch `2.14.1+cpu`, ONNX Runtime `1.26.0`, `seed = 42`.
- **Giao thức đo:** Mỗi cấu hình chạy trong một **tiến trình (subprocess) độc lập** để đo chính xác RAM đỉnh (`peak_wset` qua `psutil`), suy luận tương tác **`batch = 1`** trên `n = 100` hội thoại thật `C0` (warmup `10` lượt; trung vị độ dài `47` token, p95 `71` token).

| Cấu hình suy luận (Laptop CPU, `batch = 1`) | Dung lượng tệp | Nạp lạnh | Độ trễ `p50` / `p95` (end-to-end) | Thông lượng | RAM sau nạp / **Đỉnh** | Năng lượng ước tính / 1.000 tin *(15–28 W)* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `PyTorch FP32` (`8 luồng`) | `1.177,0 MiB` | `20,43 s` | `60,42` / `80,38 ms` | `16,10 tin/s` | `1.765` / `4.781 MiB` | `0,259 – 0,483 Wh` |
| `ONNX FP32` (`8 luồng`) | `1.177,6 MiB` | `2,00 s` | `43,97` / `58,47 ms` | `22,33 tin/s` | `1.673` / `2.419 MiB` | `0,187 – 0,348 Wh` |
| **`ONNX Hybrid INT8` (`8 luồng`)** | **`300,69 MiB`** | **`1,53 s`** | **`24,68` / `31,54 ms`** | **`41,26 tin/s`** | **`815` / `986 MiB`** | **`0,101 – 0,189 Wh`** |
| `ONNX Hybrid INT8` (`4 luồng`) | `300,69 MiB` | `1,55 s` | `28,48` / `41,29 ms` | `34,11 tin/s` | `806` / `982 MiB` | `0,122 – 0,228 Wh` |
| `ONNX Hybrid INT8` (`1 luồng`) | `300,69 MiB` | `1,41 s` | `61,46` / `92,45 ms` | `15,45 tin/s` | `811` / `982 MiB` | `0,270 – 0,504 Wh` |

- **Mức tiết kiệm đạt được (`ONNX Hybrid INT8` 8 luồng vs `PyTorch FP32` 8 luồng):**
  - **Dung lượng đĩa:** giảm **`3,91×`** (`1.177,0 → 300,69 MiB`, chỉ chiếm **`15,0%`** trần `< 2 GB`; tính cả `tokenizer.json` `32,77 MiB` là **`333,5 MiB`**).
  - **Tốc độ suy luận CPU:** nhanh hơn **`2,45×`** (`60,42 → 24,68 ms` p50, **`41,26` tin/giây** — thậm chí nhanh hơn FP32 trên GPU L4 `39,43 ms` ở `batch=1`); khởi động lạnh nhanh hơn **`13,3×`** (`20,43 → 1,53 s`).
  - **RAM đỉnh:** giảm **`4,85×`** (`4.781 → 986 MiB`, trong đó mức tăng thực của mô hình trên nền Python `406 MiB` chỉ **`~409 MiB`**).
  - **Năng lượng ước tính:** giảm **`~2,56×`** (`0,101–0,189 Wh / 1.000 tin nhắn`, quy đổi từ `mean_latency × cTDP 15–28 W` do Windows non-admin không mở bộ đếm phần cứng RAPL).

---

## 6. Cấu trúc thư mục & Danh mục tệp

```text
results/v2/official_eval/
├── README.md                                  # Tệp này — báo cáo chính thức trên bộ test khóa (1.670 mẫu)
└── dualorc_mmbert_base/                       # Hồ sơ đầy đủ của mô hình chốt SafeVi-DualORC-CRF (mmBERT-base)
    ├── config.json                            # Cấu hình huấn luyện & kiến trúc đóng băng
    ├── environment.json                       # Phần cứng, phiên bản PyTorch/Transformers/CUDA
    ├── calibration.json                       # Nhiệt độ T & ngưỡng bằng chứng đóng băng từ tập cal
    ├── benchmark.json                         # Đo VRAM & độ trễ p50/p95 trên GPU L4
    ├── history.json                           # Lịch sử 5 epoch huấn luyện
    ├── split_ids.json                         # Danh sách ID phân chia fit/dev/cal (seed 42)
    ├── dev_metrics.json                       # Số liệu đối chiếu trên dev (n=1.340)
    ├── dev_predictions.jsonl                  # Dự đoán chi tiết trên dev
    ├── dev_stress_tests.json                  # Stress test thô trên dev
    ├── dev_norm_stress_tests.json             # Stress test + chuẩn hóa trên dev
    ├── dev_accent_stress.json                 # Kiểm tra bỏ dấu trên dev
    ├── dev_context_checks.json                # Kiểm chứng ngữ cảnh đa lượt trên dev
    ├── dev_diagnostics.png                    # Biểu đồ chẩn đoán trên dev
    └── test/                                  # Trọn bộ tệp xuất thô của phiên chấm test khóa (1.670 mẫu)
        ├── REAL_TEST_LOCK.json                # Khóa SHA-256 dữ liệu test & hiệu chuẩn đóng băng
        ├── real_test_official_matrix.json     # Ma trận chính thức C0/C1/C2, 15 lát cắt & 13 kiểu (Dòng #5)
        ├── real_test_norm_official_matrix.json# Ma trận chính thức + lợi ích chuẩn hóa #6−#5
        ├── real_test_{metrics,c1_metrics,c2_metrics}.json
        ├── real_test_norm_{metrics,c1_metrics,c2_metrics}.json
        ├── real_test_{predictions,c1_predictions,c2_predictions,all_predictions}.jsonl
        ├── real_test_norm_{predictions,c1_predictions,c2_predictions,all_predictions}.jsonl
        ├── real_test_{errors,c1_errors,c2_errors}.jsonl
        ├── real_test_stress_tests.json        # 5 bộ stress test trực tiếp trên test C0
        ├── real_test_diagnostics.png          # Biểu đồ chẩn đoán trên bộ test khóa
        ├── cli_eval_report.json               # Báo cáo tái kiểm chứng từ CLI chuẩn src.v2.eval.evaluate
        ├── real_test_int8_all_predictions.jsonl      # Dự đoán ONNX Hybrid INT8 (Dòng #7, 1.670 mẫu)
        ├── int8_comparison.json               # Đối chiếu FP32 vs ONNX Hybrid INT8 trên C0/C1/C2 & McNemar
        └── bench_cpu.json                     # Đo độ trễ p50/p95, RAM đỉnh & năng lượng trên Laptop CPU
```

### Trạng thái lưu trữ tệp trong `dualorc_mmbert_base/`

| Nhóm tệp | Vị trí | Trạng thái | Danh sách tệp chính |
| :--- | :--- | :---: | :--- |
| **Hồ sơ mô hình, hiệu chuẩn & `dev`** | [`dualorc_mmbert_base/`](dualorc_mmbert_base/) | **Đã lưu (13 tệp)** | `config.json`, `environment.json`, `calibration.json`, `benchmark.json`, `history.json`, `split_ids.json`, `dev_*.json`, `dev_predictions.jsonl`, `dev_diagnostics.png` |
| **Tệp xuất thô phiên chấm test khóa** | [`dualorc_mmbert_base/test/`](dualorc_mmbert_base/test/) | **Đã lưu (23 tệp)** | `REAL_TEST_LOCK.json`, `real_test_*official_matrix.json`, `real_test_*metrics.json`, `real_test_*predictions.jsonl`, `real_test_*errors.jsonl`, `real_test_stress_tests.json`, `real_test_diagnostics.png`, `cli_eval_report.json` |
| **Lượng tử hóa ONNX INT8 & CPU đích** | [`dualorc_mmbert_base/test/`](dualorc_mmbert_base/test/) | **Đã lưu (3 tệp)** | `int8_comparison.json`, `bench_cpu.json`, `real_test_int8_all_predictions.jsonl` |
| **Trọng số & gói ONNX (ngoài git)** | `models/dualorc_mmbert_base/` | **Đã lưu cục bộ** | `model_best.safetensors` (`1,15 GiB`), `last.pt` (`4,60 GiB`), `onnx/dualorc_crf_fp32.onnx` (`1,15 GiB`), `onnx/dualorc_crf_int8.onnx` (`300,69 MiB`), `onnx/crf_params.json`, `onnx/tokenizer/` |

### Lệnh tái lập & kiểm chứng chuẩn

```bash
# 1. Xuất ONNX FP32 + Hybrid INT8 và kiểm chứng sai số số học
python -m src.v2.bench.export_onnx

# 2. Chấm 1.670 mẫu bằng ONNX FP32/INT8 và đo độ trễ + RAM đỉnh độc lập trên Laptop CPU
python -m src.v2.bench.benchmark_cpu

# 3. Tái kiểm chứng chỉ số C0/C1/C2 bằng CLI đánh giá chuẩn
python -m src.v2.eval.evaluate \
  --gold data/v2/test_day_du.jsonl data/v2/test_perturb_C1.jsonl data/v2/test_perturb_C2.jsonl \
  --pred results/v2/official_eval/dualorc_mmbert_base/test/real_test_int8_all_predictions.jsonl \
  --reference-pred results/v2/official_eval/dualorc_mmbert_base/test/real_test_all_predictions.jsonl \
  --meta data/v2/test_day_du.jsonl
```

> **Quy tắc lưu trữ:** Không commit tệp trọng số nặng (`.safetensors`, `.pt`, `.onnx`, `.bin`) vào `results/v2/`; toàn bộ checkpoint lưu cục bộ trong `models/` (đã loại trừ trong `.gitignore`).
