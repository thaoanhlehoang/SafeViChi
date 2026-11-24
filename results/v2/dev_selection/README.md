# Phần 1 — Huấn luyện, So sánh 4 Ứng viên trên `dev` & Quyết định Chọn Mô hình

Thư mục này lưu trọn bộ đầu ra nhẹ (không chứa trọng số nặng) từ quá trình huấn luyện, hiệu chuẩn và đánh giá nội bộ của **4 ứng viên song song** thuộc 2 họ kiến trúc độc lập, cùng hồ sơ phân tích và quyết định chọn mô hình chính thức trước khi mở khóa tập test.

---

## 1. Thiết lập thực nghiệm đồng nhất (Controlled Comparison)

Để đảm bảo so sánh công bằng giữa các ứng viên, cả 4 lượt huấn luyện dùng chung toàn bộ quy trình dữ liệu, hạt giống ngẫu nhiên và tiêu chí chọn checkpoint:

- **Dữ liệu & phân chia:** Commit dữ liệu `546300b7d5c3f7a8c8b78165402182d4aed23b7e`, sơ đồ `DATA_SCHEME = "dev_80_10_10"` (`fit = 10.713`, `dev = 1.340`, `cal = 1.339`), dùng chung **100% `split_ids.json`** (`seed = 42`, `split_seed = 42`). Toàn bộ bộ test người viết (`C0/C1/C2`) được khóa ngoài quy trình này.
- **Phần cứng & độ chính xác:** NVIDIA L4 (`22,03 GiB`, `torch.bfloat16`), PyTorch `2.14.1+cu130`, Transformers `4.57.6`.
- **Tăng cường trực tuyến phi điểm mù:** `accent_aug = 0.30`, `surface_aug = 0.20`, `consistency_weight = 0.25`, `stratum_weight = true`, `real_rare_cap = 2.5`. Toàn bộ kiểu điểm mù C2 và tập C1 held-out được giữ hoàn toàn ngoài quá trình huấn luyện.
- **Chọn checkpoint & hiệu chuẩn:** Cùng công thức `real_aligned` (`0.50 * macro_f1 + 0.15 * that_macro_f1 + 0.25 * that_acc + 0.10 * evidence_f1`) trên `dev`, sau đó hiệu chuẩn nhiệt độ (`T`) và ngưỡng tách riêng cho trích xuất bằng chứng trên `cal` (`1.339` mẫu) rồi đóng băng vào `calibration.json`.

### Danh sách 4 ứng viên lưu trữ

| Thư mục | Họ kiến trúc | Backbone | Tham số / Số lớp | `max_tokens` | `lr` chính | Epoch chọn |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| [`dualorc_mmbert_base/`](dualorc_mmbert_base/) | `SafeVi-DualORC-CRF` (Schema v3) | `jhu-clsp/mmBERT-base` | 307M / 22 lớp | 512 | `4,5e-5` *(LLRD `0,956`)* | 5 / 5 |
| [`dualorc_bamibert/`](dualorc_bamibert/) | `SafeVi-DualORC-CRF` (Schema v3) | `Qualcomm-AI-Research/BamiBERT` | 102M / 12 lớp | 1024 | `5,0e-5` *(LLRD `0,920`)* | 5 / 5 |
| [`dualorc_visobert/`](dualorc_visobert/) | `SafeVi-DualORC-CRF` (Schema v3) | `uitnlp/visobert` | 98M / 12 lớp | 512 | `3,0e-5` *(LLRD `0,920`)* | 4 / 5 |
| [`guardmerge_v6/`](guardmerge_v6/) | `Hier-BidirLM-0.6B GuardMerge` (Schema v6) | `BidirLM-0.6B-Base` + `Qwen3Guard-Gen-0.6B` | ~600M *(LoRA r=16)* | 192/lượt | `1,0e-4` *(adapter)* | 4 / 4 |

Mỗi thư mục ứng viên chứa đầy đủ **13 tệp đầu ra nhẹ**: `config.json`, `environment.json`, `split_ids.json`, `history.json`, `calibration.json`, `dev_metrics.json`, `dev_predictions.jsonl`, `dev_stress_tests.json`, `dev_norm_stress_tests.json`, `dev_accent_stress.json`, `dev_context_checks.json`, `benchmark.json`, và `dev_diagnostics.png`. Số liệu tổng hợp máy đọc nằm tại [`dev_summary.json`](dev_summary.json).

---

## 2. Kết quả tổng hợp trên tập `dev` (`n = 1.340`) & Chi phí suy luận

### 2.1. Chỉ số tổng thể, bằng chứng, hiệu chuẩn và chi phí GPU

| Chỉ số trên `dev` (`n = 1.340`) | `dualorc_mmbert_base` | `dualorc_bamibert` | `dualorc_visobert` | `guardmerge_v6` |
| :--- | :---: | :---: | :---: | :---: |
| **Macro-F1 (5 lớp)** `[95% CI]` | 0,9229 `[0,9067–0,9378]` | 0,9224 `[0,9031–0,9397]` | **0,9289** `[0,9122–0,9431]` | 0,9003 `[0,8806–0,9178]` |
| Macro-F1 (4 lớp nguy hại chính) | 0,9423 | 0,9375 | **0,9460** | 0,9175 |
| Binary-F1 / Accuracy | 0,9269 / 0,9112 | 0,9329 / 0,9149 | **0,9352** / **0,9201** | 0,9118 / 0,8918 |
| `FPR(AN_TOAN)` *(càng thấp càng tốt)* | 8,22% | 8,56% | **6,04%** | 10,74% |
| F1 theo lớp (`AN_TOAN` / `XUC_PHAM`) | 0,9101 / 0,8453 | 0,9160 / **0,8618** | **0,9218** / 0,8605 | 0,8904 / 0,8311 |
| F1 theo lớp (`QUAY_ROI` / `DE_DOA` / `LUA_DAO`) | 0,9264 / 0,9384 / **0,9944** | 0,9140 / 0,9286 / 0,9915 | **0,9273** / **0,9435** / 0,9914 | 0,8727 / 0,9156 / 0,9915 |
| Recall `QUAY_ROI` / Precision `DE_DOA` / Recall `LUA_DAO` | **92,24%** / **97,16%** / **100,0%** | 87,07% / 91,08% / 99,43% | 87,93% / 94,67% / 98,30% | 82,76% / 89,81% / 99,43% |
| **Bằng chứng:** Precision / Recall | **0,6851** / 0,7271 | 0,6639 / 0,7515 | 0,6238 / **0,7696** | 0,6187 / 0,7390 |
| **Bằng chứng:** Word F1 / F1 theo ý | **0,7054** / 0,6864 | 0,7050 / **0,7164** | 0,6890 / 0,7063 | 0,6735 / 0,6773 |
| `safe_mark_rate` *(tô oan `AN_TOAN`)* | **5,03%** | 6,88% | 5,70% | 9,90% |
| `word_length_ratio` *(lý tưởng gần `1,00×`)* | **1,0613×** | 1,1320× | 1,2337× | 1,1945× |
| `harm_empty_evidence_rate` | 5,86% | 3,75% | **1,11%** | 2,29% |
| Ngưỡng bằng chứng đóng băng (`cal`) | `0,400` *(bridge `0,0`)* | `0,275` *(bridge `0,75`)* | `0,200` *(bridge `0,0`)* | `0,475` *(bridge `0,0`)* |
| Hiệu chuẩn `all ECE` (trước `T`) / `real ECE` (sau `T`) | **0,0141** / 0,0250 | 0,0167 / **0,0151** | 0,0296 / 0,0435 | 0,0530 / 0,0461 |
| **Công bằng vùng miền:** Worst-group / `gap` (`max−min`) | 0,9758 / **0,0166** | 0,9731 / 0,0192 | **0,9822** / 0,0178 | 0,9178 / 0,0652 |
| Trọng số GPU FP32 / Đỉnh VRAM suy luận | 1.177,0 / 1.218,2 MiB | 396,6 / 419,9 MiB | **376,0** / **399,3 MiB** | 1.177,4 / 1.205,2 MiB |
| Độ trễ L4 (`batch=1`, p50 / p95) | 37,96 / 41,41 ms | **18,65 / 21,48 ms** | 20,82 / 24,41 ms | 82,31 / 86,57 ms |

### 2.2. Kiểm định ghép cặp McNemar trên `dev` (`n = 1.340`)

| Cặp so sánh (`A` vs `B`) | Chỉ `A` đúng | Chỉ `B` đúng | `exact p-value` | Kết luận (`α = 0,05`) |
| :--- | :---: | :---: | :---: | :--- |
| `dualorc_visobert` vs `guardmerge_v6` | 88 | 50 | `0,00154` | **DualORC thắng có ý nghĩa thống kê** |
| `dualorc_bamibert` vs `guardmerge_v6` | 72 | 41 | `0,00455` | **DualORC thắng có ý nghĩa thống kê** |
| `dualorc_mmbert_base` vs `guardmerge_v6` | 75 | 49 | `0,02437` | **DualORC thắng có ý nghĩa thống kê** |
| `dualorc_mmbert_base` vs `dualorc_visobert` | 47 | 59 | `0,28528` | **Hòa thống kê** *(không khác biệt)* |
| `dualorc_bamibert` vs `dualorc_visobert` | 53 | 60 | `0,57266` | **Hòa thống kê** *(không khác biệt)* |
| `dualorc_mmbert_base` vs `dualorc_bamibert` | 46 | 51 | `0,68486` | **Hòa thống kê** *(không khác biệt)* |

---

## 3. Phân tích lát cắt chuyên sâu, ngữ cảnh hội thoại & Stress Test trên `dev`

### 3.1. Lát cắt trọng tâm của bài toán SafeViChi v2

| Lát cắt trên `dev` | `n` | `dualorc_mmbert_base` | `dualorc_bamibert` | `dualorc_visobert` | `guardmerge_v6` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Hội thoại nhiều lượt (`nhiều lượt`)** — Macro-F1 / Ev F1 | 587 | **0,9775** / **0,7218** | 0,9770 / 0,7122 | 0,9709 / 0,6933 | 0,9457 / 0,6813 |
| **Lợi ích đọc đủ ngữ cảnh (`full − last_turn`)** | 587 | **+25,24 pp** *(0,9775 vs 0,7252)* | +24,70 pp | +23,29 pp | +22,52 pp |
| **Kịch bản hội thoại 5 lớp (`sinh`)** — Macro-F1 / Bin-F1 | 671 | **0,9786** / **0,9904** | 0,9773 / 0,9894 | 0,9741 / 0,9850 | 0,9483 / 0,9725 |
| **Ngôn ngữ ẩn ý hội thoại (`do_lo:an_y`)** — Macro-F1 / FPR / Ev F1 | 309 | **0,9679** / **1,00%** / **0,7363** | 0,9562 / 3,00% / 0,7167 | 0,9609 / 3,00% / 0,6907 | 0,9305 / 4,00% / 0,6821 |
| **Ca khó tiếng lóng (`tieng_long`)** — Macro-F1 / FPR / `safe_mark` | 178 | **0,7765** / **10,62%** / **6,19%** | 0,5545 / 14,16% / 12,39% | 0,6272 / 11,50% / 11,50% | 0,7337 / 16,81% / 16,81% |
| **Ca khó ẩn ý câu đơn (`an_y`)** — Macro-F1 / Ev F1 | 50 | 0,4835 / 0,5651 | **0,4975** / **0,6257** | 0,4444 / 0,4635 | 0,4849 / 0,5419 |
| Bình luận đơn 1 lượt (`1 lượt`) — Macro-F1 | 753 | 0,8527 | 0,8435 | **0,8775** | 0,8419 |
| Câu thật ViHSD/ViHOS (`that`) — Macro-F1 / Acc | 669 | 0,6446 / 0,8430 | 0,6137 / 0,8535 | **0,6561** / **0,8670** | 0,6526 / 0,8341 |
| Vùng miền (`bac` / `trung` / `nam`) — Macro-F1 | 103/88/54 | 0,9850 / **0,9924** / 0,9758 | 0,9731 / **0,9924** / **0,9882** | **1,0000** / 0,9822 / **0,9882** | 0,9395 / 0,9830 / 0,9178 |

### 3.2. Kiểm thử chống né lọc trực tiếp trên `dev` (Raw, không qua bộ chuẩn hóa)

| Bộ stress test trên `dev` | Thấy khi train? | `dualorc_mmbert_base` | `dualorc_bamibert` | `dualorc_visobert` | `guardmerge_v6` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `accent_strip` (`n=1.216`) — F1 (`ΔF1`) | Có | 0,8812 (`-0,0400`) | 0,8716 (`-0,0493`) | **0,9105** (`-0,0160`) | 0,8402 (`-0,0589`) |
| `c1_in_dist` (`n=1.255`) — F1 (`ΔF1`) | Có | 0,8872 (`-0,0349`) | 0,8766 (`-0,0443`) | **0,9054** (`-0,0242`) | 0,8496 (`-0,0490`) |
| `c1_heldout_controlled` (`n=1.153`) — F1 (`ΔF1`) | **Không** | 0,9180 (`-0,0080`) | 0,9124 (`-0,0159`) | **0,9361** (`+0,0056`) | 0,8834 (`-0,0182`) |
| `blindspot_homoglyph` (`n=1.203`) — F1 (`ΔF1`) | **Không** | 0,9025 (`-0,0261`) | 0,9001 (`-0,0302`) | **0,9117** (`-0,0220`) | 0,8762 (`-0,0279`) |
| `blindspot_zero_width` (`n=1.330`) — F1 (`ΔF1`) | **Không** | 0,9046 (**`-0,0189`**) | 0,8979 (`-0,0251`) | **0,9095** (`-0,0201`) | 0,8777 (`-0,0231`) |

---

## 4. Phân tích chuyên sâu & Quyết định Chọn Mô hình (Pre-Test Protocol)

Quyết định chọn mô hình được chốt hoàn toàn dựa trên bằng chứng định lượng từ tập `dev`/`cal` và đặc tính kiến trúc–triển khai, tuân thủ nghiêm quy tắc **không nhìn điểm test để chọn mô hình**. Quy trình sàng lọc gồm 3 bước:

### Bước 1 — Chọn họ kiến trúc: `SafeVi-DualORC-CRF` thắng tuyệt đối `GuardMerge`
Cả 3 mô hình thuộc nhánh **`SafeVi-DualORC-CRF` (Schema v3)** đều vượt `guardmerge_v6` có ý nghĩa thống kê trên kiểm định McNemar (`p = 0,00154` đến `0,02437`), đồng thời nhanh hơn **2,2× – 4,4×** (`18,65–37,96 ms` so với `82,31 ms`), giảm mạnh tỷ lệ báo nhầm (`FPR 6,04%–8,56%` so với `10,74%`), giảm gần một nửa tỷ lệ tô oan (`safe_mark 5,03%–6,88%` so với `9,90%`) và thu hẹp độ lệch vùng miền hơn **3,4×** (`0,0166–0,0192` so với `0,0652`). Do đó, nhánh `GuardMerge` bị loại khỏi vai trò mô hình chính và kiến trúc được chốt về **`SafeVi-DualORC-CRF`**.

### Bước 2 — Kiểm toán cân bằng thống kê và rủi ro lệch miền giữa 3 backbone `DualORC-CRF`
Trên Macro-F1 tổng thể của `dev`, 3 backbone DualORC **hòa nhau về mặt thống kê** (`mmbert_base` vs `visobert`: `p = 0,28528`; `mmbert_base` vs `bamibert`: `p = 0,68486`; `bamibert` vs `visobert`: `p = 0,57266`; khoảng tin cậy 95% chồng lấp mạnh `[0,9067–0,9378]` vs `[0,9122–0,9431]`). Khi không tồn tại khác biệt có ý nghĩa thống kê ở chỉ số gộp, việc chọn máy móc theo `+0,60 pp` thô trên `dev` của `ViSoBERT` tiềm ẩn **rủi ro quá khớp miền (domain-specialization bias)** rất rõ:

1. **Cấu trúc của `dev` thiên về bình luận 1 dòng mạng xã hội:** Đúng `669/1.340` mẫu (`49,9%`) của `dev` là câu bình luận đơn từ ViHSD/ViHOS/ViTHSD (chủ yếu 2 lớp `AN_TOAN` `396` và `XUC_PHAM` `252`, không có `LUA_DAO`, chỉ `6` `QUAY_ROI` và `15` `DE_DOA`). Toàn bộ lợi thế nhỏ của `visobert` tập trung ở lát cắt `1 lượt` (`0,8775` vs `0,8527`) — vốn trùng khớp trực tiếp với kho tiền huấn luyện bình luận mạng xã hội Việt Nam của `uitnlp/visobert`. Ngược lại, trên các lát cắt cốt lõi của bài toán v2 (`nhiều lượt`, `do_lo:an_y`, `tieng_long`, `an_y`), `visobert` đều tụt sau `mmbert_base`.
2. **Dấu hiệu quá khớp sớm và bất ổn hiệu chuẩn ở `visobert`:** Trong `history.json`, `visobert` đạt đỉnh sớm ở epoch 4 rồi suy giảm ngay trên lát cắt câu thật ở epoch 5 (`that_macro_f1: 0,6561 → 0,6404`, `that_acc: 0,8670 → 0,8625`), trong khi `mmbert_base` hội tụ đơn điệu và ổn định đến epoch 5 (`select_score: 0,7446 → 0,7921 → 0,8151 → 0,8366 → 0,8392`). Đồng thời, xác suất của `visobert` trên câu thật kém hiệu chuẩn nhất trong 3 backbone DualORC (`real ECE = 0,0435` sau temperature scaling — gần gấp đôi `mmbert_base` `0,0250`), và bộ giải mã bằng chứng phải hạ ngưỡng rất thấp (`0,20`) khiến cụm tô sáng bị phình rộng (`word_length_ratio = 1,2337×`, Precision chỉ `0,6238`).

### Bước 3 — Ma trận quyết định đa tiêu chí theo 3 yêu cầu của bài toán SafeViChi v2

Khi phá vỡ thế hòa thống kê dựa trên đúng mục tiêu cốt lõi của **SafeViChi v2**, **`SafeVi-DualORC-CRF (mmBERT-base)`** vượt trội toàn diện trên 6 trục tiêu chí then chốt:

| Tiêu chí quyết định | Yêu cầu liên quan | Dẫn chứng định lượng trên `dev` của `mmBERT-base` |
| :--- | :--- | :--- |
| **1. Hiểu hội thoại đa lượt & ngữ cảnh thực** | Bài toán cốt lõi v2 (1–8 lượt) | **Hạng #1** trên `nhiều lượt` (`Macro-F1 = 0,9775`, `Binary-F1 = 0,9903`, `Ev F1 = 0,7218`), **#1** trên `sinh` 5 lớp (`0,9786`), và có biên tăng nhờ đọc đủ hội thoại cao nhất (**`+25,24 pp`** so với chỉ đọc lượt cuối — chứng minh suy luận liên lượt tốt nhất). |
| **2. Xử lý ca khó ẩn ý & giảm báo nhầm tiếng lóng** | Yêu cầu #1 & #2 | **Hạng #1 tuyệt đối** trên `do_lo:an_y` (`Macro-F1 = 0,9679`, `FPR = 1,00%` — thấp hơn **3×** so với `ViSoBERT`/`BamiBERT`, `safe_mark = 1,00%`, `Ev F1 = 0,7363`) và **#1 bỏ xa** trên `tieng_long` (`Macro-F1 = 0,7765`, hơn `ViSoBERT` **`+14,93 pp`** và hơn `BamiBERT` **`+22,20 pp`**, đồng thời `FPR = 10,62%` và `safe_mark = 6,19%` thấp nhất cả 4 mô hình). Trên `an_y` câu đơn, `Ev F1 = 0,5651` vượt `ViSoBERT` `+10,16 pp`. |
| **3. Chất lượng bằng chứng & kỷ luật giải thích** | Yêu cầu #1 | **Hạng #1** về Word-level Evidence F1 (`0,7054`), **#1** Evidence Precision (`0,6851`), **#1** tỷ lệ tô oan `AN_TOAN` thấp nhất (`safe_mark = 5,03%`), và tỷ lệ độ dài cụm tô sát ground-truth nhất (`1,0613×`, so với `1,1320×` ở `BamiBERT` và `1,2337×` ở `ViSoBERT`). Ngưỡng hiệu chuẩn sạch và tách bạch (`marginal_threshold = 0,40`, `bridge_ratio = 0,0`, không cần fallback cưỡng bức). |
| **4. Cân bằng nhãn rủi ro cao & công bằng vùng miền** | Yêu cầu #2 | **Hạng #1** về độ chênh lệch giữa 3 vùng miền (`region_gap_max_min = 0,0166` — thấp nhất cả 4 mô hình), **#1** trên `LUA_DAO` (`F1 = 0,9944`, Recall `100%`), **#1** Recall `QUAY_ROI` (`92,24%` tổng thể; `33,33%` trên câu thật `that` — gấp đôi `ViSoBERT`/`BamiBERT`), và **#1** Precision `DE_DOA` (`97,16%`). Hiệu chuẩn xác suất tốt (`all ECE` gốc `0,0141` #1; `real ECE` sau `T` `0,0250`). |
| **5. Năng lực khái quát hóa & cấu trúc đa ngôn ngữ/Unicode** | Yêu cầu #2 | Vượt `BamiBERT` và `GuardMerge` trên **cả 5 bộ stress test** của `dev`, chịu suy giảm nhỏ nhất ở `blindspot_zero_width` (`Δ = -0,0189`) và giữ vững `c1_heldout_controlled` (`Δ = -0,0080`). Kiến trúc **ModernBERT** (`mmBERT-base`: 307M, 22 lớp, RoPE, GeGLU, từ vựng Gemma-2 256k phủ >1.800 ngôn ngữ & bảng mã Unicode) đem lại dung lượng biểu diễn gấp ~3× các encoder 98–102M và khả năng tổng quát tốt trước kịch bản mới cũng như kỹ thuật né lọc ký tự ngoại lai (Cyrillic homoglyph, fullwidth, zero-width). |
| **6. Khả thi triển khai cục bộ `< 2 GB`** | Yêu cầu #3 | Trọng số FP32 `1.177,0 MiB` (`1,15 GiB`), khi lượng tử hóa INT8 (`PerRowInt8Embedding` + `qint8`) còn khoảng **`~320–350 MiB`** — dư địa rất rộng so với trần `< 2 GB`; độ trễ L4 `37,96 ms` p50 (nhanh hơn `GuardMerge` **2,2×**). |

---

### Kết luận chốt mô hình

> **Mô hình chính thức được chọn cho Phần 2 (`../official_eval/`):** **`SafeVi-DualORC-CRF (mmBERT-base)`** (`dualorc_mmbert_base/`, checkpoint epoch 5, kèm cấu hình hiệu chuẩn đóng băng `calibration.json`: `T = 1,1807`, `marginal_threshold = 0,40`, `single_word_threshold = 0,58`).

**Ghi nhận vai trò lưu trữ của 3 ứng viên còn lại trong `dev_selection/`:**
- **`dualorc_bamibert/`** (`102M`, `396,6 MiB`, `18,65 ms` p50, `real ECE = 0,0151`, `f1_theo_y = 0,7164`): Ứng viên siêu nhẹ sáng giá nhất về tốc độ/hiệu chuẩn, tuy nhiên thua `mmbert_base` rõ ở `tieng_long` (`-22,20 pp`), `that_macro_f1` (`-3,09 pp`), `do_lo:an_y`, `safe_mark` và cả 5 bài stress test; được giữ làm hồ sơ tham chiếu kỹ thuật cho kịch bản thiết bị cực yếu.
- **`dualorc_visobert/`** (`98M`, `376,0 MiB`): Mốc đối chiếu encoder mạng xã hội tiếng Việt; không chọn làm mô hình chính do thiên lệch miền câu bình luận 1 dòng, độ tin cậy hiệu chuẩn trên câu thật kém hơn (`real ECE = 0,0435`), và xu hướng tô sáng tràn (`word_length_ratio = 1,2337×`).
- **`guardmerge_v6/`** (`0.6B LoRA`): Mốc đối chiếu kiến trúc LLM hai chiều; bị loại trên `dev` do thua có ý nghĩa thống kê trước cả 3 biến thể DualORC-CRF và chậm hơn >2×.

Hồ sơ đầy đủ và kết quả đánh giá chính thức trên bộ test khóa của mô hình chốt xem tại [`../official_eval/README.md`](../official_eval/README.md).
