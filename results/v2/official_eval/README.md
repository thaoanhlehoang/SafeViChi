# Phần 2 — Mô hình Chính thức (`SafeVi-DualORC-CRF / mmBERT-base`) & Đánh giá Bộ Test Khóa

Thư mục này tập trung toàn bộ hồ sơ của **mô hình chính thức đã chốt** từ giai đoạn lựa chọn trên `dev` ([`../dev_selection/`](../dev_selection/)) — **`SafeVi-DualORC-CRF (mmBERT-base)`** — bao gồm cấu hình, tham số hiệu chuẩn đóng băng, kết quả `dev` đối chiếu và kết quả đánh giá chính thức trên bộ test người viết đã khóa (`1.670` mẫu: `334 C0 + 668 C1 + 668 C2`).

---

## 1. Hồ sơ mô hình chốt & Cấu hình đóng băng

| Thuộc tính | Giá trị đóng băng |
| :--- | :--- |
| **Tên cấu hình** | `dualorc_mmbert_base` (`SafeVi-DualORC-CRF`, Schema v3) |
| **Backbone** | `jhu-clsp/mmBERT-base` (rev `c5955035435e2bf121cde7f3c8863ef52ff35d82`, 307M, 22 lớp, ModernBERT) |
| **Checkpoint & Hạt giống** | Epoch 5 / 5 (`best_select_score = 0,839173`), `seed = 42`, `dtype = torch.bfloat16` |
| **Dữ liệu khóa** | Commit `546300b7d5c3f7a8c8b78165402182d4aed23b7e` (`0` rò rỉ với train, xác minh tại [`../kiem_tra_ro_ri/bao_cao.json`](../kiem_tra_ro_ri/bao_cao.json)) |
| **Hiệu chuẩn đóng băng (`calibration.json`)** | `T = 1,1807`, `chosen_decoder = marginal`, `marginal_threshold = 0,40`, `single_word_threshold = 0,58`, `o_bias = -0,15`, `bridge_ratio = 0,0` |
| **Chi phí suy luận GPU (L4, `batch=1`)** | Trọng số FP32 `1.177,0 MiB` (đỉnh VRAM `1.218,2 MiB`), độ trễ `p50 = 37,96 ms` / `p95 = 41,41 ms` |

> **Ghi chú đồng bộ tệp:** Bộ 13 tệp cấu hình, hiệu chuẩn đóng băng, benchmark GPU và kết quả `dev` của mô hình chốt đã được lưu đầy đủ tại [`dualorc_mmbert_base/`](dualorc_mmbert_base/). Các tệp xuất thô của phiên chấm test khóa trong [`dualorc_mmbert_base/test/`](dualorc_mmbert_base/test/) (`REAL_TEST_LOCK.json`, `real_test_official_matrix.json`, `real_test_*metrics.json`, `real_test_*predictions.jsonl`, `real_test_diagnostics.png`) hiện chưa được đồng bộ vào kho lưu trữ và sẽ được bổ sung sau khi hoàn tất xuất gói tệp từ môi trường chạy.

---

## 2. Ma trận chính thức trên Bộ Test Khóa: `C0 / C1 / C2` (Dòng `#5` thô)

Mô hình sử dụng nguyên vẹn checkpoint và `calibration.json` đã đóng băng (không hiệu chỉnh lại bất kỳ siêu tham số hay ngưỡng nào sau khi chấm test) trên toàn bộ 3 tập test người viết:
- **`C0` (`n = 334`):** Hội thoại sạch 1–8 lượt do người tự viết (`96 AN_TOAN`, `72 LUA_DAO`, `62 DE_DOA`, `53 XUC_PHAM`, `51 QUAY_ROI`), cân bằng 3 giọng vùng (`112` Bắc, `111` Trung, `111` Nam), gồm `185` mẫu trong phân phối và `149` mẫu kịch bản chưa thấy.
- **`C1` (`n = 668`):** `334 × 2` bản biến dạng né lọc có kiểm soát (`bo_dau`, `chen_ky_tu`, `dinh_tu`, `go_sai`, `keo_dai`, `so_thay_chu`, `teencode`; biến dạng ~60% số từ, luôn trúng cụm bằng chứng).
- **`C2` (`n = 668`):** `334 × 2` bản biến dạng điểm mù (`cyrillic_homoglyph`, `double_elongation`, `exotic_separator`, `fullwidth`, `space_injection`, `zero_width`) — **100% kiểu né lọc chưa từng xuất hiện khi huấn luyện**.

### 2.1. Bảng chỉ số chính thức (Dòng `#5` — Mô hình thô)

| Chỉ số đánh giá | `C0` — Sạch (`n = 334`) | `C1` — Né lọc (`n = 668`) | `C2` — Điểm mù (`n = 668`) |
| :--- | :---: | :---: | :---: |
| **Macro-F1 (5 lớp)** | **0,8762** | **0,7914** | **0,7697** |
| Bootstrap 95% CI (Macro-F1) | `[0,8378 – 0,9103]` | `[0,7501 – 0,8265]` | `[0,7282 – 0,8039]` |
| Macro-F1 (4 lớp nguy hại chính) | 0,8599 | 0,7693 | 0,7542 |
| **Binary-F1 (Nguy hại vs `AN_TOAN`)** | **0,9417** | **0,9137** | **0,9148** |
| Accuracy | 0,8772 | 0,8039 | 0,7799 |
| `FPR(AN_TOAN)` *(càng thấp càng tốt)* | **7,29%** | 10,94% | 14,06% |
| **Evidence F1 (mức từ)** | **0,7717** | 0,6915 | 0,6678 |
| **Evidence F1 theo ý** | **0,7309** | 0,6216 | 0,6034 |
| `safe_mark_rate` *(tô oan `AN_TOAN`)* | **7,29%** | 7,29% | 8,85% |
| `word_length_ratio` | 0,8423× | 0,6895× | 0,6813× |
| `harm_empty_evidence_rate` | **2,67%** | 10,48% | 12,64% |
| **Worst-group Region Macro-F1** | **0,8690** | 0,7758 | 0,7317 |
| **Region Gap (`max − min`)** | **0,0112** | 0,0230 | 0,0659 |

### 2.2. Độ suy giảm dưới tấn công né lọc (`Δ` so với `C0`) & Đối chiếu `dev → test C0`

| Phép so sánh | Δ Macro-F1 (5 lớp) | Δ Binary-F1 | Δ Evidence F1 theo ý | Nhận xét kỹ thuật |
| :--- | :---: | :---: | :---: | :--- |
| **`C1 − C0`** *(né lọc chuẩn)* | `-0,0848` | `-0,0280` | `-0,1093` | Biến dạng mạnh ~60% từ làm giảm phân tách mịn giữa 4 nhãn nguy hại, nhưng phát hiện nguy hại nhị phân vẫn đạt `0,9137`. |
| **`C2 − C0`** *(100% điểm mù chưa thấy)* | `-0,1065` | **`-0,0269`** | `-0,1275` | Dưới 6 kiểu điểm mù chưa từng thấy khi train, **Binary-F1 chỉ tụt `-2,69 pp` (`0,9417 → 0,9148`)**, khẳng định sức chống chịu mạnh của biểu diễn đa ngôn ngữ `mmBERT-base` + `Byte-CNN`. |
| **`Test C0` vs `Dev`** *(chuyển miền sang người viết)* | `-0,0467` *(0,9229 → 0,8762)* | **`+0,0148`** *(0,9269 → 0,9417)* | **`+0,0445`** *(0,6864 → 0,7309)* | Chuyển sang hội thoại người viết hoàn toàn mới: **Binary-F1 tăng `+1,48 pp`**, **`FPR` giảm `-0,93 pp`** (`8,22% → 7,29%`), **Evidence Word F1 tăng `+6,63 pp`** (`0,7054 → 0,7717`), `empty_harm` giảm mạnh (`5,86% → 2,67%`) và `region_gap` thu hẹp (`0,0166 → 0,0112`). |

---

## 3. Phân tích lợi ích bộ chuẩn hóa: Dòng `#6` (`#5` + Chuẩn hóa) vs Dòng `#5` (Thô)

### 3.1. Tổng hợp hiệu ứng chuẩn hóa (`#6 − #5`)

| Tập / Chỉ số | Dòng `#5` (Thô) | Dòng `#6` (+ Chuẩn hóa) | Lợi ích (`#6 − #5`) |
| :--- | :---: | :---: | :---: |
| `C0` — Macro-F1 (5 lớp) | **0,8762** | 0,8644 | `-0,0118` |
| **`C1` — Macro-F1 (5 lớp)** | 0,7914 | **0,8239** | **`+0,0325`** |
| **`C1` — Evidence F1 theo ý** | 0,6216 | **0,6879** | **`+0,0663`** |
| `C2` — Macro-F1 (5 lớp) | **0,7697** | 0,7554 | `-0,0143` |
| `C2` — Evidence F1 theo ý | **0,6034** | 0,5798 | `-0,0236` |
| `strictly_unseen` *(kiểu né lọc chưa thấy)* | — | — | `-0,0136` |

### 3.2. Chi tiết hiệu ứng chuẩn hóa theo từng kiểu biến dạng (`by_kieu`, 13 kiểu)

| Nhóm | Kiểu biến dạng | `Δ Macro-F1 (#6 − #5)` | Đánh giá |
| :---: | :--- | :---: | :--- |
| **C1** | `chen_ky_tu` | **`+0,1885`** | Cải thiện đột biến (`+18,85 pp`) — chuẩn hóa khôi phục chính xác từ bị chèn ký tự xen kẽ. |
| **C1** | `keo_dai` | **`+0,0669`** | Rút gọn ký tự kéo dài giúp khôi phục biên từ và bằng chứng rõ rệt. |
| **C1** | `teencode` | **`+0,0374`** | Từ điển teencode giúp chuẩn hóa cách viết tắt tự nhiên. |
| **C1** | `so_thay_chu` | **`+0,0315`** | Phục hồi hiệu quả các biến thể leetspeak thay chữ bằng số. |
| **C1** | `go_sai` | `+0,0032` | Cải thiện nhẹ, gần trung tính. |
| **C1** | `dinh_tu` | `-0,0100` | Giảm nhẹ do tác động biên tách từ. |
| **C1** | `bo_dau` | `-0,0360` | Chuẩn hóa có thể sửa sai ngữ nghĩa tiếng Việt không dấu; mô hình thô đã được tăng cường `accent_aug = 0,30` nên tự xử lý tốt hơn. |
| **C2** | `exotic_separator` | **`+0,0249`** | Kiểu C2 duy nhất hưởng lợi rõ từ bộ chuẩn hóa nhờ loại bỏ ký tự phân cách lạ. |
| **C2** | `zero_width` | `-0,0122` | Mô hình thô tự xử lý tốt ký tự ẩn; chuẩn hóa toàn cục gây nhiễu nhẹ. |
| **C2** | `fullwidth` | `-0,0204` | Giảm nhẹ so với bản thô. |
| **C2** | `double_elongation` | `-0,0260` | Giảm nhẹ khi chuẩn hóa toàn cục tác động lên chuỗi lặp kép. |
| **C2** | `space_injection` | `-0,0266` | Giảm nhẹ do khó ghép lại đúng ranh giới âm tiết khi khoảng trắng bị chèn rời rạc. |
| **C2** | `cyrillic_homoglyph` | `-0,0290` | Tokenizer đa ngôn ngữ 256k + `Byte-CNN` của `mmBERT-base` tự xử lý homoglyph ổn định hơn bộ quy tắc cứng. |

**Chính sách triển khai khuyến nghị:** Giữ báo cáo song song cả Dòng `#5` (thô) và Dòng `#6` (+ chuẩn hóa) để định lượng rõ lợi ích trên C1 (`+3,25 pp` Macro-F1, `+6,63 pp` Evidence F1 theo ý, đặc biệt `+18,85 pp` ở `chen_ky_tu`). Khi triển khai ứng dụng thực tế/web demo, ưu tiên cấu hình **Dòng `#5` (thô)** để bảo toàn tối đa độ chính xác trên câu sạch (`C0 = 0,8762`) và điểm mù `C2` (`0,7697`), hoặc chỉ bật **chuẩn hóa chọn lọc** cho nhóm `chen_ky_tu` / `keo_dai` / `so_thay_chu` / `teencode` / `exotic_separator`.

---

## 4. Phân tích chi tiết theo lát cắt trên tập test sạch `C0` (`n = 334`)

| Nhóm lát cắt | Lát cắt | `n` | `macro_present` | `binary_f1` | `accuracy` |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Tổng thể** | `that` / `bien_dang:C0` | 334 | **0,8762** | **0,9417** | **0,8772** |
| **Độ dài hội thoại** | `nhiều lượt` *(2–8 lượt)* | 226 | **0,8991** | **0,9571** | **0,8982** |
| | `1 lượt` | 108 | 0,8252 | 0,9051 | 0,8333 |
| **Khái quát kịch bản** | `phan:trong_phan_phoi` | 185 | **0,9296** | **0,9813** | **0,9405** |
| | `phan:chua_thay` *(LD7, LD12, DD6, QR6, mới)* | 149 | 0,6438 | **0,8878** | 0,7987 |
| **Giọng vùng miền** | `vung:bac` | 112 | 0,8690 | 0,9412 | 0,8750 |
| *(gap = `0,0112`)* | `vung:trung` | 111 | 0,8740 | **0,9673** | **0,8919** |
| | `vung:nam` | 111 | **0,8802** | 0,9172 | 0,8649 |
| **Độ lộ liễu** | `do_lo:lo_lieu` | 134 | **0,9094** | **0,9653** | **0,8881** |
| | `do_lo:an_y` | 104 | 0,8305 | **0,9442** | 0,8173 |
| **Kênh giao tiếp** | `kenh:messenger_zalo` | 127 | **0,9416** | **0,9749** | **0,9449** |
| | `kenh:nhom_lop` | 63 | 0,8708 | 0,9412 | 0,8571 |
| | `kenh:binh_luan` | 63 | 0,8630 | 0,9383 | 0,8730 |
| | `kenh:chat_game` | 35 | 0,8164 | 0,9048 | 0,8000 |
| | `kenh:sms` | 46 | 0,7563 | 0,8571 | 0,7826 |
| **Nguồn gán nhãn** | `cach_gan:nguoi` | 325 | **0,8954** | **0,9441** | **0,8923** |
| | `cach_gan:llm_nguoi_duyet` *(ca ranh giới nhỏ)* | 9 | 0,2727 | 0,8750 | 0,3333 |

### Điểm nhấn phân tích lát cắt
1. **Hội thoại đa lượt & kênh nhắn tin chủ lực rất mạnh:** Trên `nhiều lượt` (`n = 226`), mô hình đạt `Macro-F1 = 0,8991`, `Binary-F1 = 0,9571`; trên `kenh:messenger_zalo` (`n = 127`) đạt `Macro-F1 = 0,9416`, `Binary-F1 = 0,9749` — phản ánh đúng hiệu quả của việc chọn `mmBERT-base` nhờ lợi thế suy luận liên lượt trên `dev`.
2. **Công bằng vùng miền xuất sắc:** Cả 3 miền Bắc (`0,8690`), Trung (`0,8740`) và Nam (`0,8802`) đều đồng đều cao với khoảng chênh cực nhỏ `region_gap_max_min = 0,0112` (thậm chí chặt hơn mức `0,0166` trên `dev`).
3. **Ngôn ngữ ẩn ý (`do_lo:an_y`, `n = 104`):** Giữ `Macro-F1 = 0,8305` và `Binary-F1 = 0,9442` trên hội thoại ẩn ý do người viết.
4. **Kịch bản chưa thấy (`phan:chua_thay`, `n = 149`) & ca ranh giới:** `Binary-F1` vẫn đạt **`0,8878`** và `Accuracy = 79,87%`. Lưu ý `macro_present = 0,6438` chịu độ nhạy lớn do lớp `XUC_PHAM` trong `phan:chua_thay` chỉ có đúng `n = 3` mẫu (xem `data/v2/test_stats.json`). Tương tự, lát cắt `cach_gan:llm_nguoi_duyet` chỉ có `n = 9` ca ranh giới (chỉ số `Binary-F1` vẫn giữ `0,8750`), còn `325` mẫu `cach_gan:nguoi` đạt `Macro-F1 = 0,8954`. Ở kênh `sms` (`n = 46`, `0,7563`), tin lừa đảo mẫu mới ngắn gọn ít ngữ cảnh hội thoại là hướng phân tích lỗi trọng tâm cho báo cáo.

---

## 5. Cấu trúc thư mục & Danh mục tệp

```text
results/v2/official_eval/
├── README.md                              # Tệp này — báo cáo chính thức trên bộ test khóa (1.670 mẫu)
└── dualorc_mmbert_base/                   # Hồ sơ đầy đủ của mô hình chốt SafeVi-DualORC-CRF (mmBERT-base)
    ├── config.json                        # Cấu hình huấn luyện & kiến trúc đóng băng
    ├── environment.json                   # Phần cứng, phiên bản PyTorch/Transformers/CUDA
    ├── calibration.json                   # Nhiệt độ T & ngưỡng bằng chứng đóng băng từ tập cal
    ├── benchmark.json                     # Đo VRAM & độ trễ p50/p95 trên GPU L4
    ├── history.json                       # Lịch sử 5 epoch huấn luyện
    ├── split_ids.json                     # Danh sách ID phân chia fit/dev/cal (seed 42)
    ├── dev_metrics.json                   # Số liệu đối chiếu trên dev (n=1.340)
    ├── dev_predictions.jsonl              # Dự đoán chi tiết trên dev
    ├── dev_stress_tests.json              # Stress test thô trên dev
    ├── dev_norm_stress_tests.json         # Stress test + chuẩn hóa trên dev
    ├── dev_accent_stress.json             # Kiểm tra bỏ dấu trên dev
    ├── dev_context_checks.json            # Kiểm chứng ngữ cảnh đa lượt trên dev
    ├── dev_diagnostics.png                # Biểu đồ chẩn đoán trên dev
    └── test/                              # Nơi lưu bộ tệp xuất thô của phiên chấm test khóa (1.670 mẫu)
        └── .gitkeep                       # Đang chờ đồng bộ tệp xuất thô (xem bảng dưới)
```

### Trạng thái đồng bộ tệp trong `dualorc_mmbert_base/`

| Nhóm tệp | Vị trí | Trạng thái | Danh sách tệp chính |
| :--- | :--- | :---: | :--- |
| **Hồ sơ mô hình, hiệu chuẩn & `dev`** | [`dualorc_mmbert_base/`](dualorc_mmbert_base/) | **Đã lưu (13 tệp)** | `config.json`, `environment.json`, `calibration.json`, `benchmark.json`, `history.json`, `split_ids.json`, `dev_*.json`, `dev_predictions.jsonl`, `dev_diagnostics.png` |
| **Tệp xuất thô phiên chấm test khóa** | [`dualorc_mmbert_base/test/`](dualorc_mmbert_base/test/) | **Chưa lưu tệp thô** | `REAL_TEST_LOCK.json`, `real_test_official_matrix.json`, `real_test_norm_official_matrix.json`, `real_test_metrics.json`, `real_test_c1_metrics.json`, `real_test_c2_metrics.json`, `real_test_all_predictions.jsonl`, `real_test_norm_all_predictions.jsonl`, `real_test_errors.jsonl`, `real_test_diagnostics.png` |
| **Benchmark lượng tử hóa & CPU đích** | [`dualorc_mmbert_base/test/`](dualorc_mmbert_base/test/) | **Chưa lưu tệp thô** | `int8_comparison.json`, `bench_cpu.json` |

### Lệnh tái kiểm chứng bằng CLI chuẩn

Sau khi đồng bộ tệp `real_test_all_predictions.jsonl` vào `dualorc_mmbert_base/test/`, có thể tái kiểm chứng toàn bộ chỉ số C0/C1/C2 và 15 lát cắt vùng miền/kịch bản bằng một lệnh duy nhất:

```bash
python -m src.v2.eval.evaluate \
  --gold data/v2/test_day_du.jsonl data/v2/test_perturb_C1.jsonl data/v2/test_perturb_C2.jsonl \
  --pred results/v2/official_eval/dualorc_mmbert_base/test/real_test_all_predictions.jsonl \
  --meta data/v2/test_day_du.jsonl \
  --out results/v2/official_eval/dualorc_mmbert_base/test/cli_eval_report.json
```

> **Quy tắc lưu trữ:** Không commit tệp trọng số nặng (`.safetensors`, `.pt`, `.onnx`, `.bin`) vào `results/v2/`; toàn bộ checkpoint lưu cục bộ trong `models/` (đã loại trừ trong `.gitignore`).
