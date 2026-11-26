# Đánh giá mô hình SafeViChi v2 trên bộ test khóa

> **Phạm vi:** Đánh giá mô hình **SafeViChi v2** (`SafeVi-DualORC-CRF / mmBERT-base`) và so sánh trực tiếp với **SafeViChi v1** và **LLM lớn qua API** (`openai/gpt-oss-120b`) trên cùng bộ test khóa `1.670` mẫu (`C0 / C1 / C2`).
>
> **Nguồn số liệu:** Toàn bộ chỉ số trích xuất từ `results/v2/official_eval/dualorc_mmbert_base/test/`, `results/v2/v1_baseline/` và `results/v2/llm_api/` (khóa dữ liệu tại [REAL_TEST_LOCK.json](../../results/v2/official_eval/dualorc_mmbert_base/test/REAL_TEST_LOCK.json)). Quy trình chọn mô hình trước test xem tại [dev_selection/README.md](../../results/v2/dev_selection/README.md); chi tiết dữ liệu xem tại [DATASET_CARD.md](../../data/v2/DATASET_CARD.md).

---

## 1. Thiết lập đánh giá & 3 hệ thống đối chiếu

### 1.1. Bộ test khóa (`1.670` mẫu, `0` rò rỉ với train)

Bộ test gồm `334` đoạn hội thoại 1–8 lượt do người viết/duyệt (`0` trùng/gần trùng với train theo [bao_cao.json](../../results/v2/kiem_tra_ro_ri/bao_cao.json)), cân bằng 3 giọng vùng (`112` Bắc, `111` Trung, `111` Nam), gồm `185` mẫu trong phân phối và `149` mẫu thuộc kịch bản chưa thấy (`DD6`, `LD7`, `LD12`, `QR6`, `moi`). Từ `334` đoạn gốc sinh ra 3 tầng đánh giá ghép cặp:

- **`C0` — Sạch (`n = 334`):** `96 AN_TOAN`, `72 LUA_DAO`, `62 DE_DOA`, `53 XUC_PHAM`, `51 QUAY_ROI`.
- **`C1` — Né lọc (`n = 668`):** `334 × 2` bản biến dạng ngẫu nhiên theo 7 kiểu (`bo_dau`, `chen_ky_tu`, `dinh_tu`, `go_sai`, `keo_dai`, `so_thay_chu`, `teencode`), biến dạng ~60% số từ và luôn trúng cụm bằng chứng.
- **`C2` — Điểm mù (`n = 668`):** `334 × 2` bản biến dạng theo 6 kiểu (`cyrillic_homoglyph`, `double_elongation`, `exotic_separator`, `fullwidth`, `space_injection`, `zero_width`) — **100% kiểu né lọc chưa từng xuất hiện khi huấn luyện**.

### 1.2. Ba hệ thống đặt lên bàn cân

| Hệ thống | Mô hình & thiết lập | Đầu ra | Cách đối chiếu công bằng |
| :--- | :--- | :--- | :--- |
| **SafeViChi v1** | ViHateT5 fine-tune đối kháng (FGM) + chuẩn hóa v1 (`v1-vongloai`) | Nhị phân `HATE / CLEAN` *(không có JSON bằng chứng v2)* | Quy chiếu qua **Binary-F1** (`HATE` = 4 nhãn nguy hại, `CLEAN` = `AN_TOAN`) kèm tỷ lệ gắn cờ `HATE` theo từng lớp v2 |
| **SafeViChi v2** | `SafeVi-DualORC-CRF` (`mmBERT-base`, 307M), suy luận trực tiếp trên văn bản gốc; hiệu chuẩn `T` & ngưỡng bằng chứng đóng băng trên `cal` | 5 nhãn + `bang_chung` nguyên văn (`TurnAwareBIOCRF`) | Chấm trọn bộ chỉ số 5 lớp, nhị phân và bằng chứng (`f1_theo_y` hệ số phủ `1,5` + `f1_muc_tu`) |
| **LLM API** | `openai/gpt-oss-120b` (OpenRouter / `venice/int4`, `temp = 0`, `reasoning = low`, độc lập với LLM tạo train) | 5 nhãn + `bang_chung` JSON *(prompt đầy đủ hướng dẫn + ví dụ ca khó, không tiền xử lý ngoài)* | Chấm bằng cùng bộ đo phân loại và bằng chứng, đối chiếu trực tiếp với SafeViChi v2 |

---

## 2. So sánh hiệu năng phân loại: v1 vs v2 vs LLM

### 2.1. Tổng hợp trên `C0 / C1 / C2`

| Hệ thống | `C0` — Sạch (`n = 334`) | `C1` — Né lọc (`n = 668`) | `C2` — Điểm mù (`n = 668`) |
| :--- | :---: | :---: | :---: |
| **Macro-F1 (5 lớp)** `[Bootstrap 95% CI]` | | | |
| LLM API (`gpt-oss-120b`) | `0,7723` `[0,7242–0,8157]` | `0,5854` `[0,5441–0,6207]` | `0,6272` `[0,5891–0,6648]` |
| **SafeViChi v2** | **`0,8647`** `[0,8255–0,8990]` | **`0,7859`** `[0,7532–0,8176]` | **`0,7708`** `[0,7399–0,8018]` |
| **Binary-F1 (Nguy hại vs `AN_TOAN`)** | | | |
| SafeViChi v1 | `0,3121` | `0,3104` | `0,3947`\* |
| LLM API (`gpt-oss-120b`) | `0,8920` | `0,7449` | `0,7793` |
| **SafeViChi v2** | **`0,9287`** | **`0,9029`** | **`0,9087`** |
| **Tỷ lệ báo nhầm `FPR(AN_TOAN)` / Bỏ lọt `FNR(Nguy hại)`** *(càng thấp càng tốt)* | | | |
| SafeViChi v1 | `0,00%` / `81,51%` | `1,56%` / `81,51%` | `4,17%` / `75,00%` |
| LLM API (`gpt-oss-120b`) | **`3,12%`** / `18,49%` | `10,94%` / `38,03%` | **`4,17%`** / `35,08%` |
| **SafeViChi v2** | `10,42%` / **`9,66%`** | `10,94%` / **`14,08%`** | `10,42%` / **`13,24%`** |

*\*Lưu ý về SafeViChi v1 trên `C2`: Binary-F1 tăng từ `0,3121` lên `0,3947` không phải do chống điểm mù tốt hơn, mà vì biến dạng `space_injection` làm vỡ từ khiến ViHateT5 báo động bừa (`FPR` nhảy lên `21,88%` trên `space_injection`), vô tình nâng recall trên tập test vốn có `71,3%` mẫu nguy hại.*

**Kiểm định ghép cặp McNemar (`α = 0,05`):**
- **SafeViChi v2 vs LLM API (5 lớp):** SafeViChi v2 thắng có ý nghĩa thống kê trên **cả 3 tầng** — `C0`: `50` vs `27` (`p = 0,0117`), `C1`: `154` vs `48` (`p < 10⁻⁶`), `C2`: `124` vs `51` (`p < 10⁻⁶`); gộp `1.670` mẫu: `328` vs `126` (`p < 10⁻⁶`).
- **SafeViChi v2 vs LLM API (nhị phân):** Hòa thống kê ở `C0` (`38` vs `24`, `p = 0,0980`), nhưng SafeViChi v2 thắng áp đảo trên `C1` (`165` vs `51`, `p < 10⁻⁶`), `C2` (`138` vs `46`, `p < 10⁻⁶`) và gộp `1.670` mẫu (`341` vs `121`, `p < 10⁻⁶`).
- **SafeViChi v2 vs SafeViChi v1 (nhị phân):** SafeViChi v2 thắng áp đảo trên cả 3 tầng (`p < 10⁻⁶`; gộp `1.670` mẫu: `804` vs `58`).

### 2.2. Phân rã hiệu năng theo 5 lớp trên `C0` và `C1`

| Nhãn (`support C0`) | SafeViChi v1 — Tỷ lệ gắn `HATE` (`C0` / `C1`) | LLM 120B — F1 / Recall (`C0`) | SafeViChi v2 — F1 / Recall (`C0`) | LLM 120B — F1 / Recall (`C1`) | SafeViChi v2 — F1 / Recall (`C1`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `AN_TOAN` (`96`) | `0,00%` / `1,56%` *(FPR)* | `0,7983` / **`96,88%`** | **`0,8390`** / `89,58%` | `0,6287` / `89,06%` | **`0,7953`** / `89,06%` |
| `XUC_PHAM` (`53`) | `73,58%` / `67,92%` | `0,8673` / **`92,45%`** | **`0,9412`** / `90,57%` | `0,7551` / `69,81%` | **`0,8858`** / **`91,51%`** |
| `QUAY_ROI` (`51`) | `0,00%` / `1,96%` | `0,5974` / `45,10%` | **`0,7826`** / **`70,59%`** | `0,3433` / `22,55%` | **`0,6341`** / **`50,98%`** |
| `DE_DOA` (`62`) | `8,06%` / `10,48%` | `0,6408` / `53,23%` | **`0,8095`** / **`82,26%`** | `0,3125` / `20,16%` | **`0,6967`** / **`68,55%`** |
| `LUA_DAO` (`72`) | `0,00%` / `0,69%` | **`0,9577`** / `94,44%` | `0,9510` / `94,44%` | `0,8874` / **`93,06%`** | **`0,9176`** / `88,89%` |

**Nhận xét chính:**
1. **Bước nhảy SafeViChi v2 vs v1:** Trên lớp trùng miền `XUC_PHAM`, v1 vẫn bắt được `73,58%` (`85,71%` nếu chỉ tính xúc phạm lộ liễu), nhưng **mù gần như hoàn toàn** trước 3 nhóm nguy hại mới không chứa từ tục (`LUA_DAO`: `0%`, `QUAY_ROI`: `0%`, `DE_DOA`: `8,06%`). SafeViChi v2 lấp đầy toàn bộ khoảng trống này với Recall `70,6%–94,4%` trên cả 4 lớp nguy hại.
2. **Vì sao SafeViChi v2 vượt LLM 120B (`+9,24 pp` C0, `+20,05 pp` C1, `+14,36 pp` C2):** LLM làm rất tốt `LUA_DAO` (`0,9577` trên C0) và giữ `FPR(AN_TOAN)` C0 rất thấp (`3,12%`), nhưng **quá bảo thủ ở `QUAY_ROI` và `DE_DOA`** — bỏ sót khoảng nửa số ca ở C0 (`Recall 45,1%` và `53,2%`) và tụt thảm xuống còn `20%–23%` ở C1/C2. Trên C1, SafeViChi v2 dẫn trước LLM về F1 trên **cả 5/5 lớp**.
3. **Đánh đổi FPR–FNR:** Trên C0, SafeViChi v2 chấp nhận `FPR = 10,42%` (`10/96` mẫu `AN_TOAN`, chủ yếu cặp sinh đôi khó) để kéo tỷ lệ bỏ lọt nguy hại xuống còn **`9,66%`** — thấp gần một nửa so với `18,49%` của LLM trên C0 và vượt xa `38,03%` của LLM trên C1.

---

## 3. So sánh chất lượng trích xuất bằng chứng

Đánh giá song song bằng **F1 theo ý** (mức mệnh đề, hệ số phủ `1,5`, tính tách biệt trên các đoạn nguy hại mô hình đã báo nguy hại; xem [DO_DO_BANG_CHUNG.md](DO_DO_BANG_CHUNG.md)) và **F1 mức từ** chặt để tránh thiên lệch độ đo:

| Chỉ số bằng chứng | Hệ thống | `C0` (Sạch) | `C1` (Né lọc) | `C2` (Điểm mù) |
| :--- | :--- | :---: | :---: | :---: |
| **F1 theo ý** *(cao tốt)* | LLM API (`gpt-oss-120b`) | `0,7073` | `0,5612` | `0,3217` |
| | **SafeViChi v2** | **`0,8303`** | **`0,7649`** | **`0,7531`** |
| **F1 mức từ** *(cao tốt)* | LLM API (`gpt-oss-120b`) | `0,6545` | `0,4982` | `0,2813` |
| | **SafeViChi v2** | **`0,7931`** | **`0,7220`** | **`0,7017`** |
| **Tô oan `AN_TOAN`** *(thấp tốt)* | LLM API / **SafeViChi v2** | **`3,12%`** / `9,38%` | `10,94%` / **`8,85%`** | **`4,17%`** / `8,85%` |
| **Bỏ trống trên câu đã báo nguy hại** *(thấp tốt)* | LLM API / **SafeViChi v2** | `4,12%` / **`3,26%`** | `10,51%` / **`6,36%`** | `41,42%` / **`4,84%`** |
| **Cụm lệch nguyên văn** (`non_verbatim`) | LLM API / **SafeViChi v2** | `7,66%` / **`0,00%`** | `12,79%` / **`0,00%`** | **`48,28%`** / **`0,00%`** |

**Nhận xét chính:**
- **Điểm yếu chí mạng của LLM sinh dưới biến dạng:** Khi gặp `fullwidth`, `zero_width` hay `cyrillic_homoglyph` ở C2, LLM tự "sửa chính tả/chuẩn hóa" lại cụm từ trong JSON thay vì chép nguyên văn → **`48,28%`** (`238/493`) cụm trả về không khớp chuỗi gốc, khiến ứng dụng không thể tô sáng và kéo `F1 theo ý` tụt xuống **`0,3217`** (`fullwidth`: `0,0951`, `zero_width`: `0,1250`).
- **Ưu thế cấu trúc của SafeViChi v2:** Đầu `TurnAwareBIOCRF` gán nhãn trực tiếp trên token kèm ánh xạ vị trí nên bảo đảm **`0,00%` non-verbatim** trên mọi tầng, đạt `F1 theo ý = 0,8303` (C0) và giữ `0,7531` trên C2 (**gấp 2,34× LLM**).
- **Đối chiếu với giải thích của SafeViChi v1:** Ở vòng loại, v1 dùng **Occlusion** (cần nhiều lượt forward, khoanh tràn `1,56×` trên ViHOS với Micro-F1 `0,5552` theo [localization.json](../../v1/results/step4/localization.json)); sang v2, bằng chứng được tách ngưỡng độc lập (`marginal_threshold = 0,30`, `single_word_threshold = 0,40`), trích trong **1 lượt suy luận duy nhất**, gọn sát nhãn vàng (`word_length_ratio = 0,98×` trên C0) và hiệu chuẩn xác suất tốt (`ECE = 0,0386`, `PR-AUC macro = 0,9262`).

---

## 4. Độ bền né lọc, lát cắt nổi bật & điểm yếu chính

### 4.1. Sức chống chịu biến dạng trên `C1 / C2` (13 kiểu biến dạng)

- **Độ suy giảm dưới tấn công (`Δ Macro-F1` so với `C0`):**
  - **SafeViChi v2:** `C1 = -7,88 pp`, `C2 = -9,39 pp` (riêng `Binary-F1` chỉ tụt **`-2,58 pp`** ở C1 và **`-2,00 pp`** ở C2). Trên tập `strictly_unseen` (`n = 764`, gồm 6 kiểu C2 + `dinh_tu`), SafeViChi v2 giữ `Macro-F1 = 0,7836`, `Binary-F1 = 0,9114` và `Ev F1 theo ý = 0,7660`.
  - **LLM API (`gpt-oss-120b`):** sụp mạnh hơn khoảng 2× (`C1 = -18,69 pp`, `C2 = -14,51 pp`; `Binary-F1` tụt `-14,71 pp` ở C1 và `-11,27 pp` ở C2).
- **Đối chiếu 13/13 kiểu biến dạng (`Macro-F1`):** SafeViChi v2 **vượt LLM API trên toàn bộ 13 kiểu**:
  - *Nhóm `C1` (v2 vs LLM):* `dinh_tu` (`0,8709` vs `0,5737`), `keo_dai` (`0,8099` vs `0,5514`), `bo_dau` (`0,8006` vs `0,5811`), `so_thay_chu` (`0,7953` vs `0,6751`), `chen_ky_tu` (`0,7672` vs `0,5041`), `go_sai` (`0,7275` vs `0,5339`), `teencode` (`0,7092` vs `0,5601`).
  - *Nhóm `C2` điểm mù (v2 vs LLM):* `fullwidth` (`0,8288` vs `0,7371`), `double_elongation` (`0,8243` vs `0,6700`), `exotic_separator` (`0,7816` vs `0,5581`), `cyrillic_homoglyph` (`0,7809` vs `0,7236`), `zero_width` (`0,7239` vs `0,5785`), `space_injection` (`0,6709` vs `0,4510`).

### 4.2. Các lát cắt nổi bật trên `C0` (`n = 334`)

| Nhóm lát cắt | Lát cắt (`n`) | LLM 120B (Macro-F1) | SafeViChi v2 (Macro-F1 / Bin-F1) | Ghi chú đối chiếu |
| :--- | :--- | :---: | :---: | :--- |
| **Số lượt thoại** | `1 lượt` (`108`) | `0,8037` | `0,8048` / `0,8759` | Tương đương trên câu đơn |
| | `nhiều lượt` (`226`) | `0,7462` *(tụt)* | **`0,8961`** / **`0,9509`** *(tăng)* | SafeViChi v2 khai thác ngữ cảnh hội thoại vượt trội (`+14,99 pp` so với LLM) |
| **Giọng vùng miền** | Bắc / Trung / Nam (`112/111/111`) | `0,7559` / `0,7279` / `0,8105` | **`0,8642`** / **`0,8448`** / **`0,8795`** | `region_gap` của v2 chỉ **`0,0347`** (so với LLM `0,0826`, v1 `0,1229`) |
| **Khái quát kịch bản** | `trong_phan_phoi` (`185`) | `0,8484` | **`0,9300`** / **`0,9736`** | `FPR` của v2 chỉ `2,00%`, `Ev F1 theo ý = 0,8679` |
| | `chua_thay` (`149`) | `0,5299` *(Bin `0,8023`)* | **`0,6209`** / **`0,8687`** | `Macro-F1` nhạy mạnh vì `XUC_PHAM` ở nhóm này chỉ có `n = 3` |
| **Kênh & độ lộ liễu** *(v2)* | `messenger_zalo` (`127`) · `do_lo:an_y` (`104`) | — | **`0,9344`** · **`0,8426`** | Rất mạnh trên kênh chat chính & hội thoại ẩn ý (`Bin-F1 = 0,9333`) |
| | `sms` (`46`) | — | `0,6712` / `0,8000` | Kênh yếu nhất của v2 (`FPR = 23,53%`, `13/45` lỗi C0) |

### 4.3. Điểm yếu còn tồn tại của SafeViChi v2

1. **Tập trung lỗi ở kịch bản chưa thấy và kênh SMS:** Trong `45` lỗi C0 của SafeViChi v2, có `34/45` nằm ở `chua_thay` (`DD6`: `13`, `QR6`: `8`, `moi`: `6`) và `13/45` ở kênh `sms` (tin nhắn ngắn, ít ngữ cảnh lượt thoại). Đáng chú ý, có `18/334` mẫu C0 mà **cả SafeViChi v2 lẫn LLM 120B đều đoán sai** (`15/18` thuộc `chua_thay`), phản ánh độ khó tự nhiên của các ca ranh giới.
2. **Báo nhầm trên cặp sinh đôi khó & nhầm lẫn nội bộ `QUAY_ROI ↔ DE_DOA`:** `10` ca FP (`AN_TOAN → nguy hại`) trên C0 chủ yếu rơi vào các câu sinh đôi `DD6` (`5`) và `LD12` (`3`), đồng thời giọng Nam có `FPR` cao hơn (`16,13%` vs miền Trung `3,12%`). Trong `12` ca nhầm giữa 4 lớp nguy hại, lớn nhất là `QUAY_ROI → DE_DOA` (`7` ca) và `XUC_PHAM → QUAY_ROI` (`3` ca — đúng vùng ranh giới bình phẩm ngoại hình `QR3`).
3. **Các kiểu né lọc khó nhất trên `C1/C2`:** 4 kiểu kéo điểm nhiều nhất của SafeViChi v2 là `space_injection` (`0,6709`), `teencode` (`0,7092`), `zero_width` (`0,7239`) và `go_sai` (`0,7275`), nơi chuỗi từ bị chia mảnh hoặc sai khác mạnh so với chính tả gốc.

---

## 5. So sánh chi phí triển khai & Kết luận

### 5.1. Đánh đổi chất lượng – tốc độ – tài nguyên (`batch = 1`)

Số liệu Laptop CPU đo trên `AMD Ryzen AI 7 H 350` (8 nhân, Windows 11, tiến trình độc lập, xem [bench_cpu.json](../../results/v2/official_eval/dualorc_mmbert_base/test/bench_cpu.json)); số liệu GPU đo trên `NVIDIA L4` ([benchmark.json](../../results/v2/official_eval/dualorc_mmbert_base/benchmark.json)); số liệu LLM đo ở `concurrency = 4` ([test_metrics.json](../../results/v2/llm_api/test_metrics.json)).

| Chỉ tiêu vận hành | SafeViChi v1 *(ONNX FP32)* | **SafeViChi v2 *(FP32)*** | LLM API *(`gpt-oss-120b`, OpenRouter)* |
| :--- | :---: | :---: | :---: |
| Dung lượng mô hình trên đĩa | `1.038,79 MiB` | **`1.177,0 MiB`** *(`< 2 GB`)* | Cloud *(không chạy cục bộ)* |
| Khởi động lạnh (cold load) | — | **`2,00 s`** *(ONNX CPU)* · `20,43 s` *(PyTorch CPU)* | Tức thì *(cần mạng)* |
| Độ trễ end-to-end `p50` / `p95` | — | **`43,97` / `58,47 ms`** *(ONNX CPU 8T, `22,3 tin/s`)*<br>`60,42` / `80,38 ms` *(PyTorch CPU 8T)* · `39,43` / `42,73 ms` *(GPU L4)* | `2.835–3.459` / `4.852–9.547 ms`<br>*(chậm hơn ~47–79× so với CPU v2)* |
| Bộ nhớ đỉnh (`peak_rss` / VRAM) | — | **`2.419 MiB`** *(ONNX CPU)* · `4.781 MiB` *(PyTorch CPU)* · `1.218 MiB` *(VRAM L4)* | Không đo phía client |
| Chi phí / Năng lượng (`1.000` tin) | `0 USD` | **`0 USD`** · `0,187–0,348 Wh` *(ONNX CPU, ước tính cTDP 15–28 W)* | **`~0,085 USD`** *(+ rủi ro `HTTP 429`)* |
| Quyền riêng tư & chạy offline | **100% cục bộ** | **100% cục bộ** | **Phải gửi tin nhắn ra ngoài** |

### 5.2. Kết luận

- **So với SafeViChi v1:** SafeViChi v2 mở rộng trọn vẹn từ bài toán câu đơn 2 lớp sang hội thoại đa lượt 5 nhãn + trích xuất bằng chứng trong 1 lượt (`Binary-F1 C0`: `0,3121 → 0,9287`, `C1`: `0,3104 → 0,9029`, `C2`: `0,3947 → 0,9087`), khắc phục hoàn toàn khoảng mù ở `LUA_DAO`, `QUAY_ROI` và `DE_DOA`.
- **So với LLM lớn qua API (`openai/gpt-oss-120b`):** SafeViChi v2 **vượt có ý nghĩa thống kê trên cả 3 tầng `C0 / C1 / C2`** cả về phân loại (`+9,24` đến `+20,05 pp` Macro-F1, thắng trên `13/13` kiểu biến dạng) lẫn trích xuất bằng chứng (gấp `2,34×` trên C2 với `0,00%` cụm lệch nguyên văn), **nhanh hơn hàng chục lần ngay trên CPU**, miễn phí API và bảo vệ tuyệt đối quyền riêng tư người dùng.
- **Hướng hoàn thiện tiếp theo:** Tập trung giảm `FPR` trên kênh `sms` và các cặp sinh đôi khó, làm rõ ranh giới `QUAY_ROI ↔ DE_DOA`, củng cố 4 kiểu né lọc khó nhất (`space_injection`, `teencode`, `zero_width`, `go_sai`), và bổ sung đối chiếu đồng thuận giữa 2 người gán thật (`Cohen's kappa`).
