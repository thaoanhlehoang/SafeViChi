# SafeViChi

[![Demo](https://img.shields.io/badge/Live%20Demo-safevichi.pages.dev-2563eb)](https://safevichi.pages.dev/)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-dminhnhat06%2Fsafevichi__v2-yellow)](https://huggingface.co/dminhnhat06/safevichi_v2)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-%E2%89%A53.10-3776ab)

Hệ thống phát hiện tin nhắn tiếng Việt nguy hại **chạy hoàn toàn trên thiết bị người dùng** (gói mô hình ~334 MiB, trần `< 2 GB`). Nội dung tin nhắn không bao giờ rời khỏi máy.

SafeViChi đọc một đoạn hội thoại hai người (1–8 lượt), trả về **nhãn tình huống** kèm **cụm từ bằng chứng chép nguyên văn** từ tin nhắn, và chịu được các kiểu viết biến dạng để lách bộ lọc: teencode, bỏ dấu, gõ sai, chèn ký tự, kéo dài ký tự, homoglyph Cyrillic, ký tự vô hình, fullwidth...

```text
$ safevichi --json "Tài khoản của quý khách sẽ bị khóa || Vui lòng xác minh tại đường link và nhập mã OTP"
{
  "nhan": "LUA_DAO",
  "ten_nhan": "Lừa đảo / Dụ OTP & link",
  "hanh_dong": "Cảnh báo lừa đảo, vô hiệu hóa liên kết",
  "tin_cay": 0.968497,
  "bang_chung": ["xác minh tại đường link và nhập mã OTP"],
  ...
}
```

---

## Mục lục

- [Bài toán & bộ nhãn](#bài-toán--bộ-nhãn)
- [Kiến trúc mô hình](#kiến-trúc-mô-hình)
- [Kết quả chính](#kết-quả-chính)
- [Cài đặt & sử dụng](#cài-đặt--sử-dụng)
- [Tái lập thực nghiệm](#tái-lập-thực-nghiệm)
- [Tài liệu chi tiết](#tài-liệu-chi-tiết)

---

## Bài toán & bộ nhãn

| Phiên bản | Vị trí | Bài toán | Đầu ra |
| --- | --- | --- | --- |
| **v2** *(hiện tại, vòng chung kết)* | thư mục gốc | Hội thoại 2 người, 1–8 lượt, **5 nhãn theo tình huống** | Nhãn + cụm bằng chứng, phân loại và tô sáng có ngưỡng riêng |
| **v1** *(baseline đóng băng)* | [v1/](v1/) · tag `v1-vongloai` | Câu đơn, `HATE` / `CLEAN` | ViHateT5 fine-tune đối kháng (FGM) + occlusion |

| Nhãn | Ý nghĩa | Ứng dụng xử lý |
| --- | --- | --- |
| `AN_TOAN` | Không thuộc 4 nhóm nguy hại | Hiển thị bình thường |
| `XUC_PHAM` | Lăng mạ, miệt thị cá nhân hoặc nhóm (kể cả vùng miền) | Làm mờ, bấm mới xem |
| `QUAY_ROI` | Gạ gẫm không mong muốn, bám riết, bêu riếu | Gợi ý chặn, báo cáo |
| `DE_DOA` | Đe dọa gây hại thân thể, tài sản, danh dự; tống tiền | Cảnh báo mạnh, gợi ý lưu bằng chứng |
| `LUA_DAO` | Dụ chuyển tiền, lấy OTP, bấm đường link giả mạo | Cảnh báo, vô hiệu hóa liên kết |

Khi nhiều nhãn cùng đúng: `DE_DOA` > `LUA_DAO` > `QUAY_ROI` > `XUC_PHAM`. Định nghĩa đầy đủ và các ca ranh giới: xem [báo cáo](docs/v2/bao_cao/bao_cao_v2.pdf).

Đầu ra JSON của mô hình:

```json
{"nhan": "LUA_DAO", "bang_chung": ["tài khoản của quý khách sẽ bị khóa", "xác minh tại đường link"]}
```

`bang_chung` luôn là cụm chép nguyên văn từ tin nhắn (tỉ lệ cụm bịa `0%` trên bộ test) và được phép rỗng.

---

## Kiến trúc mô hình

Mô hình chốt là **SafeVi-DualORC-CRF** trên backbone [`jhu-clsp/mmBERT-base`](https://huggingface.co/jhu-clsp/mmBERT-base) (307M tham số, 22 lớp ModernBERT), chọn từ 4 ứng viên huấn luyện song song trên cùng dữ liệu và seed (xem [results/v2/dev_selection/](results/v2/dev_selection/dev_summary.json)):

- **Hai luồng biểu diễn:** encoder mmBERT-base đa ngôn ngữ + **Byte-CNN** trên byte UTF-8 của từng từ, giúp mô hình vẫn đọc được từ bị biến dạng ở mức ký tự.
- **Đầu phân loại hội thoại** cho 5 nhãn, hiệu chuẩn nhiệt độ trên tập `cal` rồi đóng băng.
- **BIO-CRF mức từ có ràng buộc biên lượt thoại** cho cụm bằng chứng; giải mã Forward-Backward / Viterbi viết lại bằng NumPy, nên lúc suy luận không cần PyTorch.
- **Không dùng bộ chuẩn hóa văn bản:** mô hình đọc thẳng văn bản gốc. Ablation cho thấy bộ chuẩn hóa v1 chỉ giúp vài kiểu C1 nhưng làm giảm điểm trên câu sạch C0 (`-1,23 pp`) và trên các kiểu điểm mù C2 ([chi tiết](results/v2/official_eval/dualorc_mmbert_base/)).
- **Triển khai:** xuất ONNX opset 17, lượng tử hóa **Hybrid INT8** (embedding INT8 theo hàng, W8A8 cho attention, W8A32 riêng cho lớp chiếu GeGLU để tránh cắt ngọn giá trị ngoại lai). File mô hình còn **300,69 MiB** (nhỏ hơn `3,91×` so với FP32).

Dữ liệu huấn luyện gồm 13.392 đoạn hội thoại (`data/v2/train.jsonl`). Bộ test khóa gồm 1.670 mẫu, tất cả do người viết hoặc người duyệt và không trùng với train ([kiểm tra rò rỉ](results/v2/kiem_tra_ro_ri/bao_cao.json)):

- **C0:** 334 hội thoại sạch, chia đều 3 giọng Bắc / Trung / Nam.
- **C1:** 668 bản biến dạng né lọc thường gặp (khoảng 60% số từ bị biến dạng).
- **C2:** 668 bản biến dạng theo 6 kiểu điểm mù mà mô hình **chưa từng thấy** khi huấn luyện.

---

## Kết quả chính

Ngưỡng được dò trên validation rồi đóng băng trước khi mở bộ test. Mọi con số dưới đây do script sinh ra từ `results/v2/`. Kết quả theo từng nhãn, kiểu biến dạng, vùng miền, kênh, ablation nằm ở [docs/KET_QUA.md](docs/KET_QUA.md); phân tích lỗi và nhận xét ở [docs/DANH_GIA.md](docs/DANH_GIA.md).

### So sánh 3 hệ trên bộ test khóa

Nguồn: [results/v2/so_sanh_3_he/metrics.json](results/v2/so_sanh_3_he/metrics.json).

| Hệ | Tập | Macro-F1 (5 nhãn) | Binary-F1 | Recall nguy hại | FPR an toàn | Bằng chứng F1 theo ý |
| --- | --- | :---: | :---: | :---: | :---: | :---: |
| SafeViChi v1 (ViHateT5) | C0 / C1 / C2 | – | 0,312 / 0,310 / 0,395 | 0,185 / 0,185 / 0,250 | 0,0% / 1,6% / 4,2% | – |
| gpt-oss-120b (LLM qua API) | C0 / C1 / C2 | 0,772 / 0,585 / 0,627 | 0,892 / 0,745 / 0,779 | 0,815 / 0,620 / 0,649 | 3,1% / 10,9% / 4,2% | 0,707 / 0,561 / 0,322 |
| **SafeViChi v2 (on-device)** | C0 / C1 / C2 | **0,865 / 0,786 / 0,771** | **0,929 / 0,903 / 0,909** | **0,903 / 0,859 / 0,868** | 10,4% / 10,9% / 10,4% | **0,830 / 0,765 / 0,753** |

- Trên 6 kiểu điểm mù chưa thấy (C2), Binary-F1 của v2 chỉ giảm `2,0 pp` so với câu sạch (`0,929 → 0,909`).
- Điểm yếu hiện tại là **FPR trên câu an toàn khoảng 10%**, cao hơn LLM lớn ở C0. Lỗi tập trung ở kịch bản chưa thấy và tin SMS ngắn, ít ngữ cảnh. Phân tích lỗi chi tiết nằm trong [docs/DANH_GIA.md](docs/DANH_GIA.md).
- Độ chênh Macro-F1 giữa 3 vùng miền trên C0 là `0,035` (nhóm thấp nhất `0,845`).

### Nén INT8 và chi phí trên thiết bị

Mô hình ONNX Hybrid INT8 giữ được **98,71%** Macro-F1 của bản FP32 trên toàn bộ 1.670 mẫu. Kiểm định McNemar không thấy khác biệt có ý nghĩa thống kê ở bất kỳ tập nào (`p ≥ 0,073`). Nguồn: [int8_comparison.json](results/v2/official_eval/dualorc_mmbert_base/test/int8_comparison.json).

| Môi trường (`batch = 1`) | Độ trễ p50 / p95 | Thông lượng | RAM đỉnh | Nguồn |
| --- | :---: | :---: | :---: | --- |
| Laptop CPU AMD Ryzen AI 7 H 350, ONNX INT8, 8 luồng | 24,7 / 31,5 ms | 41,3 tin/s | 986 MiB | [bench_cpu.json](results/v2/official_eval/dualorc_mmbert_base/test/bench_cpu.json) |
| Laptop CPU Intel Core Ultra 7 155H (chạy pin), 8 luồng | 69,1 / 120,8 ms | 13,5 tin/s | 904 MiB | [cpu_laptop.json](results/v2/do_thiet_bi/cpu_laptop.json) |
| Trình duyệt Edge, ONNX Runtime Web WASM, 4 luồng (cùng máy Intel) | 336,7 / 604,7 ms | 2,8 tin/s | +949 MiB so với tab trống | [trinh_duyet.json](results/v2/do_thiet_bi/trinh_duyet.json) |

So với PyTorch FP32 trên cùng máy Ryzen, bản INT8 nhanh hơn `2,45×`, khởi động lạnh nhanh hơn `13,3×` (`1,53 s`) và dùng ít RAM đỉnh hơn `4,85×`.

---

## Cài đặt & sử dụng

### 1. CLI suy luận cục bộ `safevichi` (không cần PyTorch)

Gói chỉ phụ thuộc `onnxruntime`, `tokenizers`, `numpy`, `huggingface-hub` và đăng ký sẵn lệnh `safevichi` dùng được ở mọi thư mục:

```bash
git clone https://github.com/thaoanhlehoang/SafeViChi.git && cd SafeViChi
pip install -e .
```

Lần chạy đầu tiên, CLI tự tải gói ONNX Hybrid INT8 từ Hugging Face [`dminhnhat06/safevichi_v2`](https://huggingface.co/dminhnhat06/safevichi_v2) (`safevichi_v2_int8.onnx`, `crf_params.json`, `tokenizer/`, tổng ~334 MiB) vào `models/dualorc_mmbert_base/onnx/` và kiểm tra SHA-256. Từ lần sau CLI chạy hoàn toàn offline.

```bash
safevichi                                              # Chế độ tương tác, gõ từng tin nhắn
safevichi --chat                                       # Hội thoại trượt 1–8 lượt
safevichi "Tài khoản sẽ bị khóa, bấm link nhập OTP"    # Quét nhanh 1 tin nhắn
safevichi "Lượt 1 || Lượt 2"                           # Hội thoại nhiều lượt, ngăn bằng ||
safevichi --json "Mày là đồ ngu"                       # Trả về JSON thuần
safevichi --file messages.txt --out ket_qua.jsonl      # Quét hàng loạt
```

Tải trước mô hình hoặc chạy trong môi trường không có mạng:

```bash
python -m src.v2.download_model              # Gói ONNX INT8 (~334 MiB)
python -m src.v2.download_model --fp32       # Gói ONNX FP32 (~1,18 GiB)
python -m src.v2.download_model --all        # INT8 + FP32 + PyTorch safetensors
```

- `--no-download` hoặc `SAFEVICHI_OFFLINE=1`: không bao giờ gọi mạng.
- `--onnx-dir` hoặc `SAFEVICHI_ONNX_DIR`: trỏ tới thư mục mô hình có sẵn.
- `--legacy-normalize`: bật bộ chuẩn hóa v1. Tùy chọn này đã lỗi thời, chỉ giữ để đối chiếu ablation.

### 2. Môi trường đầy đủ (huấn luyện, đánh giá, kiểm thử)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt
pip install -e . --no-deps
pytest -q                          # Chạy toàn bộ test (tests/v2 và v1/tests)
```

Các script gọi LLM (sinh dữ liệu, gán nhãn sơ bộ) đọc cấu hình từ `.env`. Chép từ [.env.example](.env.example) rồi điền khóa API. Không commit file `.env`.

### 3. Ứng dụng web on-device

[web-v2/](web-v2/) chạy mô hình INT8 trong Web Worker qua ONNX Runtime Web (WASM), không có backend. Bản triển khai: [safevichi.pages.dev](https://safevichi.pages.dev/).

```bash
cd web-v2 && npm install
npm run dev                # http://127.0.0.1:5173
npm run verify:parity      # Đối chiếu kết quả JS với bộ test v2
```

---

## Tái lập thực nghiệm

| Bước | Lệnh / tệp |
| --- | --- |
| Huấn luyện (Colab/Kaggle, GPU) | [notebooks/v2_dualorc_crf_colab.ipynb](notebooks/v2_dualorc_crf_colab.ipynb) |
| Xuất ONNX FP32 + Hybrid INT8 | `python -m src.v2.bench.export_onnx` |
| Đo CPU laptop / trình duyệt | `python -m src.v2.bench.benchmark_cpu` · `python -m src.v2.bench.do_thiet_bi cpu` · `python -m src.v2.bench.do_thiet_bi trinh_duyet` |
| So sánh 3 hệ (v1, LLM, v2) | `python -m src.v2.eval.so_sanh_3_he` |
| Xuất số liệu cho báo cáo | `python -m src.v2.eval.xuat_so_lieu_bao_cao` (ghi `docs/v2/bao_cao/so_lieu.tex`) |
| Sinh tài liệu kết quả, đánh giá | `python -m src.v2.eval.xuat_tai_lieu` (ghi `docs/KET_QUA.md`, `docs/DANH_GIA.md`) |

Đánh giá một tệp dự đoán bất kỳ:

```bash
python -m src.v2.eval.evaluate \
  --gold data/v2/test_day_du.jsonl data/v2/test_perturb_C1.jsonl data/v2/test_perturb_C2.jsonl \
  --pred results/v2/<ten_cau_hinh>/predictions.jsonl \
  --meta data/v2/test_day_du.jsonl \
  --out  results/v2/<ten_cau_hinh>/metrics.json
```

Bộ đánh giá tính:

- **Phân loại:** Macro-F1, P/R/F1 từng nhãn, Binary-F1, FPR trên `AN_TOAN`, bootstrap 95% CI, ECE/PR-AUC, McNemar ghép cặp (`--reference-pred`).
- **Bằng chứng:** F1 theo ý, F1 mức từ, tỉ lệ tô oan câu an toàn, tỉ lệ cụm không có trong câu.
- **Lát cắt:** C0/C1/C2, từng kiểu biến dạng, vùng miền (kèm worst-group), kịch bản trong phân phối và chưa thấy, số lượt hội thoại.

Nguyên tắc tái lập: seed cố định `42`; bộ test khóa bằng SHA-256 ([REAL_TEST_LOCK.json](results/v2/official_eval/dualorc_mmbert_base/test/REAL_TEST_LOCK.json)); mỗi cấu hình ghi lại phần cứng, phiên bản thư viện và lệnh chạy.

---

## Tài liệu chi tiết

| Tài liệu | Nội dung |
| --- | --- |
| [docs/CAU_TRUC_THU_MUC.md](docs/CAU_TRUC_THU_MUC.md) | Vai trò từng thư mục và mô-đun, định dạng dữ liệu, nơi lưu kết quả |
| [docs/KET_QUA.md](docs/KET_QUA.md) | Toàn bộ bảng số: dữ liệu, chọn mô hình trên dev, kết quả theo nhãn và lát cắt, ablation, INT8, đo thiết bị |
| [docs/DANH_GIA.md](docs/DANH_GIA.md) | Nhận xét theo 3 yêu cầu cải tiến, phân tích báo nhầm/bỏ sót, ví dụ thất bại, hạn chế, hướng phát triển |
| [docs/v2/bao_cao/bao_cao_v2.pdf](docs/v2/bao_cao/bao_cao_v2.pdf) | Báo cáo vòng chung kết (mã LaTeX cùng thư mục) |

`KET_QUA.md` và `DANH_GIA.md` sinh tự động từ `results/v2/`, không sửa tay. SafeViChi v1 (vòng loại) được giữ nguyên trong [v1/](v1/) và tag `v1-vongloai` làm baseline.

---

## Giấy phép

Phát hành theo giấy phép [MIT](LICENSE).
