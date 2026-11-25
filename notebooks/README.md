# Notebooks Huấn luyện & Đánh giá SafeViChi v2

Thư mục này chứa các notebook huấn luyện, hiệu chuẩn và đánh giá trên **Google Colab / Kaggle** cho **SafeViChi v2** — hệ thống phát hiện tin nhắn tiếng Việt nguy hại chạy cục bộ trên thiết bị (`< 2 GB`), phân loại **5 nhãn theo tình huống** (`AN_TOAN`, `XUC_PHAM`, `QUAY_ROI`, `DE_DOA`, `LUA_DAO`) và trích xuất **cụm từ bằng chứng chép nguyên văn (verbatim)**.

Hai nhánh thực nghiệm dùng chung dữ liệu huấn luyện `so_sanh.jsonl` (không đổi SHA-256) và bộ test khóa tại commit `371a0f1caf3f7b62f62b87dbf557c66eda9c544a`, cùng phân hoạch `split_ids.json`, cùng giao thức hiệu chuẩn và cùng bộ test đánh giá (`1.670` mẫu).

---

## 1. Danh mục Notebook

| Tệp | Kiến trúc | Mô tả ngắn |
| :--- | :--- | :--- |
| [`v2_dualorc_crf_colab.ipynb`](v2_dualorc_crf_colab.ipynb) | **SafeVi-DualORC-CRF** *(Nhánh 2)* | Kiến trúc encoder hạng nhẹ (hỗ trợ `visobert`, `bamibert`, `mmbert_base`, `mmbert_small`) kết hợp UTF-8 Byte-CNN, HAT, BIO-CRF mức từ, ORC, nén embedding INT8, xuất ONNX và đối chiếu McNemar / Cascade. |
| [`v2_guardmerge_colab.ipynb`](v2_guardmerge_colab.ipynb) | **Hier-BidirLM-0.6B GuardMerge** *(Nhánh 1)* | Kiến trúc phân cấp hai chiều trộn trọng số `BidirLM-0.6B-Base` + `Qwen3Guard-Gen-0.6B`, tinh chỉnh LoRA + đầu phân cấp `GuardHierarchy`, Dual-View JS Consistency và nén INT8 CPU. |

---

## 2. Luồng thực nghiệm trong các notebook

| Mục | Nội dung | Tệp xuất chính trong `<RUN>/` |
| :---: | :--- | :--- |
| **1** | Cài đặt thư viện ghim (`%pip`), kiểm tra GPU/VRAM/ổ tạm, thiết lập tham số thí nghiệm & khởi tạo `RUN` hash | `config.json`, `environment.json` |
| **2** | Tải dữ liệu ghim SHA-256 (`13.392` đoạn `C0`), khóa chống rò rỉ `HUMAN_CAL_PATH` & chia tập `StratifiedGroupKFold` | `split_ids.json` |
| **3** | Token hóa hội thoại 1–8 lượt, tăng cường bề mặt C1 phi điểm mù, khớp bằng chứng (kèm LEG ở Nhánh 2), `BucketBatchSampler` & tự kiểm định | Kiểm định Offset / BIO / Aug |
| **4** | Định nghĩa kiến trúc mô hình (head chạy FP32) & tự kiểm định toán học (BIO-CRF vét cạn $3^W$ / hai chiều & bất biến padding) | Kiểm định tiền huấn luyện |
| **5** | Hàm mất mát đa nhiệm, Dual-View JS Consistency, suy luận gốc `#5` + bộ chuẩn hóa giữ dòng `#6`, vòng lặp huấn luyện & dọn sạch VRAM | `model_best.safetensors` / `adapter.safetensors`, `last.pt`, `history.json` |
| **6** | Hiệu chuẩn nhiệt độ/ngưỡng trên `cal`, báo cáo `dev` & cổng khóa cứng Bộ Test 1 lần duy nhất (`334 C0 + 668 C1 + 668 C2 = 1.670` mẫu, `#5` & `#6`) | `calibration.json`, `dev_*`, `REAL_TEST_LOCK.json`, `real_test_*` |
| **7** | 5 bộ Stress Test tất định (phân tách rõ `seen_in_train`), xem trước chuẩn hóa `#6`, kiểm chứng ngữ cảnh đa lượt, Benchmark VRAM/độ trễ & `predict_text()` | `*_stress_tests.json`, `*_context_checks.json`, `benchmark.json` |
| **8** | Lượng tử hóa INT8 trên CPU (`PerRowInt8Embedding` scale FP32 + `nn.Linear` `qint8`) & xuất ONNX opset 17 *(Nhánh 2)* | `int8_comparison.json`, `dualorc_core.onnx`, `crf_params.json` |
| **9** | **Nhánh 2:** Đối chiếu liên nhánh McNemar & `VietSafe-HierBi-Cascade` (UCCI khớp trên `cal`). **Nhánh 1:** Quy trình thực nghiệm & Bảng đối chiếu API | `cascade_vs_guardmerge_*.json` |
| **10–11** | Phụ lục: Quy trình thực nghiệm 3 giai đoạn, Ma trận Ablation, Đối chiếu API & Đặc tả quy ước đồng bộ | Lệnh tái kiểm chứng CLI |

---

## 3. Quy trình thực nghiệm

1. **Kiểm thử nhanh (`RUN_MODE = "smoke"`):** Chạy toàn bộ notebook trên GPU (1 epoch, ~15 giây) để xác minh phiên bản thư viện ghim, tự kiểm định toán học và cơ chế lưu/khôi phục checkpoint.
2. **So sánh ứng viên & Ablation (`RUN_MODE = "full"`, `DATA_SCHEME = "dev_80_10_10"`, `EVALUATE_REAL_TEST = False`):**
   - Chạy song song các preset (`visobert`, `bamibert`, `mmbert_base`, `mmbert_small`) và nhánh `GuardMerge` trên cùng `fit` (`10.713`), `dev` (`1.340`), `cal` (`1.339`).
   - Giữ `SEED = 42` và tắt lần lượt từng cờ kiến trúc / tăng cường cho ma trận Ablation.
3. **Đóng băng hiệu chuẩn & Đánh giá Bộ Test khóa 1 lần duy nhất:**
   - Sau khi hoàn tất lựa chọn cấu hình trên `dev` và đóng băng `calibration.json`, đặt `EVALUATE_REAL_TEST = True` (cờ này **không** nằm trong `RUN` hash) để chạy duy nhất một lần trên trọn bộ `1.670` mẫu (`C0 / C1 / C2`, song song `#5` thô & `#6` chuẩn hóa) và tự động khóa qua `REAL_TEST_LOCK.json`.
   - Tái kiểm chứng bằng bộ chấm CLI của kho mã nguồn:
     ```bash
     python -m src.v2.eval.evaluate \
       --gold data/v2/test_day_du.jsonl data/v2/test_perturb_C1.jsonl data/v2/test_perturb_C2.jsonl \
       --pred <RUN>/real_test_all_predictions.jsonl \
       --meta data/v2/test_day_du.jsonl \
       --out results/v2/<ten_cau_hinh>/metrics.json
     ```

---

## 4. Quy tắc toàn vẹn & Kiểm thử tự động

- **Chống rò rỉ test:** Mặc định `EVALUATE_REAL_TEST = False` và `HUMAN_CAL_PATH = None`. Mọi nỗ lực nạp nhầm 6 tệp `train`/`test`/`C1`/`C2` vào `HUMAN_CAL_PATH` hoặc sửa `calibration.json` sau khi đã khóa đều bị chặn bằng xác thực SHA-256.
- **Kiểm thử tự động cấu trúc, AST và logic cốt lõi của cả hai notebook:**
  ```bash
  pytest -q tests/v2/test_v2_notebooks.py
  ```
