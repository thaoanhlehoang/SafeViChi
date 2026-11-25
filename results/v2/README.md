# Kết quả thực nghiệm SafeViChi v2 (`results/v2/`)

Thư mục này lưu toàn bộ kết quả nhẹ, có thể kiểm tra và tái lập của **SafeViChi v2** (không chứa checkpoint nặng `.safetensors`/`.pt`/`.onnx` — xem `models/` và `.gitignore`), đóng vai trò nguồn số liệu thống nhất để sinh bảng và biểu đồ cho báo cáo vòng chung kết. Kết quả đóng băng của vòng loại v1 nằm riêng tại [`v1/results/`](../../v1/results/).

---

## Cấu trúc thư mục

```text
results/v2/
├── README.md                          # Tệp này — tổng quan bố cục kết quả v2
│
├── (Giai đoạn 0 — Kiểm định & chuẩn bị dữ liệu, giữ nguyên)
├── chi_phi_du_lieu/                   # Chi phí API sinh & gán sơ bộ dữ liệu
├── dong_thuan_nhan/                   # Đối chiếu đồng thuận nhãn gốc vs LLM
├── kiem_tra_ro_ri/                    # Xác minh 0 rò rỉ giữa train và test người viết
├── thu_model/                         # Thử nghiệm chọn LLM sinh dữ liệu
│
├── SO_SANH_BASELINE.md                # So sánh gpt-oss-120b và v1 trên test (sinh bằng src.v2.eval.bao_cao_baseline)
├── llm_api/                           # Baseline LLM lớn qua API (gpt-oss-120b); cham_lai/ = chấm lại bằng bộ chấm hiện tại
├── v1_baseline/                       # Baseline SafeViChi v1 (nhị phân); cham_lai/ = metrics từ src.v2.eval.cham_v1
│
├── dev_selection/                     # Phần 1 — Huấn luyện, so sánh 4 ứng viên trên dev & quyết định chọn mô hình
│   ├── README.md                      # So sánh toàn diện trên dev, kiểm định McNemar & luận giải chọn mmBERT-base
│   ├── dev_summary.json               # Số liệu tổng hợp máy đọc sinh tự động từ 4 thư mục ứng viên
│   ├── dualorc_mmbert_base/           # SafeVi-DualORC-CRF (mmBERT-base, Schema v3) — 13 tệp nhẹ
│   ├── dualorc_bamibert/              # SafeVi-DualORC-CRF (BamiBERT, Schema v3) — 13 tệp nhẹ
│   ├── dualorc_visobert/              # SafeVi-DualORC-CRF (ViSoBERT, Schema v3) — 13 tệp nhẹ
│   └── guardmerge_v6/                 # Hier-BidirLM-0.6B GuardMerge (Schema v6) — 13 tệp nhẹ
│
└── official_eval/                     # Phần 2 — Mô hình chốt (mmBERT-base) & đánh giá trên bộ test khóa (1.670 mẫu)
    ├── README.md                      # Báo cáo kết quả chính thức C0/C1/C2, chuẩn hóa (#6−#5), lát cắt & danh mục tệp
    └── dualorc_mmbert_base/           # Hồ sơ đầy đủ của mô hình chốt (13 tệp cấu hình/hiệu chuẩn/dev + thư mục test/)
        └── test/                      # Trọn bộ tệp xuất thô của phiên chấm test khóa & báo cáo tái kiểm chứng CLI
```

---

## Tóm tắt nhanh 2 phần thực nghiệm mô hình

| Phần | Thư mục | Mục đích | Điểm nhấn chính |
| :--- | :--- | :--- | :--- |
| **Phần 1 — Huấn luyện & Chọn mô hình** | [`dev_selection/`](dev_selection/) | Lưu trọn bộ đầu ra train + đánh giá trên `dev` (`1.340` mẫu) và `cal` (`1.339` mẫu) của 4 ứng viên song song, cùng quy trình chốt mô hình trước khi chạm test | 3 biến thể `SafeVi-DualORC-CRF` vượt có ý nghĩa thống kê `GuardMerge` (McNemar `p ≤ 0,023`) và nhanh hơn `2,1×–4,4×`. Giữa 3 backbone DualORC hòa thống kê trên Macro-F1 gộp (`p ≥ 0,290`), chốt **`mmBERT-base`** vì dẫn đầu toàn diện hội thoại nhiều lượt (`0,9801`, `+27,08 pp` ngữ cảnh), ẩn ý (`0,9702`, `FPR 1,00%`), tiếng lóng (`0,7478`), Word Evidence F1 (`0,7060`), công bằng vùng miền (`worst_group = 0,9850`, `region_gap = 0,0150`) và năng lực khái quát đa ngôn ngữ trong trần `< 2 GB`. |
| **Phần 2 — Mô hình chốt & Chấm test** | [`official_eval/`](official_eval/) | Tập trung vào mô hình chốt **`SafeVi-DualORC-CRF (mmBERT-base)`**, lưu hồ sơ đóng băng, kết quả chấm test khóa (`1.670` mẫu), bản **ONNX Hybrid INT8** (`#7`) và benchmark Laptop CPU | Dòng `#5` (thô): **`C0 = 0,8647`**, **`C1 = 0,7859`** (`+4,16 pp` lên **`0,8275`** với `#6`), **`C2 = 0,7708`**. Dòng `#7` (**ONNX Hybrid INT8**, **`300,69 MiB`**, nén `3,91×`): giữ **`99,43%`** Macro-F1 `C0` (`0,8598`, McNemar `p = 1,00`), **`98,71%`** toàn bộ `1.670` mẫu (`p = 0,073`), độ trễ Laptop CPU **`24,68 ms` p50 / `31,54 ms` p95** (`41,3 tin/s`), RAM đỉnh **`986 MiB`** (giảm `4,85×` so với PyTorch). |

---

## Nguyên tắc tổ chức & tái lập

1. **Tách bạch hoàn toàn `dev_selection/` và `official_eval/`:** Mọi quyết định chọn kiến trúc, chọn backbone, chọn checkpoint và đóng băng `calibration.json` đều hoàn tất trong `dev_selection/` trước khi mở khóa tập test.
2. **Toàn vẹn dữ liệu & hạt giống cố định:** Cả 4 lượt huấn luyện dùng chung `so_sanh.jsonl` (cùng SHA-256 ở cả `546300b` và `371a0f1`), cùng `split_ids.json` và `seed = 42`; bộ test khóa `1.670` mẫu chốt tại commit `371a0f1caf3f7b62f62b87dbf557c66eda9c544a` (`REAL_TEST_LOCK.json`).
3. **Không commit trọng số nặng:** Chỉ lưu tệp cấu hình, chỉ số, dự đoán và biểu đồ nhẹ (`.json`, `.jsonl`, `.png`, `.md`); checkpoint nặng lưu ngoài git trong `models/`.
