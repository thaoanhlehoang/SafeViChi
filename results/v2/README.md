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
        └── test/                      # Thư mục lưu tệp xuất thô của phiên chấm test khóa (đang chờ đồng bộ)
```

---

## Tóm tắt nhanh 2 phần thực nghiệm mô hình

| Phần | Thư mục | Mục đích | Điểm nhấn chính |
| :--- | :--- | :--- | :--- |
| **Phần 1 — Huấn luyện & Chọn mô hình** | [`dev_selection/`](dev_selection/) | Lưu trọn bộ đầu ra train + đánh giá trên `dev` (`1.340` mẫu) và `cal` (`1.339` mẫu) của 4 ứng viên song song, cùng quy trình chốt mô hình trước khi chạm test | 3 biến thể `SafeVi-DualORC-CRF` vượt có ý nghĩa thống kê `GuardMerge` (McNemar `p ≤ 0,024`) và nhanh hơn `2,2×–4,4×`. Giữa 3 backbone DualORC hòa thống kê trên Macro-F1 gộp (`p ≥ 0,285`), chốt **`mmBERT-base`** vì dẫn đầu toàn diện hội thoại nhiều lượt (`0,9775`), ẩn ý (`0,9679`, `FPR 1,00%`), tiếng lóng (`0,7765`), Word Evidence F1 (`0,7054`), `safe_mark` thấp nhất (`5,03%`), `region_gap` nhỏ nhất (`0,0166`) và năng lực khái quát đa ngôn ngữ trong trần `< 2 GB`. |
| **Phần 2 — Mô hình chốt & Chấm test** | [`official_eval/`](official_eval/) | Tập trung vào mô hình chốt **`SafeVi-DualORC-CRF (mmBERT-base)`**, lưu hồ sơ đóng băng và kết quả chấm 1 lần cuối trên bộ test khóa (`334 C0 + 668 C1 + 668 C2 = 1.670` mẫu) | Dòng `#5` (thô): **`C0 Macro-F1 = 0,8762`** (`Binary-F1 = 0,9417`, `FPR = 7,29%`, `Ev F1 = 0,7717`, `region_gap = 0,0112`), **`C1 = 0,7914`** (`+3,25 pp` lên `0,8239` với chuẩn hóa `#6`), **`C2 = 0,7697`** (`Binary-F1 = 0,9148`, chỉ tụt `-2,69 pp` trước 100% kiểu điểm mù chưa thấy). *(Tệp xuất thô trong `dualorc_mmbert_base/test/` hiện chưa đồng bộ.)* |

---

## Nguyên tắc tổ chức & tái lập

1. **Tách bạch hoàn toàn `dev_selection/` và `official_eval/`:** Mọi quyết định chọn kiến trúc, chọn backbone, chọn checkpoint và đóng băng `calibration.json` đều hoàn tất trong `dev_selection/` trước khi mở khóa tập test.
2. **Toàn vẹn dữ liệu & hạt giống cố định:** Cả 4 lượt huấn luyện dùng chung commit dữ liệu `546300b7d5c3f7a8c8b78165402182d4aed23b7e`, cùng `split_ids.json` và `seed = 42`.
3. **Không commit trọng số nặng:** Chỉ lưu tệp cấu hình, chỉ số, dự đoán và biểu đồ nhẹ (`.json`, `.jsonl`, `.png`, `.md`); checkpoint nặng lưu ngoài git trong `models/`.
