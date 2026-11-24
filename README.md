# SafeViChi

[![Demo](https://img.shields.io/badge/Live%20Demo-safevichi.pages.dev-2563eb)](https://safevichi.pages.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Hệ thống phát hiện tin nhắn tiếng Việt nguy hại **chạy hoàn toàn cục bộ trên thiết bị (on-device, `< 2 GB`)**, chống các kỹ thuật né bộ lọc (teencode, bỏ dấu, chèn ký tự, homoglyph, kéo dài ký tự) và không gửi nội dung tin nhắn ra ngoài.

---

## Tổng quan v2 (Vòng chung kết) vs v1 (Vòng loại)

| Phiên bản | Vị trí / Tag | Bài toán | Đầu ra |
| --- | --- | --- | --- |
| **SafeViChi v2** *(hiện tại)* | Thư mục gốc (`v2-final`) | Hội thoại 2 người (1–8 lượt), **5 nhãn theo tình huống** | JSON `{"nhan": "...", "bang_chung": [...]}` (tách ngưỡng phân loại & tô sáng) |
| **SafeViChi v1** *(baseline đóng băng)* | [v1/](v1/README.md) · tag `v1-vongloai` | Câu đơn, 2 nhãn `HATE` / `CLEAN` | ViHateT5 fine-tune đối kháng (FGM) + occlusion highlight |

### Bộ 5 nhãn tình huống (v2)

| Nhãn | Ý nghĩa | Hành động trong ứng dụng |
| --- | --- | --- |
| `AN_TOAN` | Không thuộc 4 nhóm nguy hại | Hiển thị bình thường |
| `XUC_PHAM` | Lăng mạ, miệt thị cá nhân hoặc nhóm (kể cả vùng miền) | Làm mờ, bấm mới xem |
| `QUAY_ROI` | Gạ gẫm không mong muốn, bám riết, bêu riếu | Gợi ý chặn, báo cáo |
| `DE_DOA` | Đe dọa gây hại thân thể, tài sản, danh dự; tống tiền | Cảnh báo mạnh, gợi ý lưu bằng chứng |
| `LUA_DAO` | Dụ chuyển tiền, lấy OTP, bấm đường link giả mạo | Cảnh báo, vô hiệu hóa liên kết |

Thứ tự ưu tiên khi nhiều nhãn cùng đúng: `DE_DOA` > `LUA_DAO` > `QUAY_ROI` > `XUC_PHAM`. Định nghĩa chi tiết và ca ranh giới xem tại [docs/v2/HUONG_DAN_GAN_NHAN.md](docs/v2/HUONG_DAN_GAN_NHAN.md).

---

## Cấu trúc thư mục

```text
SafeViChi/
├── README.md, AGENTS.md, CLAUDE.md, pyproject.toml, LICENSE
├── requirements.txt, requirements-lock.txt, .env.example
│
├── src/                        # Mã nguồn v2 & dùng chung (chi tiết tại src/README.md)
│   ├── v2/                     # Pipeline vòng chung kết: data/, labeling/, train/, eval/, bench/
│   └── shared/                 # Dùng chung v1 + v2: normalization/, variant_generator/, reporting/, utils/
│
├── notebooks/                  # Notebook huấn luyện v2 trên Colab/Kaggle (xem notebooks/README.md)
├── data/                       # Dữ liệu v2 (train 13.392, test C0 334, C1/C2 668 mỗi tập) & data/raw/
├── results/                    # Kết quả thực nghiệm & thống kê dữ liệu v2 (results/v2/)
├── docs/                       # Tài liệu kỹ thuật v2 (docs/v2/)
├── tests/                      # Bộ kiểm thử tự động v2 (tests/v2/)
├── demo/                       # Ứng dụng web chạy on-device qua ONNX Runtime Web
├── models/                     # Checkpoint cục bộ (không commit lên git)
│
└── v1/                         # Toàn bộ lưu trữ vòng loại (baseline đóng băng, xem v1/README.md)
    ├── src/, data/, results/, scripts/, docs/, tests/
```

---

## Cài đặt & Khởi động nhanh

### 1. Tạo môi trường Python
```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt
```

### 2. Chạy kiểm thử tự động
```bash
pytest -q                       # Chạy toàn bộ unit test (cả tests/v2 và v1/tests)
```

### 3. Đánh giá mô hình v2 (phân loại + bằng chứng + lát cắt C0/C1/C2)
```bash
python -m src.v2.eval.evaluate \
  --gold data/v2/test_day_du.jsonl data/v2/test_perturb_C1.jsonl data/v2/test_perturb_C2.jsonl \
  --pred results/v2/<ten_cau_hinh>/predictions.jsonl \
  --meta data/v2/test_day_du.jsonl \
  --out results/v2/<ten_cau_hinh>/metrics.json
```
Bộ đánh giá tính đồng thời:
- **Phân loại:** Macro-F1 (5 nhãn), P/R/F1 từng nhãn, Binary F1 (nguy hại vs `AN_TOAN` để đối chiếu v1), `FPR` trên `AN_TOAN`, Bootstrap 95% CI, ECE/PR-AUC (khi có xác suất), và kiểm định McNemar ghép cặp (`--reference-pred`).
- **Bằng chứng:** Song song **F1 theo ý** ([DO_DO_BANG_CHUNG.md](docs/v2/DO_DO_BANG_CHUNG.md)) và **F1 mức từ**, tỉ lệ tô oan `AN_TOAN`, tỉ lệ cụm bịa không có trong câu (`non_verbatim_phrase_rate`).
- **Khái quát hóa theo nhóm (in-memory):** `C0 / C1 / C2` (kèm độ tụt $\Delta\text{F1}$), 3 vùng miền `bac / trung / nam` (kèm Worst-group F1), kịch bản `trong_phan_phoi` vs `chua_thay`, số lượt hội thoại, thẻ ca khó.

### 4. Chạy Web Demo on-device
```bash
cd demo
npm install && npm run dev
```

---

## Mốc đối chiếu SafeViChi v1 (Đóng băng)

Kết quả lớp `HATE` trên tập test v1 (`v1/results/step3/`, `v1/results/step4/`) dùng làm một dòng baseline cho v2:

| Hệ thống | C0 (sạch) | C1 (né lọc) | C2 (điểm mù) |
| --- | :---: | :---: | :---: |
| Blacklist từ khóa | 0.6551 | 0.5982 | 0.4564 |
| ViHateT5 gốc | 0.7683 | 0.6046 | 0.4627 |
| ViHateT5 gốc + Chuẩn hóa | 0.7497 | 0.7218 | 0.4720 |
| **SafeViChi v1 (Chuẩn hóa + FGM)** | **0.8552** | **0.8266** | **0.6657** |

---

## Tài liệu tham khảo

- **Vòng chung kết (v2):**
  - [Kế hoạch triển khai & ma trận thực nghiệm](docs/v2/KE_HOACH.md)
  - [Thẻ dữ liệu v2](data/v2/DATASET_CARD.md)
  - [Hướng dẫn gán nhãn 5 lớp](docs/v2/HUONG_DAN_GAN_NHAN.md)
  - [Độ đo bằng chứng — F1 theo ý](docs/v2/DO_DO_BANG_CHUNG.md)
  - [Hướng dẫn viết tập test](docs/v2/HUONG_DAN_VIET_TEST.md)
  - [Notebooks huấn luyện v2](notebooks/README.md)
- **Vòng loại (v1 — lưu trữ đối chiếu):**
  - [Tổng quan thư mục v1/](v1/README.md) · [Kiến trúc v1](v1/docs/README.md)
  - [Pipeline v1](v1/docs/pipeline.md) · [Thực nghiệm v1](v1/docs/experiments.md)

---

## Giấy phép

Phát hành theo giấy phép [MIT](LICENSE).
