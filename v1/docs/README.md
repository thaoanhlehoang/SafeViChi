# Tài liệu tổng quan SafeViChi

SafeViChi là hệ thống phát hiện nội dung độc hại tiếng Việt chạy on-device (trực tiếp trên thiết bị người dùng), có khả năng chống các kỹ thuật né bộ lọc như teencode, bỏ dấu, ký tự đồng dạng (lookalike/homoglyph), chèn ký tự phân tách và lỗi gõ.

Hệ thống hoạt động hoàn toàn ở client-side, không gửi dữ liệu văn bản của người dùng lên bất kỳ máy chủ nào.

---

## 1. Kiến trúc Pipeline 4 bước

Pipeline xử lý một chiều theo chuỗi:

```
Văn bản thô
    │
    ▼
[Bước 1] Chuẩn bị dữ liệu & sinh biến thể phi chuẩn
         - Corpus 75K mẫu (ViHSD, VOZ-HSD nhãn yếu, ViHOS)
         - 12 hiện tượng biến dạng có kiểm soát + chế độ mixed
    │
    ▼
[Bước 2] Lớp chuẩn hóa theo luật (Rule-based Normalizer)
         - 5 tầng: Unicode NFC -> Ký tự đồng dạng -> Xóa phân cách -> Gộp ký tự lặp -> Tra từ điển teencode
    │
    ▼
[Bước 3] Phân loại & Huấn luyện đối kháng (Adversarial Classifier)
         - Backbone ViHateT5 fine-tune với FGM (Fast Gradient Method)
         - Dữ liệu train trộn 60% biến thể đã chuẩn hóa / 30% câu sạch / 10% điểm mù
    │
    ▼
[Bước 4] Giải thích & Khoanh vùng vi phạm (Occlusion Explainer)
         - Che từng cụm từ (n-gram occlusion) đo mức sụt giảm logit margin
         - Chỉ điểm các cụm từ độc hại gây cảnh báo
```

---

## 2. Cấu trúc thư mục dự án

```
SafeViChi/
├── data/                       # Dữ liệu phục vụ huấn luyện và đánh giá
│   ├── processed/              # Bộ biến thể 75K (v1_2) và tập đã chuẩn hóa
│   ├── data_final/             # Bộ huấn luyện đã trộn (train, val_mixed, test)
│   ├── eval_baseline/          # Bộ benchmark 3 điều kiện (C0 câu sạch, C1 né lọc, C2 điểm mù)
│   ├── data_explain/           # Bộ dữ liệu ViHOS phục vụ đánh giá module giải thích
│   └── CHECKSUMS.json          # Bảng mã băm SHA-256 xác thực tính toàn vẹn dữ liệu
│
├── src/                        # Mã nguồn chính
│   ├── normalization/          # Lớp chuẩn hóa 5 tầng và từ điển teencode
│   ├── variant_generator/      # Bộ sinh 12 hiện tượng biến thể phi chuẩn
│   ├── dataset_builder/        # Pipeline dựng dữ liệu, ghép nhãn, chống rò rỉ (dedup)
│   ├── classifier/             # Huấn luyện ViHateT5 + đối kháng FGM
│   ├── baselines/              # Blacklist và ma trận đánh giá 6 hệ thống x 3 điều kiện
│   ├── explainer/              # Thuật toán occlusion khoanh vùng vi phạm
│   ├── reporting/              # Script vẽ biểu đồ và xuất bảng số liệu PDF
│   ├── utils/                  # Thiết lập seed cố định cho khả năng tái lập
│   └── pipeline.py             # Ghép toàn bộ quy trình thành một class duy nhất
│
├── demo/                       # Ứng dụng web chạy on-device (ONNX Runtime Web)
│   ├── index.html              # Giao diện demo kiểm tra tin nhắn
│   ├── teencode_dict.json      # Từ điển teencode cho phía browser
│   ├── js/                     # Bộ chuẩn hóa JS và logic điều khiển ONNX
│   └── onnx_model/             # Checkpoint ViHateT5 định dạng ONNX
│
├── models/                     # Thư mục chứa checkpoint PyTorch (vihatet5-v2-best)
├── results/                    # Kết quả thực nghiệm, ma trận số liệu và biểu đồ PDF
├── scripts/                    # Shell script điều phối dựng dữ liệu, eval và checksums
├── tests/                      # Unit test cho từng module
└── docs/                       # Tài liệu kỹ thuật
    ├── README.md               # Tài liệu tổng quan (file này)
    ├── pipeline.md             # Hướng dẫn chi tiết cách tái lập pipeline từ đầu đến cuối
    └── experiments.md          # Báo cáo thực nghiệm, số liệu đánh giá và kiểm định thống kê
```

---

## 3. Tóm tắt kết quả thực nghiệm

### Phân loại độc hại (Bước 3)

Đo bằng F1 của lớp HATE trên tập test, ngưỡng dò trên tập validation rồi đóng băng:

| Hệ thống | C0 (câu sạch) | C1 (né lọc) | C2 (điểm mù) |
|---|:---:|:---:|:---:|
| B1 · Blacklist từ khóa | 0.6551 | 0.5982 | 0.4564 |
| B2 · ViHateT5 gốc | 0.7683 | 0.6046 | 0.4627 |
| B2 · ViHateT5 gốc + chuẩn hóa | 0.7497 | 0.7218 | 0.4720 |
| **B3 · SafeViChi đầy đủ** | **0.8552** | **0.8266** | **0.6657** |

- Kỹ thuật né lọc làm ViHateT5 gốc tụt mạnh F1 từ `0.768` xuống `0.605`.
- Thêm lớp chuẩn hóa giúp bù lại `+0.117` F1 trên câu né lọc mà không ảnh hưởng câu sạch.
- Huấn luyện đối kháng (adversarial training) giúp tăng thêm `+0.105` F1 trên câu né lọc và giải quyết hiệu quả vùng điểm mù C2 (`+0.194` F1).

### Khoanh vùng giải thích (Bước 4)

Đo trên tập ViHOS test (531 câu HATE), đối chiếu span do thuật toán occlusion dự đoán với nhãn con người đã khoanh:

- **Micro F1**: `0.5455` (Precision `0.4478`, Recall `0.6977`)
- **Macro F1**: `0.5766`
- **Tỷ lệ trượt hoàn toàn**: chỉ `1.9%` (98.1% số câu tìm trúng ít nhất một phần vùng vi phạm).

---

## 4. Quy ước Seed & Kiểm tra toàn vẹn

- Khâu sinh biến thể (Bước 1): `seed = 20260902`
- Tất cả các khâu còn lại (dựng tập trộn, huấn luyện, dò ngưỡng): `seed = 42`
- Toàn bộ thiết lập ngẫu nhiên được quản lý tập trung qua `src/utils/seed.py`.

Để kiểm tra xem dữ liệu và checkpoint có bị chỉnh sửa ngoài ý muốn hay không, chạy:

```bash
python -m scripts.checksums
```

---

## 5. Xem thêm

- [Hướng dẫn tái lập pipeline](pipeline.md): Cách chuẩn bị dữ liệu, chạy huấn luyện, benchmark và dựng demo.
- [Báo cáo thực nghiệm chi tiết](experiments.md): Chi tiết kiểm định thống kê McNemar, phân tích lỗi và các bài học kỹ thuật.
