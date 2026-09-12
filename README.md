# SafeViChi

Hệ thống phát hiện nội dung độc hại tiếng Việt chống các kỹ thuật né bộ lọc, xử lý hoàn toàn cục bộ trên thiết bị (on-device).

---

## Luận điểm cốt lõi

Các mô hình phát hiện nội dung độc hại tiếng Việt bị sụt giảm nghiêm trọng hiệu năng khi người dùng cố tình biến dạng chữ viết để lách luật (teencode, bỏ dấu, chèn ký tự phân cách, homoglyph, lỗi gõ). SafeViChi kết hợp lớp chuẩn hóa 5 tầng (rule-based normalizer) cùng cơ chế huấn luyện đối kháng (adversarial training) giúp phục hồi hầu hết hiệu năng bị mất mà vẫn đảm bảo tính riêng tư tuyệt đối cho người dùng.

```
Văn bản đầu vào
    → [1] Lớp chuẩn hóa (rule-based normalizer)       src/normalization/
    → [2] Bộ phân loại (ViHateT5 fine-tune đối kháng) src/classifier/
    → [3] Module giải thích (occlusion-based)         src/explainer/
    → Đầu ra: Nhãn rủi ro (hate/clean) + Cụm từ cảnh báo
```

---

## Cấu trúc thư mục

```
SafeViChi/
├── data/                       # Dữ liệu huấn luyện, kiểm thử và ma trận benchmark
│   ├── processed/              # Tập biến thể 75K và tập dữ liệu đã chuẩn hóa
│   ├── data_final/             # Bộ train trộn 60/30/10, validation_mixed và test
│   ├── eval_baseline/          # Ma trận đánh giá 3 điều kiện (C0, C1, C2)
│   ├── data_explain/           # Tập dữ liệu ViHOS phục vụ module giải thích
│   └── CHECKSUMS.json          # Bảng mã băm SHA-256 đối chiếu toàn vẹn dữ liệu
│
├── src/                        # Mã nguồn triển khai
│   ├── normalization/          # Bộ chuẩn hóa 5 tầng và từ điển teencode
│   ├── variant_generator/      # Trình sinh biến thể 12 hiện tượng phi chuẩn
│   ├── dataset_builder/        # Pipeline tiền xử lý, ghép nhãn, khử trùng lặp
│   ├── classifier/             # Huấn luyện mô hình ViHateT5 và đối kháng FGM
│   ├── baselines/              # Blacklist và ma trận benchmark đánh giá
│   ├── explainer/              # Thuật toán occlusion khoanh vùng vi phạm
│   ├── reporting/              # Script kết xuất biểu đồ và số liệu
│   └── pipeline.py             # Lớp đóng gói toàn bộ quy trình đầu-cuối
│
├── demo/                       # Ứng dụng web chạy trực tiếp trong trình duyệt
│   ├── index.html              # Giao diện kiểm tra tin nhắn
│   ├── js/                     # Logic chuẩn hóa và suy luận ONNX Web
│   └── onnx_model/             # Checkpoint ViHateT5 đã export ONNX
│
├── models/                     # Checkpoint mô hình PyTorch (vihatet5-v2-best)
├── results/                    # Báo cáo số liệu thực nghiệm và biểu đồ PDF
├── scripts/                    # Script điều phối dựng dữ liệu, eval và checksums
├── tests/                      # Bộ kiểm thử tự động (unit tests)
└── docs/                       # Tài liệu kỹ thuật chi tiết
    ├── README.md               # Cẩm nang tổng quan kiến trúc và hệ thống
    ├── pipeline.md             # Hướng dẫn từng bước tái lập pipeline từ A-Z
    └── experiments.md          # Báo cáo số liệu thực nghiệm và kiểm định thống kê
```

---

## Cài đặt & Khởi động nhanh

1. **Khởi tạo môi trường:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate    # Trên Windows: .venv\Scripts\activate
   pip install -r requirements-lock.txt
   ```

2. **Chạy kiểm thử:**
   ```bash
   pytest -q                    # Toàn bộ hơn 100 test case
   python -m scripts.checksums  # Kiểm tra tính toàn vẹn của dữ liệu và model
   ```

3. **Chạy thử Web Demo on-device:**
   ```bash
   cd demo
   python -m http.server 8000
   # Mở trình duyệt tại http://localhost:8000
   ```
   Toàn bộ quá trình chuẩn hóa, tokenization và suy luận mô hình diễn ra 100% bằng WebAssembly/ONNX Runtime Web trong trình duyệt của bạn.

---

## Kết quả chính

F1 của lớp HATE trên tập test qua 3 điều kiện (C0: sạch, C1: né lọc, C2: điểm mù):

| Hệ thống | C0 (câu sạch) | C1 (né lọc) | C2 (điểm mù) |
|---|:---:|:---:|:---:|
| Blacklist từ khóa | 0.6551 | 0.5982 | 0.4564 |
| ViHateT5 gốc | 0.7683 | 0.6046 | 0.4627 |
| ViHateT5 gốc + Chuẩn hóa | 0.7497 | 0.7218 | 0.4720 |
| **SafeViChi đầy đủ (Chuẩn hóa + Đối kháng)** | **0.8552** | **0.8266** | **0.6657** |

Khả năng khoanh vùng giải thích trên tập ViHOS test: **Micro F1 đạt 0.5455**, tỷ lệ trượt hoàn toàn chỉ **1.9%**.

---

## Tài liệu chi tiết

- [Tổng quan kiến trúc & Cấu trúc dự án](docs/README.md)
- [Cẩm nang tái lập Pipeline từ đầu đến cuối](docs/pipeline.md)
- [Báo cáo thực nghiệm & Phân tích chi tiết](docs/experiments.md)

---

## Giấy phép

Phát hành theo giấy phép [MIT](LICENSE).
