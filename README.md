# SafeViChi

Hệ thống phát hiện nội dung độc hại tiếng Việt chống né bộ lọc, xử lý cục bộ trên thiết bị.

> Bản demo chứng minh khái niệm (proof-of-concept) cho cuộc thi sáng tạo. Xem `docs/plan.md` để biết đầy đủ bối cảnh, ràng buộc thiết kế, và phạm vi (trong/ngoài demo).

## Luận điểm cốt lõi

Mô hình phân loại nội dung độc hại tiếng Việt sụp giảm hiệu năng khi văn bản bị biến dạng có chủ đích để né bộ lọc. Thêm một lớp chuẩn hóa và huấn luyện đối kháng phục hồi được phần lớn hiệu năng đã mất — toàn bộ chạy được ngay trên thiết bị, không gửi nội dung của trẻ lên bất kỳ máy chủ nào.

## Pipeline

```
Văn bản đầu vào
    → [1] Lớp chuẩn hóa (rule-based)         src/normalization/
    → [2] Bộ phân loại (ViSoBERT fine-tune)   src/classifier/
    → [3] Module giải thích (occlusion)       src/explainer/
    → Đầu ra: nhãn rủi ro + cụm từ gây cảnh báo
```

Bộ sinh biến thể né lọc (6 nhóm, dùng cho benchmark và huấn luyện đối kháng) nằm ở `src/variant_generator/`.

## Cấu trúc thư mục

```
safevichi/
├── data/
│   ├── raw/            # ViHSD, ViHOS gốc (không commit dữ liệu thật lên git)
│   ├── processed/      # Dữ liệu đã tiền xử lý
│   └── variants/       # Dữ liệu biến thể sinh ra từ variant_generator
├── src/
│   ├── normalization/      # Lớp chuẩn hóa rule-based (Mục 3.4 trong plan)
│   ├── variant_generator/  # 6 nhóm biến thể né lọc (Mục 3.3)
│   ├── classifier/         # Fine-tune ViSoBERT + huấn luyện đối kháng (Mục 3.5)
│   ├── explainer/          # Module giải thích occlusion-based (Mục 3.6)
│   └── pipeline.py         # Ghép toàn bộ pipeline một đường thẳng
├── notebooks/          # Notebook thử nghiệm, phân tích dữ liệu, vẽ biểu đồ đánh giá
├── demo/               # Web demo chạy on-device (ONNX Runtime Web)
├── docs/               # Tài liệu dự án (plan, baseline, kết quả)
├── models/             # Checkpoint mô hình đã huấn luyện (gitignore, không commit)
└── tests/              # Unit test cho từng module
```

## Cài đặt

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Dữ liệu

- **ViHSD**: https://huggingface.co/datasets/uitnlp/vihsd — nguồn train/eval chính cho bộ phân loại
- **ViHOS**: https://huggingface.co/datasets/htdung167/ViHOS — đối chiếu, đánh giá module giải thích
- **ViLexNorm**: nguồn tham khảo để xây từ điển teencode/chuẩn hóa (xem `docs/plan.md` Mục 3 phần làm rõ)

Tải dữ liệu về `data/raw/` (không commit dữ liệu thô lên GitHub, xem `.gitignore`).

## Baseline

| STT | Baseline                                              | Mục đích                    |
| --- | ----------------------------------------------------- | --------------------------- |
| 1   | Bộ lọc từ khóa (blacklist)                            | Mốc dưới cùng               |
| 2   | XLM-R fine-tune trên ViHSD                            | Đại diện giải pháp quốc tế  |
| 3   | ViSoBERT fine-tune, không chuẩn hóa, không đối kháng  | Đo mức sụp hiệu năng        |
| 4   | ViSoBERT fine-tune + chuẩn hóa + huấn luyện đối kháng | Hệ đầy đủ (kết quả đề xuất) |

## Demo

Xem `demo/README.md` để chạy demo web on-device (không cần backend, không cần server).

## Tài liệu

Toàn bộ kế hoạch dự án chi tiết: [`docs/plan.md`](docs/plan.md)

## Giấy phép

MIT — xem [`LICENSE`](LICENSE)
