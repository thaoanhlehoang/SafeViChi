"""
Fine-tune ViSoBERT cho bài toán phân loại 3 lớp (sạch/xúc phạm/thù ghét).

Tương ứng Mục 3.5 trong docs/plan.md.

Baseline liên quan (Mục 4.1):
  - Baseline 3: fine-tune KHÔNG có lớp chuẩn hóa, KHÔNG huấn luyện đối kháng
                -> chạy script này trực tiếp trên ViHSD gốc
  - Baseline 4: fine-tune CÓ chuẩn hóa (src/normalization) + CÓ huấn luyện
                đối kháng (thêm dữ liệu từ src/variant_generator vào tập train)
                -> chạy script này với --use_normalization --use_adversarial

Model card: https://huggingface.co/uitnlp/visobert
"""

from __future__ import annotations
import argparse

MODEL_NAME = "uitnlp/visobert"
NUM_LABELS = 3  # 0 = clean, 1 = offensive, 2 = hate
LABEL_NAMES = ["clean", "offensive", "hate"]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune ViSoBERT trên ViHSD")
    parser.add_argument("--data_path", type=str, default="data/processed/vihsd_train.csv")
    parser.add_argument("--output_dir", type=str, default="models/visobert-classifier")
    parser.add_argument("--use_normalization", action="store_true",
                         help="Đi qua lớp chuẩn hóa (src/normalization) trước khi tokenize")
    parser.add_argument("--use_adversarial", action="store_true",
                         help="Bổ sung dữ liệu biến thể (src/variant_generator) vào tập train")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    return parser


def main():
    args = build_argparser().parse_args()

    # TODO 1: load ViHSD từ args.data_path (pandas / datasets.load_dataset)
    # TODO 2: nếu args.use_normalization, áp dụng src.normalization.normalizer.normalize
    #         lên toàn bộ văn bản trước khi tokenize
    # TODO 3: nếu args.use_adversarial, dùng src.variant_generator để sinh thêm
    #         dữ liệu biến thể (giữ nguyên nhãn gốc) và trộn vào tập train
    # TODO 4: xử lý mất cân bằng nhãn (ViHSD lệch mạnh về "clean")
    #         -> class weights trong CrossEntropyLoss, hoặc oversample nhãn thiểu số
    # TODO 5: load AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,
    #         num_labels=NUM_LABELS), AutoTokenizer.from_pretrained(MODEL_NAME)
    # TODO 6: Trainer / training loop chuẩn: AdamW, warmup + linear decay,
    #         learning_rate=args.learning_rate, batch_size=args.batch_size,
    #         epochs=args.epochs
    # TODO 7: lưu checkpoint vào args.output_dir

    raise NotImplementedError(
        "Điền các TODO ở trên để hoàn thiện script fine-tune. "
        "Xem docs/plan.md Mục 3.5 và 4.1 để biết rõ 4 baseline cần chạy."
    )


if __name__ == "__main__":
    main()
