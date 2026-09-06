"""Eval-only cho bất kỳ checkpoint ViHateT5 nào (local hoặc HF hub id) trên 1
file jsonl (text/label, cột source tùy chọn) — không train gì cả.

Dùng cho:
  - B3: eval thẳng `tarudesu/ViHateT5-base-HSD` GỐC (chưa fine-tune) trên câu
    RAW (chưa qua normalize()) để đo mức tụt performance khi thiếu cả normalize
    lẫn adversarial training.
  - B2: eval checkpoint đã train trên ViHSD sạch (baseline không adversarial).
  - Kiểm tra lại checkpoint B4 trên bất kỳ file test nào khác không cần train lại.

Chạy:
    python -m src.classifier.evaluate \
        --model_name tarudesu/ViHateT5-base-HSD \
        --data_path data/data_final/test.jsonl \
        --label b3-no-normalization-no-adversarial
"""

from __future__ import annotations

import argparse

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

from src.classifier.data import read_jsonl, make_tokenize_fn
from src.classifier.metrics import build_compute_metrics, report


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Eval-only 1 checkpoint ViHateT5 trên 1 file jsonl")
    parser.add_argument("--model_name", type=str, required=True,
                         help="HF hub id (vd. tarudesu/ViHateT5-base-HSD) hoặc đường dẫn checkpoint local")
    parser.add_argument("--data_path", type=str, required=True, help="jsonl có cột text/label, source tùy chọn")
    parser.add_argument("--label", type=str, default="eval", help="tên hiển thị trong log, không ảnh hưởng logic")
    parser.add_argument("--max_input_length", type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    print(f"[{args.label}] Load model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    print(f"[{args.label}] Load data: {args.data_path}")
    dataset = read_jsonl(args.data_path)
    counts = {s: dataset["source"].count(s) for s in sorted(set(dataset["source"]))}
    print(f"  {len(dataset)} dòng {counts}")

    tokenize_fn = make_tokenize_fn(tokenizer, args.max_input_length, args.max_target_length)
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text", "label", "source"])
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, label_pad_token_id=-100)

    training_args = Seq2SeqTrainingArguments(
        output_dir="/tmp/eval_scratch",
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        report_to="none",
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=build_compute_metrics(tokenizer, list(dataset["source"])),
    )

    metrics = trainer.evaluate(eval_dataset=tokenized)
    report(f"{args.label} ({args.model_name} trên {args.data_path})", metrics)


if __name__ == "__main__":
    main()
