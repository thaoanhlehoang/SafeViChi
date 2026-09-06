"""Fine-tune ViHateT5 (seq2seq) với huấn luyện đối kháng 2 tầng:

  - Data-level: bộ trộn 60/30/10 dựng bởi src/dataset_builder/build_mixed.py
    (60% perturbed đã normalize / 30% câu gốc sạch / 10% biến thể điểm mù của
    normalizer). Mọi khâu xử lý văn bản đã xong ở bước build — KHÔNG normalize
    lại ở đây.
  - Gradient-level: FGM trên embedding "shared" của T5 (xem
    src/classifier/adversarial.py).

Input prompt: "vihsd: {text}" -> Target string: "hate" / "clean" (chữ thường —
tokenizer ViHateT5 map "HATE"/"CLEAN" viết hoa cả hai về cùng <unk>, đã verify
thực tế bằng tokenizer thật trước khi chốt).

Tính tái lập: `--seed` (mặc định 42) được nạp qua `src.utils.seed.set_global_seed`
NGAY dòng đầu của `main()`, trước khi load model hay dựng DataLoader — seed
python/numpy/torch/CUDA cùng lúc. Toàn bộ tham số của lần chạy được ghi ra
`{output_dir}/run_config.json` để đối chiếu về sau.

Chạy trên Kaggle (2x T4) — lần chạy đã tạo ra checkpoint hiện tại:
    python -m src.classifier.train \\
        --data_path   /kaggle/input/<ds>/train.jsonl \\
        --val_path    /kaggle/input/<ds>/validation_mixed.jsonl \\
        --test_path   /kaggle/input/<ds>/test.jsonl \\
        --output_dir  /kaggle/working/vihatet5-v2 \\
        --seed 42 --learning_rate 1e-5 --fgm_epsilon 0.3 \\
        --epochs 8 --early_stopping_patience 2 \\
        --per_device_train_batch_size 16 --fp16
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainingArguments,
)

from src.classifier.adversarial import FGMSeq2SeqTrainer
from src.classifier.data import read_jsonl, make_tokenize_fn
from src.classifier.metrics import build_compute_metrics, report
from src.utils.seed import SEED_DEFAULT, set_global_seed

MODEL_NAME = "tarudesu/ViHateT5-base-HSD"


def build_training_args(desired: dict, **fallbacks: str) -> Seq2SeqTrainingArguments:
    """Chỉ truyền các kwargs mà bản transformers đang cài đặt thực sự hỗ trợ.

    Tên tham số của TrainingArguments đổi qua các version (vd. evaluation_strategy
    -> eval_strategy). `fallbacks` map tên cũ -> tên mới để thử khi tên chính bị
    thiếu. Tham số nào không tồn tại ở cả hai tên sẽ bị bỏ qua kèm cảnh báo, thay
    vì làm crash script trên một phiên bản transformers khác với lúc viết code.
    """
    supported = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    kwargs = {}
    for key, value in desired.items():
        if key in supported:
            kwargs[key] = value
        elif key in fallbacks and fallbacks[key] in supported:
            kwargs[fallbacks[key]] = value
        else:
            print(f"  [!] Bỏ qua tham số không được hỗ trợ bởi transformers hiện tại: {key}={value!r}")
    return Seq2SeqTrainingArguments(**kwargs)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune ViHateT5 với huấn luyện đối kháng (FGM)")
    parser.add_argument("--data_path", type=str, default="data/data_final/train.jsonl",
                         help="bộ trộn 60/30/10, 116.943 dòng")
    parser.add_argument("--val_path", type=str, default="data/data_final/validation_mixed.jsonl",
                         help="dùng cho early stopping. Mặc định là bản MIX (4.176 dòng) chứ "
                              "không phải validation.jsonl (C0 câu sạch): early-stop phải nhìn "
                              "cùng loại phân bố với dữ liệu train, nếu chỉ nhìn câu sạch thì "
                              "không phát hiện được lúc model bắt đầu hỏng trên câu bị né lọc.")
    parser.add_argument("--test_path", type=str, default="data/data_final/test.jsonl",
                         help="eval lần cuối sau khi train xong (C0 câu sạch). Để rỗng ('') để bỏ qua. "
                              "Đánh giá đầy đủ 3 điều kiện nằm ở src/baselines/run_matrix.py.")
    parser.add_argument("--output_dir", type=str, default="models/vihatet5-v2")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--deterministic", action="store_true",
                         help="bật chế độ tất định của cuDNN/cuBLAS. Chậm hơn đáng kể; "
                              "chỉ dùng khi cần chứng minh tái lập, không dùng khi train thật.")
    parser.add_argument("--max_input_length", type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                         help="1e-5: đang fine-tune TIẾP một checkpoint đã fine-tune sẵn "
                              "(ViHateT5-base-HSD), không phải train từ đầu. Mức 3e-4 dùng "
                              "trước đây quá cao — log cho thấy val F1 dao động "
                              "0.69→0.79→0.67→0.75 rồi early-stop ở epoch 4, tức chưa hội tụ.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--epochs", type=float, default=8,
                         help="trần số epoch; early stopping thường dừng sớm hơn nhiều")
    parser.add_argument("--early_stopping_patience", type=int, default=2,
                         help="dừng khi F1 validation không cải thiện sau ngần này epoch. 0 = tắt")
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--fp16", action="store_true", default=torch.cuda.is_available())
    parser.add_argument("--no_fgm", action="store_true", help="tắt FGM, chỉ giữ data-level adversarial")
    parser.add_argument("--fgm_epsilon", type=float, default=0.3,
                         help="0.3: epsilon=1.0 cộng với LR cao gây nhiễu kép, model không "
                              "ổn định nổi. Hạ xuống để tấn công gradient vẫn có tác dụng "
                              "mà không phá trọng số đã học tốt.")
    parser.add_argument("--fgm_emb_name", type=str, default="shared")
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    # Seed TRƯỚC mọi thứ khác: trước khi load model, trước khi dựng DataLoader.
    seed_report = set_global_seed(args.seed, deterministic=args.deterministic)

    print(f"[1/6] Load train={args.data_path} val={args.val_path}")
    train_dataset = read_jsonl(args.data_path)
    val_dataset = read_jsonl(args.val_path)
    for name, ds in (("train", train_dataset), ("validation", val_dataset)):
        counts = {s: ds["source"].count(s) for s in sorted(set(ds["source"]))}
        print(f"  {name}: {len(ds)} {counts}")
    eval_sources = list(val_dataset["source"])

    print(f"[2/6] Load tokenizer/model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    print("[3/6] Tokenize (prompt 'vihsd: {text}' -> target 'hate'/'clean')")
    tokenize_fn = make_tokenize_fn(tokenizer, args.max_input_length, args.max_target_length)
    remove_columns = ["text", "label", "source"]
    tokenized_train = train_dataset.map(tokenize_fn, batched=True, remove_columns=remove_columns)
    tokenized_val = val_dataset.map(tokenize_fn, batched=True, remove_columns=remove_columns)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, label_pad_token_id=-100)

    training_args = build_training_args(
        dict(
            output_dir=args.output_dir,
            seed=args.seed,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.epochs,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=args.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            predict_with_generate=True,
            generation_max_length=args.max_target_length,
            logging_steps=args.logging_steps,
            fp16=args.fp16,
            report_to="none",
        ),
        eval_strategy="evaluation_strategy",
    )

    use_fgm = not args.no_fgm
    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))
    print(f"[4/6] Khởi tạo Trainer (FGM={'bật, epsilon=' + str(args.fgm_epsilon) if use_fgm else 'tắt'}"
          f", early_stopping_patience={args.early_stopping_patience})")
    trainer = FGMSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=build_compute_metrics(tokenizer, eval_sources),
        callbacks=callbacks,
        use_fgm=use_fgm,
        fgm_epsilon=args.fgm_epsilon,
        fgm_emb_name=args.fgm_emb_name,
    )

    # [5/6] Đóng băng cấu hình lần chạy này ra đĩa TRƯỚC khi train, để nếu job
    # bị Kaggle cắt giữa chừng thì vẫn còn bằng chứng đã chạy với tham số nào.
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "args": vars(args),
        "seed_report": seed_report.as_dict(),
        "dataset_sizes": {
            "train": len(train_dataset),
            "validation": len(val_dataset),
        },
        "label_target_strings": {"HATE": "hate", "CLEAN": "clean"},
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)
    print(f"[5/6] Đã ghi cấu hình lần chạy -> {out_dir / 'run_config.json'}")

    print("[6/6] Bắt đầu train ...")
    trainer.train()

    print(f"Lưu checkpoint tốt nhất vào {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    report("Validation (best checkpoint)", trainer.evaluate())

    if args.test_path:
        test_dataset = read_jsonl(args.test_path)
        counts = {s: test_dataset["source"].count(s) for s in sorted(set(test_dataset["source"]))}
        print(f"\nEval trên test: {args.test_path} ({len(test_dataset)} dòng, {counts})")
        trainer.compute_metrics = build_compute_metrics(tokenizer, list(test_dataset["source"]))
        tokenized_test = test_dataset.map(tokenize_fn, batched=True, remove_columns=remove_columns)
        report("Test (best checkpoint)", trainer.evaluate(eval_dataset=tokenized_test, metric_key_prefix="test"))


if __name__ == "__main__":
    main()
