"""Load dữ liệu cho fine-tune ViHateT5.

Đọc bộ đã trộn sẵn 60/30/10 từ src/dataset_builder/build_mixed.py:

  60%  perturbed_normalized — câu bị biến thể rồi qua normalize() (residual noise)
  30%  clean_original       — câu gốc sạch, giữ năng lực xử lý input bình thường
  10%  blindspot_raw        — biến thể ngoài vùng phủ normalizer, cố ý KHÔNG
                              normalize (normalizer vốn không sửa được)

Mỗi dòng có 3 cột: text / label ("HATE"|"CLEAN") / source (tên nhóm ở trên).
KHÔNG gọi normalize() ở đây — mọi thứ đã xử lý xong ở khâu build dataset.
Cột source được giữ lại để báo cáo metric tách riêng theo từng nhóm lúc eval.
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset

LABELS = ("CLEAN", "HATE")
PROMPT_PREFIX = "vihsd: "


def read_jsonl(data_path: str | Path) -> Dataset:
    texts: list[str] = []
    labels: list[str] = []
    sources: list[str] = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            label = row["label"]
            if label not in LABELS:
                raise ValueError(f"Nhãn không hợp lệ: {label!r} (chỉ chấp nhận {LABELS})")
            texts.append(row["text"])
            labels.append(label)
            sources.append(row.get("source", "unknown"))
    return Dataset.from_dict({"text": texts, "label": labels, "source": sources})


def make_tokenize_fn(tokenizer, max_input_length: int, max_target_length: int):
    def tokenize_fn(batch: dict[str, list[str]]) -> dict[str, list]:
        inputs = [PROMPT_PREFIX + t for t in batch["text"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
        # T5 sentencepiece vocab của ViHateT5 chỉ có "hate"/"clean" (chữ thường) như
        # 1 token riêng biệt; viết hoa "HATE"/"CLEAN" cả hai đều rơi vào <unk> (đã
        # verify thực tế) nên model sẽ không phân biệt được 2 nhãn nếu tokenize hoa.
        target_texts = [label.lower() for label in batch["label"]]
        labels = tokenizer(text_target=target_texts, max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return tokenize_fn
