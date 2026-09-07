"""Chạy toàn bộ ma trận baseline Step 3: {hệ thống} x {điều kiện input}.

Nguyên tắc thiết kế (đọc kỹ trước khi sửa):

  1. MỌI hệ thống nhận CÙNG một chuỗi đầu vào trong cùng một điều kiện.
     normalize() là BỘ PHẬN BÊN TRONG hệ thống B3, chạy lúc eval, KHÔNG nướng
     sẵn vào file dữ liệu. Nếu cho B1/B2 ăn câu sạch còn B3 ăn câu đã chuẩn
     hoá thì đã đổi 2 biến cùng lúc (model + input) và kết quả không diễn giải
     được.

  2. Giao thức decode ĐỒNG NHẤT cho B2 và B3: forced-decode 1 bước, so logit
     của token "hate" (9267) với "clean" (6270) tại vị trí decoder đầu tiên,
     P(hate) = softmax trên đúng 2 logit đó. Không dùng generate() tự do vì
     model CHƯA fine-tune (B2) hay sinh ra chuỗi ngoài 2 nhãn, khiến việc quy
     kết "sai" trở nên tuỳ tiện và làm B2 thiệt một cách giả tạo.

  3. KHÔNG báo cáo accuracy làm số chính: đoán bừa CLEAN hết đã được 81.8%.
     Số chính là F1 lớp HATE, kèm khoảng tin cậy bootstrap.

Ma trận hệ thống:
  b1_blacklist      blacklist(raw)            — sàn, không model
  b1_blacklist_norm blacklist(normalize(raw)) — ablation: normalizer nâng cả hệ ngây thơ
  b2_t5base         T5 gốc(raw)               — mức sập khi bị tấn công
  b2_t5base_norm    T5 gốc(normalize(raw))    — ablation: normalizer đóng góp bao nhiêu
  b3_t5ft           T5 finetune(raw)          — ablation: finetune đơn lẻ
  b3_t5ft_norm      T5 finetune(normalize)    — PIPELINE ĐẦY ĐỦ

Điều kiện input (từ src/dataset_builder/build_eval_matrix.py):
  C0_clean / C1_perturbed / C2_blindspot

Chạy:
    python -m src.baselines.run_matrix --checkpoint models/vihatet5-v2-best
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, accuracy_score)
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.baselines.blacklist import predict as blacklist_predict
from src.baselines.stats import bootstrap_ci, mcnemar
from src.normalization.normalizer import normalize
from src.utils.seed import SEED_DEFAULT, set_global_seed

PROMPT_PREFIX = "vihsd: "
LABEL2ID = {"CLEAN": 0, "HATE": 1}
CONDITIONS = ("C0_clean", "C1_perturbed", "C2_blindspot")
CONDITION_FILES = {
    "C0_clean": "C0_clean.jsonl",
    "C1_perturbed": "C1_perturbed.jsonl",
    "C2_blindspot": "C2_blindspot.jsonl",
}


def read_condition(path: Path) -> tuple[list[str], np.ndarray, list[str]]:
    texts, labels, row_ids = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            texts.append(d["text"])
            labels.append(LABEL2ID[d["label"]])
            row_ids.append(d["row_id"])
    return texts, np.array(labels), row_ids


class T5Scorer:
    """Forced-decode 1 bước -> P(hate). Dùng chung cho B2 và B3."""

    def __init__(self, model_name: str, device: str, batch_size: int = 64,
                 max_input_length: int = 256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device).eval()
        self.device = device
        self.batch_size = batch_size
        self.max_input_length = max_input_length
        self.id_hate = self.tokenizer(text_target="hate")["input_ids"][0]
        self.id_clean = self.tokenizer(text_target="clean")["input_ids"][0]
        if self.id_hate == self.id_clean:
            raise RuntimeError(
                f"Token 'hate' và 'clean' trùng id ({self.id_hate}) — hai nhãn sẽ không "
                "phân biệt được. Kiểm tra lại tokenizer/chữ hoa-thường."
            )
        self.decoder_start = (self.model.config.decoder_start_token_id
                              if self.model.config.decoder_start_token_id is not None
                              else self.model.config.pad_token_id)

    @torch.no_grad()
    def score(self, texts: list[str]) -> np.ndarray:
        out = np.empty(len(texts), dtype=np.float64)
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            # BẮT BUỘC hạ chữ thường: từ điển sentencepiece của ViHateT5 KHÔNG có
            # ký tự hoa nào — mọi chữ hoa thành <unk> (đo được: 85% câu trong tập
            # eval có chữ hoa, trung bình 10,3% token bị vứt; câu viết HOA toàn bộ
            # thì model không đọc được gì và luôn trả cùng một đáp án mặc định).
            # Chữ hoa vốn mang zero thông tin tới model nên hạ thường không mất gì.
            # Blacklist (src/baselines/blacklist.py) vốn đã hạ thường sẵn, nên đây
            # cũng là điều kiện công bằng giữa các hệ thống.
            enc = self.tokenizer([PROMPT_PREFIX + t.lower() for t in batch],
                                 max_length=self.max_input_length, truncation=True,
                                 padding=True, return_tensors="pt").to(self.device)
            dec = torch.full((len(batch), 1), self.decoder_start,
                             dtype=torch.long, device=self.device)
            logits = self.model(**enc, decoder_input_ids=dec).logits[:, 0, :]
            pair = torch.stack([logits[:, self.id_clean], logits[:, self.id_hate]], dim=-1)
            out[start:start + len(batch)] = torch.softmax(pair.float(), dim=-1)[:, 1].cpu().numpy()
        return out


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    scores: np.ndarray | None) -> dict:
    m = {
        "n": int(len(y_true)),
        "n_hate": int(y_true.sum()),
        "hate_f1": float(f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)),
        "hate_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "hate_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        # accuracy để tham khảo, KHÔNG dùng làm số chính (đoán bừa CLEAN = 81.8%)
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "predicted_hate_rate": float(y_pred.mean()),
    }
    m.update(bootstrap_ci(y_true, y_pred))
    # PR-AUC chỉ có với hệ thống cho ra điểm số liên tục; blacklist ra quyết
    # định cứng nên chỉ là 1 điểm trên mặt phẳng PR, không có đường cong.
    m["pr_auc"] = float(average_precision_score(y_true, scores)) if scores is not None else None
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description="Ma trận baseline Step 3")
    parser.add_argument("--data_dir", type=str, default="data/eval_baseline/test",
                        help="thư mục chứa C0_clean/C1_perturbed/C2_blindspot.jsonl. "
                             "Dùng data/eval_baseline/val để dò ngưỡng.")
    parser.add_argument("--checkpoint", type=str, default="models/vihatet5-v2-best",
                        help="B3: checkpoint đã fine-tune theo pipeline")
    parser.add_argument("--base_model", type=str, default="tarudesu/ViHateT5-base-HSD",
                        help="B2: T5 gốc, chưa fine-tune thêm")
    parser.add_argument("--wordlist_path", type=str,
                        default="src/baselines/blacklist_words.json")
    parser.add_argument("--out_dir", type=str, default="results/step3/test")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT,
                        help="seed cho bootstrap khoảng tin cậy")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0,
                        help=">0 để smoke-test trên N dòng đầu mỗi điều kiện")
    parser.add_argument("--dump_scores", action="store_true",
                        help="ghi P(hate) từng dòng ra scores.jsonl. Cho phép quét ngưỡng "
                             "offline trên CPU về sau mà không phải chạy lại GPU.")
    args = parser.parse_args()

    set_global_seed(args.seed)

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Thiết bị: {args.device}")

    conditions = {}
    for cond in CONDITIONS:
        path = data_dir / CONDITION_FILES[cond]
        if not path.exists():
            print(f"  [bỏ qua] {cond}: không thấy {path}")
            continue
        texts, y, row_ids = read_condition(path)
        if args.limit:
            texts, y, row_ids = texts[:args.limit], y[:args.limit], row_ids[:args.limit]
        conditions[cond] = {"texts": texts, "y": y, "row_ids": row_ids,
                            "norm_texts": [normalize(t) for t in texts]}
        print(f"  {cond:15s} {len(texts):5d} dòng, {int(y.sum())} HATE "
              f"({y.mean() * 100:.1f}%) — {path}")

    wordlist = set(json.load(open(args.wordlist_path, encoding="utf-8")))
    print(f"Blacklist: {len(wordlist)} từ khoá")

    results: dict[str, dict] = {}
    preds_store: dict[str, dict[str, np.ndarray]] = {}
    scores_store: dict[str, dict[str, np.ndarray]] = {}

    def run_system(name: str, fn, uses_norm: bool, has_scores: bool) -> None:
        results[name] = {}
        preds_store[name] = {}
        scores_store[name] = {}
        for cond, data in conditions.items():
            texts = data["norm_texts"] if uses_norm else data["texts"]
            t0 = time.time()
            y_pred, scores = fn(texts)
            results[name][cond] = compute_metrics(data["y"], y_pred, scores)
            results[name][cond]["seconds"] = round(time.time() - t0, 1)
            preds_store[name][cond] = y_pred
            if scores is not None:
                scores_store[name][cond] = scores
            r = results[name][cond]
            print(f"  {name:20s} {cond:15s} hate_F1={r['hate_f1']:.4f} "
                  f"[{r['f1_ci_low']:.3f}-{r['f1_ci_high']:.3f}]  "
                  f"P={r['hate_precision']:.3f} R={r['hate_recall']:.3f} "
                  f"macroF1={r['macro_f1']:.4f}")
        _ = has_scores

    def blacklist_fn(texts: list[str]):
        y_pred = np.array([LABEL2ID[blacklist_predict(t, wordlist)] for t in texts])
        return y_pred, None

    print("\n=== B1 blacklist (không model) ===")
    run_system("b1_blacklist", blacklist_fn, uses_norm=False, has_scores=False)
    run_system("b1_blacklist_norm", blacklist_fn, uses_norm=True, has_scores=False)

    for tag, model_name in (("b2_t5base", args.base_model), ("b3_t5ft", args.checkpoint)):
        print(f"\n=== {tag}: {model_name} ===")
        scorer = T5Scorer(model_name, args.device, args.batch_size)

        def t5_fn(texts: list[str], _s=scorer):
            scores = _s.score(texts)
            return (scores > 0.5).astype(int), scores

        run_system(tag, t5_fn, uses_norm=False, has_scores=True)
        run_system(f"{tag}_norm", t5_fn, uses_norm=True, has_scores=True)
        del scorer
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # So cặp McNemar trên cùng các dòng: mỗi cặp trả lời đúng 1 câu hỏi.
    pairs = [
        ("b2_t5base", "b2_t5base_norm", "normalizer đóng góp bao nhiêu (Step 2)"),
        ("b2_t5base_norm", "b3_t5ft_norm", "fine-tune đối kháng đóng góp bao nhiêu (Step 3)"),
        ("b2_t5base", "b3_t5ft_norm", "pipeline đầy đủ hơn T5 gốc bao nhiêu"),
        ("b3_t5ft", "b3_t5ft_norm", "model đã fine-tune còn cần normalizer không"),
        ("b1_blacklist", "b3_t5ft_norm", "pipeline hơn cách ngây thơ nhất bao nhiêu"),
    ]
    print("\n=== McNemar (ghép cặp cùng dòng) ===")
    comparisons = []
    for a, b, question in pairs:
        if a not in preds_store or b not in preds_store:
            continue
        for cond, data in conditions.items():
            res = mcnemar(data["y"], preds_store[a][cond], preds_store[b][cond])
            res.update({"system_a": a, "system_b": b, "condition": cond,
                        "question": question,
                        "hate_f1_a": results[a][cond]["hate_f1"],
                        "hate_f1_b": results[b][cond]["hate_f1"]})
            comparisons.append(res)
            mark = "có ý nghĩa" if res["significant"] else "KHÔNG có ý nghĩa"
            print(f"  {cond:15s} {a:18s} vs {b:18s} "
                  f"ΔF1={res['hate_f1_b'] - res['hate_f1_a']:+.4f} p={res['p_value']:.2e} {mark}")

    payload = {"args": vars(args), "results": results, "comparisons": comparisons}
    with open(out_dir / "matrix.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines = ["| Hệ thống | Điều kiện | hate F1 | KTC 95% | P | R | macro F1 | PR-AUC | acc |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, per_cond in results.items():
        for cond, m in per_cond.items():
            pr = f"{m['pr_auc']:.4f}" if m["pr_auc"] is not None else "—"
            lines.append(
                f"| {name} | {cond} | **{m['hate_f1']:.4f}** | "
                f"{m['f1_ci_low']:.3f}–{m['f1_ci_high']:.3f} | {m['hate_precision']:.3f} | "
                f"{m['hate_recall']:.3f} | {m['macro_f1']:.4f} | {pr} | {m['accuracy']:.3f} |")
    (out_dir / "matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nĐã ghi {out_dir / 'matrix.json'} và {out_dir / 'matrix.md'}")

    if args.dump_scores:
        # Một dòng / (hệ thống, điều kiện, câu). Đủ để quét lại ngưỡng offline
        # trên CPU mà không cần chạm GPU lần nữa. Chỉ hệ thống cho điểm liên
        # tục (T5) mới có mặt ở đây — blacklist ra quyết định cứng.
        scores_path = out_dir / "scores.jsonl"
        n = 0
        with open(scores_path, "w", encoding="utf-8") as f:
            for name, per_cond in scores_store.items():
                for cond, scores in per_cond.items():
                    data = conditions[cond]
                    for row_id, y, p in zip(data["row_ids"], data["y"], scores):
                        f.write(json.dumps({"system": name, "condition": cond,
                                            "row_id": row_id, "label": int(y),
                                            "p_hate": float(p)}) + "\n")
                        n += 1
        print(f"Đã ghi {scores_path} ({n} dòng)")


if __name__ == "__main__":
    main()
