"""Bước 4.4 — demo trực quan: giải thích 1 câu bất kỳ bằng occlusion.

Chạy occlusion trên câu người dùng nhập, in ra câu với từ được tô đậm theo
mức "gây cảnh báo". Chỉ để soi ĐỊNH TÍNH — số liệu định lượng nằm ở 4.3
(`evaluate_localization.py`, kết quả trong `results/step4/`).

Mặc định dùng ĐÚNG cấu hình chính thức đã chốt ở 4.3 (relative + window 2 +
ngưỡng 0.1) để những gì nhìn thấy ở đây khớp với số liệu đã báo cáo.

Chạy:
    python -m src.explainability.demo "Mày ngu như con chó vậy"
    python -m src.explainability.demo --format markdown "câu cần soi"
    echo "câu 1
    câu 2" | python -m src.explainability.demo   # đọc từ stdin, mỗi dòng 1 câu
"""

from __future__ import annotations

import argparse
import sys

from src.explainability.occlusion import (
    OcclusionScorer, normalize_importance, occlude_words, split_words,
)

# Ngưỡng đã quét trên ViHOS validation rồi đóng băng ở 4.3 — xem
# results/step4/localization.json ("frozen_threshold"). Đổi số này thì demo
# sẽ không còn khớp với số liệu đã báo cáo.
OFFICIAL_THRESHOLD = 0.1

ANSI = {
    "strong": "\033[1;97;41m",   # trắng đậm trên nền đỏ
    "medium": "\033[1;30;43m",   # đen trên nền vàng
    "weak": "\033[4m",           # gạch chân
    "reset": "\033[0m",
}


def bucket(score: float, threshold: float) -> str:
    """Chia mức highlight. Điểm đã chuẩn hóa nên nằm trong thang 0..1."""
    if score >= 0.5:
        return "strong"
    if score >= 0.25:
        return "medium"
    if score >= threshold:
        return "weak"
    return "none"


def render(words: list[str], importance: list[float], threshold: float, fmt: str) -> str:
    """Ghép câu, tô đậm từ vượt ngưỡng. fmt: ansi | markdown | plain."""
    out: list[str] = []
    for word, score in zip(words, importance):
        level = bucket(score, threshold)
        if level == "none":
            out.append(word)
        elif fmt == "markdown":
            out.append(f"**{word}**" if level == "strong" else f"*{word}*")
        elif fmt == "plain":
            out.append(f"[{word}]")
        else:
            out.append(f"{ANSI[level]}{word}{ANSI['reset']}")
    return " ".join(out)


def explain(text: str, scorer: OcclusionScorer, window: int, threshold: float, fmt: str) -> str:
    words = split_words(text)
    if not words:
        return "(câu rỗng)"

    margin = scorer.score(text)
    verdict = "HATE" if margin > 0 else "CLEAN"
    header = f"{verdict}  (margin={margin:+.2f}, P(hate)={scorer.probability(text):.4f})"

    # Câu CLEAN thì không có gì để định vị. Vẫn phải chặn ở đây vì điểm được
    # chuẩn hóa theo đỉnh CỦA CHÍNH CÂU ĐÓ, nên câu sạch (mọi thay đổi đều bé
    # tí) vẫn bị đẩy lên thang 0..1 và trông như có từ "gây cảnh báo".
    # evaluate_localization.py không dính lỗi này vì chỉ chạy trên câu HATE.
    if verdict == "CLEAN":
        return f"{header}\n  {' '.join(words)}\n  (câu CLEAN — không định vị cụm gây hại)"

    importance = normalize_importance(occlude_words(words, scorer, window=window))
    lines = [header, f"  {render(words, importance, threshold, fmt)}"]
    flagged = [(w, s) for w, s in zip(words, importance) if s >= threshold]
    if flagged:
        ranked = ", ".join(f"{w} ({s:.2f})" for w, s in sorted(flagged, key=lambda x: -x[1]))
        lines.append(f"  cụm gây cảnh báo: {ranked}")
    else:
        lines.append("  cụm gây cảnh báo: (không từ nào vượt ngưỡng)")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Giải thích dự đoán 1 câu bằng occlusion (Bước 4.4)")
    parser.add_argument("text", nargs="*", help="câu cần giải thích; bỏ trống để đọc từ stdin")
    parser.add_argument("--model_name", type=str, default="models/vihatet5-v2-best")
    parser.add_argument("--window", type=int, default=2, help="số từ che mỗi lần (mặc định = cấu hình chính thức)")
    parser.add_argument("--threshold", type=float, default=OFFICIAL_THRESHOLD,
                         help=f"ngưỡng flag trên thang đã chuẩn hóa 0..1 (mặc định {OFFICIAL_THRESHOLD}, "
                              "là ngưỡng đóng băng từ val ở 4.3)")
    parser.add_argument("--format", choices=["ansi", "markdown", "plain"], default="ansi",
                         help="ansi: màu cho terminal; markdown: **đậm**/*nghiêng* để paste vào báo cáo")
    args = parser.parse_args()

    texts = [" ".join(args.text)] if args.text else [l.strip() for l in sys.stdin if l.strip()]
    if not texts:
        parser.error("không có câu nào để giải thích (truyền tham số hoặc pipe qua stdin)")

    scorer = OcclusionScorer(args.model_name)
    for text in texts:
        print(explain(text, scorer, args.window, args.threshold, args.format))
        print()


if __name__ == "__main__":
    main()
