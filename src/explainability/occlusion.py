"""Occlusion scorer cho Bước 4 (giải thích dự đoán bằng cách che từng từ).

Model là seq2seq (ViHateT5): sinh 1 token duy nhất `"hate"`/`"clean"` (xem
src/classifier/data.py). Lấy 1 forward pass ép decoder bắt đầu bằng
decoder_start_token_id (không generate tự do), đọc logits ở bước decode đầu
tiên cho đúng 2 token nhãn.

**Quan trọng**: điểm dùng cho occlusion là **logit margin** (hate_logit -
clean_logit), KHÔNG phải xác suất softmax. Model thực tế rất tự tin (margin
quan sát được ~16-30), nên P(hate) tính bằng softmax bão hòa về đúng 1.0/0.0
trong float32 (vd. margin=23 -> 1 - 6.9e-11, số này không còn phân biệt được
với 1.0 ở độ chính xác float32) — occlusion sẽ mất hết tín hiệu nếu so sánh
xác suất đã bão hòa giữa câu gốc và câu bị che. Margin không bị chặn trong
[0,1] nên không bão hòa, chênh lệch giữa các câu vẫn thấy rõ (đã verify thực
tế: che 1 từ khiến margin đổi vài đơn vị trong khi P(hate) softmax vẫn y
nguyên 1.000000).

Occlusion: che lần lượt từng cửa sổ n-gram (mặc định 1 từ/lần, xoá hẳn khỏi
câu rồi ghép lại bằng khoảng trắng), đo margin giảm bao nhiêu so với câu gốc
— mức giảm càng lớn, từ càng "gây cảnh báo".
"""

from __future__ import annotations

import re

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

PROMPT_PREFIX = "vihsd: "
WORD_RE = re.compile(r"\S+")


def split_words(text: str) -> list[str]:
    return WORD_RE.findall(text)


class OcclusionScorer:
    """Load 1 checkpoint ViHateT5 và cho điểm P(hate) theo batch câu."""

    def __init__(self, model_name_or_path: str, device: str | None = None,
                 max_input_length: int = 256, batch_size: int = 64):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.max_input_length = max_input_length
        self.batch_size = batch_size
        self.clean_id = self._label_token_id("clean")
        self.hate_id = self._label_token_id("hate")

    def _label_token_id(self, word: str) -> int:
        ids = self.tokenizer(word, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(f"Nhãn {word!r} không map về đúng 1 token (tokenizer trả {ids}) — "
                              "kiểm tra lại checkpoint/tokenizer, xem ghi chú ở src/classifier/data.py")
        return ids[0]

    def score_batch(self, texts: list[str]) -> list[float]:
        """Logit margin (hate_logit - clean_logit) cho từng câu trong batch.

        Dùng margin thay vì P(hate) softmax vì model tự tin cực đoan (margin
        quan sát thực tế ~16-30) khiến xác suất bão hòa về đúng 1.0/0.0 trong
        float32 — xem docstring đầu file.

        Tự cắt thành chunk `batch_size` để câu dài (occlusion sinh N+1 câu con)
        không làm tràn bộ nhớ GPU.
        """
        scores: list[float] = []
        for start in range(0, len(texts), self.batch_size):
            scores.extend(self._score_chunk(texts[start:start + self.batch_size]))
        return scores

    @torch.no_grad()
    def _score_chunk(self, texts: list[str]) -> list[float]:
        # BẮT BUỘC hạ chữ thường: từ điển sentencepiece của ViHateT5 KHÔNG có ký
        # tự hoa nào — mọi chữ hoa bị map thành <unk>, nên câu viết HOA biến
        # thành chuỗi rỗng nghĩa và model luôn trả về cùng một kết quả mặc định
        # (đo được: "ĐỊT MẸ MÀY" -> margin -2,04 = CLEAN; hạ thường -> +15,14 =
        # HATE). Chữ hoa vốn mang zero thông tin tới model, nên hạ thường không
        # mất gì mà cứu lại phần tín hiệu đang bị vứt (~10% token trên dữ liệu
        # thật). Chỉ hạ ở đây — text hiển thị cho người đọc vẫn giữ nguyên dạng.
        inputs = [PROMPT_PREFIX + t.lower() for t in texts]
        enc = self.tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True,
            max_length=self.max_input_length,
        ).to(self.device)
        decoder_input_ids = torch.full(
            (len(texts), 1), self.model.config.decoder_start_token_id,
            dtype=torch.long, device=self.device,
        )
        logits = self.model(**enc, decoder_input_ids=decoder_input_ids).logits
        first_step_logits = logits[:, 0, :]
        margin = first_step_logits[:, self.hate_id] - first_step_logits[:, self.clean_id]
        return margin.tolist()

    def score(self, text: str) -> float:
        return self.score_batch([text])[0]

    def probability(self, text: str) -> float:
        """P(hate) = sigmoid(margin) — CHỈ để hiển thị cho người đọc, không dùng
        cho occlusion (có thể bão hòa về 0.0/1.0 khi câu rõ ràng, xem docstring
        đầu file)."""
        import math
        return 1.0 / (1.0 + math.exp(-self.score(text)))


def occlude_words(words: list[str], scorer: OcclusionScorer, window: int = 1) -> list[float]:
    """Điểm quan trọng của từng từ = tổng mức P(hate) giảm khi từ đó bị che.

    1 forward-pass batch duy nhất: [câu gốc] + [câu che ở từng vị trí cửa sổ].
    window=1: mỗi lần che đúng 1 từ. window>1: che cụm liền `window` từ, mức
    giảm được cộng dồn vào TỪNG từ trong cụm đó (để vẫn ra điểm theo từng từ).
    """
    if not words:
        return []

    texts = [" ".join(words)]
    windows: list[tuple[int, int]] = []
    for start in range(len(words)):
        end = min(start + window, len(words))
        windows.append((start, end))
        occluded = words[:start] + words[end:]
        texts.append(" ".join(occluded))

    scores = scorer.score_batch(texts)
    original_score, occluded_scores = scores[0], scores[1:]

    importance = [0.0] * len(words)
    for (start, end), occ_score in zip(windows, occluded_scores):
        drop = original_score - occ_score
        for i in range(start, end):
            importance[i] += drop
    return importance


def normalize_importance(importance: list[float]) -> list[float]:
    """Chia điểm cho từ mạnh nhất TRONG CHÍNH CÂU ĐÓ -> thang [.., 1.0].

    Lý do: thang điểm margin lệch rất mạnh giữa các câu (đo trên ViHOS test:
    điểm cao nhất mỗi câu trải từ -2.3 đến 23.1, median 6.8). Một ngưỡng tuyệt
    đối dùng chung sẽ gạt sạch các câu "model nói nhỏ" — đo được recall chỉ
    0.062 ở nhóm câu thang điểm thấp (max<1.5) so với 0.759 ở nhóm thang cao.
    Chuẩn hóa xong, ngưỡng mang nghĩa "so với từ mạnh nhất của câu này".

    Câu không có từ nào làm điểm giảm (max <= 0) -> trả về toàn 0.0, tức
    không flag từ nào (chia cho số âm sẽ lật dấu, sai hoàn toàn).
    """
    if not importance:
        return []
    peak = max(importance)
    if peak <= 0:
        return [0.0] * len(importance)
    return [v / peak for v in importance]


def top_k_spans(importance: list[float], k: int | None = None, threshold: float = 0.0) -> list[int]:
    """Chỉ số các từ được coi là "gây cảnh báo".

    k=None (mặc định): mọi từ có điểm > threshold.
    k=int: đúng top-k từ điểm cao nhất (bất kể dấu), dùng khi cần độ dài span
    cố định để so sánh công bằng giữa các câu.
    """
    if k is not None:
        ranked = sorted(range(len(importance)), key=lambda i: importance[i], reverse=True)
        return sorted(ranked[:k])
    return [i for i, v in enumerate(importance) if v > threshold]
