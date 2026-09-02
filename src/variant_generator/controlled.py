"""Controlled Vietnamese non-standard text perturbations.

The transformations in this module are deliberately local: they change the
surface form of an existing span but do not insert new semantic content.  Each
edit is recorded against the text state immediately before that edit, which
makes a multi-step trace exactly replayable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from difflib import SequenceMatcher
import random
import re
import unicodedata
from typing import Callable, Iterable, Mapping, Sequence


GENERATOR_VERSION = "1.0.0"

ABBREVIATIONS: Mapping[str, Sequence[str]] = {
    "không": ("ko", "k"),
    "được": ("đc",),
    "biết": ("bt",),
    "với": ("vs",),
    "người": ("ng",),
    "mình": ("mk",),
    "bình thường": ("bth",),
    "nói chuyện": ("nc",),
}

INTENTIONAL_SPELLINGS: Mapping[str, Sequence[str]] = {
    "quá": ("qá", "wá"),
    "gì": ("j",),
    "vậy": ("zậy",),
    "rồi": ("rùi",),
    "phải": ("fải",),
    "cũng": ("cx",),
    "chưa": ("chưaaa",),
}

PHONETIC_SPELLINGS: Mapping[str, Sequence[str]] = {
    "không": ("hông",),
    "thôi": ("thui",),
    "biết": ("bít",),
    "yêu": ("iu",),
    "nhiều": ("nhìu",),
    "hiểu": ("hỉu",),
}

DIALECTAL_VARIANTS: Mapping[str, Sequence[str]] = {
    "không": ("hổng",),
    "vào": ("dzô",),
    "về": ("dìa",),
    "thế": ("rứa",),
    "đâu": ("mô",),
}

SLANG_VARIANTS: Mapping[str, Sequence[str]] = {
    "bạn": ("fen",),
    "facebook": ("fb",),
    "điện thoại": ("đt",),
    "tin nhắn": ("tn",),
    "mọi người": ("các fen",),
}

# These phrase-level forms are only applied when the complete phrase occurs.
# Treating them separately avoids ambiguous, context-free single-token rules.
CONTEXT_VARIANTS: Mapping[str, Sequence[str]] = {
    "bây giờ": ("bg", "bi h"),
    "bao giờ": ("bh",),
    "như thế nào": ("ntn",),
    "được không": ("đc ko",),
    "tại sao": ("tsao",),
    "anh em": ("ae",),
}

BASE_PERTURBATION_TYPES = (
    "abbreviation_clipping",
    "intentional_spelling",
    "phonetic_spelling",
    "dialectal_writing",
    "slang_lexical",
    "expressive_lengthening",
    "diacritic_variation",
    "surface_obfuscation",
    "boundary_variation",
    "context_dependent",
    "typographical_noise",
)

WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?", re.UNICODE)
PROTECTED_RE = re.compile(
    r"(?:https?://\S+|www\.\S+|\b\S+@\S+\.\S+\b|[@#]\w+)", re.IGNORECASE
)
SEPARATORS = (".", "-", "_", "·", "~")
VIETNAMESE_VOWELS = frozenset(
    "aăâeêioôơuưy"
    "áàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệ"
    "íìỉĩịóòỏõọốồổỗộớờởỡợ"
    "úùủũụứừửữựýỳỷỹỵ"
)

# This is only a conservative regression guard for CLEAN examples.  It is not
# used as a classifier and intentionally remains small.  The guard rejects a
# candidate if an edit accidentally creates a new exact high-risk token.
_RISK_TOKENS = frozenset(
    {
        "dit",
        "dcm",
        "dm",
        "deo",
        "ngu",
        "cho",
        "cunt",
        "fuck",
    }
)

_KEYBOARD_NEIGHBOURS: Mapping[str, str] = {
    "a": "sqw",
    "b": "vghn",
    "c": "xdfv",
    "d": "serfcx",
    "e": "wsdr",
    "g": "ftyhbv",
    "h": "gyujnb",
    "i": "ujko",
    "k": "ijolm",
    "m": "njk",
    "n": "bhjm",
    "o": "iklp",
    "p": "ol",
    "q": "wa",
    "r": "edft",
    "s": "awedxz",
    "t": "rfgy",
    "u": "yhji",
    "v": "cfgb",
    "x": "zsdc",
    "y": "tghu",
}


@dataclass(frozen=True)
class Edit:
    """One replayable edit, indexed in the text state before this step."""

    step: int
    perturbation_type: str
    rule_id: str
    before_start: int
    before_end: int
    after_start: int
    after_end: int
    before: str
    after: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PerturbationResult:
    original_text: str
    perturbed_text: str
    perturbation_types: tuple[str, ...]
    perturbation_mode: str
    edits: tuple[Edit, ...]
    seed: int
    generator_version: str = GENERATOR_VERSION

    @property
    def perturbation_count(self) -> int:
        return len(self.edits)

    def to_metadata(self) -> dict[str, object]:
        return {
            "perturbation_types": list(self.perturbation_types),
            "perturbation_mode": self.perturbation_mode,
            "perturbation_edits": [edit.to_dict() for edit in self.edits],
            "perturbation_count": self.perturbation_count,
            "random_seed": self.seed,
            "generator_version": self.generator_version,
        }


class PerturbationError(ValueError):
    """Raised when no label-preserving surface edit can be produced."""


def _overlaps_protected(start: int, end: int, protected: Sequence[tuple[int, int]]) -> bool:
    return any(start < right and end > left for left, right in protected)


def _word_spans(text: str, *, min_length: int = 1) -> list[re.Match[str]]:
    protected = [(match.start(), match.end()) for match in PROTECTED_RE.finditer(text)]
    return [
        match
        for match in WORD_RE.finditer(text)
        if len(match.group()) >= min_length
        and not _overlaps_protected(match.start(), match.end(), protected)
    ]


def _preserve_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    pieces = [re.escape(piece) for piece in phrase.split()]
    body = r"\s+".join(pieces)
    return re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE | re.UNICODE)


def _make_edit(
    text: str,
    start: int,
    end: int,
    replacement_text: str,
    perturbation_type: str,
    rule_id: str,
) -> tuple[str, Edit]:
    before = text[start:end]
    replacement_text = _preserve_case(before, replacement_text)
    edited = text[:start] + replacement_text + text[end:]
    return edited, Edit(
        step=0,
        perturbation_type=perturbation_type,
        rule_id=rule_id,
        before_start=start,
        before_end=end,
        after_start=start,
        after_end=start + len(replacement_text),
        before=before,
        after=replacement_text,
    )


def _substitute_from_map(
    text: str,
    rng: random.Random,
    mapping: Mapping[str, Sequence[str]],
    perturbation_type: str,
) -> tuple[str, Edit] | None:
    matches: list[tuple[re.Match[str], str, Sequence[str]]] = []
    protected = [(match.start(), match.end()) for match in PROTECTED_RE.finditer(text)]
    for source, variants in mapping.items():
        for match in _phrase_pattern(source).finditer(text):
            if not _overlaps_protected(match.start(), match.end(), protected):
                matches.append((match, source, variants))
    if not matches:
        return None
    match, source, variants = rng.choice(matches)
    replacement_text = rng.choice(tuple(variants))
    return _make_edit(
        text,
        match.start(),
        match.end(),
        replacement_text,
        perturbation_type,
        f"{perturbation_type}:{source}",
    )


def _remove_marks(value: str) -> str:
    value = value.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", value)
    return unicodedata.normalize(
        "NFC", "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    )


def _abbreviation(text: str, rng: random.Random) -> tuple[str, Edit] | None:
    return _substitute_from_map(text, rng, ABBREVIATIONS, "abbreviation_clipping")


def _intentional_spelling(text: str, rng: random.Random) -> tuple[str, Edit] | None:
    return _substitute_from_map(
        text, rng, INTENTIONAL_SPELLINGS, "intentional_spelling"
    )


def _phonetic(text: str, rng: random.Random) -> tuple[str, Edit] | None:
    return _substitute_from_map(text, rng, PHONETIC_SPELLINGS, "phonetic_spelling")


def _dialectal(text: str, rng: random.Random) -> tuple[str, Edit] | None:
    return _substitute_from_map(text, rng, DIALECTAL_VARIANTS, "dialectal_writing")


def _slang(text: str, rng: random.Random) -> tuple[str, Edit] | None:
    return _substitute_from_map(text, rng, SLANG_VARIANTS, "slang_lexical")


def _context_dependent(text: str, rng: random.Random) -> tuple[str, Edit] | None:
    return _substitute_from_map(text, rng, CONTEXT_VARIANTS, "context_dependent")


def _expressive_lengthening(text: str, rng: random.Random) -> tuple[str, Edit] | None:
    candidates: list[tuple[re.Match[str], list[int]]] = []
    for match in _word_spans(text, min_length=2):
        word = match.group()
        positions = [
            index
            for index, char in enumerate(word)
            if char.casefold() in VIETNAMESE_VOWELS
            and (index == 0 or word[index - 1].casefold() != char.casefold())
            and (index + 1 == len(word) or word[index + 1].casefold() != char.casefold())
        ]
        if positions:
            candidates.append((match, positions))
    if not candidates:
        expressive_positions = [
            index for index, char in enumerate(text) if not char.isspace()
        ]
        if not expressive_positions:
            return None
        start = rng.choice(expressive_positions)
        char = text[start]
        repeated = char * rng.randint(2, 8)
        return _make_edit(
            text,
            start,
            start + 1,
            repeated,
            "expressive_lengthening",
            "expressive_lengthening:symbol",
        )
    match, positions = rng.choice(candidates)
    relative = rng.choice(positions)
    start = match.start() + relative
    char = text[start]
    repeated = char * rng.randint(2, 4)
    return _make_edit(
        text,
        start,
        start + 1,
        repeated,
        "expressive_lengthening",
        "expressive_lengthening:vowel",
    )


def _diacritic_variation(text: str, rng: random.Random) -> tuple[str, Edit] | None:
    candidates = [match for match in _word_spans(text) if _remove_marks(match.group()) != match.group()]
    if not candidates:
        return None
    match = rng.choice(candidates)
    replacement_text = _remove_marks(match.group())
    return _make_edit(
        text,
        match.start(),
        match.end(),
        replacement_text,
        "diacritic_variation",
        "diacritic_variation:strip_token",
    )


def _surface_obfuscation(text: str, rng: random.Random) -> tuple[str, Edit] | None:
    candidates = _word_spans(text, min_length=3)
    if not candidates:
        return None
    match = rng.choice(candidates)
    word = match.group()
    positions = list(range(1, len(word)))
    rng.shuffle(positions)
    count = 1 if len(word) < 7 else rng.randint(1, 2)
    selected = sorted(positions[:count], reverse=True)
    replacement_text = word
    for position in selected:
        replacement_text = (
            replacement_text[:position]
            + rng.choice(SEPARATORS)
            + replacement_text[position:]
        )
    return _make_edit(
        text,
        match.start(),
        match.end(),
        replacement_text,
        "surface_obfuscation",
        "surface_obfuscation:insert_separator",
    )


def _boundary_variation(text: str, rng: random.Random) -> tuple[str, Edit] | None:
    words = _word_spans(text, min_length=2)
    joins: list[tuple[int, int]] = []
    for left, right in zip(words, words[1:]):
        gap = text[left.end() : right.start()]
        if gap and gap.isspace() and len(left.group()) + len(right.group()) >= 6:
            joins.append((left.start(), right.end()))

    splits = [match for match in words if len(match.group()) >= 6]
    actions: list[str] = []
    if joins:
        actions.append("join")
    if splits:
        actions.append("split")
    if not actions:
        return None

    action = rng.choice(actions)
    if action == "join":
        start, end = rng.choice(joins)
        source = text[start:end]
        replacement_text = re.sub(r"\s+", "", source)
        return _make_edit(
            text,
            start,
            end,
            replacement_text,
            "boundary_variation",
            "boundary_variation:join_syllables",
        )

    match = rng.choice(splits)
    word = match.group()
    middle = len(word) // 2
    possible = [index for index in range(2, len(word) - 1)]
    split_at = min(possible, key=lambda index: (abs(index - middle), rng.random()))
    replacement_text = word[:split_at] + " " + word[split_at:]
    return _make_edit(
        text,
        match.start(),
        match.end(),
        replacement_text,
        "boundary_variation",
        "boundary_variation:split_token",
    )


def _typographical_noise(text: str, rng: random.Random) -> tuple[str, Edit] | None:
    candidates = _word_spans(text, min_length=4)
    if not candidates:
        return None
    match = rng.choice(candidates)
    word = match.group()
    actions = ["swap", "duplicate", "drop"]
    ascii_positions = [
        index
        for index, char in enumerate(word)
        if _remove_marks(char).casefold() in _KEYBOARD_NEIGHBOURS
    ]
    if ascii_positions:
        actions.append("keyboard")
    action = rng.choice(actions)

    if action == "swap":
        position = rng.randint(1, len(word) - 2)
        if word[position] == word[position + 1]:
            action = "duplicate"
        else:
            replacement_text = (
                word[:position]
                + word[position + 1]
                + word[position]
                + word[position + 2 :]
            )
    if action == "duplicate":
        position = rng.randint(1, len(word) - 2)
        replacement_text = word[:position] + word[position] + word[position:]
    elif action == "drop":
        position = rng.randint(1, len(word) - 2)
        replacement_text = word[:position] + word[position + 1 :]
    elif action == "keyboard":
        position = rng.choice(ascii_positions)
        base = _remove_marks(word[position]).casefold()
        replacement_char = rng.choice(_KEYBOARD_NEIGHBOURS[base])
        replacement_char = _preserve_case(word[position], replacement_char)
        replacement_text = word[:position] + replacement_char + word[position + 1 :]

    return _make_edit(
        text,
        match.start(),
        match.end(),
        replacement_text,
        "typographical_noise",
        f"typographical_noise:{action}",
    )


Transform = Callable[[str, random.Random], tuple[str, Edit] | None]
TRANSFORMS: Mapping[str, Transform] = {
    "abbreviation_clipping": _abbreviation,
    "intentional_spelling": _intentional_spelling,
    "phonetic_spelling": _phonetic,
    "dialectal_writing": _dialectal,
    "slang_lexical": _slang,
    "expressive_lengthening": _expressive_lengthening,
    "diacritic_variation": _diacritic_variation,
    "surface_obfuscation": _surface_obfuscation,
    "boundary_variation": _boundary_variation,
    "context_dependent": _context_dependent,
    "typographical_noise": _typographical_noise,
}


def _risk_tokens(text: str) -> set[str]:
    folded = _remove_marks(text).casefold()
    return {
        match.group()
        for match in re.finditer(r"[a-z]+", folded)
        if match.group() in _RISK_TOKENS
    }


def _similarity_form(text: str) -> str:
    folded = _remove_marks(unicodedata.normalize("NFKC", text)).casefold()
    return "".join(char for char in folded if char.isalnum())


def passes_label_preservation_guard(original: str, candidate: str, label: str) -> bool:
    """Conservative automatic guard; it complements, not replaces, human QA."""

    if not candidate.strip() or candidate == original:
        return False
    original_form = _similarity_form(original)
    candidate_form = _similarity_form(candidate)
    if not original_form:
        original_visible = "".join(char for char in original if not char.isspace())
        candidate_visible = "".join(char for char in candidate if not char.isspace())
        iterator = iter(candidate_visible)
        retained_as_subsequence = all(char in iterator for char in original_visible)
        return (
            bool(original_visible)
            and retained_as_subsequence
            and len(candidate_visible) <= len(original_visible) + 8
        )
    if not candidate_form:
        return False
    length_ratio = len(candidate_form) / len(original_form)
    maximum_ratio = 8.0 if len(original_form) <= 2 else 3.0 if len(original_form) <= 4 else 1.8
    if not 0.55 <= length_ratio <= maximum_ratio:
        return False
    similarity = SequenceMatcher(None, original_form, candidate_form).ratio()
    minimum_similarity = 0.20 if len(original_form) <= 2 else 0.45 if len(original_form) <= 4 else 0.55
    if similarity < minimum_similarity:
        return False
    if label.upper() == "CLEAN" and (_risk_tokens(candidate) - _risk_tokens(original)):
        return False
    return True


def replay_edits(original_text: str, edits: Iterable[Edit]) -> str:
    """Replay and validate a sequential edit trace."""

    text = original_text
    for expected_step, edit in enumerate(edits):
        if edit.step != expected_step:
            raise ValueError(f"Expected edit step {expected_step}, got {edit.step}")
        if text[edit.before_start : edit.before_end] != edit.before:
            raise ValueError(f"Edit {edit.step} does not match its recorded input span")
        if edit.after_start != edit.before_start:
            raise ValueError(f"Edit {edit.step} has inconsistent output start")
        if edit.after_end != edit.after_start + len(edit.after):
            raise ValueError(f"Edit {edit.step} has inconsistent output end")
        text = text[: edit.before_start] + edit.after + text[edit.before_end :]
    return text


class ControlledPerturber:
    """Create stochastic, context-aware, reproducible surface variants."""

    def __init__(self, *, generator_version: str = GENERATOR_VERSION) -> None:
        self.generator_version = generator_version

    def perturb(
        self,
        text: str,
        label: str,
        *,
        seed: int,
        required_type: str | None = None,
        min_operations: int = 1,
        max_operations: int = 3,
    ) -> PerturbationResult:
        if label.upper() not in {"HATE", "CLEAN"}:
            raise ValueError(f"Unsupported target label: {label!r}")
        if not text or not text.strip():
            raise PerturbationError("Cannot perturb an empty source text")
        if required_type not in {*BASE_PERTURBATION_TYPES, "mixed", None}:
            raise ValueError(f"Unknown required perturbation type: {required_type!r}")
        if min_operations < 1 or max_operations < min_operations:
            raise ValueError("Expected 1 <= min_operations <= max_operations")

        rng = random.Random(seed)
        word_count = len(_word_spans(text))
        safe_max = 1 if word_count <= 2 else 2 if word_count <= 7 else max_operations
        safe_max = max(min_operations, min(max_operations, safe_max))
        desired_count = rng.randint(min_operations, safe_max)
        if required_type == "mixed":
            desired_count = max(2, desired_count)

        order = list(BASE_PERTURBATION_TYPES)
        rng.shuffle(order)
        if required_type and required_type != "mixed":
            order.remove(required_type)
            order.insert(0, required_type)

        current = text
        edits: list[Edit] = []
        used_types: set[str] = set()
        # A second shuffled pass lets a failed required dictionary rule fall
        # back to a universally applicable character-level transformation.
        candidates = order + random.Random(seed ^ 0x9E3779B97F4A7C15).sample(
            list(BASE_PERTURBATION_TYPES), len(BASE_PERTURBATION_TYPES)
        )
        for perturbation_type in candidates:
            if len(edits) >= desired_count:
                break
            if perturbation_type in used_types:
                continue
            transformed = TRANSFORMS[perturbation_type](current, rng)
            if transformed is None:
                continue
            candidate, edit = transformed
            if not passes_label_preservation_guard(text, candidate, label):
                continue
            edit = replace(edit, step=len(edits))
            current = candidate
            edits.append(edit)
            used_types.add(perturbation_type)

        if not edits:
            raise PerturbationError("No safe, applicable perturbation was found")
        if (
            required_type in BASE_PERTURBATION_TYPES
            and required_type not in used_types
        ):
            raise PerturbationError(
                f"Required perturbation {required_type!r} is not applicable"
            )
        if required_type == "mixed" and len(used_types) < 2:
            raise PerturbationError("Could not create a safe mixed perturbation")

        ordered_types = tuple(edit.perturbation_type for edit in edits)
        result = PerturbationResult(
            original_text=text,
            perturbed_text=current,
            perturbation_types=ordered_types,
            perturbation_mode="mixed" if len(set(ordered_types)) > 1 else "single",
            edits=tuple(edits),
            seed=seed,
            generator_version=self.generator_version,
        )
        if replay_edits(text, result.edits) != result.perturbed_text:
            raise AssertionError("Internal error: perturbation trace is not replayable")
        return result


__all__ = [
    "BASE_PERTURBATION_TYPES",
    "ControlledPerturber",
    "Edit",
    "GENERATOR_VERSION",
    "PerturbationError",
    "PerturbationResult",
    "passes_label_preservation_guard",
    "replay_edits",
]
