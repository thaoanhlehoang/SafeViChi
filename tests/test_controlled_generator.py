from __future__ import annotations

import pytest

from src.variant_generator.controlled import (
    BASE_PERTURBATION_TYPES,
    ControlledPerturber,
    PerturbationError,
    passes_label_preservation_guard,
    replay_edits,
)


RICH_TEXT = (
    "Bây giờ mình không biết nói chuyện với mọi người, bạn về đâu rồi, "
    "được không, quá vậy thôi và yêu nhiều."
)


@pytest.mark.parametrize("perturbation_type", BASE_PERTURBATION_TYPES)
def test_each_required_type_is_applied_and_trace_is_replayable(perturbation_type: str) -> None:
    perturber = ControlledPerturber()
    result = perturber.perturb(
        RICH_TEXT,
        "CLEAN",
        seed=12345,
        required_type=perturbation_type,
        max_operations=3,
    )

    assert result.perturbed_text != RICH_TEXT
    assert perturbation_type in result.perturbation_types
    assert replay_edits(RICH_TEXT, result.edits) == result.perturbed_text
    assert result.perturbation_count == len(result.edits)


def test_reproducibility_does_not_depend_on_global_rng() -> None:
    perturber = ControlledPerturber()
    first = perturber.perturb(RICH_TEXT, "HATE", seed=20260902)
    second = perturber.perturb(RICH_TEXT, "HATE", seed=20260902)
    assert first == second


def test_mixed_mode_contains_multiple_actual_types() -> None:
    result = ControlledPerturber().perturb(
        RICH_TEXT,
        "CLEAN",
        seed=9,
        required_type="mixed",
        min_operations=2,
        max_operations=3,
    )
    assert result.perturbation_mode == "mixed"
    assert len(set(result.perturbation_types)) >= 2


def test_clean_guard_rejects_new_high_risk_token() -> None:
    assert not passes_label_preservation_guard("xin chào bạn", "xin chào bạn ngu", "CLEAN")


def test_empty_source_is_never_fabricated() -> None:
    with pytest.raises(PerturbationError):
        ControlledPerturber().perturb("  ", "CLEAN", seed=1)


def test_punctuation_only_source_can_receive_expressive_surface_edit() -> None:
    result = ControlledPerturber().perturb(
        "!", "CLEAN", seed=7, required_type="expressive_lengthening"
    )
    assert result.perturbed_text.startswith("!")
    assert len(result.perturbed_text) > 1
