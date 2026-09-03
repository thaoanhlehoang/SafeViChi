---
language:
- vi
task_categories:
- text-classification
pretty_name: SafeViChi Vietnamese Non-standard / Perturbed Text Dataset
---

# SafeViChi Vietnamese Non-standard / Perturbed Text Dataset

Version: `1.1.0`
Fingerprint: `2874b1228a2ea6c51eaff00178c3d90d8d37a5336adbe10d24934d0ac62b5268`

This build contains 75,048 controlled surface variants for
binary `HATE`/`CLEAN` classification. Every accepted row contains its original
text, perturbed text, source/revision/row provenance, mapped and original label,
sequential edit trace, per-row random seed, duplicate group, and generator
version.

The `teencode_lexical` transform was applied to
34,569/69,133 eligible rows
(50.00%), using the validated
`teencode_dict` resource at SHA-256 `23fcc07f33100fc61ca915e9f64dfb69d72387f01a41b61dc192f73c83160de3`.
The compiler accepted 589 of
625 source pairs; exclusion counts are
recorded in `manifest.json` and pair-level details in
`teencode_lexicon_audit.json`.

## Sources and label provenance

- [`uitnlp/vihsd`](https://huggingface.co/datasets/uitnlp/vihsd), pinned at
  `88e81b36ca376867640dad9df295e0f7d499ab80`: human labels; `OFFENSIVE` and `HATE` map to `HATE`.
- [`tarudesu/VOZ-HSD`](https://huggingface.co/datasets/tarudesu/VOZ-HSD), pinned
  at source revision `923915f5f633c503babf2c6dcf7003593318b04f` and converted-Parquet revision
  `2518cc1fd5287975fbc6f842fb29625880c012f4`: weak AI-generated labels (`1=HATE`, `0=CLEAN`).

VOZ-HSD's authors state that the dataset is for research and that its model-
generated labels were intended for analysis, not downstream fine-tuning.
Accordingly, these rows are marked `annotation_type=weak_ai`, forced into the
training split, and must not be treated as gold benchmark labels. Obtain any
needed permission and human re-annotation before public release or downstream
training. Validation and test contain human-labelled ViHSD rows only.

## Quality controls

- Empty texts and exact duplicate groups with contradictory binary ViHSD labels
  are quarantined in `rejected.jsonl` (hashes and reasons only).
- Exact and conservative near duplicates share one `duplicate_group_id` and are
  assigned atomically to one target split.
- Every perturbation is local, replayable, seeded, and automatically checked for
  structural label-preservation invariants.
- `human_review_sample.csv` must be completed to assess naturalness, semantic
  preservation, and label preservation; automatic checks do not replace review.

## Primary fields

`text`, `original_text`, `label`, `label_original`, `annotation_type`,
`source_dataset`, `source_revision`, `source_row_id`, `target_split`,
`perturbation_types`, `perturbation_edits`, `perturbation_count`, `random_seed`,
`generator_version`, `teencode_eligible`, `teencode_targeted`,
`teencode_applied`, `duplicate_group_id`, and `quality_flags`.

## Compact CSV exports

`train.csv`, `validation.csv`, and `test.csv` mirror their corresponding
Parquet splits while retaining only `text`, `original_text`, and `label`.
