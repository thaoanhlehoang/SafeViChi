# SafeViChi v1 (Vòng loại — Lưu trữ đóng băng)

Thư mục này gom toàn bộ mã nguồn, dữ liệu, kết quả, script, tài liệu và kiểm thử của **SafeViChi v1** (bài toán 2 nhãn `HATE` / `CLEAN`, ViHateT5 fine-tune đối kháng FGM + occlusion highlight) để làm **baseline đối chiếu** cho vòng chung kết (v2).

> **Quy tắc:** Không chỉnh sửa nội dung các tệp kết quả và dữ liệu chốt (`v1/results/step3/`, `v1/results/step4/`, `v1/data/data_final/`). Bản nguyên gốc tại thời điểm nộp vòng loại được gắn tag git `v1-vongloai`.

## Cấu trúc bên trong `v1/`

- `v1/src/`: `baselines/`, `classifier/`, `dataset_builder/`, `explainer/`, `pipeline.py`
- `v1/data/`: `data_final/`, `eval_baseline/`, `data_explain/`, `processed/`, `CHECKSUMS.json`
- `v1/results/`: `step3/` (ma trận đánh giá & biểu đồ), `step4/` (localization & predictions)
- `v1/scripts/`: `checksums.py`, `build_step3_data.sh`, `run_step3_eval.sh`, `download_dataset_sources.ps1`, `upload_model_to_hf.py`
- `v1/docs/`: `README.md`, `pipeline.md`, `experiments.md`, `report/` (hồ sơ & tài liệu dự án `.tex/.pdf`), `assets/`
- `v1/tests/`: 11 file unit test của v1 (`pytest -q v1/tests`)
