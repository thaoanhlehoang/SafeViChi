# SafeViChi: cấu trúc thư mục

Tài liệu mô tả vị trí mã, dữ liệu và kết quả trong repo. Số liệu đánh giá nằm ở [KET_QUA.md](KET_QUA.md), nhận xét ở
[DANH_GIA.md](DANH_GIA.md).

## Tổng quan

```text
SafeViChi/
├── src/                 # Mã Python: src/v2 (sản phẩm chung kết) và src/shared (dùng chung với v1)
├── notebooks/           # Notebook huấn luyện trên Colab/Kaggle (GPU)
├── data/
│   ├── raw/             # Danh sách nguồn tải (NGUON_TAI.json); dữ liệu gốc tải về không commit
│   └── v2/              # Dữ liệu v2: train, bộ test khóa C0/C1/C2, dữ liệu trung gian, checksum
├── results/v2/          # Kết quả đo: mỗi thí nghiệm một thư mục
├── docs/                # Tài liệu Markdown chi tiết và báo cáo v2 (LaTeX/PDF)
├── tests/v2/            # Unit test v2 (pytest)
├── web-v2/              # Ứng dụng web chạy mô hình trong trình duyệt
├── demo/                # Web demo của v1 (lưu trữ)
├── models/              # Trọng số tải về cục bộ, không commit
├── v1/                  # Toàn bộ SafeViChi v1 vòng loại, đóng băng làm baseline
├── pyproject.toml       # Gói `safevichi` (CLI suy luận, chỉ cần onnxruntime)
├── requirements.txt     # Phụ thuộc tối thiểu
├── requirements-lock.txt# Phiên bản ghim cho huấn luyện/đánh giá
└── .env.example         # Mẫu biến môi trường cho các script gọi LLM
```

## `src/v2/`: pipeline v2

| Thư mục / tệp | Vai trò | Tệp chính |
| --- | --- | --- |
| `data/` | Tải nguồn, gộp về định dạng chung, khử trùng, chia tập, sinh biến dạng test, kiểm tra rò rỉ | `download.py` (tải ViHSD/ViHOS/ViTHSD ghim revision), `merge.py` (gộp nguồn), `chon_gan_lai.py` (tách kho test, chọn câu cần gán lại), `mine_candidates.py` (lọc thô ứng viên theo từ khóa), `dong_thuan.py` (lọc theo đồng thuận nhãn), `load_test_nguoi_viet.py` (đọc CSV câu người viết), `split.py` (ghép train), `perturb.py` (sinh C1/C2 cho test, giữ vị trí bằng chứng), `tach_test_bien_dang.py` (tách lát cắt), `kiem_tra_ro_ri.py` |
| `labeling/` | Gọi LLM: sinh hội thoại theo kịch bản, gán sơ bộ, lọc chéo, đo đồng thuận và chi phí | `llm.py` (lớp gọi LLM dùng chung), `kich_ban.yaml` (bảng kịch bản sinh dữ liệu), `generate.py`, `prelabel.py`, `loc_sinh.py`, `agreement.py`, `chi_phi.py`, `thu_model.py` |
| `eval/` | Độ đo và báo cáo | `metrics.py` (phân loại, lát cắt, bootstrap, McNemar), `evidence.py` (F1 bằng chứng theo ý và mức từ), `evaluate.py` (CLI chấm một tệp dự đoán), `cham_v1.py`, `chuyen_du_doan_llm.py`, `so_sanh_3_he.py`, `bao_cao_baseline.py`, `xuat_so_lieu_bao_cao.py` (sinh `so_lieu.tex`), `xuat_tai_lieu.py` (sinh `docs/KET_QUA.md`, `docs/DANH_GIA.md`) |
| `bench/` | Kiến trúc mô hình, xuất ONNX/INT8, đo trên thiết bị | `dualorc_model.py` (SafeVi-DualORC-CRF, giải mã CRF bằng NumPy), `export_onnx.py`, `benchmark_cpu.py`, `do_thiet_bi.py` (CPU và trình duyệt, kèm năng lượng từ pin), `upload_hf.py` |
| `train/` | Chỗ dành cho mã huấn luyện; huấn luyện thực tế chạy bằng notebook | `__init__.py` |
| `predict.py` | CLI `safevichi` (ONNX Runtime, không cần PyTorch) | |
| `download_model.py` | Tải gói mô hình từ Hugging Face `dminhnhat06/safevichi_v2`, kiểm SHA-256 | |
| `__main__.py` | Cho phép chạy `python -m src.v2` | |

## `src/shared/`: dùng chung v1 và v2

| Thư mục | Vai trò |
| --- | --- |
| `variant_generator/` | Sinh biến dạng né lọc: `teencode.py`, `controlled.py` (C1), `blindspot.py` (điểm mù C2), `generator.py` |
| `reporting/` | Vẽ hình và định dạng báo cáo |
| `utils/` | `seed.py` cố định seed 42 |
| `normalization/` | Bộ chuẩn hóa 5 tầng của v1. **Lỗi thời**, v2 tắt mặc định, chỉ giữ để chạy ablation (`--legacy-normalize`) |

## `notebooks/`

| Tệp | Nội dung |
| --- | --- |
| `v2_dualorc_crf_colab.ipynb` | Huấn luyện mô hình chốt SafeVi-DualORC-CRF (và các ứng viên encoder) trên Colab/Kaggle |
| `v2_guardmerge_colab.ipynb` | Huấn luyện ứng viên GuardMerge (LLM nhỏ + LoRA), đã bị loại ở bước chọn mô hình |

## `data/v2/`

| Tệp / thư mục | Nội dung |
| --- | --- |
| `train.jsonl` | Tập huấn luyện (bản gốc C0; tăng cường biến dạng làm trực tuyến lúc train) |
| `test.jsonl`, `test_day_du.jsonl` | Bộ test C0; `test.jsonl` chỉ có id, text, nhãn, bằng chứng như train, `test_day_du.jsonl` có đủ trường (vùng, kênh, kịch bản, người viết) dùng để cắt lát |
| `test_perturb_C1.jsonl`, `test_perturb_C2.jsonl` | Bản biến dạng C1 (kiểu né lọc thường gặp) và C2 (điểm mù chưa thấy) của bộ test |
| `so_sanh.jsonl` | Bản đủ trường của `train.jsonl` (thêm `meta`: loại câu thật/sinh, nhãn gốc, lý do chọn), dùng để phân tích theo nhóm |
| `train_stats.json`, `test_stats.json` | Thống kê phân bố nhãn, nguồn, vùng |
| `CHECKSUMS.json` | SHA-256 các tệp dữ liệu |
| `raw/` | CSV câu test người viết (`test_A`…`test_D`) và mẫu kiểm nhãn của hai người gán (`kiem_nhan_A/B`) |
| `interim/` | Dữ liệu trung gian: `merged.jsonl` (đã gộp nguồn), `kho_test.jsonl` (**không bao giờ gửi LLM**), `ung_vien.jsonl`, `can_gan_lai.jsonl`, `that_gan_lai.jsonl`, `sinh_llm*.jsonl` (hội thoại LLM sinh, trước và sau lọc), `prelabel/` (đầu ra gán sơ bộ) |

Định dạng một mẫu (một dòng JSONL):

```json
{"id": "v2_000123", "text": "lượt 1\nlượt 2", "nhan": "DE_DOA", "bang_chung": ["..."],
 "nguon": "vihsd|vihos|vithsd|thu_thap|nguoi_viet|sinh_llm", "cach_gan": "nguoi|llm|llm_nguoi_duyet",
 "bien_dang": "C0|C1|C2", "nhom_trung": "...", "split": "train|validation|test"}
```

`text` là hội thoại 1–8 dòng ngăn bằng `\n`, mỗi dòng là toàn bộ lượt nói của một người. `bang_chung` là các cụm chép
nguyên văn từ `text`.

## `results/v2/`

| Thư mục | Nội dung |
| --- | --- |
| `dev_selection/` | Bốn ứng viên (`dualorc_mmbert_base`, `dualorc_visobert`, `dualorc_bamibert`, `guardmerge_v6`), mỗi thư mục có `config.json`, `environment.json`, `history.json`, `dev_metrics.json`, `calibration.json`, các bộ stress test và `dev_predictions.jsonl`; `dev_summary.json` tổng hợp |
| `official_eval/dualorc_mmbert_base/` | Mô hình chốt. Thư mục `test/` chứa `REAL_TEST_LOCK.json` (khóa SHA-256 bộ test), metrics và predictions trên C0/C1/C2, bản có bật chuẩn hóa (`real_test_norm_*`), `int8_comparison.json`, `bench_cpu.json`, tệp lỗi `*_errors.jsonl` |
| `so_sanh_3_he/` | `metrics.json` so sánh v1, gpt-oss-120b (API) và v2 theo tập và vùng |
| `llm_api/` | Dự đoán và metrics của gpt-oss-120b qua API; `cham_lai/` chấm lại bằng bộ chấm v2 |
| `v1_baseline/` | Dự đoán và metrics của SafeViChi v1 trên bộ test v2 |
| `do_thiet_bi/` | Đo trên laptop CPU (`cpu_laptop.json`) và trình duyệt (`trinh_duyet.json`), kèm `env.txt` |
| `kiem_tra_ro_ri/` | Báo cáo rò rỉ train–test và danh sách id train bị loại |
| `dong_thuan_nhan/` | Độ đồng thuận giữa người gán và LLM |
| `chi_phi_du_lieu/` | Token, tiền, thời gian gọi LLM để tạo dữ liệu |
| `thu_model/` | Chạy thử so sánh các LLM trước khi chọn model gán nhãn |
| `_cham_thu/` | Lượt chấm thử bộ đánh giá |

## `docs/`

| Tệp | Nội dung |
| --- | --- |
| `KET_QUA.md` | Bảng số đầy đủ (sinh tự động) |
| `DANH_GIA.md` | Đánh giá, phân tích lỗi, hạn chế (sinh tự động) |
| `CAU_TRUC_THU_MUC.md` | Tài liệu này |
| `v2/bao_cao/` | Báo cáo chung kết: `bao_cao_v2.tex` + các phần `p0`–`p4`, `so_lieu.tex` (số liệu sinh tự động), `hinh/` (hình vẽ), `bao_cao_v2.pdf` |

Luồng sinh số liệu: `results/v2/**` → `python -m src.v2.eval.xuat_so_lieu_bao_cao` → `docs/v2/bao_cao/so_lieu.tex` →
báo cáo PDF và `python -m src.v2.eval.xuat_tai_lieu` → `docs/KET_QUA.md`, `docs/DANH_GIA.md`.

## `web-v2/`: ứng dụng web on-device

| Đường dẫn | Nội dung |
| --- | --- |
| `src/engine/` | `onnx.js` (nạp ONNX Runtime Web), `tokenizer.js`, `dualorc.js` (hậu xử lý, giải mã CRF bằng JS) |
| `src/model.worker.js` | Chạy mô hình trong Web Worker để không chặn giao diện |
| `src/chat.js`, `src/labels.js`, `src/main.js`, `src/landing.js` | Giao diện trò chuyện, nhãn và trang giới thiệu |
| `src/auth.js`, `src/login.js`, `src/firebase-config.js` | Đăng nhập |
| `bench.html`, `src/bench/` | Trang đo độ trễ, RAM trong trình duyệt |
| `scripts/verify_onnx_parity.mjs` | Đối chiếu kết quả JS với bộ test v2 (`npm run verify:parity`) |
| `serve.py`, `public/_headers` | Phục vụ cục bộ và header cho bản triển khai tĩnh |

## `tests/v2/`

Mỗi mô-đun chính có một tệp test tương ứng: dữ liệu (`test_v2_merge.py`, `test_v2_split.py`, `test_v2_perturb.py`,
`test_v2_kiem_tra_ro_ri.py`, …), gán nhãn (`test_v2_prelabel.py`, `test_v2_generate.py`, `test_v2_dong_thuan.py`),
đánh giá (`test_v2_eval.py`), triển khai (`test_v2_bench.py`, `test_v2_predict.py`, `test_v2_download.py`) và notebook
(`test_v2_notebooks.py`). Chạy `pytest -q` ở thư mục gốc (gồm cả `v1/tests`).

## `v1/`: baseline vòng loại (đóng băng)

| Thư mục | Nội dung |
| --- | --- |
| `src/` | `dataset_builder/` (dựng dữ liệu, `dedup.py` khử trùng gần đúng được v2 dùng lại), `classifier/` (ViHateT5 + FGM), `explainer/` (occlusion), `baselines/` (blacklist, `stats.py` kiểm định), `pipeline.py` |
| `data/` | `data_final/` (train/validation/test v1), `data_explain/`, `eval_baseline/`, `processed/` (bộ dữ liệu phi chuẩn v1_2), `CHECKSUMS.json` |
| `results/step3/` | Kết quả phân loại v1 (`test/matrix.json`, ngưỡng, hình, log). **Không sửa** |
| `results/step4/` | Kết quả tô sáng bằng chứng v1. **Không sửa** |
| `scripts/` | `checksums.py` (`python -m v1.scripts.checksums`), script dựng dữ liệu và đánh giá |
| `tests/` | Unit test v1 |
| `docs/` | Hình và báo cáo vòng loại |

Mã v1 còn ở tag `v1-vongloai`.

## `models/` (không commit)

`models/dualorc_mmbert_base/onnx/` là nơi CLI và web-v2 đọc gói ONNX INT8 (`safevichi_v2_int8.onnx`, `crf_params.json`,
`tokenizer/`). Gói được tải tự động từ Hugging Face khi chạy lần đầu, hoặc bằng `python -m src.v2.download_model`.
