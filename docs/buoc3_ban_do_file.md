# Bản đồ file — vai trò từng thứ trong repo

> Tra cứu nhanh: file này làm gì, ai gọi nó, xóa được không.
> Cách chạy: [`buoc3_tai_lap_pipeline.md`](buoc3_tai_lap_pipeline.md).
> Số liệu: [`buoc3_bao_cao.md`](buoc3_bao_cao.md).

---

## Toàn cảnh

```
SafeViChi/
├── data/            dữ liệu — xem §1
├── models/          checkpoint — xem §2
├── results/         kết quả, log, hình — xem §3
├── src/             mã nguồn — xem §4
├── scripts/         script điều phối — xem §5
├── tests/           kiểm thử — xem §6
├── docs/            tài liệu — xem §7
└── (gốc repo)       cấu hình — xem §8
```

Luồng dữ liệu chảy một chiều:

```
ViHSD (nguồn ngoài)
   │  src/dataset_builder/build.py                          [Bước 1]
   ▼
data/processed/vietnamese_nonstandard_v1_2/    75.048 câu + biến thể
   │  src/dataset_builder/build_normalized.py               [Bước 2]
   ▼
data/processed/normalized/                     câu đã qua normalize()
   │  src/dataset_builder/build_mixed.py
   ▼
data/data_final/train.jsonl                    bộ trộn 60/30/10
   │  src/classifier/train.py                               [Bước 3]
   ▼
models/vihatet5-v2-best/
   │  src/baselines/run_matrix.py  ← data/eval_baseline/
   ▼
results/step3/{val,test}/          →  tune_threshold  →  make_figures
```

---

## 1. `data/` — dữ liệu

| Đường dẫn | Vai trò | Xóa được? |
|---|---|---|
| `processed/vietnamese_nonstandard_v1_2/` | **Đầu ra Bước 1.** 75.048 câu gốc + biến thể, kèm toàn bộ siêu dữ liệu (`perturbation_edits`, `duplicate_group_id`, seed từng dòng…). Nguồn sinh ra mọi thứ bên dưới. | ❌ **Không.** Mất là phải chạy lại toàn bộ Bước 1 |
| `processed/normalized/` | Trung gian Bước 2: `{text: normalize(text), label}`, ánh xạ 1-1 theo thứ tự dòng. Đầu vào của `build_mixed.py` | ❌ Không — bản hiện tại không tái tạo được (xem báo cáo §6.3) |
| `data_final/train.jsonl` | **Bộ huấn luyện.** 116.943 dòng, trộn 60/30/10, 49,19% HATE | ⚠️ Dựng lại được, nhưng sẽ khác bộ đã train |
| `data_final/validation_mixed.jsonl` | 4.176 dòng, bản trộn. **File thật dùng cho early stopping** | ⚠️ như trên |
| `data_final/validation.jsonl` | 2.506 dòng, C0 câu sạch. Bản sao của `eval_baseline/val/C0_clean.jsonl` | ✓ Copy lại được |
| `data_final/test.jsonl` | 2.376 dòng, C0 câu sạch. Bản sao của `eval_baseline/test/C0_clean.jsonl` | ✓ Copy lại được |
| `data_final/manifest_mix.json` | Thống kê bộ trộn: tỉ lệ, seed, `by_source_label`. **Kiểm file này sau mỗi lần dựng lại** | ✓ |
| `eval_baseline/test/C0_clean.jsonl` | Điều kiện C0 — `original_text`, chưa cấy biến thể | ⚠️ |
| `eval_baseline/test/C1_perturbed.jsonl` | Điều kiện C1 — đã cấy biến thể, **chưa** chuẩn hóa | ⚠️ |
| `eval_baseline/test/C2_blindspot.jsonl` | Điều kiện C2 — điểm mù, `normalize(x) == x` được assert | ⚠️ |
| `eval_baseline/val/…` | Ba điều kiện tương ứng trên validation, **chỉ dùng để dò ngưỡng** | ⚠️ |
| `eval_baseline/*/manifest.json` | Nguồn, seed, số dòng, phân bố nhãn từng điều kiện | ✓ |
| `CHECKSUMS.json` | SHA-256 của 15 file dữ liệu + model. Chốt an toàn chống sửa nhầm | ✓ (`scripts.checksums --write`) |
| `README.md` | Sơ đồ bố cục dữ liệu, schema JSONL, cách tái tạo | ✓ |
| `raw/`, `variants/` | Rỗng, chỉ có `.gitkeep`. Chỗ để dữ liệu tải về của Bước 1 | ✓ |

**Vì sao `data_final/test.jsonl` trùng hash với `eval_baseline/test/C0_clean.jsonl`:**
chúng là một. Nhân đôi cho tiện dùng — `train.py` cần một file test đơn giản,
`run_matrix.py` cần cả ba điều kiện nằm cùng thư mục.

---

## 2. `models/` — checkpoint

| Đường dẫn | Vai trò |
|---|---|
| `vihatet5-v2-best/model.safetensors` | **Trọng số của checkpoint tốt nhất** — epoch 4, F1 val 0,7881. Xác minh bằng SHA-256 là trùng `checkpoint-14620` |
| `vihatet5-v2-best/config.json` | Cấu hình kiến trúc T5 |
| `vihatet5-v2-best/generation_config.json` | Tham số sinh chuỗi |
| `vihatet5-v2-best/tokenizer.json`, `tokenizer_config.json` | Tokenizer ViHateT5 |
| `vihatet5-v2-best/trainer_state.json` | Nhật ký huấn luyện: F1 từng epoch, checkpoint tốt nhất. Giữ làm bằng chứng |

Không có `optimizer.pt` / `scheduler.pt` — chúng chỉ cần để **train tiếp**, không
cần để suy luận, mà lại nặng ~2 GB mỗi checkpoint.

`models/` nằm trong `.gitignore` (file 1,2 GB, quá nặng cho git).

---

## 3. `results/step3/` — kết quả

| Đường dẫn | Vai trò |
|---|---|
| `test/matrix.json` | Kết quả ma trận trên test ở **ngưỡng mặc định 0,5**, kèm 15 phép McNemar |
| `test/matrix.md` | Bản markdown của trên, đọc nhanh |
| `test/scores.jsonl` | **P(hate) từng dòng**, 26.688 dòng. Schema: `{system, condition, row_id, label, p_hate}`. Nhờ file này mà dò ngưỡng và vẽ ROC/PR **không cần GPU** |
| `threshold/threshold.json` | **Số liệu báo cáo chính**: ngưỡng đã chọn + chỉ số tại ngưỡng đó cho 4 hệ thống T5 |
| `logs/train_eval_v2.log` | **Log gốc của lần chạy thật.** Bằng chứng: F1 từng epoch, thời gian, dò ngưỡng |
| `logs/train_v1_lr3e-4_DEPRECATED.log` | Log lần chạy đầu (lr 3e-4) — giữ để đối chiếu vì sao phải hạ learning rate. **Không dùng số liệu trong đây** |
| `figures/*.pdf` | 8 hình/bảng, PDF vector. Xem báo cáo §8 |

> **Không có `val/`**: kết quả validation của lần chạy lại (sau khi sửa lỗi chữ
> hoa) không được tải về — bản cũ đã xóa để tránh lẫn số liệu hai lần chạy.
> Ngưỡng đã dò trên validation vẫn được lưu nguyên trong `threshold/threshold.json`,
> nên báo cáo không thiếu gì. Chỉ khi muốn **dò lại ngưỡng từ đầu** mới cần
> chạy lại ma trận trên validation (xem `buoc3_tai_lap_pipeline.md` §3).

---

## 4. `src/` — mã nguồn

### 4.1 `src/utils/` — hạ tầng dùng chung

| File | Vai trò |
|---|---|
| `seed.py` | **Quản lý ngẫu nhiên tập trung.** `set_global_seed()` seed cùng lúc `random`/`numpy`/`torch`/`torch.cuda`/`PYTHONHASHSEED`. `derive_seed()` sinh seed con bằng BLAKE2b (không dùng `hash()` vì nó bị ngẫu nhiên hóa theo tiến trình). Chứa hằng `SEED_STEP1_DATASET = 20260902` và `SEED_DEFAULT = 42` |

### 4.2 `src/normalization/` — Bước 2

| File | Vai trò |
|---|---|
| `normalizer.py` | **Bộ chuẩn hóa 5 tầng.** NFC → lookalike (`4`→`a`) → separator (`n.g.u`→`ngu`) → elongation (`nguuuu`→`ngu`) → teencode. Triết lý bảo thủ: "thà không sửa còn hơn sửa sai" |
| `teencode_dict.json` | Từ điển teencode, 625 mục |
| `build_dictionary.py` | Sinh `teencode_dict.json` từ dữ liệu |
| `evaluate_variants.py` | Đo độ phủ của bộ chuẩn hóa trên các loại biến thể |

### 4.3 `src/variant_generator/` — Bước 1

| File | Vai trò |
|---|---|
| `controlled.py` | 12 luật cấy biến thể, dùng để dựng bộ 75K |
| `blindspot.py` | 6 kỹ thuật **điểm mù**: homoglyph Cyrillic, ký tự rộng-0, separator lạ (`* / \| + ~ ^`), chèn khoảng trắng, kéo dài kép, Unicode fullwidth. Mọi kết quả phải thỏa `normalize(x) == x` |
| `teencode.py` | Từ vựng teencode cho bộ sinh |
| `generator.py` | Điều phối sinh biến thể |

### 4.4 `src/dataset_builder/` — dựng dữ liệu

| File | Vai trò | Chạy khi nào |
|---|---|---|
| `build.py` | **Bước 1.** Dựng bộ 75K từ ViHSD. Seed 20260902, dẫn seed từng dòng bằng hash | Chỉ khi dựng lại từ nguồn |
| `sources.py`, `dedup.py`, `quality.py`, `schema.py`, `constants.py` | Bộ phận của `build.py`: tải nguồn, chống trùng lặp/rò rỉ, kiểm chất lượng, định nghĩa schema | (gọi gián tiếp) |
| `build_normalized.py` | **Bước 2 áp lên dữ liệu.** Sinh `data/processed/normalized/`. Mặc định từ chối ghi đè | Khâu 1 của `build_step3_data.sh` |
| `build_mixed.py` | Dựng bộ trộn 60/30/10 | Khâu 2 |
| `build_eval_matrix.py` | Dựng 3 điều kiện C0/C1/C2 | Khâu 3 và 4 |

### 4.5 `src/classifier/` — Bước 3, huấn luyện

| File | Vai trò |
|---|---|
| `train.py` | **Điểm vào huấn luyện.** Nạp dữ liệu, tokenize, dựng `FGMSeq2SeqTrainer`, train, eval. Ghi `run_config.json` trước khi train |
| `adversarial.py` | `FGMSeq2SeqTrainer` — tấn công đối kháng ở tầng gradient. Double backward mỗi bước, tấn công embedding `shared`, khôi phục. Chạy đúng dưới `nn.DataParallel` |
| `data.py` | Đọc JSONL + tokenize. Prompt `"vihsd: {text}"` → đích `"hate"`/`"clean"` |
| `metrics.py` | Tính chỉ số, tách theo nhóm nguồn (`{source}_f1`). Chứa `LABEL2ID` |
| `evaluate.py` | Eval độc lập một checkpoint bất kỳ trên một file JSONL, không train |

### 4.6 `src/baselines/` — Bước 3, đánh giá

| File | Vai trò |
|---|---|
| `run_matrix.py` | **Điểm vào đánh giá.** Chấm 6 hệ thống × 3 điều kiện. Quyết định bằng **1 bước forced-decode** (so logit token `hate` id=9267 với `clean` id=6270), không sinh chuỗi đầy đủ. Cờ `--dump_scores` ghi `scores.jsonl` |
| `blacklist.py` | Baseline B1: xây danh sách từ khóa từ dữ liệu train, và chấm bằng cách khớp từ khóa |
| `blacklist_words.json` | Danh sách từ khóa đã xây |
| `stats.py` | `bootstrap_ci()` (1.000 lần lấy mẫu lại, seed 42) và `mcnemar()` (binomial test chính xác, dùng được cả khi số dòng khác biệt ít) |
| `tune_threshold.py` | Dò ngưỡng trên validation → đóng băng → áp lên test. **Chạy CPU** |

### 4.7 `src/reporting/` — Bước 3, hình và bảng

| File | Vai trò |
|---|---|
| `style.py` | Bảng màu định danh (3 màu + gạch chéo), `rcParams`, `add_caption()`, `save_pdf()` (chặn cứng mọi đuôi khác PDF) |
| `load.py` | Đọc và **hợp nhất** kết quả: B2/B3 lấy số ở ngưỡng đã dò, B1 lấy số ở ngưỡng mặc định. Ghép nhầm hai loại là báo cáo sai |
| `make_figures.py` | Vẽ 6 hình + 2 bảng, xuất PDF. **Chạy CPU trong vài giây** |

### 4.8 Bước 4 — chưa bắt đầu

| File | Vai trò |
|---|---|
| `src/explainer/occlusion.py` | Khung sườn module giải thích bằng occlusion. **Chưa cài đặt** — xem `plan.md` |
| `src/pipeline.py` | Khung sườn ghép chuẩn hóa → phân loại → giải thích cho demo. **Chưa hoàn chỉnh** |

---

## 5. `scripts/` — điều phối

| File | Vai trò |
|---|---|
| `build_step3_data.sh` | Dựng lại **toàn bộ** dữ liệu Bước 3 qua 5 khâu, tự đối chiếu checksum ở cuối |
| `run_step3_eval.sh` | Chạy ma trận trên val + test + dò ngưỡng. Nhận đường dẫn checkpoint làm tham số |
| `checksums.py` | Băm SHA-256 15 file và đối chiếu. `--write` để chốt bảng mới. Mã thoát 1 nếu lệch — dùng được trong CI |
| `download_dataset_sources.ps1` | Tải nguồn dữ liệu cho Bước 1 (PowerShell, dùng trên Windows) |

---

## 6. `tests/` — kiểm thử

91 test, chạy bằng `python -m pytest -q`, mất ~3 giây.

| File | Kiểm cái gì |
|---|---|
| `test_normalizer.py` | Bộ chuẩn hóa 5 tầng |
| `test_controlled_generator.py` | 12 luật cấy biến thể |
| `test_teencode_lexicon.py` | Từ điển teencode |
| `test_dataset_sources.py` | Tải và ánh xạ nguồn |
| `test_dedup_and_splits.py` | Chống trùng lặp, chống rò rỉ giữa các split |
| `test_build_pipeline.py` | Toàn tuyến dựng dữ liệu Bước 1 |
| `test_seed.py` | Module seed. Có test chạy 3 tiến trình con với `PYTHONHASHSEED` khác nhau để chứng minh `derive_seed()` không phụ thuộc nó |
| `test_reporting.py` | Ràng buộc chỉ-xuất-PDF; ghép đúng nguồn số liệu; **mỗi hệ thống chỉ dùng một ngưỡng cho cả ba điều kiện**; khoảng tin cậy phải bao quanh giá trị đo |

---

## 7. `docs/` — tài liệu

| File | Nội dung |
|---|---|
| `buoc3_bao_cao.md` | **Hồ sơ kết quả Bước 3.** Thiết kế, số liệu, seed, quan sát, điểm yếu |
| `buoc3_tai_lap_pipeline.md` | **Hướng dẫn chạy lại.** Từng khâu, cần GPU hay không, xử lý sự cố |
| `buoc3_ban_do_file.md` | File này |
| `dataset_pipeline.md` | Tài liệu kỹ thuật Bước 1 |
| `plan.md` | Kế hoạch tổng thể dự án, gồm cả Bước 4 |
| `kaggle/RUN_V2.md` | Đóng gói và chạy trên Kaggle (2×T4) |
| `kaggle/RUN_EVAL_V1_DEPRECATED.md` | Hướng dẫn của vòng chạy đầu. **Đã lỗi thời**, giữ để đối chiếu |

---

## 8. Gốc repo

| File | Vai trò |
|---|---|
| `CLAUDE.md` | Ngữ cảnh dự án cho trợ lý lập trình |
| `README.md` | Giới thiệu chung |
| `plan.md` | Kế hoạch Bước 4 (occlusion) |
| `requirements.txt` | Khoảng phiên bản chấp nhận được |
| `requirements-lock.txt` | **Bản khóa chính xác** 72 gói của môi trường đã tạo ra kết quả |
| `.gitignore` | Bỏ qua `data/`, `models/`, `*.log` — **trừ** `results/step3/logs/*.log` (giữ làm bằng chứng) |

---

## 9. Tra ngược: "tôi muốn sửa X thì vào đâu"

| Muốn sửa | Vào file |
|---|---|
| Luật chuẩn hóa | `src/normalization/normalizer.py` |
| Từ điển teencode | `src/normalization/teencode_dict.json` |
| Kỹ thuật cấy biến thể | `src/variant_generator/controlled.py` |
| Kỹ thuật điểm mù | `src/variant_generator/blindspot.py` |
| Tỉ lệ 60/30/10 | `src/dataset_builder/build_mixed.py`, cờ `--clean_ratio` / `--blind_ratio` |
| Siêu tham số huấn luyện | `src/classifier/train.py`, `build_argparser()` |
| Cách tấn công FGM | `src/classifier/adversarial.py` |
| Thêm/bớt hệ thống baseline | `src/baselines/run_matrix.py`, hàm `run_system()` |
| Danh sách từ khóa blacklist | `src/baselines/blacklist_words.json` |
| Cách tính khoảng tin cậy | `src/baselines/stats.py` |
| Màu sắc, font, kiểu hình | `src/reporting/style.py` |
| Bố cục từng hình | `src/reporting/make_figures.py` |
| Danh sách file được băm | `scripts/checksums.py`, biến `TRACKED` |
| Hằng seed | `src/utils/seed.py` |
