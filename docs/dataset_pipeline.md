# Pipeline dữ liệu tiếng Việt phi chuẩn

## Kết quả đích

Pipeline tạo đúng 75.048 mẫu với phân bố nhãn nhị phân:

| Nhãn đích | Số mẫu |
| --- | ---: |
| `HATE` | 35.162 |
| `CLEAN` | 39.886 |

Ánh xạ ViHSD là `HATE → HATE`, `OFFENSIVE → HATE`, `CLEAN → CLEAN`.
VOZ-HSD dùng `1 → HATE`, `0 → CLEAN`.

Nguồn được pin để kết quả không thay đổi âm thầm:

| Nguồn | Revision nguồn | Artifact dùng |
| --- | --- | --- |
| `uitnlp/vihsd` | `88e81b36ca376867640dad9df295e0f7d499ab80` | `train.csv` |
| `tarudesu/VOZ-HSD` | `923915f5f633c503babf2c6dcf7003593318b04f` | Parquet revision `2518cc1fd5287975fbc6f842fb29625880c012f4` |

## Kiểm tra nguồn và quota thực tế

Revision ViHSD đã pin có 24.048 dòng train: 19.886 `CLEAN`, 1.606
`OFFENSIVE`, 2.556 `HATE`. Sau ánh xạ là 19.886 `CLEAN` và 4.162 `HATE`.

Không thể vừa khẳng định label preservation vừa giữ nguyên mọi dòng lỗi:

- có 2 văn bản rỗng (một `CLEAN`, một `OFFENSIVE`);
- có 110 nhóm exact duplicate mang hai nhãn nhị phân trái ngược, tính cả nhóm
  rỗng; sau khi tách nhóm rỗng, 245 dòng còn lại thuộc các nhóm mâu thuẫn.

Mặc định, 247 dòng này được quarantine, chỉ ghi source row id, nhãn, hash và lý
do — không ghi lại nội dung. ViHSD hợp lệ còn 4.042 `HATE` và 19.759 `CLEAN`.
Để vẫn đạt đúng tổng và phân bố nhãn đích, quota VOZ-HSD tự điều chỉnh thành
31.120 `HATE` và 20.127 `CLEAN` (51.247 mẫu, vẫn xấp xỉ 51K). Có thể dùng
`--conflict-policy keep` để tái tạo đúng phép tính 31K/20K ban đầu, nhưng không
khuyến nghị vì các dòng mâu thuẫn không có cơ sở kiểm chứng nhãn.

## Nhãn yếu của VOZ-HSD

VOZ-HSD có 10.747.733 bình luận với các cột `texts`, `labels`, `probs`.
Các nhãn được mô hình ViSoBERT-HSD tự gán. Dataset card của tác giả nói nguồn
chỉ phục vụ nghiên cứu và các nhãn tự động được dùng cho phân tích, không dùng
fine-tune downstream.

Vì vậy pipeline:

- ghi `annotation_type=weak_ai` và giữ `source_confidence`;
- mặc định chỉ lấy mẫu có `probs >= 0.90`;
- loại exact duplicate trong phần VOZ đã chọn và ưu tiên bản ViHSD nếu trùng;
- ép toàn bộ VOZ vào `target_split=train`;
- không coi VOZ là dữ liệu benchmark vàng;
- ghi `distribution_status=rights_and_weak_label_review_required_before_release`.

Cần xin phép nếu cần và tái gán nhãn bằng người trước khi phát hành hoặc dùng
để huấn luyện chính thức. Không được báo cáo validation/test dựa trên nhãn VOZ.

## Lấy mẫu tái lập

VOZ-HSD được đọc một lượt theo các shard Parquet đã pin. Với từng lớp, pipeline
giữ một priority reservoir gồm các hash nhỏ nhất, trong đó hash phụ thuộc vào
master seed, revision, shard/row id và hash nội dung. Phương pháp này:

- phân tầng đúng quota `HATE`/`CLEAN`;
- không phụ thuộc global RNG;
- dùng bộ nhớ tỉ lệ với kích thước mẫu, không phải 10,7 triệu dòng;
- cho cùng lựa chọn khi seed, revision và tham số không đổi.

Mặc định scan hết nguồn để tránh thiên lệch theo thứ tự. `--voz-max-rows` chỉ
dành cho smoke test; manifest sẽ ghi `scan_complete=false`. Nếu đã tải bốn
Parquet về máy, truyền chúng qua `--voz-parquet path1 ... path4`.

Nếu một bước hậu xử lý lỗi sau khi đã ghi được `train.jsonl`, có thể khôi phục
đúng tập VOZ đã chọn mà không quét lại nguồn bằng
`--voz-selection-jsonl <train.jsonl> --overwrite`. Loader sẽ kiểm tra revision,
quota, confidence, uniqueness và overlap với ViHSD trước khi chấp nhận cache;
manifest ghi rõ đây là lượt recovery.

## Augmentation và label preservation

Mỗi dòng có từ một đến ba edit cục bộ. Không có paraphrase tự do và không chèn
nội dung độc hại mới. Mười hai loại cơ sở gồm:

1. `abbreviation_clipping`;
2. `intentional_spelling`;
3. `phonetic_spelling`;
4. `dialectal_writing`;
5. `slang_lexical`;
6. `expressive_lengthening`;
7. `diacritic_variation`;
8. `surface_obfuscation`;
9. `boundary_variation`;
10. `context_dependent`;
11. `typographical_noise`;
12. `teencode_lexical`.

`src/normalization/teencode_dict.json` có chiều `teencode → dạng chuẩn`. Trước
khi sinh dữ liệu, compiler chuẩn hóa NFC, phát hiện xung đột/chain, loại các cặp
trùng với luật curated rồi tạo index ổn định `dạng chuẩn → các biến thể
teencode`. Với dictionary hiện tại, 589/625 cặp được chấp nhận; 36 cặp bị loại
có lý do cụ thể (20 trùng curated, 2 xung đột curated, 14 normalization chain).
`teencode_lexicon_audit.json` ghi summary và toàn bộ cặp bị loại để có thể rà
soát thủ công mà không âm thầm bỏ qua xung đột.

Mặc định `teencode_lexical` được bắt buộc trên quota làm tròn 50% số mẫu đủ điều
kiện trong từng nhóm `(source_dataset, label, target_split)`. Mẫu được chọn bằng
hash ổn định từ master seed và provenance nguồn. Mẫu không thuộc quota bị tắt
riêng loại này, vì vậy số áp dụng không bị tăng ngoài kế hoạch do thứ tự random
của các transform. Trong mỗi edit, generator chọn span chuẩn trước rồi mới chọn
biến thể; standard có nhiều biến thể không được ưu tiên chỉ vì bucket lớn.

`perturbation_mode=mixed` có ít nhất hai loại edit thực tế. Các phép thay từ/cụm
từ chỉ dùng bảng tương đương đã duyệt và khớp trọn token/cụm; URL, email,
mention và hashtag được bảo vệ. Lỗi gõ chỉ tác động một ký tự nội bộ. Guard tự
động yêu cầu đầu ra khác nguồn, không rỗng, giới hạn tỷ lệ độ dài/độ tương tự và
không cho một edit vô tình tạo high-risk token mới ở mẫu `CLEAN`.

Các guard này chỉ là điều kiện cần. Không có span độc hại hay mô hình semantic
nào có thể tự chứng minh tuyệt đối việc giữ nhãn; vì thế kiểm tra người là bắt
buộc.

## Trace và schema

Mỗi phần tử `perturbation_edits` chứa:

- `step`;
- `perturbation_type`, `rule_id`;
- `before_start`, `before_end`: offset trong trạng thái text trước bước đó;
- `after_start`, `after_end`: offset sau thay thế;
- `before`, `after`.

Edit `teencode_lexical` còn có `resource_name`, `resource_version`,
`resource_sha256`, `canonical_form`, `selected_variant` và `candidate_count`.
`rule_id` là hash ổn định của cặp, và QA bắt buộc cặp sinh ra ánh xạ trực tiếp
trở lại `canonical_form` trong đúng dictionary đã ghi hash.

Áp lần lượt trace lên `original_text` phải thu được chính xác `text`. Các trường
chính khác gồm:

- định danh: `sample_id`, `dataset_version`, `generator_version`;
- nhãn: `label_original`, `label_mapped`, `label`, `label_id`;
- nguồn: `source_dataset`, `source_revision`, `source_artifact_revision`,
  `source_artifact`, `source_split`, `source_row_id`, `source_confidence`,
  `annotation_type`;
- chia tập/dedup: `target_split`, `duplicate_group_id`,
  `duplicate_group_size`;
- augmentation: `perturbation_primary_requested`, `perturbation_types`,
  `perturbation_mode`, `perturbation_count`, `random_seed`, trace;
- audit: hash SHA-256 của source/output và `quality_flags`.

`random_seed` được lưu dưới dạng chuỗi hexadecimal 16 ký tự để không mất độ
chính xác khi đọc JSON bằng JavaScript; chuyển lại bằng `int(value, 16)` trong
Python.

## Duplicate, near-duplicate và data leakage

Exact duplicate được nhận diện sau NFKC, case-fold, bỏ control/format character
và chuẩn hóa khoảng trắng. Near-duplicate dùng SimHash LSH để sinh ứng viên,
sau đó xác nhận bằng character-trigram Jaccard (mặc định `>= 0.90`), khoảng cách
Hamming (`<= 8`) và tỷ lệ độ dài (`>= 0.85`). Các chuỗi chỉ gồm emoji/dấu câu
với khác biệt lặp ký tự cũng được gom nhóm.

Các mẫu cùng nhóm luôn vào cùng split. Nếu nhóm có ít nhất một nhãn yếu VOZ,
cả nhóm vào train. Những nhóm chỉ có ViHSD được hash ổn định theo tỷ lệ mặc định
80/10/10. Exact duplicate ViHSD cùng nhãn được giữ để bảo toàn coverage nguồn,
nhưng nhận seed/biến thể riêng và không thể rò sang split khác. Manifest báo đầy
đủ kích thước nhóm và số cạnh exact/near duplicate.

## QA tự động và thủ công

Build chỉ hoàn tất khi các invariant sau qua hết:

- đúng tổng và quota nhãn;
- 100% mẫu có thay đổi, không rỗng và có trace replay được;
- đủ 12 loại perturbation và có mẫu mixed;
- quota `teencode_lexical` đúng theo từng nguồn/nhãn/split, mọi target đều được
  áp dụng và mọi edit đều round-trip qua dictionary;
- không group nào đi qua nhiều split;
- không exact hoặc near-duplicate group nào của cả source lẫn perturbed output
  xuất hiện ở hai split;
- không nhãn `weak_ai` nào vào validation/test;
- schema nhị phân và số edit nhất quán.

`human_review_sample.csv` chọn 250 mẫu tái lập, đảm bảo phủ nhãn, nguồn, split,
loại perturbation và mixed trước khi lấp phần còn lại ngẫu nhiên theo hash. Nên
có ít nhất hai annotator độc lập chấm:

1. naturalness từ 1–5;
2. semantic preservation yes/no;
3. label preservation yes/no;
4. ghi chú lỗi và loại biến đổi.

Ngưỡng phát hành đề xuất: ít nhất 90% mẫu có naturalness `>=3`, ít nhất 95%
giữ ngữ nghĩa, ít nhất 98% giữ nhãn, đồng thời công bố agreement giữa annotator.
Nếu một nhóm biến đổi không đạt, chỉnh rule/weight rồi dựng lại toàn bộ bằng
generator version mới — không sửa tay âm thầm trên artifact.

## Lệnh chạy

```powershell
python -m pip install -r requirements.txt
hf auth login
./scripts/download_dataset_sources.ps1
python -m src.dataset_builder.build
python -m pytest -q --basetemp .pytest_tmp
```

Hai tham số liên quan đến teencode:

```powershell
python -m src.dataset_builder.build `
  --teencode-dict src/normalization/teencode_dict.json `
  --teencode-rate 0.50
```

Ví dụ dùng Parquet VOZ đã tải và đổi seed:

```powershell
python -m src.dataset_builder.build `
  --voz-parquet data/raw/voz/default/train/0000.parquet `
                 data/raw/voz/default/train/0001.parquet `
                 data/raw/voz/default/train/0002.parquet `
                 data/raw/voz/default/train/0003.parquet `
  --seed 20260902
```

Không dùng `--overwrite` nếu cần giữ artifact cũ. Khi dùng, pipeline chỉ thay
thế các tên artifact đã biết trong output directory, không xóa cả thư mục.

Artifact phát hành `v1_1` còn có `train.csv`, `validation.csv` và `test.csv`
được xuất một lần từ các file Parquet tương ứng. Ba file chỉ giữ đúng các cột
`text`, `original_text`, `label`; chúng không phải output mặc định của build CLI.
