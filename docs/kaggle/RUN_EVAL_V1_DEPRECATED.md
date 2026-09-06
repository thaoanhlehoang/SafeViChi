# Ma trận đánh giá baseline (Step 3) — chạy trên Kaggle

## 3 dataset cần upload lên Kaggle

| # | Tên dataset gợi ý | Nội dung | Cỡ |
|---|---|---|---|
| 1 | `safevichi-eval-code` | giải nén `safevichi_eval_code.zip` | ~1 MB |
| 2 | `safevichi-eval-data` | giải nén `safevichi_eval_data.zip` | ~1.1 MB |
| 3 | `vihatet5-adversarial` | 5 file checkpoint (xem bên dưới) | ~1.2 GB |

Checkpoint (#3) chỉ cần **5 file ở thư mục gốc** `vihatet5-adversarial/`:
`config.json`, `generation_config.json`, `model.safetensors`,
`tokenizer.json`, `tokenizer_config.json`.

Bỏ hẳn 3 thư mục `checkpoint-*` (10.8 GB toàn optimizer state, eval không dùng).
File `model.safetensors` ở thư mục gốc **đã chính là checkpoint-7310** — bản tốt
nhất theo val F1 = 0.7894, đã verify bằng md5.

## Bật GPU
Notebook Settings → Accelerator → **GPU T4 x2** (1 GPU cũng đủ; script chỉ
forward pass, ~28K câu, chừng 5-10 phút).

## Cách chạy

```bash
!cp -r /kaggle/input/datasets/truongdinhv/safevichi-eval-code/* /kaggle/working/
%cd /kaggle/working
!python -m src.baselines.run_matrix \
    --data_dir /kaggle/input/datasets/truongdinhv/safevichi-eval-data/eval_matrix \
    --checkpoint /kaggle/input/datasets/truongdinhv/vihatet5-adversarial \
    --base_model tarudesu/ViHateT5-base-HSD \
    --out_dir /kaggle/working/results
```

Ra 2 file: `results/matrix.json` (đầy đủ) và `results/matrix.md` (bảng markdown).

Smoke-test trước cho nhanh: thêm `--limit 64`.

## Ma trận

6 hệ thống × 3 điều kiện input. Mọi hệ thống nhận **cùng một chuỗi đầu vào**
trong cùng một điều kiện; `normalize()` là bộ phận bên trong hệ thống, chạy
lúc eval, không nướng sẵn vào file dữ liệu.

| Hệ thống | Xử lý |
|---|---|
| `b1_blacklist` | blacklist(thô) — sàn, không model |
| `b1_blacklist_norm` | blacklist(normalize(thô)) |
| `b2_t5base` | T5 gốc(thô) |
| `b2_t5base_norm` | T5 gốc(normalize(thô)) |
| `b3_t5ft` | T5 fine-tune(thô) |
| `b3_t5ft_norm` | T5 fine-tune(normalize(thô)) — **pipeline đầy đủ** |

| Điều kiện | File | Dòng | HATE |
|---|---|---|---|
| C0_clean | `test_clean.jsonl` | 2376 | 432 (18.2%) |
| C1_perturbed | `test_perturbed.jsonl` | 2376 | 432 (18.2%) |
| C2_blindspot | `test_blindspot.jsonl` | 1920 | 321 (16.7%) |

C0 và C1 khớp 1-1 từng dòng qua `row_id`. C2 ít hơn vì 456 dòng sinh điểm mù
không thành công (đã bỏ).

## Đọc kết quả

Số chính là **F1 lớp HATE** kèm khoảng tin cậy bootstrap. **Không dùng
accuracy** — đoán bừa CLEAN hết đã được 81.8%.

| So sánh | Trả lời câu hỏi |
|---|---|
| `b2_t5base`: C0 → C1 | né lọc làm T5 gốc sập bao nhiêu (tiền đề của đề tài) |
| `b2_t5base` → `b2_t5base_norm` trên C1 | normalizer (Step 2) đóng góp bao nhiêu |
| `b2_t5base_norm` → `b3_t5ft_norm` trên C1 | fine-tune đối kháng (Step 3) đóng góp bao nhiêu |
| `b3_t5ft` → `b3_t5ft_norm` | model đã fine-tune còn cần normalizer không |
| cột C2 | normalizer bó tay, chỉ fine-tune cứu được |
| `b1_blacklist` → `b3_t5ft_norm` | pipeline hơn cách ngây thơ nhất bao nhiêu |

Mỗi cặp đều có kiểm định McNemar ghép cặp trên cùng từng dòng (mạnh hơn so
hai con số F1 rời rạc) trong `matrix.json` → `comparisons`.

## Lưu ý phải ghi vào báo cáo

Dataset dựng từ **ViHSD train.csv**, mà `tarudesu/ViHateT5-base-HSD` cũng đã
fine-tune trên chính ViHSD. Nên cột **C0 của b2_t5base là đang test trên dữ
liệu nó từng học** → điểm sẽ cao ảo. Điều này *có lợi* cho lập luận (B2 được
ưu ái mà vẫn thua khi bị tấn công), nhưng phải nói rõ. Muốn chặt chẽ hơn thì
dựng thêm eval set từ ViHSD `test.csv`.
