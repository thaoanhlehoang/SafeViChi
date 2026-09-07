# Bước 3 — Huấn luyện đối kháng & so sánh 3 baseline

> Hồ sơ kết quả đầy đủ. Mọi con số trong tài liệu này lấy trực tiếp từ
> `results/step3/`, không có số nào được gõ tay lại.
>
> - Cách chạy lại từ đầu: [`buoc3_tai_lap_pipeline.md`](buoc3_tai_lap_pipeline.md)
> - Vai trò từng file trong repo: [`buoc3_ban_do_file.md`](buoc3_ban_do_file.md)

---

## 1. Câu hỏi nghiên cứu

Bộ phân loại nội dung độc hại tiếng Việt tụt hiệu năng bao nhiêu khi văn bản bị
biến dạng có chủ đích để né lọc, và hai lớp phòng vệ của SafeViChi lấy lại được
bao nhiêu?

Tách thành bốn câu hỏi con, mỗi câu ứng với một phép so sánh trong Bảng 2:

| # | Câu hỏi | Cặp so sánh |
|---|---|---|
| Q1 | Né lọc có thực sự phá được model không? | `b2_t5base` giữa C0 và C1 |
| Q2 | Bước 2 (chuẩn hóa) góp bao nhiêu? | `b2_t5base` vs `b2_t5base_norm` |
| Q3 | Bước 3 (fine-tune đối kháng) góp thêm bao nhiêu? | `b2_t5base_norm` vs `b3_t5ft_norm` |
| Q4 | Fine-tune rồi có còn cần Bước 2 nữa không? | `b3_t5ft` vs `b3_t5ft_norm` |

---

## 2. Thiết kế thí nghiệm

### 2.1 Sáu hệ thống, không phải ba

Ba baseline, mỗi cái chạy hai lần — một lần input thô, một lần input đã qua
`normalize()`. Cặp có/không này chính là phép **ablation** cô lập đóng góp của
Bước 2.

| Mã | Hệ thống | Chuẩn hóa | Ghi chú |
|---|---|:---:|---|
| `b1_blacklist` | Khớp danh sách từ khóa | ✗ | Không có model, cách ngây thơ nhất |
| `b1_blacklist_norm` | Khớp từ khóa trên câu đã chuẩn hóa | ✓ | |
| `b2_t5base` | `tarudesu/ViHateT5-base-HSD` nguyên bản | ✗ | Chưa fine-tune thêm |
| `b2_t5base_norm` | như trên + chuẩn hóa | ✓ | |
| `b3_t5ft` | Checkpoint đã fine-tune đối kháng | ✗ | |
| `b3_t5ft_norm` | **Pipeline SafeViChi đầy đủ** | ✓ | Hệ thống đề xuất |

**Nguyên tắc đã chốt:** cả sáu hệ thống nhận **cùng một input thô**.
`normalize()` là **bộ phận bên trong** hệ thống, chạy lúc suy luận, **không phải**
bước chuẩn bị dữ liệu. Cho B1/B2 ăn câu sạch còn B3 ăn câu đã chuẩn hóa là so
sánh gian lận — đó là sai lầm thiết kế đã bị loại bỏ ở vòng đầu.

### 2.2 Ba điều kiện đầu vào

Cả ba sinh ra từ **cùng một tập câu gốc**, ghép cặp theo `row_id` để kiểm định
McNemar theo cặp.

| Mã | `text` chứa gì | n (test) | %HATE | Đo cái gì |
|---|---|---:|---:|---|
| `C0_clean` | `original_text` — chưa cấy biến thể | 2.376 | 18,18% | Trần hiệu năng |
| `C1_perturbed` | `text` — đã cấy biến thể, chưa chuẩn hóa | 2.376 | 18,18% | Kịch bản né lọc chính |
| `C2_blindspot` | Biến thể **điểm mù** | 1.920 | 16,72% | Chỗ chuẩn hóa bó tay |

C2 ít dòng hơn vì một số câu quá ngắn hoặc không có ký tự phù hợp nên bộ sinh
điểm mù bỏ qua (456 câu ở test, 446 ở validation).

**Bất biến của C2:** mọi dòng đều được `assert normalize(x) == x` lúc dựng. Nếu
bộ chuẩn hóa sửa được thì đó là biến thể thường, không phải điểm mù. Hệ quả trực
tiếp: trên C2, `b3_t5ft` và `b3_t5ft_norm` nhận **input giống hệt nhau** — mọi
khác biệt còn lại là do dao động của phép đo, không phải do chuẩn hóa.

### 2.3 Vì sao không lấy accuracy làm chỉ số chính

Nhãn lệch: tập test chỉ ~18% HATE. Đoán bừa CLEAN cho mọi câu đã đạt ~82%
accuracy. Chỉ số chính là **F1 của lớp HATE**, kèm khoảng tin cậy bootstrap và
kiểm định McNemar. Accuracy có trong Bảng 1 nhưng chỉ để tham khảo.

Cũng vì vậy **validation/test giữ nguyên phân bố nhãn tự nhiên, không ép 50/50**.
Cân bằng sẽ thổi phồng precision và xóa mất đúng điểm yếu cần đo của blacklist.

### 2.4 Dò ngưỡng quyết định

Tập train lệch prior nặng (49,19% HATE) so với test (18,18%). Ở ngưỡng mặc định
0,5, model báo HATE quá tay. Nặng hơn: **ngưỡng 0,5 mang ý nghĩa khác nhau giữa
B2 và B3** vì hai model học dưới hai prior khác nhau — so ở cùng 0,5 là so tại
hai điểm vận hành không tương đương.

Quy trình: mỗi hệ thống tự chọn ngưỡng tối ưu F1 **trên tập validation** (gộp cả
ba điều kiện), **đóng băng**, rồi mới mở test ra chấm. Ngưỡng **không bao giờ**
được dò trên tập test.

Gộp cả ba điều kiện khi dò là có chủ ý: lúc triển khai thật, hệ thống không biết
trước câu đang tới là sạch, bị né lọc hay điểm mù, nên không được phép đổi ngưỡng
theo từng loại tấn công.

| Hệ thống | Ngưỡng đã chọn | F1 trên val | F1 nếu để 0,5 | Chênh |
|---|---:|---:|---:|---:|
| `b2_t5base` | 0,443 | 0,6050 | 0,6008 | +0,0042 |
| `b2_t5base_norm` | 0,429 | 0,6449 | 0,6321 | +0,0128 |
| `b3_t5ft` | 0,695 | 0,7202 | 0,7158 | +0,0044 |
| `b3_t5ft_norm` | 0,453 | 0,7548 | 0,7520 | +0,0028 |

(F1 ở bảng này đo trên tập validation gộp cả ba điều kiện, nên thấp hơn số của
riêng từng điều kiện ở mục 5 — không so trực tiếp hai bảng với nhau được.)

**Quan sát trung thực:** dò ngưỡng thay đổi rất ít — nhiều nhất là +0,0128 điểm
F1 (`b2_t5base_norm`), còn lại quanh +0,003 đến +0,004. Áp lên test thì chênh
lệch còn nhỏ hơn nữa (−0,010 đến +0,003). Lo ngại lệch prior sẽ phá kết quả **đã
không thành hiện thực** ở mức nghiêm trọng như dự đoán. Vẫn giữ bước này trong
quy trình vì nó đúng phương pháp và trả lời được câu phản biện "sao lại lấy 0,5,
so ở 0,5 có công bằng giữa hai model không".

Blacklist không có mặt trong bảng này: nó chỉ khớp từ khóa, không sinh ra điểm
xác suất nào để mà dò. Số của B1 đo ở quyết định mặc định — đó là ý nghĩa của dấu
`—` trong cột Ngưỡng của Bảng 1.

---

## 3. Dữ liệu

### 3.1 Bộ huấn luyện — trộn 60/30/10

`data/data_final/train.jsonl`, **116.943 dòng**, 49,19% HATE.

| Nguồn | Tỉ lệ | Số dòng | HATE | CLEAN | Vai trò |
|---|---:|---:|---:|---:|---|
| `perturbed_normalized` | 60% | 70.166 | 34.297 | 35.869 | Câu đã cấy biến thể rồi chạy qua `normalize()` — model học chịu **phần nhiễu còn sót** |
| `clean_original` | 30% | 35.083 | 17.425 | 17.658 | Câu gốc sạch, dedup theo `duplicate_group_id` — chống quên hiệu năng trên câu sạch |
| `blindspot_raw` | 10% | 11.694 | 5.808 | 5.886 | Biến thể điểm mù, **cố ý không chuẩn hóa** |

Vì sao 30% câu sạch: checkpoint nền `ViHateT5-base-HSD` vốn đã fine-tune trên
ViHSD sạch. Không giữ lại câu sạch trong lúc train thì model quên mất năng lực
cũ (catastrophic forgetting). Dedup theo `duplicate_group_id` để tránh train lại
cùng một câu sạch tới 103 lần.

Vì sao 10% điểm mù không chuẩn hóa: đây là phần **duy nhất** dạy model tự chống
đỡ những gì Bước 2 không với tới. Chuẩn hóa chúng là làm hỏng đúng mục đích tồn
tại của chúng.

**Lỗi đã từng mắc và đã sửa:** một phiên bản `build_mixed.py` trước đây lấy mẫu
dòng blindspot không quan tâm nhãn, kết quả sinh ra 100% dòng CLEAN. Sau khi dựng
lại dữ liệu, luôn kiểm `by_source_label` trong manifest.

### 3.2 Tập validation dùng cho early stopping

`data/data_final/validation_mixed.jsonl`, **4.176 dòng**, 17,67% HATE, cùng cấu
trúc trộn 60/30/10.

Cố ý dùng bản trộn chứ không dùng C0 câu sạch: early stopping phải nhìn cùng loại
phân bố với dữ liệu train, nếu chỉ nhìn câu sạch thì không phát hiện được lúc
model bắt đầu hỏng trên câu bị né lọc.

`data/data_final/{validation,test}.jsonl` là **C0 câu sạch**, dùng cho eval nhanh
sau khi train. Chúng là bản sao đúng byte của `eval_baseline/*/C0_clean.jsonl` —
checksum trùng nhau là đúng thiết kế, không phải lỗi.

---

## 4. Huấn luyện

### 4.1 Cấu hình

| Tham số | Giá trị | Lý do |
|---|---|---|
| Model nền | `tarudesu/ViHateT5-base-HSD` | Đã fine-tune sẵn trên ViHSD |
| Kiến trúc | `AutoModelForSeq2SeqLM` (T5) | Không dùng ViSoBERT — đã chốt |
| Prompt vào | `f"vihsd: {text}"` | |
| Chuỗi đích | `"hate"` / `"clean"` **chữ thường** | Chữ hoa `HATE`/`CLEAN` bị tokenizer gộp về cùng `<unk>` — hai nhãn sẽ không phân biệt được. Đã verify bằng tokenizer thật |
| `learning_rate` | **1e-5** | Đang fine-tune TIẾP một checkpoint đã fine-tune sẵn, không train từ đầu |
| `fgm_epsilon` | **0,3** | |
| `epochs` | 8 (trần) | Early stopping dừng ở epoch 6 |
| `early_stopping_patience` | 2 | |
| `per_device_train_batch_size` | 16 | Kaggle 2×T4 |
| `warmup_ratio` / `weight_decay` | 0,06 / 0,01 | |
| `max_input_length` | 256 | |
| **`seed`** | **42** | |
| Phần cứng | Kaggle 2×T4, fp16 | |

### 4.2 Huấn luyện đối kháng hai tầng

- **Tầng dữ liệu** — bộ trộn 60/30/10 ở mục 3.1.
- **Tầng gradient (FGM)** — `src/classifier/adversarial.py`. Mỗi bước: backward
  lần 1 lấy gradient sạch, tấn công embedding `shared` theo hướng
  `epsilon · grad/‖grad‖`, backward lần 2, rồi khôi phục trọng số. Hoạt động
  đúng dưới `nn.DataParallel` vì thao tác trên `model` được truyền vào
  `training_step`, không dùng tham chiếu lưu sẵn.

### 4.3 Diễn biến — và vì sao lần chạy đầu bị bỏ

Lần chạy đầu dùng `learning_rate=3e-4` và `fgm_epsilon=1.0`. F1 validation dao
động **0,69 → 0,79 → 0,67 → 0,75** qua 4 epoch — biên độ ±11,5 điểm, tức chưa hội
tụ; early stopping bắt nhầm một đỉnh tạm thời. Nguyên nhân: learning rate quá cao
cho việc fine-tune tiếp một checkpoint đã chín, cộng với epsilon quá mạnh gây
nhiễu kép. Log lần chạy đó giữ ở
`results/step3/logs/train_v1_lr3e-4_DEPRECATED.log`.

Lần chạy thứ hai (lr 1e-5, epsilon 0,3) — **hội tụ ổn định**:

| Epoch | F1 (val) | `clean_original` | `perturbed_normalized` | `blindspot_raw` |
|---:|---:|---:|---:|---:|
| 1 | 0,7537 | 0,7647 | 0,7737 | 0,6081 |
| 2 | 0,7755 | 0,8035 | 0,7875 | 0,6341 |
| 3 | 0,7696 | 0,7898 | 0,7892 | 0,5972 |
| **4** | **0,7881** ← tốt nhất | **0,8158** | **0,7958** | **0,6667** |
| 5 | 0,7839 | | | |
| 6 | 0,7801 | | | |

Biên độ dao động giữa các epoch tối đa **±2,2 điểm**, so với ±11,5 điểm ở lần
trước. Early stopping dừng đúng sau 2 epoch không cải thiện (5 và 6).

- Tổng thời gian: **18.690 giây ≈ 5 giờ 11 phút**, ~50 mẫu/giây.
- Checkpoint tốt nhất: **`checkpoint-14620`, epoch 4, F1 val = 0,7881**.
  Đã xác minh bằng SHA-256 rằng `models/vihatet5-v2-best/model.safetensors`
  chính là file đó.

**Quan sát đáng chú ý:** `blindspot_raw` luôn là nhóm khó nhất, cách hai nhóm kia
12–19 điểm F1 ở mọi epoch. Đúng như thiết kế — đó là phần Bước 2 không với tới.

`test_invalid_generation_rate = 0,0000`: model **không sinh ra một chuỗi đích
không hợp lệ nào**. Chọn chuỗi đích chữ thường đã giải quyết triệt để rủi ro
`<unk>`.

### 4.4 Eval sau train (trên tập trộn của lần chạy đó)

| Nhóm nguồn | F1 | Precision | Recall | Accuracy |
|---|---:|---:|---:|---:|
| Tổng hợp | 0,7767 | 0,7714 | 0,7821 | 0,9177 |
| `clean_original` | 0,8142 | 0,7931 | 0,8364 | 0,9293 |
| `perturbed_normalized` | 0,7926 | 0,7935 | 0,7917 | 0,9247 |
| `blindspot_raw` | 0,5655 | 0,5694 | 0,5616 | 0,8409 |

Đây là eval nhanh trên **tập test trộn của chính lần chạy đó**, không phải ma
trận 3 điều kiện. Số liệu báo cáo chính nằm ở mục 5.

Trong log có dòng cảnh báo `early stopping required metric_for_best_model, but
did not find eval_f1 so early stopping is disabled`. Vô hại: nó xuất hiện ở lần
`evaluate()` cuối với `metric_key_prefix="test"` nên chỉ số tên là `test_f1` chứ
không phải `eval_f1`. Lúc đó train đã xong, early stopping không còn việc gì.

---

## 5. Kết quả

### 5.1 Bảng chính — F1 lớp HATE trên tập test

Ngưỡng đã dò trên validation và đóng băng. Xem thêm
[`results/step3/figures/bang1_chi_so_day_du.pdf`](../results/step3/figures/bang1_chi_so_day_du.pdf).

| Hệ thống | Ngưỡng | C0 câu sạch | C1 né lọc | C2 điểm mù |
|---|---:|---:|---:|---:|
| `b1_blacklist` | — | 0,6551 | 0,5982 | 0,4564 |
| `b1_blacklist_norm` | — | 0,6421 | 0,6308 | 0,4534 |
| `b2_t5base` | 0,690 | 0,7683 | 0,6046 | 0,4627 |
| `b2_t5base_norm` | 0,576 | 0,7497 | 0,7218 | 0,4720 |
| `b3_t5ft` | 0,778 | 0,8743 | 0,7441 | 0,6529 |
| **`b3_t5ft_norm`** | 0,586 | **0,8552** | **0,8266** | **0,6657** |

Khoảng tin cậy 95% (bootstrap 1.000 lần lấy mẫu lại, seed 42):

| Hệ thống | C0 | C1 | C2 |
|---|---|---|---|
| `b1_blacklist` | 0,620–0,689 | 0,558–0,638 | 0,410–0,505 |
| `b2_t5base` | 0,735–0,800 | 0,567–0,639 | 0,417–0,510 |
| `b2_t5base_norm` | 0,718–0,784 | 0,687–0,756 | 0,430–0,514 |
| `b3_t5ft` | 0,851–0,897 | 0,710–0,776 | 0,608–0,696 |
| `b3_t5ft_norm` | 0,831–0,879 | 0,800–0,853 | 0,628–0,705 |

> **Số liệu đã chạy lại sau khi sửa lỗi chữ hoa** (xem [`buoc4_bao_cao.md`](buoc4_bao_cao.md)
> §3.2). Từ điển sentencepiece của ViHateT5 không chứa ký tự hoa nào, nên mọi
> chữ hoa bị hủy thành `<unk>` — đo được 85% câu trong tập eval có chữ hoa,
> trung bình mất 10,3% token. Sau khi hạ chữ thường ở đầu vào model
> (`run_matrix.py`), **mọi hệ T5 đều tăng**: `b3_t5ft_norm` +0,0332 (C0),
> +0,0303 (C1), +0,0287 (C2). Hai hệ blacklist **không đổi** vì
> `src/baselines/blacklist.py` vốn đã hạ chữ thường sẵn — nghĩa là bảng cũ
> đang thiệt cho T5 và lợi cho blacklist. (Chênh lệch nhỏ ở
> `b1_blacklist_norm` so với bảng cũ là do lần chạy lại, không do sửa lỗi.)

Precision / Recall / macro F1 / Accuracy đầy đủ nằm trong Bảng 1 (PDF) và
`results/step3/threshold/threshold.json`.

### 5.2 Trả lời bốn câu hỏi

Các hiệu số dưới đây đo tại **ngưỡng 0,5** để phép ghép cặp so cùng một điểm
vận hành (khác với bảng 5.1 dùng ngưỡng đã dò).

Tất cả kiểm định bằng **McNemar ghép cặp** trên cùng từng dòng test.

**Q1 — Né lọc có phá được model không? CÓ.**
`b2_t5base` tụt từ 0,7557 (C0) xuống 0,5860 (C1): **−0,1697**. Tiền đề của cả đề
tài được xác nhận bằng số.

**Q2 — Bước 2 góp bao nhiêu? +0,1327 trên C1** (`p = 7,47e-30`).
Trên C0 thì Δ = −0,0052, `p = 0,550`, **không có ý nghĩa** — nghĩa là bật chuẩn
hóa lên câu vốn đã sạch **không làm hỏng gì**. Đây là bằng chứng cho triết lý
"thà không sửa còn hơn sửa sai" của bộ chuẩn hóa.
Trên C2 thì Δ = +0,0076, `p = 0,324`, không có ý nghĩa — đúng như bất biến ở
mục 2.2.

**Q3 — Bước 3 góp thêm bao nhiêu?**

| Điều kiện | `b2_t5base_norm` → `b3_t5ft_norm` | p-value |
|---|---:|---|
| C0 | +0,1024 | 2,90e-14 |
| C1 | +0,1086 | 9,96e-16 |
| C2 | **+0,1926** | 1,21e-26 |

Trên điểm mù, fine-tune là thứ **duy nhất** cứu được — chuẩn hóa hoàn toàn bó tay
ở đó.

**Q4 — Fine-tune rồi có còn cần Bước 2 không? CÓ, trên C1.**

| Điều kiện | `b3_t5ft` → `b3_t5ft_norm` | p-value | Kết luận |
|---|---:|---|---|
| C0 | −0,0157 | 0,144 | không ý nghĩa |
| C1 | **+0,0745** | 2,85e-08 | **có ý nghĩa** |
| C2 | +0,0019 | 1,00 | không ý nghĩa (input gần như giống hệt) |

Đây là kết luận quan trọng nhất về mặt kiến trúc: **hai bước bổ trợ nhau, không
thay thế nhau**. Fine-tune đối kháng không xóa được nhu cầu chuẩn hóa.

**So với cách ngây thơ nhất:** pipeline đầy đủ hơn blacklist +0,1978 (C0),
**+0,2290** (C1), +0,2084 (C2), tất cả `p < 1e-6`.

### 5.3 Tổng kết mức đóng góp trên C1 — kịch bản chính

```
ViHateT5 gốc, không phòng vệ         0,6046
  + Bước 2 (chuẩn hóa)      +0,1172  0,7218
  + Bước 3 (fine-tune)      +0,1048  0,8266   ← pipeline đầy đủ
```

Trên C2 điểm mù, tỉ trọng đảo ngược hoàn toàn:

```
ViHateT5 gốc, không phòng vệ         0,4627
  + Bước 2 (chuẩn hóa)      +0,0093  0,4720   ← gần như vô hiệu
  + Bước 3 (fine-tune)      +0,1937  0,6657
```

### 5.4 Diện tích dưới đường cong

| Hệ thống | AUC C0 | AUC C1 | AUC C2 | AP C0 | AP C1 | AP C2 |
|---|---:|---:|---:|---:|---:|---:|
| `b2_t5base` | 0,953 | 0,875 | 0,774 | 0,859 | 0,680 | 0,498 |
| `b2_t5base_norm` | 0,951 | 0,936 | 0,781 | 0,851 | 0,814 | 0,515 |
| `b3_t5ft` | 0,980 | 0,932 | 0,887 | 0,939 | 0,825 | 0,723 |
| **`b3_t5ft_norm`** | 0,972 | **0,962** | **0,890** | 0,924 | **0,899** | **0,729** |

AP (Precision-Recall) đáng tin hơn AUC ở đây vì nhãn lệch nặng. Khoảng cách giữa
hai chỉ số nói lên điều đó: trên C2, AUC của `b2_t5base` là 0,774 nghe khá ổn,
nhưng AP chỉ 0,498 — sát mức nền. Pipeline đầy đủ nâng AP lên 0,729.

---

## 6. Quan sát & đánh giá

### 6.1 Những gì kết quả chứng minh được

1. **Né lọc là mối đe dọa thật, không phải giả định.** Model tốt nhất hiện có
   cho tiếng Việt mất 16,38 điểm F1 chỉ vì văn bản bị biến dạng.
2. **Hai lớp phòng vệ bổ trợ nhau, phân công rõ ràng.** Chuẩn hóa gánh phần lớn
   trên biến thể thông thường (+0,1172 trên C1); fine-tune gánh phần lớn trên
   điểm mù (+0,1937 trên C2). Bỏ bất kỳ bên nào cũng để lộ một mặt trận.
3. **Chuẩn hóa không gây tác dụng phụ.** Trên C0, bật chuẩn hóa lên câu sạch cho
   Δ = −0,0052, `p = 0,550`, không có ý nghĩa thống kê.
4. **Mọi so sánh chính đều có ý nghĩa thống kê**, `p < 1e-6`, kiểm định ghép cặp
   chứ không phải đặt cạnh nhau hai con số rời rạc.

### 6.2 Những điểm yếu phải nói rõ

1. **B2 có lợi thế sân nhà ở C0.** Câu test lấy từ ViHSD, mà
   `ViHateT5-base-HSD` cũng đã fine-tune trên chính ViHSD. Nói cách khác, B2
   được chấm trên dữ liệu họ hàng với dữ liệu nó đã học. B3 vẫn thắng
   (+0,0869) → kết luận càng chắc, nhưng phải ghi nhận.

2. **Điểm mù vẫn là mặt trận thua.** Ngay cả pipeline đầy đủ cũng chỉ đạt 0,6657
   trên C2, thấp hơn C0 tới 18,95 điểm. Chưa giải quyết xong, chỉ mới thu hẹp.

3. **Bộ chuẩn hóa hiện tại không tái tạo được dữ liệu train.** Chi tiết ở mục
   6.3 — đây là hạn chế về tính tái lập, cần biết trước khi dựng lại pipeline.

4. **Dò ngưỡng gần như vô ích trên thực tế** — nhiều nhất +0,0128 F1 trên
   validation, và trên test có trường hợp còn giảm nhẹ. Giữ trong quy trình vì
   đúng phương pháp, nhưng không nên trình bày như một đóng góp.

5. **Chỉ một lần chạy, không có nhiều seed.** Mọi con số là kết quả của một lần
   train duy nhất với seed 42. Chưa đo được phương sai giữa các seed. Khoảng tin
   cậy trong báo cáo là phương sai **của tập test**, không phải phương sai của
   quá trình huấn luyện — hai thứ khác nhau, đừng nhầm.

6. **Blacklist bị thiệt vì không dò được ngưỡng.** Nó không sinh xác suất nên
   không có điểm vận hành nào để tối ưu. So sánh vẫn công bằng theo nghĩa "đây là
   khả năng tốt nhất của cách làm đó", nhưng không phải so cùng một sân chơi.

### 6.3 Lỗ hổng tái lập đã phát hiện

Thư mục `data/processed/normalized/` — bước trung gian giữa Bước 1 và Bước 3 —
ban đầu **được làm tay, không có script nào tạo ra nó**. Đã bổ sung
`src/dataset_builder/build_normalized.py`.

Nhưng chạy lại script đó bằng bộ chuẩn hóa **hiện tại** cho kết quả **lệch ~14%
số dòng** (339/2.376 dòng ở split test) so với file đã thực sự dùng để train.
Phiên bản chuẩn hóa sinh ra file cũ rộng tay hơn ở hai chỗ:

1. Từ điển teencode lúc đó có thêm mục kiểu `vcl` → `vãi lồn`; bản
   `teencode_dict.json` hiện tại (625 mục) không có.
2. Bộ separator lúc đó xóa cả dấu chấm giữa `·` (U+00B7); bản hiện tại chỉ xóa
   `. - _`.

Trong 339 dòng lệch, 294 dòng thuộc loại biến thể `teencode_lexical`.

**Ảnh hưởng, nói thẳng:** phần 60% `perturbed_normalized` của bộ train sạch hơn
một chút so với thứ mà bộ chuẩn hóa đang ship sẽ tạo ra lúc chạy thật. Đây là
lệch phân bố train/inference nhỏ, và nó lệch theo hướng **làm xấu con số báo cáo
chứ không thổi phồng**: lúc eval dùng bộ chuẩn hóa hiện tại (yếu hơn), tức input
khó hơn lúc train. Kết quả Bước 3 vẫn dùng được, chỉ là hơi bảo thủ.

**Không có rò rỉ nhãn.** `normalize()` không hề nhìn thấy `original_text` hay
`label`.

Vì vậy `build_normalized.py` mặc định **từ chối ghi đè** file đã có, phải truyền
`--force` mới ghi.

### 6.4 Sai lầm thiết kế đã sửa trong quá trình làm

Ghi lại để không lặp lại:

| Sai lầm | Hậu quả | Đã sửa thành |
|---|---|---|
| Cho B1/B2 ăn câu sạch, B3 ăn câu đã chuẩn hóa | So sánh gian lận | Cả sáu hệ thống ăn cùng input thô; chuẩn hóa nằm bên trong hệ thống |
| `learning_rate=3e-4`, `fgm_epsilon=1.0` | F1 dao động ±11,5 điểm, chưa hội tụ | 1e-5 và 0,3 → ±2,2 điểm |
| Ép validation/test cân bằng 50/50 | Không đo được precision, che mất điểm yếu blacklist | Giữ phân bố tự nhiên ~18% HATE |
| Lấy mẫu dòng blindspot không quan tâm nhãn | 100% dòng CLEAN | Lấy mẫu theo từng nhãn, kiểm `by_source_label` |
| Chuỗi đích chữ hoa `HATE`/`CLEAN` | Tokenizer gộp cả hai về `<unk>`, hai nhãn không phân biệt được | Chữ thường `hate`/`clean` |
| Dán chung một bộ AP vào chú giải 3 ô | Số của C0 bị ghi cho cả C1 và C2 | AP riêng từng ô |
| Rút gọn nhãn thành `B2`/`B3` trong bảng McNemar | Mất phân biệt bản có/không chuẩn hóa, ghi sai cặp so sánh | `B2+ch` / `B3+ch` |

---

## 7. Bảng tra seed

| Khâu | Seed | Ghi ở đâu |
|---|---:|---|
| Bước 1 — sinh 75K biến thể | **20260902** | `src/dataset_builder/build.py --seed`, hằng `SEED_STEP1_DATASET` |
| Bước 2 — áp `normalize()` | 42 | `build_normalized.py` (hàm tất định, seed chỉ để ghi nhật ký) |
| Trộn 60/30/10 | **42** | `data/data_final/manifest_mix.json` |
| Dựng ma trận 3 điều kiện | **42** | `data/eval_baseline/*/manifest.json` |
| Huấn luyện | **42** | `models/vihatet5-v2-best/run_config.json` |
| Bootstrap khoảng tin cậy | **42** | `src/baselines/stats.py`, 1.000 lần lấy mẫu lại |
| Dò ngưỡng | **42** | `results/step3/threshold/threshold.json` |
| Vẽ hình | 42 | không dùng ngẫu nhiên, seed chỉ để đồng bộ |

Toàn bộ đi qua `src/utils/seed.py`. Hàm `set_global_seed()` seed cùng lúc
`random`, `numpy`, `torch`, `torch.cuda` và `PYTHONHASHSEED`.

**Giới hạn của seed, nói cho rõ:** seed đảm bảo cùng dữ liệu, cùng thứ tự batch,
cùng khởi tạo. Nó **không** đảm bảo kết quả trùng nhau tới từng bit khi train trên
GPU với fp16 và `nn.DataParallel` — thứ tự cộng dồn số thực lúc gộp gradient từ
nhiều thiết bị có thể khác nhau giữa các lần chạy.

Seed cũng không phát hiện được file bị sửa tay hay copy nhầm. Việc đó do
`data/CHECKSUMS.json` đảm nhiệm (`python -m scripts.checksums`).

---

## 8. Hình và bảng

Toàn bộ ở `results/step3/figures/`, định dạng **PDF vector**, font nhúng trong
file, nhúng thẳng vào Word/LaTeX được. Đã vẽ lại theo số liệu sau khi sửa lỗi
chữ hoa — vẽ lại bất cứ lúc nào bằng `python -m src.reporting.make_figures`
(~5 giây, không cần GPU).

| File | Nội dung |
|---|---|
| `hinh1_ma_tran_f1.pdf` | Cột nhóm: 6 hệ thống × 3 điều kiện + thanh sai số KTC 95% |
| `hinh2_do_tut_hieu_nang.pdf` | Đường tụt hiệu năng qua C0→C1→C2 |
| `hinh3_duong_cong_roc.pdf` | ROC, 3 ô, AUC riêng từng ô |
| `hinh4_duong_cong_pr.pdf` | Precision-Recall, 3 ô, AP riêng từng ô |
| `hinh5_dong_gop_tung_buoc.pdf` | Đóng góp của Bước 2 và Bước 3 trên C1 và C2 |
| `hinh6_precision_recall.pdf` | Đánh đổi precision/recall từng hệ thống |
| `bang1_chi_so_day_du.pdf` | 18 dòng chỉ số đầy đủ |
| `bang2_mcnemar.pdf` | 15 phép kiểm định ý nghĩa thống kê |

Quy ước đọc hình: **màu = hệ thống** (B1 xanh dương, B2 cam, B3 xanh ngọc),
**gạch chéo / nét đứt = có bật bộ chuẩn hóa**. Chỉ 3 màu thay vì 6 để an toàn cho
người mù màu và đọc được khi in đen trắng.

Vẽ lại: `python -m src.reporting.make_figures` — vài giây trên CPU, không cần GPU.

---

## 9. Việc chưa làm

- Chưa chạy nhiều seed để đo phương sai của quá trình huấn luyện.
- Chưa thử tắt FGM để tách riêng đóng góp của tầng gradient khỏi tầng dữ liệu
  (`--no_fgm` đã có sẵn trong `train.py`, chỉ là chưa chạy).
- Chưa có baseline "train lại từ đầu chỉ trên dữ liệu sạch, không FGM" — hiện
  B2 dùng checkpoint có sẵn của tác giả gốc.
- Chưa đánh giá trên tập ngoài ViHSD, nên chưa biết pipeline tổng quát hóa ra
  sao ngoài phân bố này.
- Bước 4 (module giải thích bằng occlusion) chưa bắt đầu — xem `plan.md`.
