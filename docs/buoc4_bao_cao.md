# Bước 4 — Module giải thích bằng occlusion

> **Câu hỏi**: khi hệ thống kết luận một câu là HATE, nó có chỉ ra được *chính
> xác cụm từ nào* gây ra kết luận đó không — và cụm đó có trùng với cụm mà con
> người khoanh không?
>
> **Trả lời ngắn**: có, ở mức **micro F1 = 0,5552** trên ViHOS test. Occlusion
> bắt được **64,7%** số từ con người khoanh, và chỉ **1,5%** số câu là chỉ sai
> hoàn toàn. Đổi lại, vùng nó khoanh **rộng gấp 1,33 lần** vùng gold.

Mọi số trong tài liệu này lấy từ [`results/step4/localization.json`](../results/step4/localization.json)
và [`results/step4/predictions_test.jsonl`](../results/step4/predictions_test.jsonl),
chạy bằng checkpoint `models/vihatet5-v2-best` của Bước 3.

---

## 1. Vấn đề và cách tiếp cận

### 1.1 Vì sao cần bước này

Sau Bước 3, hệ thống trả về đúng/sai ở mức **cả câu** (HATE / CLEAN). Với công
cụ kiểm duyệt thực tế, chừng đó chưa đủ: người kiểm duyệt cần biết *vì sao* —
cụm từ nào khiến câu bị gắn cờ — để quyết định nhanh và để phát hiện khi model
bắt nhầm.

### 1.2 Chọn occlusion, không chọn SHAP/attention

Occlusion = che một cụm từ đi, đo xem điểm "hate" tụt bao nhiêu. Cụm nào bị che
mà điểm tụt mạnh nhất chính là cụm gây cảnh báo.

Chọn cách này vì:

- **Không cần train thêm gì cả** — chỉ chạy inference trên model đã có.
- **Giải thích được cho người không chuyên** trong một câu: "che chữ này đi thì
  máy hết coi là độc hại".
- **Không phụ thuộc kiến trúc** — không cần bóc attention weight, không cần
  giả định về cách model hoạt động bên trong.

SHAP và attention rollout mạnh hơn về lý thuyết nhưng phức tạp hơn nhiều để
trình bày và không tương xứng với quy mô demo này.

### 1.3 Ranh giới cần nói rõ ngay từ đầu

Model **chưa bao giờ được học nhãn span**. Bước 3 chỉ huấn luyện trên nhãn nhị
phân cả câu. Occlusion là **giải thích hậu kỳ** cho model đó, **không phải** một
model định vị span chuyên dụng.

Vì vậy con số ở đây **không so sánh trực tiếp được** với các hệ thống token
classification huấn luyện thẳng trên span ViHOS — đó là bài toán khác, có giám
sát trực tiếp cho đúng thứ cần đo. Xem thêm §6.2.

---

## 2. Dữ liệu — ViHOS

### 2.1 Nguồn và cách xử lý

Tải bằng `load_dataset("htdung167/ViHOS")`. Bộ gốc có 2 cột:

| Cột | Nội dung |
|---|---|
| `content` | Câu tiếng Việt (sạch, không nhiễu) |
| `index_spans` | Chuỗi kiểu `"[9, 10]"` — **chỉ số KÝ TỰ** thuộc vùng gây hại |

**Điểm dễ sai**: `index_spans` là chỉ số **ký tự**, không phải chỉ số từ. Ví dụ
`"Bấp bênh vl thế"` với `[9, 10]` trỏ tới hai ký tự `v` và `l` — tức từ `"vl"`.

Occlusion trượt theo **từ**, nên [`src/dataset_builder/build_vihos.py`](../src/dataset_builder/build_vihos.py)
phải quy đổi: **một từ được coi là gold nếu có ít nhất 1 ký tự của nó nằm trong
`index_spans`**. Kết quả lưu ở `data/data_explain/`:

```json
{"words": ["Bấp","bênh","vl","thế"], "span_mask": [0,0,1,0], "label": "HATE"}
```

### 2.2 Quy mô

| Split | Tổng câu | CLEAN | HATE |
|---|---:|---:|---:|
| train | 8 844 | 4 552 | 4 292 |
| validation | 1 106 | 569 | 537 |
| test | 1 106 | 575 | 531 |

0 dòng lỗi định dạng. Bước 4 **không dùng split train** (không train gì) —
validation dùng để dò ngưỡng, test để báo cáo.

### 2.3 Chỉ đánh giá trên câu HATE

Đánh giá chỉ chạy trên **531 câu HATE có span gold không rỗng** trong test. Câu
CLEAN không có cụm gây hại nào để định vị; đưa vào chỉ làm loãng số liệu.

### 2.4 ViHOS là câu sạch — và đó là chủ ý

ViHOS không có nhiễu né lọc. Bước 4 vì vậy đo **chất lượng định vị**, tách bạch
khỏi **độ bền trước né lọc** (việc đó Bước 3 đã đo bằng ba điều kiện C0/C1/C2).
Trộn hai câu hỏi vào một phép đo sẽ không diễn giải được kết quả.

---

## 3. Phương pháp

### 3.1 Điểm số phải lấy ở dạng logit margin, không phải xác suất

Model là seq2seq: sinh đúng một token `"hate"` hoặc `"clean"`. Điểm số lấy từ
một forward pass ép decoder (không generate tự do), đọc logits ở bước decode
đầu tiên cho đúng hai token nhãn:

```
margin = logit("hate") - logit("clean")
```

**Vì sao không dùng P(hate) = softmax(margin)?** Đây là lỗi đã mắc và đã sửa
trong quá trình làm. Checkpoint Bước 3 rất tự tin: margin quan sát được nằm
trong khoảng ~16–30. Với margin = 23, xác suất là `1 − 6,9e-11` — con số này
**không phân biệt được với đúng 1,0** ở độ chính xác float32.

Hệ quả: che một từ đi, xác suất vẫn in ra `1,000000` y hệt → occlusion mất
sạch tín hiệu. Bảng đo thực tế trên câu `"Mày ngu như con chó vậy"`:

| Câu | margin | P(hate) float32 |
|---|---:|---:|
| Nguyên câu | +23,37 | 1,000000 |
| Bỏ `"ngu"` | +18,26 | 1,000000 |
| Bỏ `"ngu"` và `"chó"` | +16,19 | 1,000000 |

Margin phân biệt rõ ràng (23,37 → 18,26 → 16,19) trong khi xác suất phẳng lì.
Margin không bị chặn trong [0, 1] nên không bao giờ bão hòa — đó là lý do nó
được chọn làm điểm số.

`probability()` vẫn còn trong code nhưng **chỉ để hiển thị cho người đọc**,
không dùng để tính occlusion.

### 3.2 Bắt buộc hạ chữ thường trước khi đưa vào model

Phát hiện muộn, khi chạy demo bằng tay: câu chửi viết HOA bị phân loại **CLEAN**.

```
"ĐỊT MẸ MÀY"   -> margin −2,04  -> CLEAN   (SAI)
"địt mẹ mày"   -> margin +15,14 -> HATE    (đúng)
```

Nguyên nhân: **từ điển sentencepiece của ViHateT5 không chứa một ký tự hoa
nào**. Mọi chữ hoa bị map thành `<unk>`:

```
'ĐỊT MẸ MÀY' -> ['▁','<unk>','▁','<unk>','▁','<unk>']    ← model không đọc được gì
'Chó'        -> ['▁','<unk>','h','ó']                     ← hoa 1 chữ cũng vỡ từ
```

Vì thế mọi câu viết HOA đều cho margin gần y hệt nhau (−2,04 / −2,03 / −1,95):
model nhận vào một chuỗi rỗng nghĩa nên luôn trả cùng một đáp án mặc định.

**Đây không phải trường hợp hiếm.** Tiếng Việt viết hoa đầu câu và tên riêng,
nên vấn đề chạm gần như mọi câu:

| | ViHOS test | Bước 3 eval C0 |
|---|---:|---:|
| Câu có chứa chữ hoa | 87,6% | 85,0% |
| Token bị `<unk>` (trung bình) | **10,1%** | **10,3%** |
| Câu hỏng nặng (>30% `<unk>`) | 1,7% | 1,7% |

Ví dụ trong chính tập test, gold span lại rơi đúng vào từ viết hoa:

```
'Cảnh Sát GiựT Tiền = CSGT'  ->  6/19 token là <unk>
gold = [GiựT, Tiền]   ← cả hai từ độc hại đều bị hủy thành <unk>
```

**Cách sửa**: hạ chữ thường ngay trước khi tokenize (`_score_chunk`). Chữ hoa
vốn mang **zero thông tin** tới model — nó đã bị hủy sẵn rồi — nên hạ chữ
thường không mất gì mà chỉ cứu lại phần tín hiệu đang bị vứt. Text hiển thị cho
người đọc vẫn giữ nguyên dạng gốc.

Ảnh hưởng lên số liệu (cùng cấu hình, chỉ khác hạ chữ thường):

| Chỉ số | Trước | Sau |
|---|---:|---:|
| micro F1 | 0,5455 | **0,5552** |
| micro precision | 0,4478 | **0,4861** |
| micro recall | 0,6977 | 0,6472 |
| Số từ được flag | 3 953 | **3 378** (gold: 2 537) |
| Câu trượt hoàn toàn | 10 | **8** |

F1 chỉ nhích +0,0097 — **lợi ích thật không nằm ở con số tổng** mà ở hai chỗ:
span khoanh gọn hơn hẳn (rộng gấp 1,56× → 1,33×) và, quan trọng hơn cả, hệ
thống thôi phân loại sai hoàn toàn các câu chửi viết hoa.

Câu ví dụ ở trên sau khi sửa:

```
'Cảnh Sát GiựT Tiền = CSGT'   gold = [GiựT, Tiền]
  trước: [Cảnh, Tiền, =, CSGT]   F1 = 0,33
  sau  : [Sát, GiựT, Tiền]       F1 = 0,80
```

> **Bước 3 vẫn đang dính đúng lỗi này.** `src/classifier/data.py` không hạ chữ
> thường, `normalize()` của Bước 2 cũng không (còn chủ động khôi phục chữ hoa).
> Toàn bộ số liệu Bước 3 vì vậy được đo trong tình trạng mất ~10% token. Chưa
> sửa, chưa chạy lại — ghi nhận ở §7.

### 3.3 Chuẩn hóa điểm theo từng câu

Điểm quan trọng của một từ = mức margin **tụt xuống** khi che nó đi.

Nhưng thang điểm này lệch rất mạnh giữa các câu. Đo trên test: điểm cao nhất
của mỗi câu trải từ **−2,3 đến 23,1** (trung vị 6,8). Một ngưỡng tuyệt đối dùng
chung cho mọi câu sẽ gạt sạch những câu "model nói nhỏ":

| Nhóm câu | Recall ở ngưỡng tuyệt đối |
|---|---:|
| Điểm đỉnh cao (> 10) | 0,759 |
| Điểm đỉnh thấp (< 1,5) | **0,062** |

Câu nhóm dưới gần như không được flag từ nào — không phải vì model không biết,
mà vì thang điểm của nó vốn nhỏ.

**Cách sửa**: chia điểm cho **từ mạnh nhất của chính câu đó**
(`normalize_importance()`), đưa mọi câu về thang 0–1. Ngưỡng khi đó mang nghĩa
tương đối: *"mạnh bằng ít nhất X% cụm mạnh nhất trong câu này"*.

Trường hợp biên: câu mà không từ nào làm điểm tụt (đỉnh ≤ 0) trả về toàn 0 —
không flag gì. Chia cho số âm sẽ lật dấu và cho kết quả sai hoàn toàn.

### 3.4 Che theo cụm 2 từ

Che từng từ đơn quá hẹp so với cách ViHOS khoanh span (theo **cụm ngữ nghĩa**,
trung vị 3 từ). Cấu hình chốt là che **2 từ liền kề** mỗi lần, điểm tụt được
cộng vào cả hai từ trong cụm.

### 3.5 Dò ngưỡng trên validation, đóng băng, rồi mới áp lên test

Giữ đúng kỷ luật của Bước 3: quét ngưỡng trên **validation**, chọn theo micro
F1, **đóng băng**, rồi áp con số đó lên **test**. Không bao giờ chọn ngưỡng
theo kết quả test.

Bảng quét trên validation (537 câu HATE):

| Ngưỡng | Precision | Recall | micro F1 | Số từ được flag |
|---:|---:|---:|---:|---:|
| 0,00 | 0,3608 | 0,7956 | 0,4965 | 6 117 |
| 0,05 | 0,4088 | 0,7271 | 0,5234 | 4 934 |
| 0,10 | 0,4461 | 0,6543 | 0,5305 | 4 069 |
| **0,15** | **0,4788** | **0,6020** | **0,5334** | **3 488** |
| 0,20 | 0,5031 | 0,5555 | 0,5280 | 3 063 |
| 0,30 | 0,5554 | 0,4917 | 0,5216 | 2 456 |
| 0,50 | 0,6631 | 0,3825 | 0,4851 | 1 600 |
| 0,80 | 0,8056 | 0,2390 | 0,3686 | 823 |

Ngưỡng **0,15** thắng và **nằm giữa dải quét**, không dính biên — dải 0–0,8 đủ
rộng, không cần mở rộng thêm. Đường cong đúng dạng đánh đổi P/R kinh điển:
ngưỡng càng cao, precision càng lên, recall càng xuống.

---

## 4. Kết quả

### 4.1 Bảng chính — ViHOS test, 531 câu HATE, ngưỡng 0,10 đóng băng từ validation

| Chỉ số | Giá trị |
|---|---:|
| **micro F1** | **0,5455** |
| micro precision | 0,4478 |
| micro recall | 0,6977 |
| macro F1 (trung bình theo câu) | 0,5766 |
| Số từ được flag / số từ gold | 3 953 / 2 537 |

**micro** gộp toàn bộ TP/FP/FN của cả tập rồi mới tính — câu dài có trọng số
lớn hơn. **macro** tính F1 cho từng câu rồi lấy trung bình — mọi câu có trọng
số bằng nhau. Macro (0,5766) cao hơn micro (0,5455) vì hệ thống làm tốt hơn
trên câu ngắn, mà câu ngắn lại đông (xem §4.3).

### 4.2 Phân bố chất lượng theo từng câu

| Nhóm | Số câu | Tỉ lệ |
|---|---:|---:|
| Khớp hoàn hảo (F1 = 1) | 36 | 6,8% |
| Khá (0,5 ≤ F1 < 1) | 321 | **60,5%** |
| Kém (0 < F1 < 0,5) | 164 | 30,9% |
| Trượt hoàn toàn (F1 = 0) | 10 | **1,9%** |

Trung vị F1 theo câu là **0,571**; tứ phân vị 0,414 – 0,741.

**Đây là con số quan trọng nhất của cả bước**, quan trọng hơn F1 tổng:
**67,2% số câu đạt F1 ≥ 0,5**, và **chỉ 1,9% là chỉ sai hoàn toàn**.

"Trượt hoàn toàn" nghĩa là tập từ occlusion chỉ ra và tập từ con người khoanh
**không giao nhau một từ nào**. Tỉ lệ 1,9% cho thấy: hệ thống hầu như luôn
**nhìn đúng vùng**, sai chủ yếu ở **biên** của vùng đó — rộng hơn hoặc hẹp hơn
người. Đó là loại sai khác hẳn về mức độ nghiêm trọng so với việc chỉ sai chỗ.

### 4.3 Kết quả theo độ dài câu

| Độ dài câu | Số câu | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| 1–10 từ | 214 | 0,5938 | 0,8336 | **0,6935** |
| 10–20 từ | 170 | 0,4712 | 0,7787 | 0,5871 |
| 20–40 từ | 102 | 0,3804 | 0,6042 | 0,4668 |
| 40+ từ | 45 | 0,3745 | 0,5789 | 0,4548 |

Câu càng dài, kết quả càng kém — F1 tụt từ 0,69 xuống 0,45. Lý do: câu dài có
nhiều từ trung tính, mỗi từ khi bị che đều làm margin nhiễu nhẹ, và với ngưỡng
tương đối thì nhiễu đó dễ vượt mốc 10% của đỉnh.

Đây là điểm may mắn về mặt ứng dụng: bình luận độc hại thực tế thường ngắn.

### 4.4 Kết quả theo độ dài span gold — nơi lộ ra điểm yếu rõ nhất

| Độ dài span gold | Số câu | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| 1 từ | 103 | **0,2225** | **0,9806** | 0,3627 |
| 2–3 từ | 175 | 0,4018 | 0,8381 | 0,5432 |
| 4–7 từ | 151 | 0,4735 | 0,7689 | 0,5861 |
| 8+ từ | 102 | 0,5279 | 0,5833 | 0,5542 |

Nhóm **gold chỉ 1 từ** là trường hợp cực đoan: recall gần như hoàn hảo
(**0,9806** — hầu như luôn tìm ra đúng từ đó) nhưng precision chỉ **0,2225**
(flag thừa gấp ~4,5 lần).

Nguyên nhân mang tính cấu trúc: cấu hình chốt che **cụm 2 từ**, nên mỗi lần che
đều cộng điểm cho **cả hai** từ trong cụm. Nó **về mặt cơ chế không thể** tạo ra
một span gọn đúng 1 từ. Ví dụ thật:

```
câu : Dm vtv mặt trời ^^
gold: [Dm]
pred: [Dm, vtv, mặt]        → tìm đúng "Dm" nhưng kéo theo 2 từ thừa
```

Ngược lại, span gold càng dài thì precision càng lên (0,53 ở nhóm 8+ từ) vì
vùng khoanh rộng của occlusion khớp tự nhiên hơn.

### 4.5 Xu hướng bao trùm: khoanh rộng gấp 1,56 lần

| | Trung vị | Trung bình |
|---|---:|---:|
| Độ dài span gold | 3 từ | 4,8 từ |
| Độ dài span dự đoán | 6 từ | 7,4 từ |

Tổng cộng flag 3 953 từ trong khi gold chỉ có 2 537 — **rộng gấp 1,56 lần**.
Đây là lời giải thích gốc rễ cho toàn bộ hình dạng số liệu: precision 0,4478
thấp hơn recall 0,6977 **không phải vì tìm sai chỗ**, mà vì **khoanh quá tay**.

### 4.6 Ví dụ định tính

**Khớp hoàn hảo** — câu ngắn, span gold gọn:

```
câu : Haha lại ba que xỏ lá        gold: [ba que xỏ lá]   pred: [ba que xỏ lá]
câu : con điên                     gold: [con điên]       pred: [con điên]
```

**Khoanh rộng** — dạng lỗi phổ biến nhất (30,9% nhóm "kém" chủ yếu là dạng này):

```
câu : Từ lúc mấy bro cmt cực kì cl gì đấy là rất khó để ngó được cái cmt hay buồn thực sự
gold: [cl]
pred: [cmt, cực, kì, cl, gì, đấy, khó, cái]     → trúng "cl" nhưng kéo theo 7 từ thừa
```

**Trượt hoàn toàn** — chỉ 10 câu, và phần lớn là câu dài với gold rất hẹp:

```
câu : Thối CC chỉ không ngửi đuợc thôi
gold: [Thối, CC]
pred: [không, ngửi, đuợc, thôi]                 → bắt nhầm hẳn sang nửa sau câu
```

---

## 5. Kiểm chứng và tái lập

### 5.1 Test tự động

**107 test, chạy ~13 giây**, không cần tải model thật (dùng `FakeScorer` giả
lập), nên chạy được trong CI:

| File | Số test | Kiểm gì |
|---|---:|---|
| `tests/test_build_vihos.py` | 3 | Quy đổi chỉ số ký tự → chỉ số từ, kể cả khi chỉ số rơi vào khoảng trắng |
| `tests/test_occlusion.py` | 9 | `occlude_words` đúng độ dài & bắt đúng từ "xấu"; `top_k_spans` theo ngưỡng/top-k; `normalize_importance` chia theo đỉnh và không lật dấu khi toàn điểm âm |
| `tests/test_demo.py` | 4 | Phân mức highlight và render 3 định dạng |

### 5.2 Chạy lại

```bash
# 1. Dựng lại dữ liệu (cần internet để tải ViHOS từ HF Hub)
python -m src.dataset_builder.build_vihos          # → data/data_explain/

# 2. Chạy đánh giá — ~9 phút CPU, ~2 phút trên T4
python -m src.explainability.evaluate_localization --out_dir results/step4

# 3. Giải thích một câu bất kỳ
python -m src.explainability.demo "Con này óc chó thế"
```

Cấu hình chính thức (`--scoring relative --window 2`) **đã là mặc định** trong
code — không cần truyền tay. Chạy trên Kaggle: xem [`kaggle/RUN_STEP4.md`](kaggle/RUN_STEP4.md).

Occlusion hoàn toàn tất định (không có yếu tố ngẫu nhiên), nên chạy lại cùng
checkpoint sẽ ra đúng số cũ.

### 5.3 Chi phí tính toán

Mỗi câu N từ cần **N+1 forward pass** (1 lần cho câu gốc + N lần che), gộp
thành một batch duy nhất cho mỗi câu. Tổng cho val + test: **19,3 nghìn forward
pass**. Đo thực tế trên CPU máy dev: 37,6 pass/giây → **~9 phút**.

**Bước 4 không cần GPU.** Đây là điểm khác biệt cơ bản với Bước 3 (~5 giờ trên
2×T4): ở đây không train gì cả, chỉ inference.

---

## 6. Đánh giá thẳng thắn

### 6.1 Những gì kết quả này chứng minh được

1. **Model đã học đúng thứ cần học.** 98,1% số câu occlusion chỉ ra ít nhất một
   từ trùng với vùng con người khoanh. Model không "đoán mò rồi đúng" — nó thật
   sự dựa vào cụm từ độc hại để ra quyết định. Đây là bằng chứng gián tiếp
   nhưng mạnh cho chất lượng của Bước 3.
2. **Giải thích được mà không tốn thêm gì.** Không train, không nhãn span,
   không GPU, không đổi kiến trúc.
3. **Đủ dùng cho kiểm duyệt có người trong vòng lặp.** Vùng khoanh rộng hơn
   1,56 lần là chấp nhận được khi mục đích là *hướng mắt người kiểm duyệt vào
   đúng chỗ*, chứ không phải tự động cắt bỏ.

### 6.2 Những điểm yếu phải nói rõ

1. **Precision thấp (0,4478)** — cứ 100 từ được flag thì ~55 từ không nằm trong
   vùng gold. Không dùng được cho tác vụ tự động cắt/che chữ chính xác.
2. **Không thể tạo span 1 từ.** Che cụm 2 từ khiến precision ở nhóm gold-1-từ
   chỉ còn 0,2225. Đây là giới hạn cấu trúc của cấu hình, không phải lỗi cài đặt.
3. **Xuống cấp theo độ dài câu** — F1 rơi từ 0,69 (câu ngắn) xuống 0,45 (40+ từ).
4. **Không so sánh được với hệ thống có giám sát span.** Model này chưa từng
   thấy một nhãn span nào. Con số 0,5455 phải đọc trong ngữ cảnh "giải thích hậu
   kỳ, không giám sát", không phải "hệ thống phát hiện span".
5. **Chưa có baseline để đối chiếu.** Chưa dựng baseline ngẫu nhiên hay baseline
   từ điển để chứng minh 0,5455 tốt hơn cách làm ngây thơ đến mức nào. Đây là
   thiếu sót thật của bước này (xem §7).
6. **Bản thân nhãn ViHOS cũng mơ hồ.** Khoanh biên cụm từ độc hại là việc con
   người cũng bất đồng — một phần khoảng cách F1 là do ranh giới vốn không rõ,
   không phải do model sai.

### 6.3 Vì sao không train lại trên ViHOS

Đã cân nhắc và **quyết định không làm**, vì hai lý do:

- **Lệch mục tiêu.** Bước 4 tồn tại để *giải thích* model của Bước 3. Train một
  model token-classification trên span ViHOS sẽ tạo ra **một model khác**, không
  còn giải thích gì cho model đang chạy trong pipeline nữa.
- **Không phải cách rẻ nhất để tăng số.** Hai lần cải thiện lớn nhất trong bước
  này (§3.1 margin, §3.2 chuẩn hóa) đều đến từ sửa lỗi phương pháp đo, không
  tốn một giây huấn luyện nào.

### 6.4 Hai lỗi phương pháp đã mắc và đã sửa

Ghi lại để không lặp lại:

1. **Dùng xác suất softmax làm điểm số.** Bão hòa về đúng 1,0/0,0 trong float32
   → occlusion cho ra toàn số 0. Phát hiện khi smoke-test thấy mọi từ đều có
   điểm quan trọng bằng 0,000. Sửa: dùng logit margin (§3.1).
2. **Dùng ngưỡng tuyệt đối chung cho mọi câu.** Phát hiện khi tách nhóm theo
   thang điểm và thấy recall chênh nhau 12 lần (0,062 vs 0,759). Sửa: chuẩn hóa
   theo đỉnh của từng câu (§3.2).

Cả hai đều **không phát hiện được bằng cách nhìn F1 tổng** — phải bóc số liệu
theo nhóm mới thấy. Bài học: luôn tách nhóm trước khi kết luận.

---

## 7. Việc chưa làm

| Việc | Vì sao đáng làm |
|---|---|
| Baseline ngẫu nhiên / từ điển | Để biết 0,5455 hơn cách làm ngây thơ bao nhiêu — hiện chưa có mốc so sánh nào |
| Ngưỡng thích ứng theo độ dài câu | Câu dài đang bị flag thừa; ngưỡng cao hơn cho câu dài có thể gỡ được |
| Occlusion 2 chiều | Hiện chỉ đo "che đi thì điểm tụt bao nhiêu"; có thể đo thêm chiều ngược lại |
| Kiểm chứng trên câu bị né lọc | ViHOS toàn câu sạch; chưa biết occlusion còn định vị đúng không khi input bị nhiễu như C1/C2 của Bước 3 |

---

## 8. Bản đồ file

| File | Vai trò |
|---|---|
| [`src/dataset_builder/build_vihos.py`](../src/dataset_builder/build_vihos.py) | Tải ViHOS, quy đổi span ký tự → span từ |
| [`src/explainability/occlusion.py`](../src/explainability/occlusion.py) | Lõi: chấm điểm margin, che cụm từ, chuẩn hóa, chọn span |
| [`src/explainability/evaluate_localization.py`](../src/explainability/evaluate_localization.py) | Dò ngưỡng trên val, đóng băng, chấm điểm trên test |
| [`src/explainability/demo.py`](../src/explainability/demo.py) | Giải thích một câu bất kỳ, tô đậm cụm gây cảnh báo |
| `data/data_explain/` | Bộ ViHOS đã quy đổi span mức từ (3 split + manifest) |
| `results/step4/localization.json` | Bảng dò ngưỡng + số liệu test |
| `results/step4/predictions_test.jsonl` | Dự đoán từng câu — dùng để soi định tính |
| [`docs/kaggle/RUN_STEP4.md`](kaggle/RUN_STEP4.md) | Hướng dẫn chạy trên Kaggle |
