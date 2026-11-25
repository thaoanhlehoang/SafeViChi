# Baseline SafeViChi v1 trên bộ test v2

> Tệp này sinh tự động bằng `python -m src.v2.eval.bao_cao_baseline`. Không sửa tay.

v1 (ViHateT5 fine-tune đối kháng FGM, vòng loại) chỉ trả HATE/CLEAN và không trả bằng chứng, nên chỉ chấm **nhị phân nguy hại/an toàn**: HATE ứng với 4 nhãn nguy hại của v2, CLEAN ứng với `AN_TOAN`. Không có Macro-F1 5 nhãn và không có số bằng chứng. Script: `src/v2/eval/cham_v1.py`; dự đoán: `results/v2/v1_baseline/*_predictions.jsonl`.

## Nhị phân nguy hại/an toàn

| Chỉ số | C0 (334 mẫu) | C1 (668 mẫu) | C2 (668 mẫu) |
| --- | --- | --- | --- |
| Precision | 1,0000 | 0,9670 | 0,9370 |
| **Recall** | **0,1849** | **0,1849** | **0,2500** |
| Binary-F1 | 0,3121 | 0,3104 | 0,3947 |
| FPR câu an toàn | 0,0% | 1,6% | 4,2% |
| TP / FP / FN / TN | 44 / 0 / 194 / 96 | 88 / 3 / 388 / 189 | 119 / 8 / 357 / 184 |

v1 gán CLEAN cho 290/334 câu C0, nên precision cao chỉ phản ánh việc nó hiếm khi báo HATE.

## Recall theo nhãn v2

| Nhãn | C0 | C1 | C2 |
| --- | --- | --- | --- |
| XUC_PHAM | 0,7358 | 0,6792 | 0,6792 |
| QUAY_ROI | 0,0000 | 0,0196 | 0,0980 |
| DE_DOA | 0,0806 | 0,1048 | 0,2500 |
| LUA_DAO | 0,0000 | 0,0069 | 0,0417 |

v1 được huấn luyện cho hate speech nên chủ yếu bắt được `XUC_PHAM`; `QUAY_ROI` và `LUA_DAO` gần như bị bỏ qua.

## Theo giọng vùng

| Tập | Vùng | n | Precision | Recall | Binary-F1 | FPR an toàn |
| --- | --- | --- | --- | --- | --- | --- |
| C0 | Bắc | 112 | 1,0000 | 0,1899 | 0,3191 | 0,0% |
| C0 | Trung | 111 | 1,0000 | 0,1392 | 0,2444 | 0,0% |
| C0 | Nam | 111 | 1,0000 | 0,2250 | 0,3673 | 0,0% |
| C1 | Bắc | 224 | 1,0000 | 0,1772 | 0,3011 | 0,0% |
| C1 | Trung | 222 | 0,9091 | 0,1266 | 0,2222 | 3,1% |
| C1 | Nam | 222 | 0,9756 | 0,2500 | 0,3980 | 1,6% |
| C2 | Bắc | 224 | 0,9184 | 0,2848 | 0,4348 | 6,1% |
| C2 | Trung | 222 | 0,9667 | 0,1835 | 0,3085 | 1,6% |
| C2 | Nam | 222 | 0,9375 | 0,2812 | 0,4327 | 4,8% |

| Tập | Vùng yếu nhất (Binary-F1) | Chênh lệch cao nhất − thấp nhất |
| --- | --- | --- |
| C0 | 0,2444 | 0,1229 |
| C1 | 0,2222 | 0,1758 |
| C2 | 0,3085 | 0,1263 |

## Theo kiểu biến dạng

| Tập | Kiểu | n | Recall | Binary-F1 | Báo nhầm câu an toàn |
| --- | --- | --- | --- | --- | --- |
| C1 | `bo_dau` | 102 | 0,2368 | 0,3789 | 1/26 |
| C1 | `teencode` | 103 | 0,2329 | 0,3778 | 0/30 |
| C1 | `go_sai` | 104 | 0,1831 | 0,3023 | 2/33 |
| C1 | `dinh_tu` | 96 | 0,1642 | 0,2821 | 0/29 |
| C1 | `so_thay_chu` | 93 | 0,2121 | 0,3500 | 0/27 |
| C1 | `keo_dai` | 89 | 0,0968 | 0,1765 | 0/27 |
| C1 | `chen_ky_tu` | 81 | 0,1475 | 0,2571 | 0/20 |
| C2 | `fullwidth` | 122 | 0,2000 | 0,3333 | 0/32 |
| C2 | `zero_width` | 116 | 0,2184 | 0,3585 | 0/29 |
| C2 | `space_injection` | 114 | 0,4512 | 0,5873 | 7/32 |
| C2 | `exotic_separator` | 108 | 0,1711 | 0,2921 | 0/32 |
| C2 | `cyrillic_homoglyph` | 110 | 0,1216 | 0,2143 | 1/36 |
| C2 | `double_elongation` | 98 | 0,3433 | 0,5111 | 0/31 |
