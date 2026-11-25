# Baseline LLM lớn qua API: gpt-oss-120b trên bộ test v2

> Tệp này sinh tự động bằng `python -m src.v2.eval.bao_cao_baseline`. Không sửa tay.

Chấm lại ngày 25/11/2026 bằng bộ chấm hiện tại (`src/v2/eval/`, sau commit `20989e4` đổi cách tính bằng chứng). Dự đoán lấy nguyên từ lượt chạy API ngày 24/11 (`results/v2/llm_api/*_predictions.jsonl`), không gọi API lại; chỉ đổi tên trường `nhan_llm`, `bang_chung_llm` thành `nhan`, `bang_chung` (`src/v2/eval/chuyen_du_doan_llm.py`). Các file `results/v2/llm_api/test*_metrics.json` cũ giữ lại vì còn log lượt gọi API, nhưng **số bằng chứng trong đó tính theo bộ chấm cũ, không dùng cho báo cáo.**

## Phân loại và bằng chứng

| Chỉ số | C0 (334 mẫu) | C1 (668 mẫu) | C2 (668 mẫu) |
| --- | --- | --- | --- |
| Macro-F1 5 nhãn | 0,7723 [0,7242; 0,8157] | 0,5854 [0,5441; 0,6207] | 0,6272 [0,5891; 0,6648] |
| Binary-F1 nguy hại/an toàn | 0,8920 [0,8591; 0,9204] | 0,7449 [0,7063; 0,7763] | 0,7793 [0,7447; 0,8102] |
| Precision nguy hại | 0,9848 | 0,9335 | 0,9748 |
| Recall nguy hại | 0,8151 | 0,6197 | 0,6492 |
| FPR câu an toàn | 3,1% | 10,9% | 4,2% |
| Bằng chứng: precision theo ý | 0,6783 | 0,4561 | 0,2870 |
| Bằng chứng: recall theo ý | 0,5403 | 0,3130 | 0,1844 |
| **Bằng chứng: F1 theo ý** | **0,5765** | **0,3492** | **0,2088** |
| Bằng chứng: F1 mức từ | 0,5335 | 0,3102 | 0,1826 |
| Câu nguy hại không có bằng chứng | 21,9% | 44,3% | 62,0% |
| Tỉ lệ đầu ra hợp lệ | 100,0% | 100,0% | 100,0% |

Khoảng trong ngoặc vuông là khoảng tin cậy 95% bootstrap (1.000 lần, seed 42).

## F1 theo nhãn

| Nhãn | C0 | C1 | C2 |
| --- | --- | --- | --- |
| AN_TOAN | 0,7983 | 0,6287 | 0,6777 |
| XUC_PHAM | 0,8673 | 0,7551 | 0,7662 |
| QUAY_ROI | 0,5974 | 0,3433 | 0,4818 |
| DE_DOA | 0,6408 | 0,3125 | 0,3171 |
| LUA_DAO | 0,9577 | 0,8874 | 0,8935 |

## Theo giọng vùng

| Tập | Vùng | n | Macro-F1 | Binary-F1 | FPR an toàn | BC F1 theo ý |
| --- | --- | --- | --- | --- | --- | --- |
| C0 | Bắc | 112 | 0,7559 | 0,8889 | 3,0% | 0,6152 |
| C0 | Trung | 111 | 0,7279 | 0,8811 | 3,1% | 0,5174 |
| C0 | Nam | 111 | 0,8105 | 0,9054 | 3,2% | 0,5967 |
| C1 | Bắc | 224 | 0,5551 | 0,7176 | 15,2% | 0,3576 |
| C1 | Trung | 222 | 0,5779 | 0,7433 | 9,4% | 0,3392 |
| C1 | Nam | 222 | 0,6017 | 0,7732 | 8,1% | 0,3508 |
| C2 | Bắc | 224 | 0,6179 | 0,7490 | 6,1% | 0,1886 |
| C2 | Trung | 222 | 0,5794 | 0,7597 | 3,1% | 0,2125 |
| C2 | Nam | 222 | 0,6779 | 0,8261 | 3,2% | 0,2252 |

| Tập | Vùng yếu nhất (Macro-F1) | Chênh lệch cao nhất − thấp nhất |
| --- | --- | --- |
| C0 | 0,7279 | 0,0826 |
| C1 | 0,5551 | 0,0466 |
| C2 | 0,5794 | 0,0985 |

Mỗi vùng chỉ có khoảng 32 câu an toàn ở C0 (khoảng 64 ở C1/C2), nên một câu sai làm FPR theo vùng đổi khoảng 3 điểm phần trăm ở C0. Chênh lệch giữa vùng cần đọc kèm n.

## Theo kiểu biến dạng

| Tập | Kiểu | n | Macro-F1 | Binary-F1 | FPR an toàn | BC F1 theo ý |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | `bo_dau` | 102 | 0,5811 | 0,7107 | 7,7% | 0,3271 |
| C1 | `teencode` | 103 | 0,5601 | 0,8033 | 0,0% | 0,4258 |
| C1 | `go_sai` | 104 | 0,5339 | 0,7167 | 18,2% | 0,3370 |
| C1 | `dinh_tu` | 96 | 0,5737 | 0,7387 | 10,3% | 0,4271 |
| C1 | `so_thay_chu` | 93 | 0,6751 | 0,8136 | 14,8% | 0,4355 |
| C1 | `keo_dai` | 89 | 0,5514 | 0,7143 | 3,7% | 0,2308 |
| C1 | `chen_ky_tu` | 81 | 0,5041 | 0,7059 | 25,0% | 0,2406 |
| C2 | `fullwidth` | 122 | 0,7371 | 0,8765 | 3,1% | 0,0750 |
| C2 | `zero_width` | 116 | 0,5785 | 0,7917 | 0,0% | 0,0819 |
| C2 | `space_injection` | 114 | 0,4510 | 0,5862 | 0,0% | 0,2096 |
| C2 | `exotic_separator` | 108 | 0,5581 | 0,7969 | 3,1% | 0,2582 |
| C2 | `cyrillic_homoglyph` | 110 | 0,7236 | 0,7937 | 5,6% | 0,2910 |
| C2 | `double_elongation` | 98 | 0,6700 | 0,7863 | 12,9% | 0,4057 |
