# Kế hoạch SafeViChi v2 – vòng chung kết

Hạn nộp: **26/11/2026**. Nhóm: 2 người. Máy: Kaggle (miễn phí), Colab, máy riêng, có ngân sách API LLM.
Bản đầy đủ có giải thích nằm trong tài liệu dự án trên Claude; file này là bản làm việc trong repo.

## Mục tiêu

Thay hai nhãn HATE/CLEAN bằng **5 nhãn theo tình huống** (`AN_TOAN`, `XUC_PHAM`, `QUAY_ROI`, `DE_DOA`, `LUA_DAO`),
mô hình nhỏ dưới 2 GB trả về nhãn kèm cụm từ bằng chứng, và trả lời đủ 3 yêu cầu của ban tổ chức.

## Lịch theo ngày

| Ngày | Việc | Máy | Xong |
| --- | --- | --- | --- |
| T5 19/11 | Chốt `HUONG_DAN_GAN_NHAN.md`; tạo nhánh `v2`, tag `v1-vong-loai`; script LLM gán sơ bộ ViHSD, ViHOS, ViTHSD sang 5 nhãn; bắt đầu gom tin lừa đảo thật (đã che thông tin); sửa số lệch của v1 | API, máy riêng | [ ] |
| T6 20/11 | Hai người gán thử 100 câu, đo Cohen's kappa, sửa hướng dẫn; LLM sinh câu theo kịch bản cho `DE_DOA`, `QUAY_ROI`, `LUA_DAO` | API | [ ] |
| T7 21/11 | Hai người duyệt độc lập tập test (~150 câu thật mỗi nhãn, kèm bằng chứng); khử trùng, chia tập, sinh biến dạng C1/C2; **đóng băng dataset** | Máy riêng | [ ] |
| CN 22/11 | Train song song: Qwen3-1.7B QLoRA và encoder đa nhiệm; chạy LLM API làm baseline trên test | Colab, Kaggle, API | [ ] |
| T2 23/11 | Ablation: bỏ biến dạng, bỏ chuẩn hóa, 4-bit vs 16-bit; zero-shot mô hình nhỏ chưa fine-tune | Colab, Kaggle | [ ] |
| T3 24/11 | Đo chi phí trên laptop và trình duyệt; phân tích lỗi theo nhóm; chọn ví dụ thất bại; gắn mô hình mới vào demo | Máy riêng | [ ] |
| T4 25/11 | Viết báo cáo, sinh bảng và hình từ `results/v2`; ngày dự phòng | | [ ] |
| T5 26/11 | Nộp | | [ ] |

Mốc quan trọng nhất: **tối 21/11 đóng băng dataset.** Nếu trễ, giữ đủ 5 nhãn nhưng giảm test của nhãn còn thiếu
xuống ~100 câu và ghi rõ trong báo cáo.

## Dataset

| Nhãn | Câu thật lấy từ | Bổ sung bằng LLM sinh (chỉ train) |
| --- | --- | --- |
| `AN_TOAN` | Phần CLEAN của ViHSD, ViHOS, ViTHSD; tin giao dịch thật đã che (OTP, giao hàng) | Không cần |
| `XUC_PHAM` | ViHSD, ViHOS, ViTHSD (đã có nhãn người gán) | Không cần |
| `DE_DOA` | Lọc lại phần HATE/OFFENSIVE của 3 bộ trên và kho VOZ (chỉ lấy câu, gán lại) | Dọa tung ảnh, tống tiền, đòi nợ khủng bố |
| `QUAY_ROI` | Lọc lại từ cùng kho: gạ gẫm, bêu riếu ngoại hình | Gạ gẫm qua tin riêng, nhắn tiếp sau khi bị từ chối |
| `LUA_DAO` | Tin rác SMS/Zalo/Messenger của nhóm và người quen (xin phép, che thông tin); ví dụ trong bài cảnh báo của Cục An toàn thông tin | Theo từng hình thức lừa đảo trong danh sách cảnh báo |

Quy mô mục tiêu: test ~150 câu thật mỗi nhãn; train 1.500–3.000 câu mỗi nhãn nguy hại, `AN_TOAN` gấp 2–3 lần.
Train do LLM gán: người duyệt mọi câu LLM không chắc hoặc lệch nhãn gốc, cộng mẫu ngẫu nhiên 10–20% để ước tỉ lệ sai.

Các bước:

1. [ ] `src/v2/data/merge.py`: gộp 3 bộ về định dạng chung trong `CLAUDE.md`.
2. [ ] `src/v2/labeling/prelabel.py`: LLM gán sơ bộ theo hướng dẫn, lưu cả lý do và độ chắc.
3. [ ] `src/v2/labeling/generate.py`: sinh câu theo bảng kịch bản × kênh × giọng văn.
4. [ ] `src/v2/labeling/agreement.py`: kappa giữa 2 người, danh sách câu bất đồng.
5. [ ] `src/v2/data/split.py`: khử trùng (dùng `dedup.py`), chia theo nhóm, giữ 2 kiểu điểm mù ngoài train.
6. [ ] `src/v2/data/perturb.py`: sinh C1/C2 theo từng từ để vị trí bằng chứng đi theo.
7. [ ] Thẻ dữ liệu `data/v2/DATASET_CARD.md`, checksum, test tự động.

## Bảng thực nghiệm (yêu cầu 3)

| # | Cấu hình | Trả lời câu hỏi | Train? |
| --- | --- | --- | --- |
| 1 | Blacklist từ khóa | Mốc sàn | Không |
| 2 | SafeViChi v1 | v2 có hơn vòng trước không (so trên nhị phân nguy hại/an toàn) | Không |
| 3 | Encoder đa nhiệm (ViSoBERT hoặc PhoBERT) | Mô hình tiếng Việt cơ sở | 1 lượt |
| 4 | Qwen3-1.7B chưa fine-tune, zero/few-shot | Fine-tune đóng góp bao nhiêu | Không |
| 5 | Qwen3-1.7B fine-tune chỉ câu sạch | Mốc cho dữ liệu biến dạng | 1 lượt |
| 6 | #5 + dữ liệu biến dạng | Lợi ích của đối kháng mức dữ liệu | 1 lượt |
| 7 | #6 + chuẩn hóa đầu vào (v2 đầy đủ) | Lợi ích của chuẩn hóa | Không |
| 8 | #7 bản 4-bit | Lượng tử hóa mất bao nhiêu, tiết kiệm bao nhiêu | Không |
| 9 | LLM lớn qua API, zero/few-shot | Trần chất lượng và cái giá | Không |

Cột đo cho mỗi dòng:

- Phân loại: F1 từng nhãn, macro-F1, F1 nhị phân nguy hại/an toàn; trên C0, C1, C2, kiểu điểm mù chưa thấy,
  bộ câu khó (tách theo thẻ `tieng_long`, `vung_mien`, `an_y`, `trich_dan`), câu thật và câu sinh.
- Bằng chứng: precision, recall, F1 mức từ; tỉ lệ câu `AN_TOAN` bị tô oan; tỉ lệ cụm không có trong câu (mô hình sinh).
- Chi phí: dung lượng, RAM đỉnh, độ trễ p50/p95 mỗi câu trên laptop CPU và trình duyệt, năng lượng ước tính, tiền/1.000 câu với API.

## Cắt khỏi bản nộp 26/11

- Nhãn phụ.
- Các lượt train lại để giữ riêng từng kiểu biến dạng (thay bằng giữ 2 kiểu điểm mù ngoài train ngay khi dựng dữ liệu).
- Tập test thu riêng từ nền tảng khác (tin lừa đảo thật từ SMS/Zalo đã là nguồn khác với dữ liệu train).

## Rủi ro

- Train chủ yếu do LLM gán và sinh: phải nói rõ, báo tỉ lệ nhãn sai ước từ mẫu kiểm tay, và báo điểm riêng trên câu thật.
- Độ trễ mô hình sinh trên thiết bị: nếu 23/11 thấy quá chậm, encoder đa nhiệm làm mô hình chính, Qwen3 làm dòng so sánh.
- Sức chống né lọc phải chứng minh lại trên C1/C2.
- Hỏi ban tổ chức sớm: có được đổi bộ nhãn và dùng LLM gán dữ liệu train không.
