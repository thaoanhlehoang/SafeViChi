# Kế hoạch SafeViChi v2 – vòng chung kết

Hạn nộp: **26/11/2026**. Nhóm: 2 người. Máy: Kaggle (miễn phí), Colab, máy riêng, có ngân sách API LLM.
Bản đầy đủ có giải thích nằm trong tài liệu dự án trên Claude; file này là bản làm việc trong repo.

## Mục tiêu

Thay hai nhãn HATE/CLEAN bằng **5 nhãn theo tình huống** (`AN_TOAN`, `XUC_PHAM`, `QUAY_ROI`, `DE_DOA`, `LUA_DAO`),
mô hình nhỏ dưới 2 GB trả về nhãn kèm cụm từ bằng chứng, và trả lời đủ 3 yêu cầu của ban tổ chức.

## Lịch theo ngày

| Ngày | Việc | Máy | Xong |
| --- | --- | --- | --- |
| T5 19/11 | Chốt `HUONG_DAN_GAN_NHAN.md`; tạo nhánh `v2-final`, tag `v1-vong-loai`; script LLM gán sơ bộ ViHSD, ViHOS, ViTHSD sang 5 nhãn; bắt đầu gom tin lừa đảo thật (đã che thông tin); sửa số lệch của v1 | API, máy riêng | [ ] |
| T6 20/11 | Hai người gán thử 100 câu, đo Cohen's kappa, sửa hướng dẫn; LLM sinh câu theo kịch bản cho `DE_DOA`, `QUAY_ROI`, `LUA_DAO` | API | [ ] |
| T7 21/11 | Hai người duyệt độc lập tập test (~150 câu thật mỗi nhãn, kèm bằng chứng); khử trùng, chia tập, sinh biến dạng C1/C2 **chỉ cho test**; **đóng băng dataset** | Máy riêng | [ ] |
| CN 22/11 | Train song song: Qwen3-1.7B QLoRA và encoder đa nhiệm; chạy LLM API làm baseline trên test | Colab, Kaggle, API | [ ] |
| T2 23/11 | Ablation: bỏ chuẩn hóa, 4-bit vs 16-bit; zero-shot mô hình nhỏ chưa fine-tune | Colab, Kaggle | [ ] |
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

Quy mô mục tiêu: test ~150 câu thật mỗi nhãn; **train khoảng 15.000 đoạn** (chốt 20/11, xem bảng dưới).
Train do LLM gán: người duyệt mọi câu LLM không chắc hoặc lệch nhãn gốc, cộng mẫu ngẫu nhiên 10–20% để ước tỉ lệ sai.

### Quyết định 19–20/11

- **Dữ liệu gốc:** tải đủ ViHSD (train/dev/test) và ViTHSD từ Hugging Face, ghim revision
  (`data/raw/NGUON_TAI.json`). ViHOS và ViTHSD gần như toàn là câu của ViHSD, nên gộp theo text còn **30.580 câu
  khác nhau**; nhãn gốc của từng bộ giữ ở `nguon_goc`, câu các bộ gán mâu thuẫn có `nhan_goc_bat_dong=true`.
- **Kho test tách trước khi gọi LLM:** sau khử trùng gần đúng, nhóm câu nào chưa từng nằm trong split train của
  ViHSD/ViHOS (ViHateT5 nền của v1 đã học phần đó) và chưa từng gửi LLM thì vào kho test (6.724 câu,
  `data/v2/interim/kho_test.jsonl`). Kho test **không bao giờ gửi LLM**; câu test/validation chọn từ đây do người gán.
- **Gán lại có chọn lọc 10.062 câu** của kho train (`can_gan_lai.jsonl`): mọi câu nhãn gốc nguy hại, ứng viên từ
  khóa, câu CLEAN có từ trong blacklist v1 (ca dễ báo nhầm), cộng 3.000 câu CLEAN ngẫu nhiên (seed 42).
- **LLM:** DeepSeek V4.1 Flash 1017 qua OpenRouter, khóa endpoint `wafer/us`, tắt thinking, giá trần 0,04/0,5 USD
  mỗi triệu token. Đã so với Qwen3.8 27B trên cùng 10 mẫu (`results/v2/thu_model/`): DeepSeek sinh `DE_DOA` rõ
  hơn, không hỏng định dạng, rẻ hơn khoảng 3 lần; bỏ Qwen.
- **Sinh 10.720 đoạn** (2.144 lượt gọi): `LUA_DAO` ~1.880, `DE_DOA` 1.500, `QUAY_ROI` 1.500, `XUC_PHAM` ~480 (chỉ
  hội thoại nhiều dòng), cộng đoạn sinh đôi `AN_TOAN` cùng số. Sau đó **lọc lại**: LLM gán lại từng đoạn mà không
  biết nhãn yêu cầu, chỉ giữ đoạn trùng nhãn.
- **Xử lý nhãn LLM lệch nhãn gốc (chốt 20/11):** không gán lại bằng tay. Chỉ giữ câu thật mà nhãn LLM (sạch/nguy hại)
  khớp với ít nhất một người gán gốc; câu mà mọi người gán gốc cùng một ý còn LLM nói ngược lại thì bỏ khỏi train
  (`python -m src.v2.data.dong_thuan` -> `that_gan_lai.jsonl`, trường `dong_thuan`). Kết quả: giữ 8.649, bỏ 1.413
  (865 HATE→`AN_TOAN`, 475 CLEAN→`XUC_PHAM`, 73 CLEAN→nhãn hiếm). Hai người gán độc lập mẫu 200 câu (100 giữ,
  100 bỏ, ẩn nhãn) trong `data/v2/raw/kiem_nhan_A.csv`, `kiem_nhan_B.csv`; `python -m src.v2.labeling.agreement`
  tính kappa, ước tỉ lệ nhãn sai của câu giữ, và cho biết câu bị bỏ thường đúng theo LLM hay theo nhãn gốc.
- **Chi phí và thời gian** ghi ở `results/v2/chi_phi_du_lieu/chi_phi.json` (sinh bằng `python -m src.v2.labeling.chi_phi`).

| Nhãn | Nguồn | Dự kiến lấy vào train |
| --- | --- | --- |
| `AN_TOAN` | ~4.500 câu thật gán lại + đoạn sinh đôi | ~6.000 |
| `XUC_PHAM` | ~5.000 câu thật gán lại + ~480 hội thoại sinh | ~3.000 |
| `LUA_DAO` | Đoạn sinh (+ tin lừa đảo thật thu thập, nếu có) | ~1.750 |
| `DE_DOA` | Đoạn sinh + câu thật gán lại | ~1.450 |
| `QUAY_ROI` | Đoạn sinh + câu thật gán lại | ~1.450 |

Các bước:

1. [x] `src/v2/data/download.py`, `src/v2/data/merge.py`: tải và gộp 3 bộ về định dạng chung, mỗi text một bản ghi.
2. [x] `src/v2/data/chon_gan_lai.py`: khử trùng, tách kho test, chọn câu cần gán lại.
3. [x] `src/v2/labeling/prelabel.py`: LLM gán sơ bộ 10.062 câu (chạy 20/11, 7,8 phút).
4. [x] `src/v2/labeling/generate.py` + `loc_sinh.py`: sinh 10.557 đoạn, giữ 9.625 sau lọc (chạy 20/11).
   Chi phí cả bước 3–4: `results/v2/chi_phi_du_lieu/chi_phi.json`.
5. [ ] `src/v2/labeling/agreement.py`: kappa giữa 2 người, danh sách câu bất đồng (script xong 20/11; chờ
   hai người gán `kiem_nhan_A.csv`, `kiem_nhan_B.csv`).
6. [ ] `src/v2/data/split.py` (phần train xong 20/11: `data/v2/train.jsonl` 13.392 dòng, thẻ `data/v2/DATASET_CARD.md`): chọn test/validation từ kho test, lập train ~15.000 đoạn cân nhãn, giữ các kiểu điểm mù C2
   ngoài train, loại đoạn sinh gần trùng câu test người viết. Tệp train lưu bản gốc `C0`; huấn luyện được phép tăng cường trực tuyến phi điểm mù.
7. [x] `src/v2/data/perturb.py`: sinh C1/C2 **chỉ cho test** (và validation), theo từng từ để bằng chứng đi theo.
   Mỗi câu có 13 bản một kiểu (7 C1, 6 C2) và 4 bản trộn (C1, C2, đã thấy, chưa thấy), biến dạng mạnh (~60% số từ,
   luôn trúng cụm bằng chứng). "Đã thấy" = kiểu xuất hiện tự nhiên trong >= 1% câu train (script đo bằng `--train`,
   chỉ đọc). Lệnh: `python -m src.v2.data.perturb data/v2/test_day_du.jsonl --train data/v2/train.jsonl
   --out data/v2/interim/test_bien_dang.jsonl`.
   Bản gọn dùng khi chạy mô hình (23/11): `--nhom C1 --so-ban 2` và `--nhom C2 --so-ban 2` trên `data/v2/test.jsonl` ra
   `data/v2/test_perturb_C1.jsonl`, `test_perturb_C2.jsonl` (mỗi câu 2 dòng bốc ngẫu nhiên theo kiểu, 668 dòng mỗi file,
   không giữ bản gốc; `goc_id` trỏ về câu gốc).
   `python -m src.v2.data.tach_test_bien_dang` gắn `giong_vung`, `phan_test` từ `test_day_du.jsonl` (theo `goc_id`) vào
   `test_perturb_C1/C2.jsonl` và có thể sinh cục bộ 15 file trong `data/v2/test_perturb/` (đã đưa vào `.gitignore` để gọn
   repo; thống kê gộp trong `data/v2/test_stats.json` và `python -m src.v2.eval.evaluate` tự cắt lát in-memory).
8. [x] Test người viết (21–23/11): `data/v2/test.jsonl` 334 câu (cùng định dạng train), `data/v2/test_day_du.jsonl`
   thêm `giong_vung` (bac/trung/nam), `phan_test` (trong_phan_phoi/chua_thay), `kich_ban`, nhãn người/LLM.
   Tạo lại từ 4 file CSV: `python -m src.v2.data.load_test_nguoi_viet data/v2/raw/test_A.csv data/v2/raw/test_B.csv
   data/v2/raw/test_C.csv data/v2/raw/test_D.csv --xuat-test`. 23/11: thêm `test_D.csv` 45 câu `XUC_PHAM`
   chửi rủa, lăng mạ người đối diện và gia đình họ (XP1, 3 giọng vùng, không có đe dọa hay quấy rối), vì 8 câu
   `XUC_PHAM` cũ đều là ca ranh giới (mỉa mai, bình phẩm ngoại hình). Nay `XUC_PHAM` 53 câu, `QUAY_ROI` 51 câu.
   Kiểm tra rò rỉ với train và kho test: `python -m src.v2.data.kiem_tra_ro_ri` (0 trùng id, 0 gần trùng).
9. [ ] Thẻ dữ liệu `data/v2/DATASET_CARD.md`, checksum, test tự động.

## Bảng thực nghiệm (yêu cầu 3)

| # | Cấu hình | Trả lời câu hỏi | Train? |
| --- | --- | --- | --- |
| 1 | Blacklist từ khóa | Mốc sàn | Không |
| 2 | SafeViChi v1 | v2 có hơn vòng trước không (so trên nhị phân nguy hại/an toàn) | Không |
| 3 | Encoder đa nhiệm (ViSoBERT hoặc PhoBERT) | Mô hình tiếng Việt cơ sở | 1 lượt |
| 4 | Qwen3-1.7B chưa fine-tune, zero/few-shot | Fine-tune đóng góp bao nhiêu | Không |
| 5 | Mô hình fine-tune (tăng cường trực tuyến bỏ dấu / C1 đơn nghĩa phi điểm mù; kèm ablation tắt tăng cường) | Mô hình tự chống né lọc được bao nhiêu | 1 lượt |
| 6 | #5 + chuẩn hóa đầu vào (v2 đầy đủ) | Lợi ích của chuẩn hóa: #6 − #5 trên C1 | Không |
| 7 | #6 bản lượng tử hóa (INT8/4-bit) | Lượng tử hóa mất bao nhiêu, tiết kiệm bao nhiêu | Không |
| 8 | LLM lớn qua API, zero/few-shot | Trần chất lượng và cái giá | Không |

Quy tắc tăng cường dữ liệu (đã gỡ bỏ giới hạn chỉ học câu sạch): tệp `data/v2/train.jsonl` vẫn lưu bản gốc `C0`,
còn quá trình huấn luyện được phép tăng cường trực tuyến (bỏ dấu, tập C1 đơn nghĩa phi điểm mù) và đo ablation khi
bật/tắt. **Bắt buộc giữ hoàn toàn C2 (6 kiểu điểm mù) và tập C1 held-out ngoài train** để làm cột "kiểu né lọc
chưa thấy" cho yêu cầu 2.

**Chưa chốt dòng #8.** Đã bỏ Qwen3.8 27B. Yêu cầu 3 của ban tổ chức có nhắc "LLM lớn qua API"; nếu giữ dòng này thì
dùng chính DeepSeek V4.1 Flash và ghi rõ trong báo cáo rằng nó cũng là LLM gán sơ bộ dữ liệu train, nên điểm có thể
cao hơn thực tế.

Cột đo cho mỗi dòng:

- Phân loại: F1 từng nhãn, macro-F1, F1 nhị phân nguy hại/an toàn; trên C0, C1, C2 (kiểu né lọc chưa thấy), theo giọng vùng miền (`giong_vung`),
  bộ câu khó (tách theo thẻ `tieng_long`, `vung_mien`, `an_y`, `trich_dan`), câu thật và câu sinh.
- Bằng chứng: precision, recall, F1 mức từ; tỉ lệ câu `AN_TOAN` bị tô oan; tỉ lệ cụm không có trong câu (mô hình sinh).
- Chi phí: dung lượng, RAM đỉnh, độ trễ p50/p95 mỗi câu trên laptop CPU và trình duyệt, năng lượng ước tính, tiền/1.000 câu với API.

## Cắt khỏi bản nộp 26/11

- Nhãn phụ.
- Tập test thu riêng từ nền tảng khác (tin lừa đảo thật từ SMS/Zalo đã là nguồn khác với dữ liệu train).

## Rủi ro

- Train chủ yếu do LLM gán và sinh: phải nói rõ, báo tỉ lệ nhãn sai ước từ mẫu kiểm tay, và báo điểm riêng trên câu thật.
- Câu test người viết phải viết trước khi đọc câu LLM sinh. Ngày 19–20/11 một thành viên đã đọc khoảng 70 đoạn sinh
  thử, nên thành viên còn lại viết câu test `nguoi_viet`; sau đó chạy lại `generate` + `loc_sinh` (dùng cache, không
  tốn tiền) để loại đoạn sinh gần trùng câu test.
- Độ trễ mô hình sinh trên thiết bị: nếu 23/11 thấy quá chậm, encoder đa nhiệm làm mô hình chính, Qwen3 làm dòng so sánh.
- Sức chống né lọc phải chứng minh lại trên C1/C2.
- Hỏi ban tổ chức sớm: có được đổi bộ nhãn và dùng LLM gán dữ liệu train không.
