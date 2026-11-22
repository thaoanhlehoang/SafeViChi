# SafeViChi – hướng dẫn cho Claude Code

## Bối cảnh

SafeViChi phát hiện tin nhắn tiếng Việt nguy hại, kể cả khi bị viết biến dạng để lách bộ lọc
(teencode, bỏ dấu, chèn ký tự, homoglyph). Hệ thống chạy trên thiết bị người dùng, không gửi tin nhắn ra ngoài.

- **v1 (vòng loại, nhánh `main`, tag `v1-vong-loai`):** chuẩn hóa 5 tầng → ViHateT5 fine-tune đối kháng (FGM)
  → occlusion để tô sáng. Nhãn HATE/CLEAN. Kết quả nằm trong `results/step3`, `results/step4`.
- **v2 (vòng chung kết, nhánh `v2-final`):** đổi sang 5 nhãn theo tình huống và một mô hình ngôn ngữ nhỏ
  (dưới 2 GB khi lượng tử hóa) trả về nhãn kèm cụm từ bằng chứng.
- **Hạn nộp: 26/11/2026.** Lịch từng ngày nằm trong `docs/v2/KE_HOACH.md`. Ưu tiên giữ lịch hơn làm đẹp code.

## Đề cải tiến của ban tổ chức (phải trả lời đủ cả 3)

1. Giảm lỗi giải thích và kiểm duyệt sai: phân tích false positive (tiếng lóng, vô hại) và false negative,
   hiệu chỉnh ngưỡng, tách kết quả phân loại khỏi bằng chứng tô sáng. Đo precision/recall riêng cho hai nhiệm vụ.
2. Khái quát trước kiểu né lọc mới: F1 theo nhóm (nguồn, kiểu né lọc chưa thấy, vùng miền), không để câu gần trùng
   giữa train và test, kèm ví dụ mô hình vẫn thất bại.
3. So sánh phương pháp và chi phí: baseline (v1, encoder tiếng Việt, LLM lớn qua API) và ablation từng mô-đun,
   đo độ trễ, bộ nhớ, năng lượng trên môi trường đích (laptop CPU và trình duyệt). Mọi cấu hình phải tái hiện được.

## Bộ nhãn v2

| Nhãn | Ý nghĩa ngắn | Ứng dụng làm gì |
| --- | --- | --- |
| `AN_TOAN` | Không thuộc 4 nhãn dưới | Hiện bình thường |
| `XUC_PHAM` | Lăng mạ, miệt thị người hoặc nhóm (kể cả vùng miền) | Làm mờ, bấm mới xem |
| `QUAY_ROI` | Gạ gẫm không mong muốn, bám riết, bêu riếu | Gợi ý chặn, báo cáo |
| `DE_DOA` | Dọa gây hại thân thể, tài sản, danh dự; tống tiền | Cảnh báo mạnh, gợi ý lưu bằng chứng |
| `LUA_DAO` | Thông tin giả để dụ chuyển tiền, đưa OTP, bấm link | Cảnh báo, vô hiệu link |

Định nghĩa đầy đủ và ca khó: `docs/v2/HUONG_DAN_GAN_NHAN.md`. Khi nhiều nhãn cùng đúng: `DE_DOA` > `LUA_DAO` > `QUAY_ROI` > `XUC_PHAM`.

Đầu ra của mô hình là JSON:

```json
{"nhan": "LUA_DAO", "bang_chung": ["tài khoản của quý khách sẽ bị khóa", "xác minh tại đường link"]}
```

`bang_chung` là các cụm chép nguyên văn từ tin nhắn, được phép rỗng. Phân loại và tô sáng được đo và đặt ngưỡng riêng.

## Định dạng dữ liệu v2 (`data/v2/*.jsonl`, mỗi dòng JSONL một mẫu)

```json
{"id": "v2_000123", "text": "...", "nhan": "DE_DOA", "bang_chung": ["..."],
 "nguon": "vihsd|vihos|vithsd|thu_thap|nguoi_viet|sinh_llm", "nguon_id": "...",
 "cach_gan": "nguoi|llm|llm_nguoi_duyet", "the": ["tieng_long", "vung_mien", "an_y", "trich_dan"],
 "so_dong": 3,
 "bien_dang": "C0|C1|C2", "nhom_trung": "...", "split": "train|validation|test"}
```

Mỗi mẫu là một **đoạn hội thoại** hai người, 1–8 dòng ngăn bằng `"\n"`. Mỗi dòng là **toàn bộ lượt nói của một
người**: các tin nhắn liên tiếp của cùng người được gộp vào một dòng (nối bằng dấu cách), nên các dòng luôn luân
phiên hai người. **Không** có nhãn người nói trong `text`. Ứng dụng gửi vào mô hình đúng dạng này (cửa sổ trượt tối
đa 8 dòng gần nhất). Câu một dòng của bộ cũ (ViHSD, ViHOS) là hội thoại 1 dòng. Nhãn gán cho cả đoạn theo tình
huống nặng nhất; mỗi cụm `bang_chung` nằm gọn trong một dòng. Sinh biến dạng (C1/C2) áp dụng theo từng dòng, giữ
nguyên `"\n"`.

## Quy tắc bắt buộc

- **Không sửa gì trong `results/step3`, `results/step4`, `data/data_final`, `models/`** của v1. v1 là một dòng baseline.
- **Test chỉ chứa câu thật do người duyệt hoặc câu do người trong nhóm tự viết** (`cach_gan` là `nguoi` hoặc
  `llm_nguoi_duyet`; câu người viết có `nguon="nguoi_viet"`, `cach_gan="nguoi"`, xem `docs/v2/HUONG_DAN_VIET_TEST.md`).
  Câu do LLM sinh (`nguon="sinh_llm"`) chỉ được vào train. Câu người viết phải được viết **trước** khi đọc câu LLM sinh,
  để không bắt chước; câu sinh gần trùng câu người viết bị loại khỏi train.
- **Kho test (`data/v2/interim/kho_test.jsonl`) không bao giờ gửi cho LLM**, kể cả để chạy thử. Kho này gồm các
  nhóm câu chưa từng nằm trong split train của ViHSD/ViHOS (dữ liệu ViHateT5 của v1 đã học) và chưa từng gửi LLM;
  tạo bằng `python -m src.v2.data.chon_gan_lai`. `prelabel.py` tự bỏ câu thuộc kho này. Quy trình chạy LLM:
  `docs/v2/CHAY_SINH_DU_LIEU.md`.
- **Không dùng nhãn của VOZ-HSD làm nhãn train.** Đó là nhãn do máy gán; tác giả không khuyến nghị dùng để fine-tune
  (xem `data/processed/vietnamese_nonstandard_v1_2/DATASET_CARD.md`). Chỉ dùng làm kho câu thô để LLM và người gán lại.
- **Khử trùng gần đúng trước khi chia tập**, dùng lại `src/dataset_builder/dedup.py`; cả nhóm trùng vào cùng một split.
- **Không đưa câu biến dạng nhân tạo (C1/C2) vào train** (quyết định 20/11), để đo đúng tác dụng của bộ chuẩn hóa.
  C1/C2 chỉ sinh cho validation và test; C2 (điểm mù) là cột "kiểu né lọc chưa thấy" cho yêu cầu 2.
- **Ngưỡng dò trên validation rồi đóng băng** trước khi chạm vào test. Không chỉnh gì sau khi đã nhìn điểm test.
- **Mọi con số trong báo cáo phải sinh từ file kết quả** (`results/v2/...`) bằng script, không gõ tay.
  v1 từng có số trong `docs/` lệch với `results/`; không lặp lại lỗi này.
- **Seed cố định (42)**, ghi phần cứng, lệnh chạy và phiên bản thư viện cho mỗi lượt train và đo.
- **Che thông tin cá nhân** (số điện thoại, số tài khoản, tên thật) ngay khi thu thập tin nhắn thật. Không commit bản gốc.
- **Khóa API chỉ đặt trong biến môi trường** (`.env` đã có trong `.gitignore`), không bao giờ commit.

## Bố cục mã cho v2 (thêm mới, không đụng mã v1 trừ khi cần sửa lỗi)

```
src/v2/
  labeling/   # gọi LLM gán sơ bộ, sinh câu theo kịch bản, đo đồng thuận
  data/       # gộp nguồn về định dạng chung, khử trùng, chia tập, sinh biến dạng giữ vị trí bằng chứng
  train/      # QLoRA cho mô hình nhỏ, encoder đa nhiệm
  eval/       # F1 theo nhãn và theo nhóm, precision/recall bằng chứng, McNemar
  bench/      # độ trễ, RAM, dung lượng, năng lượng; ONNX/GGUF
data/v2/      # dữ liệu v2 + CHECKSUMS
results/v2/   # mỗi cấu hình một thư mục: config.yaml, metrics.json, predictions.jsonl, env.txt
docs/v2/      # kế hoạch, hướng dẫn gán nhãn, báo cáo
```

Tái sử dụng từ v1: `src/normalization/` (chuẩn hóa), `src/variant_generator/` (sinh biến dạng),
`src/dataset_builder/dedup.py` (khử trùng), `src/baselines/stats.py` (kiểm định), `src/reporting/` (vẽ hình), `scripts/checksums.py`.

## Lệnh thường dùng

```bash
pip install -r requirements-lock.txt
pytest -q                      # phải pass trước khi commit
python -m scripts.checksums     # kiểm tra toàn vẹn dữ liệu v1
```

## Cách làm việc

- Báo cáo, tài liệu và comment viết bằng tiếng Việt; tên biến, tên hàm bằng tiếng Anh hoặc không dấu.
- Mỗi việc trong `docs/v2/KE_HOACH.md` là một commit hoặc PR nhỏ. Cập nhật ô trạng thái trong kế hoạch khi xong.
- Việc cần GPU (train, ablation) viết sao cho chạy được trên Kaggle hoặc Colab: một lệnh, đọc dữ liệu từ `data/v2`,
  ghi kết quả vào `results/v2/<ten_cau_hinh>/`.
- Khi không chắc một quyết định có làm thay đổi kết quả nộp hay không, hỏi trước.
