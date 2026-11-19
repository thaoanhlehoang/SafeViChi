# Chạy sinh dữ liệu và gán nhãn sơ bộ bằng LLM (v2)

Hướng dẫn cho người chạy trên máy khác. Mọi lệnh chạy ở thư mục gốc repo. Chi phí API tính vào khóa đặt trong `.env`.

## 1. Chuẩn bị

```bash
git fetch origin
git checkout v2-final
git pull origin v2-final
pip install -r requirements-lock.txt
```

`data/v2/interim/merged.jsonl` đã có trong repo. Nếu thiếu, chạy `git lfs pull` (dữ liệu v1_2 dùng Git LFS, khoảng
400 MB) rồi `python -m src.v2.data.merge`.

Tạo file `.env` từ mẫu rồi điền nhà cung cấp, model và khóa:

```bash
cp .env.example .env
```

```
SAFEVICHI_LLM_PROVIDER=anthropic        # hoặc openai, gemini
SAFEVICHI_LLM_MODEL=<tên model>
ANTHROPIC_API_KEY=<khóa>                # hoặc OPENAI_API_KEY / GEMINI_API_KEY
```

**Không bao giờ commit `.env`.**

## 2. Sinh câu train (generate.py)

Bảng kịch bản và dạng hội thoại: `docs/v2/KICH_BAN.md`. Mỗi **ô** là một lần gọi LLM, sinh 5 đoạn hội thoại cho
một tổ hợp kịch bản × kênh × giọng × độ lộ × kiểu lộ. Mỗi ô nguy hại có một ô sinh đôi `AN_TOAN`. Tổng 800 ô,
khoảng 4.000 đoạn (LUA_DAO 800, DE_DOA 600, QUAY_ROI 600, AN_TOAN 2.000; chỉnh ở `muc_tieu_doan` trong
`src/v2/labeling/kich_ban.yaml`).

```bash
# (a) Xem prompt, không gọi API, không tốn tiền
python -m src.v2.labeling.generate --dry-run

# (b) Thử 2 ô: một ô LD1 và ô sinh đôi an toàn của nó (khoảng 10 đoạn hội thoại)
python -m src.v2.labeling.generate --kich-ban LD1 --limit-o 2

# (c) Chạy hết, 4 lượt gọi song song
python -m src.v2.labeling.generate --workers 4
```

Sau bước (b), **mở `data/v2/interim/sinh_llm.jsonl` đọc kỹ khoảng 10 đoạn** xem có tự nhiên, các dòng có luân
phiên hai người, đúng nhãn, đúng bằng chứng không, rồi mới chạy (c). Thống kê nằm ở `data/v2/interim/sinh_llm_stats.json`.

Ô bị LLM từ chối (hay gặp ở QR1, DD3) được ghi log và liệt kê ở mục `o_bi_tu_choi` trong file thống kê. Không cần
sửa prompt để lách. Ô lỗi mạng hoặc API (`o_loi_api`) sẽ được thử lại ở lần chạy sau.

## 3. Gán nhãn sơ bộ câu thật (prelabel.py)

Gán 5 nhãn mới cho câu ViHSD và ViHOS. Câu trùng text chỉ gọi API một lần.

```bash
# Thử 20 câu ViHOS (có cả CLEAN và HATE), xem kết quả trước
python -m src.v2.labeling.prelabel --nguon vihos --limit 20

# Chạy thật, khuyên dùng: giữ mọi câu OFFENSIVE/HATE, chỉ lấy mẫu 4.000 câu CLEAN (seed 42)
python -m src.v2.labeling.prelabel --mau-clean 4000
```

Với `--mau-clean 4000` có khoảng 9.450 text khác nhau, khoảng 950 lượt gọi (10 câu mỗi lượt). Không có cờ này thì
gán toàn bộ 24.019 text, khoảng 2.400 lượt. Lệnh in số text và số lượt gọi trước khi chạy; thêm `--dry-run` để chỉ
xem con số mà không gọi API.

Kết quả nằm trong `data/v2/interim/prelabel/`:

- `prelabel_<provider>_<model>_<version>.jsonl`: mỗi câu có `nhan`, `bang_chung`, `llm_do_chac`, `llm_ly_do`,
  `can_duyet` và `ly_do_duyet` (độ chắc dưới 0,7, lệch nhãn gốc, lỗi đầu ra...).
- `prelabel_..._stats.json`: phân bố nhãn, số câu cần người duyệt.

## 4. Bị ngắt giữa chừng

Chạy lại **đúng lệnh cũ**. Cả hai script ghi cache sau mỗi lô và bỏ qua phần đã xong, nên không gọi API lại câu cũ.
Đổi model hoặc sửa prompt sẽ tạo cache mới (tên file có mã phiên bản prompt).

## 5. Gửi lại kết quả

| File | Commit? |
| --- | --- |
| `.env` | **Không bao giờ** |
| `data/v2/interim/sinh_llm_cache/`, `data/v2/interim/prelabel/cache_*.jsonl` | Không (đã ignore). Giữ trên máy để chạy tiếp |
| `data/v2/interim/sinh_llm.jsonl`, `sinh_llm_stats.json` | Gửi qua Drive; commit hay không do nhóm chốt |
| `data/v2/interim/prelabel/prelabel_*.jsonl`, `*_stats.json` | Gửi qua Drive; commit hay không do nhóm chốt |

Trước khi `git add`, chạy `git status` để chắc không có `.env` hay cache.
