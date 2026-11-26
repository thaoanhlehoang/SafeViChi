# Chạy tạo dữ liệu train v2 bằng LLM

Hướng dẫn cho người chạy lại trên máy khác. Mọi lệnh chạy ở thư mục gốc repo. Quyết định và lý do nằm trong
`docs/v2/KE_HOACH.md` (mục "Quyết định 19–20/11"); kịch bản sinh trong `docs/v2/KICH_BAN.md`.

## Tóm tắt quy trình

| Bước | Lệnh | Gọi API? | Đầu ra chính |
| --- | --- | --- | --- |
| 1. Tải dữ liệu gốc | `python -m src.v2.data.download` | Không (cần `HF_TOKEN`) | `data/raw/vihsd/`, `data/raw/vithsd/`, `data/raw/NGUON_TAI.json` |
| 2. Gộp 3 bộ | `python -m src.v2.data.merge` | Không | `data/v2/interim/merged.jsonl` (30.580 câu khác nhau) |
| 3. Ứng viên từ khóa | `python -m src.v2.data.mine_candidates` | Không | `data/v2/interim/ung_vien.jsonl` |
| 4. Tách kho test, chọn câu cần gán | `python -m src.v2.data.chon_gan_lai` | Không | `kho_test.jsonl` (6.724), `can_gan_lai.jsonl` (10.062) |
| 5. Gán lại nhãn | `python -m src.v2.labeling.prelabel --batch-size 20 --workers 8` | Có | `data/v2/interim/prelabel/prelabel_*.jsonl` |
| 6. Sinh hội thoại | `python -m src.v2.labeling.generate --workers 8` | Có | `data/v2/interim/sinh_llm.jsonl` (~10.720 đoạn) |
| 7. Lọc lại đoạn sinh | `python -m src.v2.labeling.loc_sinh --batch-size 20 --workers 8` | Có | `sinh_llm_loc.jsonl`, `sinh_llm_bi_loai.jsonl` |
| 8. Tổng hợp chi phí | `python -m src.v2.labeling.chi_phi --bat-dau "YYYY-MM-DD HH:MM"` | Không | `results/v2/chi_phi_du_lieu/chi_phi.json` |

Lượt chạy ngày 20/11/2026 (DeepSeek V4.1 Flash, `wafer/us`, 8 luồng): ước tính khoảng 1–1,5 giờ và khoảng
80.000 đ cho bước 5–7. Số thật ghi ở `results/v2/chi_phi_du_lieu/chi_phi.json`.

## 1. Chuẩn bị

```bash
git fetch origin && git checkout v2-final && git pull origin v2-final
pip install -r requirements-lock.txt
cp .env.example .env
```

Điền `.env` (**không bao giờ commit**):

```env
SAFEVICHI_LLM_PROVIDER=openai                         # OpenRouter dùng API tương thích OpenAI
SAFEVICHI_LLM_MODEL=deepseek/deepseek-v4.1-flash-1017
SAFEVICHI_LLM_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=sk-or-v1-...                           # khóa OpenRouter
SAFEVICHI_LLM_REASONING=none                          # BẮT BUỘC: model này bật thinking mặc định
SAFEVICHI_LLM_PROVIDER_ORDER=wafer/us                 # đúng endpoint Wafer 🇺🇸 0,036/0,50 USD
SAFEVICHI_LLM_MAX_PRICE=0.04,0.5                      # chặn endpoint đắt hơn (vd. wafer/mxfp4 0,066)
HF_TOKEN=hf_...                                       # chỉ cần cho bước 1
```

Bước 1 cần chấp nhận điều khoản trên trang Hugging Face của `uitnlp/vihsd` và `sonlam1102/vithsd` trước.

## 2. Thử nhỏ trước khi chạy thật

```bash
python -m src.v2.labeling.prelabel --dry-run --limit 3           # xem prompt, không gọi API
python -m src.v2.labeling.prelabel --limit 20 --batch-size 20    # khoảng 20 đồng
python -m src.v2.labeling.generate --dry-run                     # xem kế hoạch: số ô, số đoạn từng nhãn
python -m src.v2.labeling.generate --kich-ban XP1 --limit-o 2     # 1 ô và ô sinh đôi của nó
```

Dòng cuối mỗi lệnh in số token. **Số token suy luận phải bằng 0**; khác 0 nghĩa là thinking chưa tắt, dừng lại kiểm
tra `.env`. Đọc kỹ vài đoạn trong `data/v2/interim/sinh_llm.jsonl` trước khi chạy hết.

## 3. Chạy thật và lưu log

Ghi log từng bước vào `logs_v2/` (đã ignore) để bước 8 đọc lại số token:

```bash
mkdir -p logs_v2
python -m src.v2.labeling.prelabel --batch-size 20 --workers 8 > logs_v2/1_prelabel.log 2>&1
python -m src.v2.labeling.generate --workers 8 > logs_v2/2_generate.log 2>&1
python -m src.v2.labeling.loc_sinh --batch-size 20 --workers 8 > logs_v2/3_loc_sinh.log 2>&1
python -m src.v2.labeling.chi_phi --bat-dau "2026-11-20 00:30"
```

Nếu phải chạy lại một bước, ghi vào log **mới** (ví dụ `1_prelabel_lan2.log`) để không mất số token của lần trước;
`chi_phi.py` cộng mọi log khớp `1_prelabel*.log`.

## 4. Bị ngắt giữa chừng

Chạy lại **đúng lệnh cũ**. Cả ba script ghi cache sau mỗi lô hoặc mỗi ô và bỏ qua phần đã xong. Lô lỗi (mạng, hết
tiền, nhà cung cấp ngừng) chỉ bị bỏ qua lượt đó và được báo trong log. Đổi model hoặc sửa prompt tạo cache mới (tên
file có mã phiên bản prompt).

## 5. Quy tắc dữ liệu cần nhớ

- **Kho test không bao giờ gửi LLM.** `prelabel.py` tự bỏ mọi câu có trong `kho_test.jsonl`. Câu test/validation do
  người gán (xem `HUONG_DAN_VIET_TEST.md` và `HUONG_DAN_GAN_NHAN.md`).
- Câu do LLM sinh chỉ vào train (`nguon=sinh_llm`).
- Sau khi có câu test người viết (`data/v2/test_day_du.jsonl`), chạy lại bước 6 và 7: không tốn tiền nhờ
  cache, và loại đoạn sinh gần trùng câu test.

## 6. Commit gì

| File | Commit? |
| --- | --- |
| `.env`, `logs_v2/` | **Không** |
| `data/raw/vihsd/`, `data/raw/vithsd/` (CSV gốc, bộ gated) | Không (đã ignore); chỉ commit `data/raw/NGUON_TAI.json` |
| `data/v2/interim/sinh_llm_cache/`, `data/v2/interim/prelabel/cache_*.jsonl` | Không (đã ignore); giữ trên máy để chạy tiếp |
| `merged.jsonl`, `kho_test.jsonl`, `can_gan_lai.jsonl`, kết quả gán và sinh, file thống kê | Có |
| `results/v2/chi_phi_du_lieu/`, `results/v2/thu_model/` | Có |

Trước khi `git add`, chạy `git status` để chắc không có `.env`, log hay cache.
