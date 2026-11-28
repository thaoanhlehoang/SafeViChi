"""Nghiệm thu quyền riêng tư cho DB phản hồi v3 (SQLite demo).

Kiểm tra:
1. Không bảng nào có cột nhạy cảm (mã tài khoản Firebase, SĐT, IP, tên liên hệ, nội dung tin ở mức 1).
2. Mọi chuỗi văn bản trong DB không còn PII lộ rõ (SĐT, email, số dài giống STK/CCCD).
3. Bảng mức 1 không có cột ngoài hợp đồng (nếu tìm thấy bảng phan_hoi_muc_1).

Cách chạy:  python web-v2/scripts/kiem_tra_db.py <duong_dan.sqlite>
Trả mã thoát 1 nếu có vi phạm.
"""

import re
import sqlite3
import sys

COT_CAM = re.compile(
    r"(uid|firebase|tai_khoan|account|so_dien_thoai|sdt|phone|\bip\b|ip_|_ip|dia_chi_ip|ten_lien_he|contact|email)",
    re.IGNORECASE,
)

# Trường cho phép của gói mức 1 (mục "Hợp đồng dữ liệu" trong DE_XUAT_TRIEN_KHAI_CHANG_2) + cột kỹ thuật của máy chủ.
TRUONG_MUC_1 = {
    "loai", "ma_lan_gui", "ma_bao_cao", "nhan_mo_hinh", "nhan_de_xuat", "ma_ly_do", "ma_chi_tiet",
    "ly_do_khac", "co_to_sang", "khoang_diem", "phien_ban_mo_hinh", "phien_ban_nguong",
}
COT_KY_THUAT = {"id", "ngay", "ngay_tiep_nhan", "ngay_nhan"}

PII = {
    "sdt": re.compile(r"(?:\+?84|0)(?:[\s.\-]?\d){8,10}"),
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "so_dai": re.compile(r"\b\d{9,19}\b"),
}


def main(path: str) -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # console Windows mặc định cp1252
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    loi = []
    bang = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    for b in bang:
        cot = [r[1] for r in con.execute(f'PRAGMA table_info("{b}")')]
        # nhat_ky và nguoi_dung_noi_bo là dữ liệu nội bộ (người duyệt), không phải người dùng cuối.
        if b not in ("nguoi_dung_noi_bo",):
            for c in cot:
                if COT_CAM.search(c):
                    loi.append(f"[cột cấm] {b}.{c}")
        if b == "phan_hoi_muc_1":
            for c in cot:
                if c not in TRUONG_MUC_1 | COT_KY_THUAT:
                    loi.append(f"[ngoài hợp đồng] {b}.{c}")
        if b == "nguoi_dung_noi_bo":
            continue
        for row in con.execute(f'SELECT * FROM "{b}"'):
            for c, v in zip(cot, row):
                if not isinstance(v, str):
                    continue
                for ten, rx in PII.items():
                    m = rx.search(v)
                    if m and not c.startswith(("ma_", "phien_ban")):
                        loi.append(f"[PII lộ: {ten}] {b}.{c}: …{m.group(0)}…")
    print(f"Đã kiểm tra {len(bang)} bảng: {', '.join(bang) or '(trống)'}")
    for l in loi:
        print(l)
    print("ĐẠT" if not loi else f"KHÔNG ĐẠT: {len(loi)} vi phạm")
    return 1 if loi else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
