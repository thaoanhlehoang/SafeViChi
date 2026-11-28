-- Bổ sung cho phần kiểm duyệt (28/11/2026). Không sửa bảng của 0001, chỉ thêm cột và bảng mới.

-- Thời điểm và ghi chú ngắn của từng lượt duyệt. API điền `luc` khi lưu (SQLite không cho
-- ADD COLUMN có mặc định không cố định); lượt duyệt cũ có `luc` NULL, giờ thật nằm trong nhat_ky.
ALTER TABLE luot_duyet ADD COLUMN luc TEXT;
ALTER TABLE luot_duyet ADD COLUMN ghi_chu TEXT CHECK (ghi_chu IS NULL OR length(ghi_chu) <= 200);

-- Quản trị viên chốt mẫu "bất đồng" bằng cách chọn một trong các lượt duyệt đã có.
-- Kết quả cuối vì vậy luôn được ít nhất hai người đồng ý (người duyệt đó và quản trị viên),
-- giữ đúng luật hai người duyệt cùng nhãn cho Đe dọa và Xúc phạm nhắm nhóm.
CREATE TABLE IF NOT EXISTS chot_bat_dong (
  mau_id TEXT PRIMARY KEY REFERENCES phan_hoi(id),
  luot_duyet_id TEXT NOT NULL REFERENCES luot_duyet(id),
  nguoi TEXT NOT NULL,
  ly_do TEXT NOT NULL CHECK (length(trim(ly_do)) BETWEEN 1 AND 200),
  luc TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TRIGGER IF NOT EXISTS chot_bat_dong_da_luu AFTER INSERT ON chot_bat_dong
BEGIN
  UPDATE phan_hoi SET trang_thai = 'da_duyet', quyet_dinh_id = NEW.luot_duyet_id
    WHERE id = NEW.mau_id AND trang_thai = 'bat_dong';
  INSERT INTO nhat_ky (nguoi, hanh_dong, doi_tuong, chi_tiet)
    VALUES (NEW.nguoi, 'chot_bat_dong', NEW.mau_id, NEW.ly_do);
END;
