-- Cho phép áp dụng lại cùng lược đồ trên D1 cục bộ đã dùng tên tệp cũ.
CREATE TABLE IF NOT EXISTS phan_hoi (
  id TEXT PRIMARY KEY,
  loai TEXT NOT NULL CHECK (loai IN ('muc_1', 'muc_2')),
  ngay TEXT NOT NULL,
  du_lieu TEXT NOT NULL CHECK (json_valid(du_lieu)),
  trang_thai TEXT CHECK (trang_thai IN ('cho_duyet', 'cho_nguoi_thu_hai', 'bat_dong', 'da_duyet')),
  quyet_dinh_id TEXT,
  CHECK (
    (loai = 'muc_1' AND trang_thai IS NULL AND json_type(du_lieu, '$.doan_da_che') IS NULL)
    OR (loai = 'muc_2' AND trang_thai IS NOT NULL AND json_type(du_lieu, '$.doan_da_che') = 'text')
  )
);

CREATE INDEX IF NOT EXISTS phan_hoi_thong_ke ON phan_hoi (loai, ngay);
CREATE INDEX IF NOT EXISTS phan_hoi_hang_doi ON phan_hoi (loai, trang_thai, ngay, id);

CREATE TABLE IF NOT EXISTS gioi_han (
  ngay TEXT NOT NULL,
  ma TEXT NOT NULL,
  loai TEXT NOT NULL,
  so_lan INTEGER NOT NULL,
  PRIMARY KEY (ngay, ma, loai)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS nhat_ky (
  id INTEGER PRIMARY KEY,
  luc TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  nguoi TEXT NOT NULL,
  hanh_dong TEXT NOT NULL,
  doi_tuong TEXT NOT NULL,
  chi_tiet TEXT
);
CREATE INDEX IF NOT EXISTS nhat_ky_doi_tuong ON nhat_ky (doi_tuong, id);

CREATE TABLE IF NOT EXISTS luot_duyet (
  id TEXT PRIMARY KEY,
  mau_id TEXT NOT NULL REFERENCES phan_hoi(id),
  nguoi TEXT NOT NULL,
  nhan TEXT NOT NULL CHECK (nhan IN ('AN_TOAN', 'XUC_PHAM', 'QUAY_ROI', 'DE_DOA', 'LUA_DAO')),
  nham_nhom INTEGER NOT NULL CHECK (nham_nhom IN (0, 1)),
  can_duyet_hai INTEGER NOT NULL CHECK (can_duyet_hai IN (0, 1)),
  bang_chung TEXT NOT NULL CHECK (json_valid(bang_chung)),
  UNIQUE (mau_id, nguoi)
);

CREATE TRIGGER IF NOT EXISTS luot_duyet_da_luu AFTER INSERT ON luot_duyet
BEGIN
  UPDATE phan_hoi SET quyet_dinh_id = NEW.id, trang_thai = 'da_duyet'
    WHERE id = NEW.mau_id AND (
      NEW.can_duyet_hai = 0 OR EXISTS (
        SELECT 1 FROM luot_duyet WHERE mau_id = NEW.mau_id AND nguoi <> NEW.nguoi
          AND nhan = NEW.nhan AND nham_nhom = NEW.nham_nhom
      )
    );
  UPDATE phan_hoi SET trang_thai = CASE
    WHEN (SELECT count(*) FROM luot_duyet WHERE mau_id = NEW.mau_id) > 1 THEN 'bat_dong'
    ELSE 'cho_nguoi_thu_hai' END
    WHERE id = NEW.mau_id AND trang_thai <> 'da_duyet';
  INSERT INTO nhat_ky (nguoi, hanh_dong, doi_tuong, chi_tiet)
    VALUES (NEW.nguoi, 'luu_luot_duyet', NEW.mau_id, NEW.nhan);
END;

CREATE TABLE IF NOT EXISTS de_xuat (
  id TEXT PRIMARY KEY,
  phien_ban TEXT NOT NULL,
  phien_ban_mo_hinh TEXT NOT NULL,
  nguoi_de_xuat TEXT NOT NULL,
  du_lieu TEXT NOT NULL CHECK (json_valid(du_lieu)),
  nguoi_phe_duyet TEXT,
  ly_do_phe_duyet TEXT,
  UNIQUE (phien_ban_mo_hinh, phien_ban)
);

CREATE TABLE IF NOT EXISTS nguong_hieu_luc (
  phien_ban_mo_hinh TEXT PRIMARY KEY,
  de_xuat_id TEXT NOT NULL REFERENCES de_xuat(id)
);

CREATE TRIGGER IF NOT EXISTS de_xuat_da_tao AFTER INSERT ON de_xuat
BEGIN
  INSERT INTO nhat_ky (nguoi, hanh_dong, doi_tuong)
    VALUES (NEW.nguoi_de_xuat, 'tao_de_xuat', NEW.id);
END;

CREATE TRIGGER IF NOT EXISTS de_xuat_da_phe_duyet AFTER UPDATE OF nguoi_phe_duyet ON de_xuat
  WHEN OLD.nguoi_phe_duyet IS NULL AND NEW.nguoi_phe_duyet IS NOT NULL
BEGIN
  INSERT INTO nguong_hieu_luc (phien_ban_mo_hinh, de_xuat_id) VALUES (NEW.phien_ban_mo_hinh, NEW.id)
    ON CONFLICT (phien_ban_mo_hinh) DO UPDATE SET de_xuat_id = excluded.de_xuat_id;
  INSERT INTO nhat_ky (nguoi, hanh_dong, doi_tuong, chi_tiet)
    VALUES (NEW.nguoi_phe_duyet, 'phe_duyet_nguong', NEW.id, NEW.ly_do_phe_duyet);
END;
