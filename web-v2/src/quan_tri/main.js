// Trang riêng quan-tri.html: giữ làm đường dự phòng, dùng chung mã với tab trong trang chính.
import '../fonts.css';
import './quan_tri.css';
import { dungKhuVucQuanTri } from './khu_vuc.js';

dungKhuVucQuanTri(document.getElementById('qt-host')).batDau();
