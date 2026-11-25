// Đăng nhập bằng Firebase Authentication (email/mật khẩu).
// Đặt cấu hình web của Firebase trong src/firebase-config.js (được phép công khai;
// quyền truy cập dữ liệu được kiểm soát bằng Firebase Security Rules, không phải bằng khóa này).
import { firebaseConfig } from './firebase-config.js';

let authPromise = null;

export const isConfigured = () => Boolean(firebaseConfig?.apiKey);

async function getAuth() {
  // TODO: khi có firebaseConfig, cài `firebase` và khởi tạo ở đây (tải lười để trang chủ nhẹ).
  throw Object.assign(new Error('Firebase chưa được cấu hình.'), { code: 'not-configured' });
}

export async function signIn(email, password) {
  authPromise ||= getAuth();
  return (await authPromise).signIn(email, password);
}

export async function signUp(email, password, displayName) {
  authPromise ||= getAuth();
  return (await authPromise).signUp(email, password, displayName);
}
