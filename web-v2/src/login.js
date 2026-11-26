// Form đăng nhập (chưa có đăng ký). Phần gọi Firebase nằm trong auth.js.
import { signIn, isConfigured } from './auth.js';

const MESSAGES = {
  'auth/invalid-credential': 'Email hoặc mật khẩu không đúng.',
  'auth/invalid-email': 'Email chưa đúng định dạng.',
  'auth/email-already-in-use': 'Email này đã có tài khoản. Hãy đăng nhập.',
  'auth/weak-password': 'Mật khẩu cần ít nhất 6 ký tự.',
  'auth/too-many-requests': 'Thử sai quá nhiều lần. Đợi một lúc rồi thử lại.',
  'auth/network-request-failed': 'Không kết nối được mạng.',
};

export function setupLoginForm({ onSignedIn }) {
  const form = document.getElementById('login-form');
  const msg = document.getElementById('login-msg');
  const submit = document.getElementById('login-submit');
  const pass = document.getElementById('pass');
  const eye = document.getElementById('toggle-pass');

  eye.addEventListener('click', () => {
    const show = pass.type === 'password';
    pass.type = show ? 'text' : 'password';
    eye.setAttribute('aria-pressed', String(show));
    eye.setAttribute('aria-label', show ? 'Ẩn mật khẩu' : 'Hiện mật khẩu');
  });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    msg.textContent = '';
    const email = form.email.value.trim();
    const password = form.password.value;
    if (!email || !password) {
      msg.textContent = 'Nhập tài khoản và mật khẩu.';
      return;
    }
    if (!isConfigured()) {
      // TẠM THỜI: chưa nối Firebase nên cho vào thẳng để thiết kế màn chat. Xóa khi có tài khoản thật.
      form.reset();
      onSignedIn();
      return;
    }
    submit.disabled = true;
    try {
      await signIn(email, password);
      form.reset();
      onSignedIn();
    } catch (err) {
      msg.textContent = MESSAGES[err?.code] || 'Có lỗi, thử lại sau.';
    } finally {
      submit.disabled = false;
    }
  });
}
