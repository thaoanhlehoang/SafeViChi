import { defineConfig } from 'vite';

// Header bảo mật dùng khi chạy thử; khi deploy Cloudflare Pages dùng public/_headers.
const securityHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
  'X-Content-Type-Options': 'nosniff',
  'Referrer-Policy': 'no-referrer',
};

export default defineConfig({
  server: { headers: securityHeaders },
  preview: { headers: securityHeaders },
  worker: { format: 'es' },
});
