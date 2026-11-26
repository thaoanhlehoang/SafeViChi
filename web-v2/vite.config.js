import { createReadStream } from 'node:fs';
import { stat, writeFile } from 'node:fs/promises';
import { dirname, extname, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig, loadEnv } from 'vite';

const CONFIG_DIR = dirname(fileURLToPath(import.meta.url));
const DEFAULT_ONNX_DIR = resolve(CONFIG_DIR, '..', 'models', 'dualorc_mmbert_base', 'onnx');
const DEFAULT_HF_CONNECT = ['https://huggingface.co', 'https://*.hf.co'];

const MODEL_MIME = {
  '.onnx': 'application/octet-stream',
  '.json': 'application/json; charset=utf-8',
};

const BASE_SECURITY_HEADERS = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
  'Cross-Origin-Resource-Policy': 'same-origin',
  'X-Content-Type-Options': 'nosniff',
  'X-Frame-Options': 'DENY',
  'Referrer-Policy': 'no-referrer',
  'Permissions-Policy': 'camera=(), microphone=(), geolocation=(), payment=()',
};

function resolveLocalOnnxDir(env) {
  const configured = (process.env.SAFEVICHI_ONNX_DIR || env.SAFEVICHI_ONNX_DIR || '').trim();
  return configured ? resolve(configured) : DEFAULT_ONNX_DIR;
}

function modelOriginFromUrl(raw) {
  const trimmed = String(raw || '').trim();
  if (!trimmed || trimmed.startsWith('/')) return '';
  const url = new URL(trimmed);
  const isLocal = url.hostname === '127.0.0.1' || url.hostname === 'localhost';
  if (url.protocol !== 'https:' && !isLocal) {
    throw new Error('VITE_MODEL_BASE_URL phải dùng HTTPS khi triển khai ngoài localhost.');
  }
  return url.origin;
}

/** Phục vụ trực tiếp gói ONNX cục bộ tại `/model/*` trong dev & preview mà không sao chép vào public/. */
function localModelServer(onnxDir) {
  const rootPrefix = onnxDir.endsWith(sep) ? onnxDir : onnxDir + sep;

  const middleware = async (req, res, next) => {
    if (req.method !== 'GET' && req.method !== 'HEAD') return next();
    const pathname = new URL(req.url || '/', 'http://localhost').pathname;
    if (!pathname.startsWith('/model/')) return next();

    const rel = decodeURIComponent(pathname.slice('/model/'.length));
    if (!rel || rel.includes('..') || rel.includes('\\') || rel.includes('\0')) {
      res.statusCode = 400;
      return res.end('Invalid model path');
    }

    const filePath = resolve(onnxDir, rel);
    if (!filePath.startsWith(rootPrefix)) {
      res.statusCode = 403;
      return res.end('Forbidden');
    }

    const mime = MODEL_MIME[extname(filePath).toLowerCase()];
    if (!mime) return next();

    try {
      const info = await stat(filePath);
      if (!info.isFile()) return next();
      res.statusCode = 200;
      res.setHeader('Content-Type', mime);
      res.setHeader('Content-Length', String(info.size));
      res.setHeader('Cache-Control', 'no-store');
      res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
      res.setHeader('X-Content-Type-Options', 'nosniff');
      if (req.method === 'HEAD') return res.end();
      createReadStream(filePath).pipe(res);
    } catch {
      res.statusCode = 404;
      res.setHeader('Content-Type', 'text/plain; charset=utf-8');
      res.end('Model file not found locally');
    }
  };

  return {
    name: 'safevichi-v2-local-model-server',
    configureServer(server) {
      server.middlewares.use(middleware);
    },
    configurePreviewServer(server) {
      server.middlewares.use(middleware);
    },
  };
}

/** Sinh `dist/_headers` cho Cloudflare Pages với CSP connect-src khớp nguồn mô hình. */
function buildHeadersPlugin(customOrigin) {
  const connectSources = ["'self'", ...DEFAULT_HF_CONNECT, ...(customOrigin ? [customOrigin] : [])];
  const uniqueConnect = [...new Set(connectSources)].join(' ');
  const headersText = `/*
  Content-Security-Policy: default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self'; img-src 'self' data:; font-src 'self'; connect-src ${uniqueConnect}; worker-src 'self' blob:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Embedder-Policy: require-corp
  Cross-Origin-Resource-Policy: same-origin
  X-Frame-Options: DENY
  X-Content-Type-Options: nosniff
  Referrer-Policy: no-referrer
  Permissions-Policy: camera=(), microphone=(), geolocation=(), payment=()
`;

  return {
    name: 'safevichi-v2-headers',
    async closeBundle() {
      await writeFile(resolve(CONFIG_DIR, 'dist', '_headers'), headersText, 'utf8');
    },
  };
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, CONFIG_DIR, '');
  const onnxDir = resolveLocalOnnxDir(env);
  const customOrigin = modelOriginFromUrl(process.env.VITE_MODEL_BASE_URL || env.VITE_MODEL_BASE_URL || '');

  return {
    plugins: [localModelServer(onnxDir), buildHeadersPlugin(customOrigin)],
    optimizeDeps: {
      exclude: ['onnxruntime-web'],
    },
    server: { headers: BASE_SECURITY_HEADERS },
    preview: { headers: BASE_SECURITY_HEADERS },
    worker: { format: 'es' },
    build: {
      assetsInlineLimit: 0,
    },
  };
});
