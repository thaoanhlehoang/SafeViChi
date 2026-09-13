import { createReadStream } from 'node:fs';
import { copyFile, mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { defineConfig, loadEnv } from 'vite';

const TOKENIZER_FILES = [
  'config.json',
  'generation_config.json',
  'tokenizer.json',
  'tokenizer_config.json',
];

function localModelServer() {
  let manifestPromise;
  return {
    name: 'safevichi-local-model-server',
    configureServer(server) {
      server.middlewares.use(async (request, response, next) => {
        if (request.method !== 'GET' && request.method !== 'HEAD') return next();
        try {
          manifestPromise ||= readFile(resolve('model-manifest.json'), 'utf8').then(JSON.parse);
          const manifest = await manifestPromise;
          const asset = Object.values(manifest.assets).find((candidate) =>
            new URL(request.url, 'http://localhost').pathname
              === '/models/' + manifest.version + '/' + candidate.path
          );
          if (!asset) return next();

          const path = resolve('onnx_model', 'onnx', asset.path);
          const info = await stat(path);
          response.statusCode = 200;
          response.setHeader('Content-Type', 'application/octet-stream');
          response.setHeader('Content-Length', info.size);
          response.setHeader('Cache-Control', 'no-store');
          if (request.method === 'HEAD') return response.end();
          createReadStream(path).pipe(response);
        } catch (error) {
          next(error);
        }
      });
    },
  };
}

function normalizeModelOrigin(value) {
  const trimmed = value.trim().replace(/\/+$/, '');
  if (!trimmed) return '';

  const url = new URL(trimmed);
  if (url.protocol !== 'https:' && url.hostname !== '127.0.0.1' && url.hostname !== 'localhost') {
    throw new Error('VITE_MODEL_ORIGIN must use HTTPS outside local development.');
  }
  return url.origin;
}

function deploymentArtifacts(modelOrigin) {
  const connectSources = ["'self'", ...(modelOrigin ? [modelOrigin] : [])].join(' ');
  const headers = `/*
  Content-Security-Policy: default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self'; img-src 'self' data:; font-src 'self'; connect-src ${connectSources}; worker-src 'self' blob:; object-src 'none'; base-uri 'self'; frame-ancestors 'none'; form-action 'self'; upgrade-insecure-requests
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Embedder-Policy: require-corp
  X-Content-Type-Options: nosniff
  Referrer-Policy: no-referrer
  Permissions-Policy: camera=(), microphone=(), geolocation=(), payment=(), usb=(), serial=(), bluetooth=()

/
  Cache-Control: no-cache, must-revalidate

/*.html
  Cache-Control: no-cache, must-revalidate

/model-manifest.json
  Cache-Control: no-cache, must-revalidate

/onnx_model/*
  Cache-Control: public, max-age=3600, must-revalidate

/assets/*
  Cache-Control: public, max-age=31536000, immutable
`;

  return {
    name: 'safevichi-deployment-artifacts',
    async closeBundle() {
      const outputDir = resolve('dist');
      const tokenizerDir = resolve(outputDir, 'onnx_model');
      await mkdir(tokenizerDir, { recursive: true });
      await Promise.all(TOKENIZER_FILES.map((file) =>
        copyFile(resolve('onnx_model', file), resolve(tokenizerDir, file)),
      ));
      await copyFile(resolve('model-manifest.json'), resolve(outputDir, 'model-manifest.json'));
      await writeFile(resolve(outputDir, '_headers'), headers, 'utf8');
    },
  };
}

function previewSecurityHeaders(modelOrigin, allowInlineStyles = false) {
  const connectSources = ["'self'", ...(modelOrigin ? [modelOrigin] : [])].join(' ');
  const styleSources = allowInlineStyles ? "'self' 'unsafe-inline'" : "'self'";
  return {
    'Content-Security-Policy': "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src " + styleSources + "; img-src 'self' data:; font-src 'self'; connect-src " + connectSources + "; worker-src 'self' blob:; object-src 'none'; base-uri 'self'; frame-ancestors 'none'; form-action 'self'",
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Embedder-Policy': 'require-corp',
    'X-Content-Type-Options': 'nosniff',
    'Referrer-Policy': 'no-referrer',
    'Permissions-Policy': 'camera=(), microphone=(), geolocation=(), payment=(), usb=(), serial=(), bluetooth=()',
  };
}

function disableThirdPartyRuntimeFallbacks() {
  return {
    name: 'safevichi-disable-third-party-runtime-fallbacks',
    enforce: 'pre',
    transform(code, id) {
      if (!id.includes('@huggingface/transformers')) return null;
      const updated = code.replaceAll(
        'https://cdn.jsdelivr.net/npm/@huggingface/transformers@',
        '/vendor/transformers@',
      );
      return updated === code ? null : { code: updated, map: null };
    },
  };
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const modelOrigin = normalizeModelOrigin(
    process.env.VITE_MODEL_ORIGIN || env.VITE_MODEL_ORIGIN || '',
  );
  const isPagesBuild = (process.env.CF_PAGES || env.CF_PAGES) === '1';
  if (isPagesBuild && !modelOrigin) {
    throw new Error('VITE_MODEL_ORIGIN is required for Cloudflare Pages builds.');
  }

  return {
    publicDir: false,
    plugins: [localModelServer(), disableThirdPartyRuntimeFallbacks(), deploymentArtifacts(modelOrigin)],
    server: { headers: previewSecurityHeaders(modelOrigin, true) },
    preview: { headers: previewSecurityHeaders(modelOrigin) },
    build: {
      outDir: 'dist',
      emptyOutDir: true,
      assetsInlineLimit: 0,
      chunkSizeWarningLimit: 1024,
    },
    test: {
      environment: 'jsdom',
      include: ['tests/**/*.test.js'],
      restoreMocks: true,
    },
  };
});
