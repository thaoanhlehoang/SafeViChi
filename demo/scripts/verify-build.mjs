import { readFile, readdir, stat } from 'node:fs/promises';
import { extname, join, relative, resolve } from 'node:path';

const DIST = resolve('dist');
const MAX_PAGES_ASSET_BYTES = 25 * 1024 * 1024;
const TEXT_EXTENSIONS = new Set(['.css', '.html', '.js', '.json', '.map', '']);
const FORBIDDEN_REMOTE_SOURCES = ['cdn.jsdelivr.net', 'fonts.googleapis.com', 'fonts.gstatic.com'];

async function walk(directory) {
  const output = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) output.push(...await walk(path));
    else output.push(path);
  }
  return output;
}

const files = await walk(DIST);
const errors = [];
for (const file of files) {
  const name = relative(DIST, file).replaceAll('\\', '/');
  const info = await stat(file);
  if (info.size > MAX_PAGES_ASSET_BYTES) errors.push(`${name} exceeds 25 MiB.`);
  if (name.endsWith('.onnx')) errors.push(`${name} must be served from R2, not Pages.`);
  if (name === 'qa' || name.startsWith('qa/')) errors.push(`${name} is a QA artifact.`);

  if (TEXT_EXTENSIONS.has(extname(file))) {
    const content = await readFile(file, 'utf8');
    for (const source of FORBIDDEN_REMOTE_SOURCES) {
      if (content.includes(source)) errors.push(`${name} still references ${source}.`);
    }
  }
}

for (const required of ['_headers', 'index.html', 'model-manifest.json', 'onnx_model/tokenizer.json']) {
  if (!files.some((file) => relative(DIST, file).replaceAll('\\', '/') === required)) {
    errors.push(`Missing required deployment artifact: ${required}.`);
  }
}

const deploymentHeaders = await readFile(join(DIST, '_headers'), 'utf8');
for (const fragment of [
  "script-src 'self' 'wasm-unsafe-eval'",
  "style-src 'self'",
  'Cross-Origin-Opener-Policy: same-origin',
  'Cross-Origin-Embedder-Policy: require-corp',
  'X-Content-Type-Options: nosniff',
  'Referrer-Policy: no-referrer',
  'Permissions-Policy:',
  'Cache-Control: public, max-age=31536000, immutable',
  'Cache-Control: no-cache, must-revalidate',
]) {
  if (!deploymentHeaders.includes(fragment)) errors.push(`_headers is missing: ${fragment}`);
}
if (deploymentHeaders.includes("'unsafe-inline'") || deploymentHeaders.includes("'unsafe-eval'")) {
  errors.push('_headers weakens the production CSP with a general unsafe source.');
}

const configuredOrigin = String(process.env.VITE_MODEL_ORIGIN || '').trim().replace(/\/+$/, '');
if (configuredOrigin && !deploymentHeaders.includes(new URL(configuredOrigin).origin)) {
  errors.push('_headers does not allow the configured model origin.');
}

if (errors.length) {
  console.error(errors.map((error) => `- ${error}`).join('\n'));
  process.exitCode = 1;
} else {
  console.log(`Verified ${files.length} Pages assets; no file exceeds 25 MiB.`);
}
