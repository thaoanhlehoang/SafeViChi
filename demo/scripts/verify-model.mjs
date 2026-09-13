import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { readFile, stat } from 'node:fs/promises';
import { resolve } from 'node:path';

const manifest = JSON.parse(await readFile(resolve('model-manifest.json'), 'utf8'));
const modelOrigin = String(process.env.MODEL_ORIGIN || '').replace(/\/+$/, '');
const verifyRemote = process.argv.includes('--remote');

async function digest(path) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}

for (const [name, asset] of Object.entries(manifest.assets)) {
  const localPath = resolve('onnx_model', 'onnx', asset.path);
  const localStat = await stat(localPath);
  if (localStat.size !== asset.bytes) throw new Error(`${name} byte size does not match the manifest.`);
  if (await digest(localPath) !== asset.sha256) throw new Error(`${name} SHA-256 does not match the manifest.`);

  if (verifyRemote) {
    if (!modelOrigin) throw new Error('MODEL_ORIGIN is required with --remote.');
    const url = `${modelOrigin}/models/${manifest.version}/${asset.path}`;
    const requestHeaders = { Origin: 'https://safevichi-pages-verifier.invalid' };
    const response = await fetch(url, { method: 'HEAD', headers: requestHeaders });
    if (!response.ok) throw new Error(`${url} returned ${response.status}.`);
    if (Number(response.headers.get('content-length')) !== asset.bytes) {
      throw new Error(`${url} content-length does not match the manifest.`);
    }
    if (response.headers.get('access-control-allow-origin') !== '*') {
      throw new Error(`${url} is missing Access-Control-Allow-Origin: *.`);
    }
    if (!response.headers.get('cache-control')?.includes('immutable')) {
      throw new Error(`${url} is missing immutable caching.`);
    }
    if (response.headers.get('content-type') !== 'application/octet-stream') {
      throw new Error(`${url} has the wrong Content-Type.`);
    }
    if (response.headers.get('accept-ranges')?.toLowerCase() !== 'bytes') {
      throw new Error(`${url} does not advertise byte ranges.`);
    }
    if (!response.headers.get('etag')) throw new Error(`${url} is missing an ETag.`);

    const exposed = new Set(
      response.headers.get('access-control-expose-headers')
        ?.toLowerCase()
        .split(',')
        .map((value) => value.trim())
        .filter(Boolean) || [],
    );
    for (const header of ['content-length', 'etag', 'accept-ranges']) {
      if (!exposed.has(header)) throw new Error(`${url} does not expose ${header}.`);
    }

    const rangeResponse = await fetch(url, {
      method: 'GET',
      headers: { ...requestHeaders, Range: 'bytes=0-0' },
    });
    if (rangeResponse.status !== 206) {
      await rangeResponse.body?.cancel();
      throw new Error(`${url} range GET returned ${rangeResponse.status}, expected 206.`);
    }
    const expectedContentRange = `bytes 0-0/${asset.bytes}`;
    if (rangeResponse.headers.get('content-range') !== expectedContentRange) {
      await rangeResponse.body?.cancel();
      throw new Error(`${url} returned an invalid Content-Range.`);
    }
    if ((await rangeResponse.arrayBuffer()).byteLength !== 1) {
      throw new Error(`${url} range GET did not return exactly one byte.`);
    }
  }

  console.log(`Verified ${name}: ${asset.bytes} bytes, ${asset.sha256.slice(0, 12)}…`);
}
