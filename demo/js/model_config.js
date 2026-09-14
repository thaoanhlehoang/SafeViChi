const MODEL_ROOT = 'models';
const SHA256_PATTERN = /^[a-f0-9]{64}$/;
const SAFE_SEGMENT_PATTERN = /^[a-zA-Z0-9._-]+$/;

function assertPositiveInteger(value, name) {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }
}

function validateAsset(asset, name) {
  if (!asset || typeof asset !== 'object') {
    throw new Error(`Missing ${name} asset metadata.`);
  }
  if (!SAFE_SEGMENT_PATTERN.test(asset.path || '')) {
    throw new Error(`${name}.path must be a single safe file name.`);
  }
  assertPositiveInteger(asset.bytes, `${name}.bytes`);
  if (!SHA256_PATTERN.test(asset.sha256 || '')) {
    throw new Error(`${name}.sha256 must be a lowercase SHA-256 digest.`);
  }
  return { path: asset.path, bytes: asset.bytes, sha256: asset.sha256 };
}

export function validateModelManifest(value) {
  if (!value || typeof value !== 'object' || value.schemaVersion !== 1) {
    throw new Error('Unsupported model manifest schema.');
  }
  if (!SAFE_SEGMENT_PATTERN.test(value.version || '')) {
    throw new Error('Model manifest version is invalid.');
  }

  const encoder = validateAsset(value.assets?.encoder, 'encoder');
  const decoder = validateAsset(value.assets?.decoder, 'decoder');
  assertPositiveInteger(value.totalBytes, 'totalBytes');
  if (encoder.bytes + decoder.bytes !== value.totalBytes) {
    throw new Error('Model manifest totalBytes does not match its assets.');
  }

  return {
    schemaVersion: 1,
    version: value.version,
    totalBytes: value.totalBytes,
    assets: { encoder, decoder },
  };
}

export function normalizeModelOrigin(value) {
  const trimmed = String(value || '').trim().replace(/\/+$/, '');
  if (!trimmed) throw new Error('Model origin is not configured.');

  const base = globalThis.location?.href || 'https://local.invalid/';
  const url = new URL(trimmed, base);
  if (url.protocol !== 'https:' && url.hostname !== '127.0.0.1' && url.hostname !== 'localhost') {
    throw new Error('Model origin must use HTTPS.');
  }
  return url.origin;
}

export function buildModelAssetUrl(modelOrigin, manifest, assetName) {
  const origin = normalizeModelOrigin(modelOrigin);
  const checked = validateModelManifest(manifest);
  const asset = checked.assets[assetName];
  if (!asset) throw new Error(`Unknown model asset: ${assetName}`);
  return new URL(`${MODEL_ROOT}/${checked.version}/${asset.path}`, `${origin}/`).href;
}

export function configuredModelOrigin(explicitOrigin) {
  if (explicitOrigin) return normalizeModelOrigin(explicitOrigin);
  const configured = import.meta.env.VITE_MODEL_ORIGIN;
  if (configured) return normalizeModelOrigin(configured);
  if (import.meta.env.DEV && globalThis.location?.origin) return globalThis.location.origin;
  throw new Error('VITE_MODEL_ORIGIN is not configured for this deployment.');
}
