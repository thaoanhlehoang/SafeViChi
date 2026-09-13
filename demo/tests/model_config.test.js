import { describe, expect, it } from 'vitest';
import {
  buildModelAssetUrl,
  normalizeModelOrigin,
  validateModelManifest,
} from '../js/model_config.js';

const manifest = {
  schemaVersion: 1,
  version: 'vihatet5-v2-fp32-test',
  totalBytes: 30,
  assets: {
    encoder: { path: 'encoder_model.onnx', bytes: 10, sha256: 'a'.repeat(64) },
    decoder: { path: 'decoder_model.onnx', bytes: 20, sha256: 'b'.repeat(64) },
  },
};

describe('model deployment configuration', () => {
  it('validates and normalizes the manifest contract', () => {
    expect(validateModelManifest(manifest)).toEqual(manifest);
  });

  it('rejects inconsistent byte totals and unsafe paths', () => {
    expect(() => validateModelManifest({ ...manifest, totalBytes: 31 })).toThrow('totalBytes');
    expect(() => validateModelManifest({
      ...manifest,
      assets: { ...manifest.assets, encoder: { ...manifest.assets.encoder, path: '../model.onnx' } },
    })).toThrow('single safe file name');
  });

  it('builds an immutable R2 object URL', () => {
    expect(buildModelAssetUrl('https://models.example.com/', manifest, 'decoder')).toBe(
      'https://models.example.com/models/vihatet5-v2-fp32-test/decoder_model.onnx',
    );
  });

  it('requires HTTPS outside localhost', () => {
    expect(normalizeModelOrigin('http://127.0.0.1:4173')).toBe('http://127.0.0.1:4173');
    expect(() => normalizeModelOrigin('http://models.example.com')).toThrow('HTTPS');
  });
});
