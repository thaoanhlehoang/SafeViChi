import assert from 'node:assert/strict';
import test from 'node:test';
import { onRequest } from '../functions/api/[[path]].js';
import { sha256 } from '../src/feedback/contract.js';

function request(env, path, headers = {}) {
  return onRequest({
    request: new Request('https://safevichi.example/api/' + path, { headers }),
    env,
    params: { path: path.split('/') },
  });
}

test('Bản demo dùng binding riêng và chỉ công bố một tài khoản kiểm duyệt', async () => {
  const env = { FEEDBACK_DEMO: '1', DEMO_FEEDBACK_DB: {} };
  const mode = await request(env, 'noi-bo/che-do');
  assert.equal(mode.headers.get('Cache-Control'), 'no-store');
  const config = await mode.json();
  assert.equal(config.demo, true);
  assert.deepEqual(config.vai_tro.map(({ id }) => id), ['kiem_duyet']);
  const actor = await request(env, 'noi-bo/phien', { 'X-SafeViChi-Demo-Role': 'duyet_b' });
  assert.deepEqual(await actor.json(), { id: 'duyet_b', vai_tro: 'nguoi_duyet' });
  const missing = await request({ FEEDBACK_DEMO: '1', FEEDBACK_DB: {} }, 'noi-bo/che-do');
  assert.equal(missing.status, 503);
});

test('Mã vai trò demo không mở quyền trên cơ sở dữ liệu thật', async () => {
  const token = crypto.randomUUID();
  const env = {
    FEEDBACK_DB: {},
    INTERNAL_TOKEN_HASHES: JSON.stringify({ [await sha256(token)]: { id: 'quan_tri', vai_tro: 'quan_tri' } }),
  };
  const fakeRole = { 'X-SafeViChi-Demo-Role': 'quan_tri' };
  assert.equal((await request(env, 'noi-bo/phien', fakeRole)).status, 401);
  assert.equal((await request(env, 'nhat-ky', fakeRole)).status, 401);
  const session = await request(env, 'noi-bo/phien', { Authorization: 'Bearer ' + token });
  assert.deepEqual(await session.json(), { id: 'quan_tri', vai_tro: 'quan_tri' });
});
