import { ValidationError, isValidCode, sha256 } from '../../src/feedback/contract.js';

export const DEMO_ACTORS = {
  duyet_a: { id: 'duyet_a', vai_tro: 'nguoi_duyet', ten: 'Người duyệt A' },
  duyet_b: { id: 'duyet_b', vai_tro: 'nguoi_duyet', ten: 'Người duyệt B' },
  quan_tri: { id: 'quan_tri', vai_tro: 'quan_tri', ten: 'Quản trị viên' },
};

export function isDemoMode(env) {
  return env.FEEDBACK_DEMO === '1';
}

export class ApiError extends Error {
  constructor(status, code) {
    super(code);
    this.status = status;
  }
}

export function json(data, status = 200) {
  return Response.json(data, {
    status,
    headers: { 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' },
  });
}

export function handleApiError(err) {
  if (err instanceof ValidationError) return json({ loi: 'du_lieu_khong_hop_le', truong: err.field }, 400);
  if (err instanceof ApiError) return json({ loi: err.message }, err.status);
  // Chi tiết lỗi hoặc body có thể chứa dữ liệu của người dùng.
  console.error('SafeViChi API: lỗi nội bộ.');
  return json({ loi: 'loi_may_chu' }, 500);
}

export async function readJson(request) {
  const contentType = request.headers.get('Content-Type')?.split(';', 1)[0].trim().toLowerCase();
  if (contentType !== 'application/json') throw new ApiError(415, 'can_json');
  const origin = request.headers.get('Origin');
  if ((origin && origin !== new URL(request.url).origin) ||
      request.headers.get('Sec-Fetch-Site') === 'cross-site') throw new ApiError(403, 'khac_origin');
  const limit = 24 * 1024;
  if (Number(request.headers.get('Content-Length')) > limit) throw new ApiError(413, 'goi_qua_lon');
  const reader = request.body?.getReader();
  if (!reader) throw new ApiError(400, 'json_khong_hop_le');
  const decoder = new TextDecoder('utf-8', { fatal: true });
  let bytes = 0;
  let text = '';
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      bytes += value.byteLength;
      if (bytes > limit) {
        await reader.cancel();
        throw new ApiError(413, 'goi_qua_lon');
      }
      text += decoder.decode(value, { stream: true });
    }
    return JSON.parse(text + decoder.decode());
  } catch (err) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(400, 'json_khong_hop_le');
  } finally {
    reader.releaseLock();
  }
}

export async function getInternalActor(request, env) {
  if (isDemoMode(env)) {
    const role = request.headers.get('X-SafeViChi-Demo-Role');
    if (!Object.hasOwn(DEMO_ACTORS, role)) throw new ApiError(401, 'chon_vai_tro_demo');
    const { id, vai_tro } = DEMO_ACTORS[role];
    return { id, vai_tro };
  }
  if (typeof env.INTERNAL_TOKEN_HASHES !== 'string' || !env.INTERNAL_TOKEN_HASHES) {
    throw new ApiError(503, 'chua_cau_hinh_noi_bo');
  }
  const match = /^Bearer ([A-Za-z0-9_-]{32,128})$/i.exec(request.headers.get('Authorization') ?? '');
  if (!match) throw new ApiError(401, 'can_xac_thuc');
  const tokens = JSON.parse(env.INTERNAL_TOKEN_HASHES);
  const hash = await sha256(match[1]);
  const actor = Object.hasOwn(tokens, hash) ? tokens[hash] : null;
  if (!actor || !isValidCode(actor.id)) throw new ApiError(401, 'can_xac_thuc');
  if (!['nguoi_duyet', 'quan_tri'].includes(actor.vai_tro)) throw new ApiError(403, 'thieu_quyen');
  return { id: actor.id, vai_tro: actor.vai_tro };
}

export async function requireRole(request, env, ...roles) {
  const actor = await getInternalActor(request, env);
  if (!roles.includes(actor.vai_tro)) throw new ApiError(403, 'thieu_quyen');
  return actor.id;
}
