import { UUID_PATTERN } from '../../src/feedback/contract.js';
import { listReviewQueue, saveReview, openReviewSample, getAuditLog } from '../../cloudflare/api/review.js';
import { DEMO_REVIEWER, getInternalActor, isDemoMode, requireRole, readJson, json, ApiError, handleApiError } from '../../cloudflare/api/http.js';
import { getProposal, listPendingProposals, getActiveThreshold, approveProposal, createProposal } from '../../cloudflare/api/thresholds.js';
import { saveFeedback, getFeedbackStats } from '../../cloudflare/api/feedback.js';

function requireId(id) {
  if (!UUID_PATTERN.test(id ?? '')) throw new ApiError(404, 'khong_tim_thay');
  return id;
}

export async function onRequest({ request, env, params }) {
  try {
    const [area, id, action, ...extra] = params.path ?? [];
    if (extra.length) throw new ApiError(404, 'khong_tim_thay');
    const demo = isDemoMode(env);
    // Chế độ trình diễn chỉ làm việc với binding dữ liệu demo.
    const db = demo ? env.DEMO_FEEDBACK_DB : env.FEEDBACK_DB;
    if (!db) throw new ApiError(503, 'chua_cau_hinh');
    const method = request.method;
    const url = new URL(request.url);
    switch (area) {
      case 'phan-hoi': {
        if (id) break;
        if (method !== 'POST') return methodNotAllowed('POST');
        const result = await saveFeedback(db, env.FEEDBACK_RATE_SECRET, await readJson(request));
        return json(result, result.gui_lai ? 200 : 201);
      }
      case 'nguong':
        if (id) break;
        if (method !== 'GET') return methodNotAllowed('GET');
        return json(await getActiveThreshold(db, url.searchParams.get('phien_ban_mo_hinh')));
      case 'noi-bo':
        if (!['che-do', 'phien'].includes(id) || action) break;
        if (method !== 'GET') return methodNotAllowed('GET');
        if (id === 'che-do') return json({ demo, vai_tro: demo ? [DEMO_REVIEWER] : [] });
        return json(await getInternalActor(request, env));
      case 'duyet': {
        if (action) break;
        const actor = await requireRole(request, env, 'nguoi_duyet');
        if (!id) {
          if (method !== 'GET') return methodNotAllowed('GET');
          return json(await listReviewQueue(db, actor, url.searchParams));
        }
        if (id === 'tiep-theo') {
          if (method !== 'GET') return methodNotAllowed('GET');
          return json(await openReviewSample(db, actor));
        }
        if (method === 'GET') return json(await openReviewSample(db, actor, requireId(id)));
        if (method === 'POST') return json(await saveReview(db, actor, requireId(id), await readJson(request)), 201);
        return methodNotAllowed('GET, POST');
      }
      case 'thong-ke':
        if (id) break;
        if (method !== 'GET') return methodNotAllowed('GET');
        await requireRole(request, env, 'nguoi_duyet', 'quan_tri');
        return json(await getFeedbackStats(db, url));
      case 'nhat-ky':
        if (action) break;
        if (method !== 'GET') return methodNotAllowed('GET');
        await requireRole(request, env, 'quan_tri');
        return json(await getAuditLog(db, id ? requireId(id) : null));
      case 'de-xuat': {
        if (action === 'phe-duyet') {
          if (method !== 'POST') return methodNotAllowed('POST');
          const actor = await requireRole(request, env, 'quan_tri');
          return json(await approveProposal(db, actor, requireId(id), await readJson(request)));
        }
        if (action) break;
        const actor = await requireRole(request, env, 'nguoi_duyet', 'quan_tri');
        if (method === 'GET') return json(id ? await getProposal(db, requireId(id)) : await listPendingProposals(db));
        if (!id && method === 'POST') return json(await createProposal(db, actor, await readJson(request)), 201);
        return methodNotAllowed(id ? 'GET' : 'GET, POST');
      }
    }
    throw new ApiError(404, 'khong_tim_thay');
  } catch (err) {
    return handleApiError(err);
  }
}

function methodNotAllowed(allow) {
  const response = json({ loi: 'phuong_thuc_khong_hop_le' }, 405);
  response.headers.set('Allow', allow);
  return response;
}
