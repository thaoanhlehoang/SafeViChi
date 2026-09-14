const LOAD_STATES = new Set(['loading_encoder', 'loading_decoder']);

export function createModelLoadController({ load, onChange = () => {} }) {
  let snapshot = { status: 'idle', detail: null, error: null };
  let inFlight = null;

  const transition = (status, detail = null, error = null) => {
    snapshot = { status, detail, error };
    onChange(snapshot);
  };

  const start = () => {
    if (snapshot.status === 'ready') return Promise.resolve(snapshot.detail);
    if (inFlight) return inFlight;

    transition('loading_encoder');
    inFlight = Promise.resolve()
      .then(() => load((status, detail) => {
        if (LOAD_STATES.has(status)) transition(status, detail);
      }))
      .then((detail) => {
        transition('ready', detail);
        return detail;
      })
      .catch((error) => {
        transition('error', null, error instanceof Error ? error : new Error(String(error)));
        throw snapshot.error;
      })
      .finally(() => {
        inFlight = null;
      });
    return inFlight;
  };

  return {
    start,
    retry: start,
    getSnapshot: () => snapshot,
  };
}
