import { describe, expect, it, vi } from 'vitest';
import { createModelLoadController } from '../js/model_loader.js';

describe('model load controller', () => {
  it('reports ordered load stages and deduplicates concurrent starts', async () => {
    let finish;
    const load = vi.fn((onStage) => new Promise((resolve) => {
      onStage('loading_encoder', { bytes: 10 });
      onStage('loading_decoder', { bytes: 20 });
      finish = () => resolve({ threads: 4 });
    }));
    const changes = [];
    const controller = createModelLoadController({ load, onChange: (state) => changes.push(state.status) });

    const first = controller.start();
    const second = controller.start();
    expect(first).toBe(second);
    await vi.waitFor(() => expect(finish).toBeTypeOf('function'));
    finish();

    await expect(first).resolves.toEqual({ threads: 4 });
    expect(load).toHaveBeenCalledOnce();
    expect(changes).toEqual(['loading_encoder', 'loading_encoder', 'loading_decoder', 'ready']);
  });

  it('can retry after a failed load', async () => {
    const load = vi.fn()
      .mockRejectedValueOnce(new Error('R2 unavailable'))
      .mockResolvedValueOnce({ threads: 1 });
    const controller = createModelLoadController({ load });

    await expect(controller.start()).rejects.toThrow('R2 unavailable');
    expect(controller.getSnapshot().status).toBe('error');
    await expect(controller.retry()).resolves.toEqual({ threads: 1 });
    expect(controller.getSnapshot().status).toBe('ready');
  });
});
