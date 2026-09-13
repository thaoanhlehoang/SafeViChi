import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const ai = vi.hoisted(() => ({
  initModel: vi.fn(() => new Promise(() => {})),
  isReady: vi.fn(() => false),
  scanMessage: vi.fn(),
}));

vi.mock('../js/ai_scanner.js', () => ai);

describe('send gate before the model is ready', () => {
  beforeEach(() => {
    vi.resetModules();
    ai.initModel.mockReset().mockImplementation(() => new Promise(() => {}));
    ai.isReady.mockReturnValue(false);
    ai.scanMessage.mockReset();
    const html = readFileSync(resolve('index.html'), 'utf8');
    document.open();
    document.write(html);
    document.close();
    window.history.replaceState({}, '', '/');
  });

  it('starts loading while preserving the unsent draft', async () => {
    await import('../js/app.js');
    const input = document.getElementById('messageInput');
    input.value = 'Bản nháp phải được giữ nguyên';
    document.getElementById('messageComposer').dispatchEvent(new Event('submit', {
      bubbles: true,
      cancelable: true,
    }));

    await vi.waitFor(() => expect(ai.initModel).toHaveBeenCalledOnce());
    expect(input.value).toBe('Bản nháp phải được giữ nguyên');
    expect(document.getElementById('modelLoadButton').disabled).toBe(true);
  });

  it('preserves the draft and exposes Retry after a download failure', async () => {
    ai.initModel.mockRejectedValueOnce(new Error('R2 unavailable'));
    await import('../js/app.js');
    const input = document.getElementById('messageInput');
    input.value = 'Nội dung chưa được gửi';
    document.getElementById('messageComposer').dispatchEvent(new Event('submit', {
      bubbles: true,
      cancelable: true,
    }));

    await vi.waitFor(() => {
      expect(document.getElementById('modelLoadButton').dataset.state).toBe('error');
    });
    expect(input.value).toBe('Nội dung chưa được gửi');
    expect(document.getElementById('modelLoadButton').disabled).toBe(false);
    expect(document.getElementById('modelStatusText').textContent).toContain('Thử lại');
  });

  it('shows the warning choice for a harmful QA fixture', async () => {
    window.history.replaceState({}, '', '/?qa=1');
    await import('../js/app.js');
    const input = document.getElementById('messageInput');
    input.value = 'Đồ ngu, biến đi';
    document.getElementById('messageComposer').dispatchEvent(new Event('submit', {
      bubbles: true,
      cancelable: true,
    }));

    await vi.waitFor(() => {
      expect(document.querySelector('[data-pending-action="send-anyway"]')).not.toBeNull();
    });
    expect(document.body.textContent).toContain('Hãy dừng lại một chút');
    expect(document.querySelector('[data-pending-action="edit"]')).not.toBeNull();
    expect(input.disabled).toBe(true);
  });

  it('sends a clean QA fixture after scanning it', async () => {
    window.history.replaceState({}, '', '/?qa=1');
    await import('../js/app.js');
    const input = document.getElementById('messageInput');
    const cleanText = 'Một tin nhắn an toàn duy nhất';
    input.value = cleanText;
    document.getElementById('messageComposer').dispatchEvent(new Event('submit', {
      bubbles: true,
      cancelable: true,
    }));

    await vi.waitFor(() => {
      expect(document.getElementById('messageList').textContent).toContain(cleanText);
    });
    expect(document.querySelector('[data-pending-action="send-anyway"]')).toBeNull();
    expect(input.value).toBe('');
  });
});
