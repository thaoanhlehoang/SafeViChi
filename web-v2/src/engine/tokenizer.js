// Tokenizer BPE + Metaspace + ByteFallback cho mmBERT-base, đọc trực tiếp từ tokenizer.json.
// Khớp 100% với `tokenizers.Tokenizer` (Rust/Python) và trả về offsets theo chỉ số UTF-16 của JS
// để ánh xạ bằng chứng thẳng vào `line.slice(start, end)` kể cả khi có emoji/surrogate pairs.

const UTF8 = new TextEncoder();
const HEX = Array.from({ length: 256 }, (_, b) => `<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`);
const PAIR_STRIDE = 262144; // >= 256000 vocab size; max pairKey ~ 6.7e10 << Number.MAX_SAFE_INTEGER
const WORD_CHAR_RE = /[\p{L}\p{N}_]/u;

function escapeRegex(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

export class MMBertTokenizer {
  constructor(json) {
    const model = json?.model;
    if (!model || model.type !== 'BPE' || !model.vocab || !model.merges) {
      throw new Error('tokenizer.json không đúng định dạng BPE của mmBERT.');
    }

    this.replacement = json.pre_tokenizer?.replacement || '▁';
    this.prependAlways = (json.pre_tokenizer?.prepend_scheme || 'always') === 'always';
    this.splitMetaspace = json.pre_tokenizer?.split !== false;
    this.byteFallback = Boolean(model.byte_fallback);
    this.fuseUnk = Boolean(model.fuse_unk);

    // 1. Nạp bảng từ vựng: string -> id
    this.vocab = new Map();
    for (const [tok, id] of Object.entries(model.vocab)) {
      this.vocab.set(tok, id);
    }

    this.unkId = this.vocab.get(model.unk_token ?? '<unk>') ?? 3;
    this.byteIds = new Int32Array(256);
    for (let b = 0; b < 256; b++) {
      this.byteIds[b] = this.vocab.get(HEX[b]) ?? this.unkId;
    }

    // 2. Nạp bảng merges dưới dạng số nguyên (idA * PAIR_STRIDE + idB) -> [rank, mergedId]
    // Lưu trong 2 Map số giúp tiết kiệm RAM và không phải ghép chuỗi trong vòng lặp BPE.
    const merges = model.merges;
    this.mergeRank = new Map();
    this.mergeOut = new Map();
    for (let rank = 0; rank < merges.length; rank++) {
      const m = merges[rank];
      let a;
      let b;
      if (Array.isArray(m)) {
        a = m[0];
        b = m[1];
      } else {
        const sp = m.indexOf(' ');
        a = m.slice(0, sp);
        b = m.slice(sp + 1);
      }
      const idA = this.vocab.get(a);
      const idB = this.vocab.get(b);
      if (idA === undefined || idB === undefined) continue;
      const mergedId = this.vocab.get(a + b);
      if (mergedId === undefined) continue;
      const key = idA * PAIR_STRIDE + idB;
      if (!this.mergeRank.has(key)) {
        this.mergeRank.set(key, rank);
        this.mergeOut.set(key, mergedId);
      }
    }

    // 3. Added tokens (đều normalized=false; hỗ trợ lstrip/rstrip/single_word như HF tokenizers)
    const added = (json.added_tokens || [])
      .filter((t) => t && t.content)
      .sort((x, y) => y.content.length - x.content.length);

    this.addedByContent = new Map();
    for (const t of added) {
      this.addedByContent.set(t.content, t);
      if (!this.vocab.has(t.content)) this.vocab.set(t.content, t.id);
    }

    if (added.length > 0) {
      this.addedRegex = new RegExp(added.map((t) => escapeRegex(t.content)).join('|'), 'g');
    } else {
      this.addedRegex = null;
    }
  }

  /** Tách chuỗi gốc theo added_tokens trước khi chuẩn hóa/Metaspace. */
  _splitAdded(text) {
    if (!this.addedRegex || !text) return [{ added: false, text, start: 0, end: text.length }];
    const parts = [];
    this.addedRegex.lastIndex = 0;
    let cursor = 0;
    let m;
    while ((m = this.addedRegex.exec(text)) !== null) {
      const tok = this.addedByContent.get(m[0]);
      let s = m.index;
      let e = s + m[0].length;
      if (tok.single_word) {
        if ((s > 0 && WORD_CHAR_RE.test(text[s - 1])) || (e < text.length && WORD_CHAR_RE.test(text[e]))) {
          continue;
        }
      }
      if (tok.lstrip) {
        while (s > cursor && /\s/.test(text[s - 1])) s--;
      }
      if (tok.rstrip) {
        while (e < text.length && /\s/.test(text[e])) e++;
      }
      if (s > cursor) {
        parts.push({ added: false, text: text.slice(cursor, s), start: cursor, end: s });
      }
      parts.push({ added: true, id: tok.id, start: s, end: e });
      cursor = e;
      this.addedRegex.lastIndex = e;
    }
    if (cursor < text.length) {
      parts.push({ added: false, text: text.slice(cursor), start: cursor, end: text.length });
    }
    return parts;
  }

  /** Chạy BPE trên một mảnh đã qua Metaspace. */
  _bpePiece(elems, outIds, outOffsets) {
    const ids = [];
    const starts = [];
    const ends = [];

    for (let i = 0; i < elems.length; i++) {
      const { ch, start, end } = elems[i];
      const id = this.vocab.get(ch);
      if (id !== undefined) {
        ids.push(id);
        starts.push(start);
        ends.push(end);
      } else if (this.byteFallback) {
        const bytes = UTF8.encode(ch);
        for (let bi = 0; bi < bytes.length; bi++) {
          ids.push(this.byteIds[bytes[bi]]);
          starts.push(start);
          ends.push(end);
        }
      } else {
        ids.push(this.unkId);
        starts.push(start);
        ends.push(end);
      }
    }

    // Lặp gộp cặp có thứ hạng nhỏ nhất (ưu tiên bên trái nếu đồng hạng)
    let len = ids.length;
    while (len > 1) {
      let bestRank = Infinity;
      let bestIdx = -1;
      let bestOut = -1;
      for (let i = 0; i < len - 1; i++) {
        const key = ids[i] * PAIR_STRIDE + ids[i + 1];
        const rank = this.mergeRank.get(key);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestIdx = i;
          bestOut = this.mergeOut.get(key);
        }
      }
      if (bestIdx === -1) break;
      ids[bestIdx] = bestOut;
      if (starts[bestIdx + 1] < starts[bestIdx]) starts[bestIdx] = starts[bestIdx + 1];
      if (ends[bestIdx + 1] > ends[bestIdx]) ends[bestIdx] = ends[bestIdx + 1];
      ids.splice(bestIdx + 1, 1);
      starts.splice(bestIdx + 1, 1);
      ends.splice(bestIdx + 1, 1);
      len--;
    }

    // fuse_unk: gộp các <unk> liên tiếp
    for (let i = 0; i < len; i++) {
      if (this.fuseUnk && ids[i] === this.unkId && outIds.length > 0 && outIds[outIds.length - 1] === this.unkId) {
        const prev = outOffsets[outOffsets.length - 1];
        if (ends[i] > prev[1]) prev[1] = ends[i];
        continue;
      }
      outIds.push(ids[i]);
      outOffsets.push([starts[i], ends[i]]);
    }
  }

  /** Mã hóa 1 dòng (add_special_tokens=False), trả { ids: number[], offsets: [number, number][] }. */
  encode(line) {
    const outIds = [];
    const outOffsets = [];
    if (!line) return { ids: outIds, offsets: outOffsets };

    const rep = this.replacement;
    const parts = this._splitAdded(line);

    for (const part of parts) {
      if (part.added) {
        outIds.push(part.id);
        outOffsets.push([part.start, part.end]);
        continue;
      }
      if (!part.text) continue;

      // Bước 1: Normalizer Replace(" " -> "▁"), tách theo code point Unicode (giữ chỉ số UTF-16)
      const elems = [];
      let pos = part.start;
      for (const cp of part.text) {
        const nextPos = pos + cp.length;
        elems.push({ ch: cp === ' ' ? rep : cp, start: pos, end: nextPos });
        pos = nextPos;
      }
      if (elems.length === 0) continue;

      // Bước 2: Metaspace(prepend_scheme='always', split=True)
      if (this.prependAlways && elems[0].ch !== rep) {
        elems.unshift({ ch: rep, start: elems[0].start, end: elems[0].end });
      }

      if (!this.splitMetaspace) {
        this._bpePiece(elems, outIds, outOffsets);
        continue;
      }

      let pieceStart = 0;
      for (let i = 1; i < elems.length; i++) {
        if (elems[i].ch === rep) {
          this._bpePiece(elems.slice(pieceStart, i), outIds, outOffsets);
          pieceStart = i;
        }
      }
      this._bpePiece(elems.slice(pieceStart), outIds, outOffsets);
    }

    return { ids: outIds, offsets: outOffsets };
  }
}
