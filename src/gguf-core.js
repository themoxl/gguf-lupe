/* GGUF core: header parser + dequantization. Pure JS, no deps.
   Works in browser (File/Blob) and Node (for tests). */
const GGUF = (() => {
  // ggml_type -> [name, block size, bytes per block]  (gguf-py 0.19 / ggml 2026)
  const TYPES = {
    0: ['F32', 1, 4], 1: ['F16', 1, 2], 2: ['Q4_0', 32, 18], 3: ['Q4_1', 32, 20],
    6: ['Q5_0', 32, 22], 7: ['Q5_1', 32, 24], 8: ['Q8_0', 32, 34], 9: ['Q8_1', 32, 40],
    10: ['Q2_K', 256, 84], 11: ['Q3_K', 256, 110], 12: ['Q4_K', 256, 144], 13: ['Q5_K', 256, 176],
    14: ['Q6_K', 256, 210], 15: ['Q8_K', 256, 292], 16: ['IQ2_XXS', 256, 66], 17: ['IQ2_XS', 256, 74],
    18: ['IQ3_XXS', 256, 98], 19: ['IQ1_S', 256, 50], 20: ['IQ4_NL', 32, 18], 21: ['IQ3_S', 256, 110],
    22: ['IQ2_S', 256, 82], 23: ['IQ4_XS', 256, 136], 24: ['I8', 1, 1], 25: ['I16', 1, 2],
    26: ['I32', 1, 4], 27: ['I64', 1, 8], 28: ['F64', 1, 8], 29: ['IQ1_M', 256, 56],
    30: ['BF16', 1, 2], 34: ['TQ1_0', 256, 54], 35: ['TQ2_0', 256, 66], 39: ['MXFP4', 32, 17],
    40: ['NVFP4', 64, 36], 41: ['Q1_0', 128, 18],
  };
  const FILE_TYPES = {
    0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 7: 'Q8_0', 8: 'Q5_0', 9: 'Q5_1', 10: 'Q2_K',
    11: 'Q3_K_S', 12: 'Q3_K_M', 13: 'Q3_K_L', 14: 'Q4_K_S', 15: 'Q4_K_M', 16: 'Q5_K_S', 17: 'Q5_K_M',
    18: 'Q6_K', 19: 'IQ2_XXS', 20: 'IQ2_XS', 21: 'Q2_K_S', 22: 'IQ3_XS', 23: 'IQ3_XXS', 24: 'IQ1_S',
    25: 'IQ4_NL', 26: 'IQ3_S', 27: 'IQ3_M', 28: 'IQ2_S', 29: 'IQ2_M', 30: 'IQ4_XS', 31: 'IQ1_M',
    32: 'BF16', 36: 'TQ1_0', 37: 'TQ2_0', 38: 'MXFP4_MOE', 39: 'NVFP4', 40: 'Q1_0',
  };
  // gguf metadata value types
  const VT = ['u8', 'i8', 'u16', 'i16', 'u32', 'i32', 'f32', 'bool', 'string', 'array', 'u64', 'i64', 'f64'];
  const VT_SIZE = [1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8];

  class NeedMore extends Error { constructor(at) { super('need more bytes at ' + at); this.at = at; } }

  const utf8 = new TextDecoder('utf-8');

  function parse(buf, fileSize) {
    const dv = new DataView(buf), u8 = new Uint8Array(buf), len = buf.byteLength;
    let p = 0;
    const need = n => { if (p + n > len) throw new NeedMore(p + n); };
    need(8);
    if (u8[0] !== 0x47 || u8[1] !== 0x47 || u8[2] !== 0x55 || u8[3] !== 0x46)
      throw new Error('Keine GGUF-Datei (Magic "GGUF" fehlt am Dateianfang).');
    let le = true, version = dv.getUint32(4, true);
    if ((version & 0xffff) === 0) { le = false; version = dv.getUint32(4, false); }
    if (version < 1 || version > 3) throw new Error('Nicht unterstützte GGUF-Version ' + version);
    p = 8;
    const u32 = () => { need(4); const v = dv.getUint32(p, le); p += 4; return v; };
    const u64 = () => {
      need(8);
      const a = dv.getUint32(p, le), b = dv.getUint32(p + 4, le); p += 8;
      return le ? b * 4294967296 + a : a * 4294967296 + b;
    };
    const len64 = version === 1 ? u32 : u64; // v1 used 32-bit lengths/counts
    const str = () => {
      const n = len64(); need(n);
      let s;
      if (n < 64) { // fast path for short ASCII (tokens, keys)
        let ascii = true;
        for (let i = p; i < p + n; i++) if (u8[i] > 127) { ascii = false; break; }
        s = ascii ? String.fromCharCode.apply(null, u8.subarray(p, p + n)) : utf8.decode(u8.subarray(p, p + n));
      } else s = utf8.decode(u8.subarray(p, p + n));
      p += n; return s;
    };
    const scalar = t => {
      const sz = VT_SIZE[t]; need(sz); let v;
      switch (t) {
        case 0: v = dv.getUint8(p); break; case 1: v = dv.getInt8(p); break;
        case 2: v = dv.getUint16(p, le); break; case 3: v = dv.getInt16(p, le); break;
        case 4: v = dv.getUint32(p, le); break; case 5: v = dv.getInt32(p, le); break;
        case 6: v = dv.getFloat32(p, le); break; case 7: v = dv.getUint8(p) !== 0; break;
        case 10: v = Number(dv.getBigUint64(p, le)); break; case 11: v = Number(dv.getBigInt64(p, le)); break;
        case 12: v = dv.getFloat64(p, le); break;
        default: throw new Error('Unbekannter Metadaten-Typ ' + t + ' bei Byte ' + p);
      }
      p += sz; return v;
    };
    const TA = { 0: Uint8Array, 1: Int8Array, 2: Uint16Array, 3: Int16Array, 4: Uint32Array, 5: Int32Array, 6: Float32Array, 12: Float64Array };
    const value = t => {
      if (t === 8) return str();
      if (t === 9) {
        const et = u32(), n = len64();
        if (et === 8) { const a = new Array(n); for (let i = 0; i < n; i++) a[i] = str(); return { arrayType: 'string', items: a }; }
        if (et === 9) { const a = new Array(n); for (let i = 0; i < n; i++) a[i] = value(9); return { arrayType: 'array', items: a }; }
        const sz = VT_SIZE[et]; if (!sz) throw new Error('Unbekannter Array-Typ ' + et);
        need(n * sz);
        let items;
        if (le && TA[et]) items = new TA[et](buf.slice(p, p + n * sz)); // copy -> aligned
        else { items = new Array(n); for (let i = 0; i < n; i++) items[i] = scalar(et); p -= n * sz; }
        p += n * sz;
        return { arrayType: VT[et], items };
      }
      return scalar(t);
    };

    const nTensors = len64(), nKV = len64();
    const meta = {}, metaTypes = {};
    for (let i = 0; i < nKV; i++) {
      const k = str(), t = u32();
      if (t > 12) throw new Error(`Unbekannter Metadaten-Typ ${t} für Schlüssel "${k}"`);
      meta[k] = value(t); metaTypes[k] = VT[t];
    }
    const tensors = new Array(nTensors);
    for (let i = 0; i < nTensors; i++) {
      const name = str(), nd = u32(), dims = [];
      for (let d = 0; d < nd; d++) dims.push(len64());
      const type = u32(), offset = u64();
      tensors[i] = { name, dims, type, offset, idx: i };
    }
    const headerEnd = p;
    const align = typeof meta['general.alignment'] === 'number' ? meta['general.alignment'] : 32;
    const dataStart = Math.ceil(headerEnd / align) * align;

    finalize(tensors, dataStart, fileSize);
    return { version, littleEndian: le, nTensors, nKV, meta, metaTypes, tensors, headerEnd, align, dataStart, fileSize };
  }


  /** Adds nElements, nBytes, rowBytes, absOffset, fileOrder to tensor infos. */
  function finalize(tensors, dataStart, fileSize) {
    const byOff = tensors.slice().sort((a, b) => a.offset - b.offset);
    for (let i = 0; i < byOff.length; i++) {
      const t = byOff[i];
      t.nElements = t.dims.reduce((a, b) => a * b, 1);
      const tr = TYPES[t.type];
      // unknown type: size from the gap to the next tensor
      const gapEnd = i + 1 < byOff.length ? byOff[i + 1].offset : (fileSize != null ? fileSize - dataStart : NaN);
      t.typeName = tr ? tr[0] : 'TYPE_' + t.type;
      t.nBytes = tr ? (t.nElements / tr[1]) * tr[2] : gapEnd - t.offset;
      t.rowBytes = tr && t.dims[0] % tr[1] === 0 ? (t.dims[0] / tr[1]) * tr[2] : NaN;
      t.absOffset = dataStart + t.offset;
      t.fileOrder = i;
    }
    return tensors;
  }

  /** Reads only the header of a Blob/File: grows the slice until parse succeeds. */
  async function parseBlob(blob, onProgress) {
    let size = Math.min(blob.size, 16 << 20), bytesRead = 0;
    for (;;) {
      const buf = await blob.slice(0, size).arrayBuffer();
      bytesRead += buf.byteLength;
      try {
        const h = parse(buf, blob.size);
        h.bytesRead = bytesRead; h.sliceSize = size;
        return h;
      } catch (e) {
        if (!(e instanceof NeedMore) || size >= blob.size) throw e;
        size = Math.min(blob.size, Math.max(size * 2, e.at + (1 << 20)));
        onProgress && onProgress(size);
      }
    }
  }

  // ---------- dequantization ----------
  const F16 = (() => { // lookup table for all 65536 half values
    const t = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) {
      const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
      t[h] = e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
    }
    return t;
  })();
  const bf16buf = new Float32Array(1), bf16u = new Uint32Array(bf16buf.buffer);
  const BF16 = h => { bf16u[0] = h << 16; return bf16buf[0]; };
  const KV_IQ4NL = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];
  const KV_MXFP4 = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
  const E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6];
  const UE4M3 = x => { // NVFP4 block scale (unsigned e4m3)
    const e = (x >> 3) & 0xf, m = x & 7;
    if (x === 0x7f) return 0; // ggml treats the NaN pattern as 0
    return e === 0 ? m * 2 ** -9 : (1 + m / 8) * 2 ** (e - 7);
  };
  function scaleMinK4(j, s, o) { // get_scale_min_k4 on scales at byte offset o
    if (j < 4) return [s[o + j] & 63, s[o + j + 4] & 63];
    return [(s[o + j + 4] & 0xf) | ((s[o + j - 4] >> 6) << 4), (s[o + j + 4] >> 4) | ((s[o + j] >> 6) << 4)];
  }

  /** Dequantize `nBlocks` whole blocks of `type` from bytes (Uint8Array) -> Float32Array, or null if unsupported. */
  function dequant(type, bytes, nBlocks, out) {
    const tr = TYPES[type]; if (!tr) return null;
    const [, QK, BS] = tr, n = nBlocks * QK;
    const y = out && out.length >= n ? out : new Float32Array(n); // every branch writes all n values
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength), q = bytes;
    const h = o => F16[dv.getUint16(o, true)];
    switch (TYPES[type][0]) {
      case 'F32': for (let i = 0; i < n; i++) y[i] = dv.getFloat32(i * 4, true); break;
      case 'F64': for (let i = 0; i < n; i++) y[i] = dv.getFloat64(i * 8, true); break;
      case 'F16': for (let i = 0; i < n; i++) y[i] = F16[dv.getUint16(i * 2, true)]; break;
      case 'BF16': for (let i = 0; i < n; i++) y[i] = BF16(dv.getUint16(i * 2, true)); break;
      case 'I8': for (let i = 0; i < n; i++) y[i] = dv.getInt8(i); break;
      case 'I16': for (let i = 0; i < n; i++) y[i] = dv.getInt16(i * 2, true); break;
      case 'I32': for (let i = 0; i < n; i++) y[i] = dv.getInt32(i * 4, true); break;
      case 'Q4_0': for (let b = 0; b < nBlocks; b++) {
        const o = b * BS, d = h(o), Y = b * 32;
        for (let j = 0; j < 16; j++) { const v = q[o + 2 + j]; y[Y + j] = ((v & 15) - 8) * d; y[Y + j + 16] = ((v >> 4) - 8) * d; }
      } break;
      case 'Q4_1': for (let b = 0; b < nBlocks; b++) {
        const o = b * BS, d = h(o), m = h(o + 2), Y = b * 32;
        for (let j = 0; j < 16; j++) { const v = q[o + 4 + j]; y[Y + j] = (v & 15) * d + m; y[Y + j + 16] = (v >> 4) * d + m; }
      } break;
      case 'Q5_0': for (let b = 0; b < nBlocks; b++) {
        const o = b * BS, d = h(o), qh = dv.getUint32(o + 2, true), Y = b * 32;
        for (let j = 0; j < 16; j++) {
          const v = q[o + 6 + j], h0 = ((qh >>> j) << 4) & 0x10, h1 = (qh >>> (j + 12)) & 0x10;
          y[Y + j] = (((v & 15) | h0) - 16) * d; y[Y + j + 16] = (((v >> 4) | h1) - 16) * d;
        }
      } break;
      case 'Q5_1': for (let b = 0; b < nBlocks; b++) {
        const o = b * BS, d = h(o), m = h(o + 2), qh = dv.getUint32(o + 4, true), Y = b * 32;
        for (let j = 0; j < 16; j++) {
          const v = q[o + 8 + j], h0 = ((qh >>> j) << 4) & 0x10, h1 = (qh >>> (j + 12)) & 0x10;
          y[Y + j] = ((v & 15) | h0) * d + m; y[Y + j + 16] = ((v >> 4) | h1) * d + m;
        }
      } break;
      case 'Q8_0': for (let b = 0; b < nBlocks; b++) {
        const o = b * BS, d = h(o), Y = b * 32;
        for (let j = 0; j < 32; j++) y[Y + j] = dv.getInt8(o + 2 + j) * d;
      } break;
      case 'Q2_K': for (let b = 0; b < nBlocks; b++) {
        const o = b * BS, sc = o, qs = o + 16, d = h(o + 80), mn = h(o + 82);
        let Y = b * 256, is = 0;
        for (let n0 = 0; n0 < 256; n0 += 128) {
          const qq = qs + (n0 >> 2);
          for (let sh = 0; sh < 8; sh += 2) {
            for (let half = 0; half < 2; half++) {
              const s = q[sc + is++], dl = d * (s & 15), ml = mn * (s >> 4);
              for (let l = 0; l < 16; l++) y[Y++] = dl * ((q[qq + l + 16 * half] >> sh) & 3) - ml;
            }
          }
        }
      } break;
      case 'Q3_K': for (let b = 0; b < nBlocks; b++) {
        const o = b * BS, hm = o, qs = o + 32, sc = o + 96, d = h(o + 108);
        const a0 = dv.getUint32(sc, true), a1 = dv.getUint32(sc + 4, true), tmp = dv.getUint32(sc + 8, true);
        const K1 = 0x03030303, K2 = 0x0f0f0f0f;
        const aux = new Uint32Array([
          (a0 & K2) | (((tmp >>> 0) & K1) << 4), (a1 & K2) | (((tmp >>> 2) & K1) << 4),
          ((a0 >>> 4) & K2) | (((tmp >>> 4) & K1) << 4), ((a1 >>> 4) & K2) | (((tmp >>> 6) & K1) << 4)]);
        const scales = new Int8Array(aux.buffer);
        let Y = b * 256, is = 0, m = 1;
        for (let n0 = 0; n0 < 256; n0 += 128) {
          const qq = qs + (n0 >> 2);
          for (let sh = 0; sh < 8; sh += 2) {
            for (let half = 0; half < 2; half++) {
              const dl = d * (scales[is++] - 32), off = 16 * half;
              for (let l = 0; l < 16; l++)
                y[Y++] = dl * (((q[qq + l + off] >> sh) & 3) - ((q[hm + l + off] & m) ? 0 : 4));
            }
            m <<= 1;
          }
        }
      } break;
      case 'Q4_K': for (let b = 0; b < nBlocks; b++) {
        const o = b * BS, d = h(o), mn = h(o + 2), sc = o + 4;
        let qs = o + 16, Y = b * 256, is = 0;
        for (let j = 0; j < 256; j += 64) {
          const [s1, m1] = scaleMinK4(is, q, sc), [s2, m2] = scaleMinK4(is + 1, q, sc);
          const d1 = d * s1, mm1 = mn * m1, d2 = d * s2, mm2 = mn * m2;
          for (let l = 0; l < 32; l++) y[Y + l] = d1 * (q[qs + l] & 15) - mm1;
          for (let l = 0; l < 32; l++) y[Y + 32 + l] = d2 * (q[qs + l] >> 4) - mm2;
          Y += 64; qs += 32; is += 2;
        }
      } break;
      case 'Q5_K': for (let b = 0; b < nBlocks; b++) {
        const o = b * BS, d = h(o), mn = h(o + 2), sc = o + 4, qh = o + 16;
        let ql = o + 48, Y = b * 256, is = 0, u1 = 1, u2 = 2;
        for (let j = 0; j < 256; j += 64) {
          const [s1, m1] = scaleMinK4(is, q, sc), [s2, m2] = scaleMinK4(is + 1, q, sc);
          const d1 = d * s1, mm1 = mn * m1, d2 = d * s2, mm2 = mn * m2;
          for (let l = 0; l < 32; l++) y[Y + l] = d1 * ((q[ql + l] & 15) + (q[qh + l] & u1 ? 16 : 0)) - mm1;
          for (let l = 0; l < 32; l++) y[Y + 32 + l] = d2 * ((q[ql + l] >> 4) + (q[qh + l] & u2 ? 16 : 0)) - mm2;
          Y += 64; ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
        }
      } break;
      case 'Q6_K': for (let b = 0; b < nBlocks; b++) {
        const o = b * BS, d = h(o + 208);
        let ql = o, qh = o + 128, sc = o + 192, Y = b * 256;
        for (let n0 = 0; n0 < 256; n0 += 128) {
          for (let l = 0; l < 32; l++) {
            const is = l >> 4, H = q[qh + l];
            const q1 = ((q[ql + l] & 15) | ((H & 3) << 4)) - 32;
            const q2 = ((q[ql + l + 32] & 15) | (((H >> 2) & 3) << 4)) - 32;
            const q3 = ((q[ql + l] >> 4) | (((H >> 4) & 3) << 4)) - 32;
            const q4 = ((q[ql + l + 32] >> 4) | (((H >> 6) & 3) << 4)) - 32;
            y[Y + l] = d * dv.getInt8(sc + is) * q1;
            y[Y + l + 32] = d * dv.getInt8(sc + is + 2) * q2;
            y[Y + l + 64] = d * dv.getInt8(sc + is + 4) * q3;
            y[Y + l + 96] = d * dv.getInt8(sc + is + 6) * q4;
          }
          Y += 128; ql += 64; qh += 32; sc += 8;
        }
      } break;
      case 'Q8_K': for (let b = 0; b < nBlocks; b++) {
        const o = b * BS, d = dv.getFloat32(o, true), Y = b * 256;
        for (let j = 0; j < 256; j++) y[Y + j] = dv.getInt8(o + 4 + j) * d;
      } break;
      case 'IQ4_NL': for (let b = 0; b < nBlocks; b++) {
        const o = b * BS, d = h(o), Y = b * 32;
        for (let j = 0; j < 16; j++) { const v = q[o + 2 + j]; y[Y + j] = d * KV_IQ4NL[v & 15]; y[Y + j + 16] = d * KV_IQ4NL[v >> 4]; }
      } break;
      case 'IQ4_XS': for (let b = 0; b < nBlocks; b++) {
        const o = b * BS, d = h(o), sh = dv.getUint16(o + 2, true), sl = o + 4;
        let qs = o + 8, Y = b * 256;
        for (let ib = 0; ib < 8; ib++) {
          const ls = ((q[sl + (ib >> 1)] >> (4 * (ib & 1))) & 15) | (((sh >> (2 * ib)) & 3) << 4), dl = d * (ls - 32);
          for (let j = 0; j < 16; j++) { y[Y + j] = dl * KV_IQ4NL[q[qs + j] & 15]; y[Y + j + 16] = dl * KV_IQ4NL[q[qs + j] >> 4]; }
          Y += 32; qs += 16;
        }
      } break;
      case 'TQ2_0': for (let b = 0; b < nBlocks; b++) {
        const o = b * BS, d = h(o + 64); let Y = b * 256;
        for (let j = 0; j < 64; j += 32) for (let l = 0; l < 4; l++) for (let m = 0; m < 32; m++)
          y[Y++] = (((q[o + j + m] >> (l * 2)) & 3) - 1) * d;
      } break;
      case 'MXFP4': for (let b = 0; b < nBlocks; b++) {
        const o = b * BS, e = q[o], d = e < 2 ? (e === 0 ? 2 ** -128 : 2 ** -127) : 2 ** (e - 128), Y = b * 32;
        for (let j = 0; j < 16; j++) { const v = q[o + 1 + j]; y[Y + j] = KV_MXFP4[v & 15] * d; y[Y + j + 16] = KV_MXFP4[v >> 4] * d; }
      } break;
      case 'NVFP4': for (let b = 0; b < nBlocks; b++) { // 4 sub-blocks of 16, ue4m3 scales
        const o = b * BS; let Y = b * 64;
        for (let s = 0; s < 4; s++) {
          const d = UE4M3(q[o + s]), qs = o + 4 + s * 8;
          for (let j = 0; j < 8; j++) { const v = q[qs + j]; y[Y + j] = E2M1[v & 15] * d; y[Y + j + 8] = E2M1[v >> 4] * d; }
          Y += 16;
        }
      } break;
      default: return null;
    }
    return y;
  }
  const canDequant = type => TYPES[type] && ['IQ2_XXS', 'IQ2_XS', 'IQ3_XXS', 'IQ1_S', 'IQ3_S', 'IQ2_S', 'IQ1_M', 'TQ1_0', 'Q8_1', 'Q1_0', 'I64'].indexOf(TYPES[type][0]) < 0;

  // ---------- tokenizer: port of llama.cpp's BPE path ----------
  // special-token split -> regex pre-split (table extracted from llama-vocab.cpp) -> ranked merges -> byte fallback
  const PRE = typeof GGUF_PRE !== 'undefined' ? GGUF_PRE
    : (typeof require !== 'undefined' ? require('./pre_table.json') : { types: {}, names: {}, flags: {} });
  const B2U = (() => { // GPT-2 bytes_to_unicode
    const bs = []; for (let i = 33; i <= 126; i++) bs.push(i); for (let i = 161; i <= 172; i++) bs.push(i); for (let i = 174; i <= 255; i++) bs.push(i);
    const cs = bs.slice(); let n = 0;
    for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n++); }
    const out = new Array(256); bs.forEach((b, i) => { out[b] = String.fromCodePoint(cs[i]); }); return out;
  })();
  const U2B = new Map(B2U.map((u, b) => [u, b]));
  const te = new TextEncoder();
  const hexTok = b => '<0x' + b.toString(16).toUpperCase().padStart(2, '0') + '>';

  class Tokenizer {
    constructor(meta) {
      const get = k => meta['tokenizer.ggml.' + k];
      this.model = get('model') || '';
      this.pre = get('pre') || (this.model === 'gemma4' ? 'gemma4' : 'default');
      const tk = get('tokens'), ty = get('token_type'), mg = get('merges');
      this.tokens = tk ? tk.items : []; this.types = ty ? ty.items : null; this.merges = mg ? mg.items : null;
      this.bosId = get('bos_token_id'); this.eosId = get('eos_token_id');
      if (this.model !== 'gpt2' && this.model !== 'gemma4') { this.supported = false; this.reason = `Eingebaut ist die Live-Zerlegung für BPE-Tokenizer (Qwen, Llama 3, Mistral Nemo, Gemma 4 …). Dieses Modell nutzt „${this.model || 'unbekannt'}“.`; return; }
      if (!this.merges || !this.tokens.length) { this.supported = false; this.reason = 'Die Datei enthält keine Merge-Regeln.'; return; }
      this.supported = true;
      this.preType = PRE.names[this.pre] || 'DEFAULT';
      this.approx = !PRE.names[this.pre];
      let ent = PRE.types[this.preType] || PRE.types.DEFAULT;
      try { this.res = ent.re.map(r => new RegExp(r.replace(/\\p\{Han\}/g, '\\p{Script=Han}'), 'gu')); }
      catch { ent = PRE.types.DEFAULT; this.approx = true; this.res = ent.re.map(r => new RegExp(r, 'gu')); }
      this.byteEncode = ent.byte;
      const fl = PRE.flags[this.pre] || {};
      this.ignoreMerges = !!fl.ignoreMerges; this.escapeWs = !!fl.escapeWs;
      const ab = get('add_bos_token');
      this.addBos = typeof ab === 'boolean' ? ab : ['LLAMA3', 'TEKKEN'].includes(this.preType);
      if (this.preType === 'GEMMA4') this.addBos = true; // llama.cpp workaround for Gemma 4
      this.addEos = get('add_eos_token') === true;
    }
    init() {
      if (this.t2id) return;
      const t2id = new Map(); for (let i = 0; i < this.tokens.length; i++) t2id.set(this.tokens[i], i); // last wins, like llama.cpp
      const ranks = new Map();
      for (let i = 0; i < this.merges.length; i++) { const w = this.merges[i], p = w.indexOf(' ', 1); if (p > 0) ranks.set(w.slice(0, p) + '\u0000' + w.slice(p + 1), i); }
      const sp = [];
      if (this.types) for (let i = 0; i < this.tokens.length; i++) { const y = this.types[i]; if ((y === 2 || y === 3 || y === 4) && this.tokens[i]) sp.push(i); }
      const blen = i => te.encode(this.tokens[i]).length;
      sp.sort((a, b) => blen(b) - blen(a));
      this.t2id = t2id; this.ranks = ranks; this.specials = sp;
    }
    preSplit(text) {
      let pieces = [text];
      for (const re of this.res) {
        const out = [];
        for (const p of pieces) {
          re.lastIndex = 0; let last = 0, m;
          while ((m = re.exec(p))) {
            if (!m[0].length) { re.lastIndex++; continue; }
            if (m.index > last) out.push(p.slice(last, m.index));
            out.push(m[0]); last = m.index + m[0].length;
          }
          if (last < p.length) out.push(p.slice(last));
        }
        pieces = out;
      }
      return pieces;
    }
    bpeWord(word, trace) {
      const init = this.byteEncode ? Array.from(te.encode(word), b => B2U[b]) : Array.from(word);
      const joined = init.join('');
      if (this.ignoreMerges && this.t2id.has(joined)) return { init, syms: [joined], steps: trace ? [{ whole: true, merged: joined }] : null };
      if (this.preType === 'GEMMA4' && /^\n+$/.test(word) && this.t2id.has(word)) return { init: [word], syms: [word], steps: trace ? [] : null };
      const s = init.slice(), steps = trace ? [] : null;
      for (;;) {
        let best = -1, bestRank = Infinity;
        for (let i = 0; i < s.length - 1; i++) { const r = this.ranks.get(s[i] + '\u0000' + s[i + 1]); if (r !== undefined && r < bestRank) { bestRank = r; best = i; } }
        if (best < 0) break;
        const a = s[best], b = s[best + 1];
        s.splice(best, 2, a + b);
        if (trace) steps.push({ rank: bestRank, at: best, a, b, merged: a + b, syms: s.slice() });
      }
      return { init, syms: s, steps };
    }
    tokenize(text, opt = {}) {
      this.init();
      const parseSpecial = opt.parseSpecial !== false, addSpecial = opt.addSpecial !== false, trace = !!opt.trace;
      let frags = [{ t: text }];
      for (const id of this.specials) {
        const y = this.types[id];
        if (!parseSpecial && (y === 3 || y === 2)) continue;
        const s = this.tokens[id], out = [];
        for (const f of frags) {
          if (f.id !== undefined) { out.push(f); continue; }
          let pos = 0, k;
          while ((k = f.t.indexOf(s, pos)) >= 0) { if (k > pos) out.push({ t: f.t.slice(pos, k) }); out.push({ id }); pos = k + s.length; }
          if (pos < f.t.length) out.push({ t: f.t.slice(pos) });
        }
        frags = out;
      }
      const ids = [], pieces = [], words = [];
      const emit = (id, extra) => { ids.push(id); pieces.push({ id, ...extra }); };
      if (addSpecial && this.addBos && this.bosId != null) emit(this.bosId, { kind: 'bos' });
      for (const f of frags) {
        if (f.id !== undefined) { emit(f.id, { kind: 'special' }); continue; }
        const t = this.escapeWs ? f.t.replace(/ /g, '▁') : f.t;
        for (const w of this.preSplit(t)) {
          const wi = words.length, r = this.bpeWord(w, trace);
          words.push({ text: w, init: r.init, steps: r.steps, syms: r.syms });
          for (const s of r.syms) {
            const id = this.t2id.get(s);
            if (id !== undefined) { emit(id, { word: wi }); continue; }
            if (this.byteEncode) { for (const ch of s) { const b = this.t2id.get(ch); if (b !== undefined) emit(b, { word: wi, fallback: true }); } }
            else for (const byte of te.encode(s)) { const b = this.t2id.get(hexTok(byte)); if (b !== undefined) emit(b, { word: wi, fallback: true }); }
          }
        }
      }
      if (addSpecial && this.addEos && this.eosId != null) emit(this.eosId, { kind: 'eos' });
      return { ids, pieces, words };
    }
  }

  return { TYPES, FILE_TYPES, parse, parseBlob, finalize, dequant, canDequant, NeedMore, F16, Tokenizer, B2U, U2B };
})();
if (typeof module !== 'undefined') module.exports = GGUF;
