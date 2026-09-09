struct LinearInfo {
    M: u32, // Batch * Seq
    K: u32, // Hidden Size
    N: u32, // Vocab Size
}

@group(0) @binding(0) var<uniform> info: LinearInfo;
@group(0) @binding(1) var<storage, read> Input: array<f32>;

// We use two bindings for weights. Only one is active per pipeline.
@group(0) @binding(2) var<storage, read> W_F32: array<f32>;  // [N, K]
@group(0) @binding(3) var<storage, read> W_BF16: array<u32>; // [N, K/2]

@group(0) @binding(4) var<storage, read_write> Output: array<f32>; // [M, N]

// Q4_K weights, read as raw words. A block is 144 bytes covering 256 weights:
// d (f16), dmin (f16), scales[12], qs[128]. Kept packed so decode streams a
// quarter of what the BF16 path streams.
@group(0) @binding(5) var<storage, read> W_Q4K: array<u32>;

// Q6_K weights. A block is 210 bytes -- ql[128], qh[64], scales[16], d -- which is
// not a multiple of four, so the upload pads each block to 212 (53 words) and the
// shader can index words directly instead of straddling boundaries.
@group(0) @binding(9) var<storage, read> W_Q6K: array<u32>;

// Q8_0 weights. The on-disk block is 34 bytes -- d (f16) then qs[32] -- which
// leaves `qs` two bytes out of word alignment, so the upload repacks each block
// into 9 words: d alone in word 0, then the 32 values as eight clean words. That
// costs 5.9% of the buffer and buys an aligned `unpack4xI8` per word.
@group(0) @binding(10) var<storage, read> W_Q8: array<u32>;

// Reduction cache for Wide kernel
var<workgroup> wg_sum: array<f32, 256>;

// Four row accumulators per thread, reduced together in one barrier chain.
//
// The GEMV kernels run 64 threads rather than 256. At K=3072 a row is only 384 qs
// words, so with 256 threads each one did about 1.5 words of work and then paid an
// eight-step barrier reduction; the reduction cost as much as the arithmetic. Fewer,
// busier threads move that ratio the right way.
var<workgroup> wg_sum4: array<vec4<f32>, 64>;

fn unpack_bf16(packed: u32) -> vec2<f32> {
    let x = bitcast<f32>(packed << 16u);
    let y = bitcast<f32>(packed & 0xFFFF0000u);
    return vec2<f32>(x, y);
}

// Wide GEMV (BF16)
@compute @workgroup_size(256)
fn gemv_bf16_wide(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32> 
) {
    let n = wg_id.y * grid.x + wg_id.x; 
    
    let tid = local_id.x;

    if (n >= info.N) { return; }
    
    let k_pairs = info.K / 2u;
    let weight_base = n * k_pairs;
    var partial_sum = 0.0;

    for (var k = tid; k < k_pairs; k = k + 256u) {
        let in0 = Input[k * 2u];
        let in1 = Input[k * 2u + 1u];
        
        let w_packed = W_BF16[weight_base + k];
        let w_vec = unpack_bf16(w_packed);
        
        partial_sum += in0 * w_vec.x + in1 * w_vec.y;
    }

    wg_sum[tid] = partial_sum;
    workgroupBarrier();

    // Tree Reduction
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            wg_sum[tid] += wg_sum[tid + s];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        Output[n] = wg_sum[0];
    }
}

// Standard GEMV (BF16) - 1 Thread per Output
@compute @workgroup_size(256)
fn gemv_bf16(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = id.x;
    if (n >= info.N) { return; }

    var sum = vec4<f32>(0.0);
    let k_pairs = info.K / 2u;
    let weight_offset = n * k_pairs;

    // Process 4 pairs (8 floats) per iteration
    var k = 0u;
    for (; k + 3u < k_pairs; k += 4u) {
        let w0 = unpack_bf16(W_BF16[weight_offset + k]);
        let w1 = unpack_bf16(W_BF16[weight_offset + k + 1u]);
        let w2 = unpack_bf16(W_BF16[weight_offset + k + 2u]);
        let w3 = unpack_bf16(W_BF16[weight_offset + k + 3u]);

        // Input index is k*2
        let idx = k * 2u;
        let in0 = Input[idx];      let in1 = Input[idx + 1u];
        let in2 = Input[idx + 2u]; let in3 = Input[idx + 3u];
        let in4 = Input[idx + 4u]; let in5 = Input[idx + 5u];
        let in6 = Input[idx + 6u]; let in7 = Input[idx + 7u];

        sum.x += in0 * w0.x + in1 * w0.y;
        sum.y += in2 * w1.x + in3 * w1.y;
        sum.z += in4 * w2.x + in5 * w2.y;
        sum.w += in6 * w3.x + in7 * w3.y;
    }

    // Cleanup tail
    var tail_sum = 0.0;
    for (; k < k_pairs; k += 1u) {
        let w = unpack_bf16(W_BF16[weight_offset + k]);
        let in0 = Input[k * 2u];
        let in1 = Input[k * 2u + 1u];
        tail_sum += in0 * w.x + in1 * w.y;
    }

    Output[n] = sum.x + sum.y + sum.z + sum.w + tail_sum;
}

// BMM (BF16) - 2D Tiled
@compute @workgroup_size(16, 16)
fn bmm_bf16(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = id.x;
    let m = id.y;
    
    if (m >= info.M || n >= info.N) { return; }

    var sum = 0.0;
    let input_offset = m * info.K;
    let k_pairs = info.K / 2u;
    let weight_offset = n * k_pairs;
    
    for (var k = 0u; k < k_pairs; k = k + 1u) {
        let w_vec = unpack_bf16(W_BF16[weight_offset + k]);
        let in0 = Input[input_offset + k * 2u];
        let in1 = Input[input_offset + k * 2u + 1u];
        sum += in0 * w_vec.x + in1 * w_vec.y;
    }
    Output[m * info.N + n] = sum;
}

// GEMV (F32)
@compute @workgroup_size(256)
fn gemv_f32(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = id.x;
    if (n >= info.N) { return; }

    var sum = 0.0;
    let weight_offset = n * info.K;
    
    for (var k = 0u; k < info.K; k = k + 1u) {
        sum += Input[k] * W_F32[weight_offset + k];
    }
    Output[n] = sum;
}

// BMM (F32)
@compute @workgroup_size(16, 16)
fn bmm_f32(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = id.x;
    let m = id.y;
    if (m >= info.M || n >= info.N) { return; }

    var sum = 0.0;
    let input_offset = m * info.K;
    let weight_offset = n * info.K;

    for (var k = 0u; k < info.K; k = k + 1u) {
        sum += Input[input_offset + k] * W_F32[weight_offset + k];
    }
    Output[m * info.N + n] = sum;
}

// ---------------------------------------------------------------------------
// Q4_K GEMV
//
// The BF16 kernels above require the weights to be expanded before upload, which
// costs 3.4x the memory traffic per token and, on a 12GB card, the difference
// between a 3B model fitting and spilling to host memory. This kernel reads the
// Q4_K blocks as they sit in the GGUF.
//
// Layout of one 144-byte block, as 36 u32 words:
//   word  0      : d (f16) in the low half, dmin (f16) in the high half
//   words 1..3   : scales[12], six-bit scales and mins for the 8 sub-blocks
//   words 4..35  : qs[128], two 4-bit weights per byte
//
// Dequantisation matches `dequantize_q4_k_block` on the CPU side exactly: each
// group of 64 weights takes its low nibbles from one sub-block scale/min pair and
// its high nibbles from the next.
// ---------------------------------------------------------------------------

const Q4K_WORDS: u32 = 36u; // 144 bytes per block

// The three scale words are passed by value, never re-read. Indexing the storage
// buffer per extracted byte costs a memory op each time and the compiler will not
// hoist it out, which measured 4x slower than the BF16 kernel on the same weights.
fn q4k_sbyte(s0: u32, s1: u32, s2: u32, i: u32) -> u32 {
    var w = s0;
    if (i >= 8u) { w = s2; } else if (i >= 4u) { w = s1; }
    return (w >> ((i & 3u) * 8u)) & 0xFFu;
}

// Mirrors `get_scale_min_k4`: sub-blocks 0-3 are plain 6-bit values, 4-7 are
// reassembled from a low nibble plus the top two bits of an earlier byte.
fn q4k_scale_min(s0: u32, s1: u32, s2: u32, j: u32) -> vec2<f32> {
    if (j < 4u) {
        return vec2<f32>(
            f32(q4k_sbyte(s0, s1, s2, j) & 63u),
            f32(q4k_sbyte(s0, s1, s2, j + 4u) & 63u),
        );
    }
    let sc = (q4k_sbyte(s0, s1, s2, j + 4u) & 0xFu)
           | ((q4k_sbyte(s0, s1, s2, j - 4u) >> 6u) << 4u);
    let m  = (q4k_sbyte(s0, s1, s2, j + 4u) >> 4u)
           | ((q4k_sbyte(s0, s1, s2, j) >> 6u) << 4u);
    return vec2<f32>(f32(sc), f32(m));
}

// Dequantises one qs word (8 weights) into two vec4 lanes: the low nibbles and the
// high nibbles, each already scaled and offset.
//
// It returns the weights rather than a dot product so the caller can load the input
// activations once and reuse them across several output rows. That reuse is the
// point: the kernel moves a quarter of the bytes the BF16 kernel does in the same
// wall time, so it is ALU- and latency-bound, and giving each thread more rows is
// the only change measured to help.
struct Q4KPair {
    lo: vec4<f32>,
    hi: vec4<f32>,
}

fn q4k_word_weights(base: u32, wi: u32) -> Q4KPair {
    let dd = unpack2x16float(W_Q4K[base]);
    let s0 = W_Q4K[base + 1u];
    let s1 = W_Q4K[base + 2u];
    let s2 = W_Q4K[base + 3u];
    let qword = W_Q4K[base + 4u + wi];

    let j = wi / 8u;
    let sm1 = q4k_scale_min(s0, s1, s2, j * 2u);
    let sm2 = q4k_scale_min(s0, s1, s2, j * 2u + 1u);

    // One mask over the whole word extracts four nibbles at once.
    let lo_n = vec4<f32>(unpack4xU8(qword & 0x0F0F0F0Fu));
    let hi_n = vec4<f32>(unpack4xU8((qword >> 4u) & 0x0F0F0F0Fu));

    var out: Q4KPair;
    out.lo = lo_n * (dd.x * sm1.x) - vec4<f32>(dd.y * sm1.y);
    out.hi = hi_n * (dd.x * sm2.x) - vec4<f32>(dd.y * sm2.y);
    return out;
}

// Output rows each GEMV workgroup computes. Shared by the Q4_K and Q6_K kernels:
// the input activations are loaded once and reused across all of them, which is what
// took these kernels from 12% of the card's bandwidth to competitive with BF16.
// Measured: 4 beats 1, and 8 is no better than 4 once register pressure bites.
const ROWS_PER_WG: u32 = 4u;

@compute @workgroup_size(64)
fn gemv_q4k(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>
) {
    let row0 = (wg_id.y * grid.x + wg_id.x) * ROWS_PER_WG;
    let tid = local_id.x;
    if (row0 >= info.N) { return; }

    let blocks_per_row = info.K / 256u;
    let words_per_row = blocks_per_row * Q4K_WORDS;
    let total_words = blocks_per_row * 32u;

    var p0 = 0.0;
    var p1 = 0.0;
    var p2 = 0.0;
    var p3 = 0.0;

    for (var w = tid; w < total_words; w = w + 64u) {
        let blk = w / 32u;
        let wi = w % 32u;
        let l0 = blk * 256u + (wi / 8u) * 64u + (wi % 8u) * 4u;

        // Loaded once, used by every row this workgroup owns.
        let a_lo = vec4<f32>(Input[l0], Input[l0 + 1u], Input[l0 + 2u], Input[l0 + 3u]);
        let a_hi = vec4<f32>(
            Input[l0 + 32u], Input[l0 + 33u], Input[l0 + 34u], Input[l0 + 35u],
        );
        let blk_off = blk * Q4K_WORDS + wi;
        let base = row0 * words_per_row + blk * Q4K_WORDS;

        let w0 = q4k_word_weights(base, wi);
        p0 += dot(a_lo, w0.lo) + dot(a_hi, w0.hi);
        if (row0 + 1u < info.N) {
            let w1 = q4k_word_weights(base + words_per_row, wi);
            p1 += dot(a_lo, w1.lo) + dot(a_hi, w1.hi);
        }
        if (row0 + 2u < info.N) {
            let w2 = q4k_word_weights(base + words_per_row * 2u, wi);
            p2 += dot(a_lo, w2.lo) + dot(a_hi, w2.hi);
        }
        if (row0 + 3u < info.N) {
            let w3 = q4k_word_weights(base + words_per_row * 3u, wi);
            p3 += dot(a_lo, w3.lo) + dot(a_hi, w3.hi);
        }
    }

    wg_sum4[tid] = vec4<f32>(p0, p1, p2, p3);
    workgroupBarrier();
    for (var s = 32u; s > 0u; s >>= 1u) {
        if (tid < s) {
            wg_sum4[tid] += wg_sum4[tid + s];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        let r = wg_sum4[0];
        Output[row0] = r.x;
        if (row0 + 1u < info.N) { Output[row0 + 1u] = r.y; }
        if (row0 + 2u < info.N) { Output[row0 + 2u] = r.z; }
        if (row0 + 3u < info.N) { Output[row0 + 3u] = r.w; }
    }
}

// Batched Q4_K. Prefill runs many rows at once, so the grid is (n, m) and each
// workgroup reduces one output element. Same unpacking as the GEMV above; only
// the input offset and the output index differ.
@compute @workgroup_size(256)
fn bmm_q4k(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let n = wg_id.x;
    let row = wg_id.y;
    let tid = local_id.x;
    if (n >= info.N || row >= info.M) { return; }

    let blocks_per_row = info.K / 256u;
    let row_base = n * blocks_per_row * Q4K_WORDS;
    let total_words = blocks_per_row * 32u;
    let input_offset = row * info.K;

    var partial = 0.0;
    for (var w = tid; w < total_words; w = w + 256u) {
        let blk = w / 32u;
        let wi = w % 32u;
        let l0 = input_offset + blk * 256u + (wi / 8u) * 64u + (wi % 8u) * 4u;
        let a_lo = vec4<f32>(Input[l0], Input[l0 + 1u], Input[l0 + 2u], Input[l0 + 3u]);
        let a_hi = vec4<f32>(
            Input[l0 + 32u], Input[l0 + 33u], Input[l0 + 34u], Input[l0 + 35u],
        );
        let ww = q4k_word_weights(row_base + blk * Q4K_WORDS, wi);
        partial += dot(a_lo, ww.lo) + dot(a_hi, ww.hi);
    }

    wg_sum[tid] = partial;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            wg_sum[tid] += wg_sum[tid + s];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        Output[row * info.N + n] = wg_sum[0];
    }
}


// ---------------------------------------------------------------------------
// Q8_0 GEMV
//
// Repacked block layout, 9 u32 words:
//   word    0 : d (f16) in the low half
//   words 1..8: qs[32], four signed 8-bit values per word
//
// A block covers 32 weights, so each of the 8 work units per block is one word
// and four weights. Dequantisation is `d * qs`, matching `dequantize_q8_0_block`.
// ---------------------------------------------------------------------------

const Q8_WORDS: u32 = 9u;

fn q8_unit_weights(base: u32, jg: u32) -> vec4<f32> {
    let d = unpack2x16float(W_Q8[base]).x;
    return vec4<f32>(unpack4xI8(W_Q8[base + 1u + jg])) * d;
}

@compute @workgroup_size(64)
fn gemv_q8(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>
) {
    let row0 = (wg_id.y * grid.x + wg_id.x) * ROWS_PER_WG;
    let tid = local_id.x;
    if (row0 >= info.N) { return; }

    let blocks_per_row = info.K / 32u;
    let words_per_row = blocks_per_row * Q8_WORDS;
    let total_units = blocks_per_row * 8u;

    var p0 = 0.0;
    var p1 = 0.0;
    var p2 = 0.0;
    var p3 = 0.0;

    for (var u = tid; u < total_units; u = u + 64u) {
        let blk = u / 8u;
        let jg = u % 8u;
        let ob = blk * 32u + jg * 4u;

        // Loaded once, reused by every row this workgroup owns.
        let a = vec4<f32>(Input[ob], Input[ob + 1u], Input[ob + 2u], Input[ob + 3u]);
        let base = row0 * words_per_row + blk * Q8_WORDS;

        p0 += dot(a, q8_unit_weights(base, jg));
        if (row0 + 1u < info.N) {
            p1 += dot(a, q8_unit_weights(base + words_per_row, jg));
        }
        if (row0 + 2u < info.N) {
            p2 += dot(a, q8_unit_weights(base + words_per_row * 2u, jg));
        }
        if (row0 + 3u < info.N) {
            p3 += dot(a, q8_unit_weights(base + words_per_row * 3u, jg));
        }
    }

    wg_sum4[tid] = vec4<f32>(p0, p1, p2, p3);
    workgroupBarrier();
    for (var s = 32u; s > 0u; s >>= 1u) {
        if (tid < s) {
            wg_sum4[tid] += wg_sum4[tid + s];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        let r = wg_sum4[0];
        Output[row0] = r.x;
        if (row0 + 1u < info.N) { Output[row0 + 1u] = r.y; }
        if (row0 + 2u < info.N) { Output[row0 + 2u] = r.z; }
        if (row0 + 3u < info.N) { Output[row0 + 3u] = r.w; }
    }
}

// Batched Q8_0: grid is (n, m), one workgroup per output element.
@compute @workgroup_size(256)
fn bmm_q8(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let n = wg_id.x;
    let row = wg_id.y;
    let tid = local_id.x;
    if (n >= info.N || row >= info.M) { return; }

    let blocks_per_row = info.K / 32u;
    let row_base = n * blocks_per_row * Q8_WORDS;
    let total_units = blocks_per_row * 8u;
    let input_offset = row * info.K;

    var partial = 0.0;
    for (var u = tid; u < total_units; u = u + 256u) {
        let blk = u / 8u;
        let jg = u % 8u;
        let ob = input_offset + blk * 32u + jg * 4u;

        let a = vec4<f32>(Input[ob], Input[ob + 1u], Input[ob + 2u], Input[ob + 3u]);
        partial += dot(a, q8_unit_weights(row_base + blk * Q8_WORDS, jg));
    }

    wg_sum[tid] = partial;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            wg_sum[tid] += wg_sum[tid + s];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        Output[row * info.N + n] = wg_sum[0];
    }
}

// Q6_K GEMV
//
// Padded block layout, 53 u32 words:
//   words  0..31 : ql[128], the low 4 bits of each weight
//   words 32..47 : qh[64],  the high 2 bits, four weights per byte
//   words 48..51 : scales[16], signed 8-bit
//   word     52  : d (f16) in the low half
//
// Each work unit covers four consecutive `j` within one half-block, which is 16
// weights drawn from one ql word, one ql word 32 bytes on, and one qh word. Four
// consecutive `j` always share the same scale index, so the scales are read once.
// Unpacking matches `dequantize_q6_k_block` exactly.
// ---------------------------------------------------------------------------

const Q6K_WORDS: u32 = 53u;

fn q6k_scale(base: u32, idx: u32) -> f32 {
    let word = W_Q6K[base + 48u + (idx >> 2u)];
    let b = (word >> ((idx & 3u) * 8u)) & 0xFFu;
    return f32(i32(b << 24u) >> 24u); // scales are signed
}

struct Q6KQuad {
    a: vec4<f32>,
    b: vec4<f32>,
    c: vec4<f32>,
    d: vec4<f32>,
}

fn q6k_unit_weights(base: u32, i: u32, jg: u32) -> Q6KQuad {
    let d = unpack2x16float(W_Q6K[base + 52u]).x;
    let scb = i * 8u + jg / 4u;
    let s0 = d * q6k_scale(base, scb);
    let s1 = d * q6k_scale(base, scb + 2u);
    let s2 = d * q6k_scale(base, scb + 4u);
    let s3 = d * q6k_scale(base, scb + 6u);

    let qlw = W_Q6K[base + i * 16u + jg];
    let qlw2 = W_Q6K[base + i * 16u + jg + 8u];
    let qhw = W_Q6K[base + 32u + i * 8u + jg];

    // Each mask keeps the two high bits in place for its quarter, so the OR lands
    // them directly on top of the low nibble without any per-byte shifting.
    let n0 = unpack4xU8((qlw & 0x0F0F0F0Fu) | ((qhw & 0x03030303u) << 4u));
    let n1 = unpack4xU8((qlw2 & 0x0F0F0F0Fu) | ((qhw & 0x0C0C0C0Cu) << 2u));
    let n2 = unpack4xU8(((qlw >> 4u) & 0x0F0F0F0Fu) | (qhw & 0x30303030u));
    let n3 = unpack4xU8(((qlw2 >> 4u) & 0x0F0F0F0Fu) | ((qhw & 0xC0C0C0C0u) >> 2u));

    let bias = vec4<f32>(32.0);
    var o: Q6KQuad;
    o.a = (vec4<f32>(n0) - bias) * s0;
    o.b = (vec4<f32>(n1) - bias) * s1;
    o.c = (vec4<f32>(n2) - bias) * s2;
    o.d = (vec4<f32>(n3) - bias) * s3;
    return o;
}

@compute @workgroup_size(64)
fn gemv_q6k(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>
) {
    let row0 = (wg_id.y * grid.x + wg_id.x) * ROWS_PER_WG;
    let tid = local_id.x;
    if (row0 >= info.N) { return; }

    let blocks_per_row = info.K / 256u;
    let words_per_row = blocks_per_row * Q6K_WORDS;
    let total_units = blocks_per_row * 16u;

    var p0 = 0.0;
    var p1 = 0.0;
    var p2 = 0.0;
    var p3 = 0.0;

    for (var u = tid; u < total_units; u = u + 64u) {
        let blk = u / 16u;
        let ub = u % 16u;
        let i = ub / 8u;
        let jg = ub % 8u;
        let ob = blk * 256u + i * 128u + jg * 4u;

        // Loaded once, reused by every row this workgroup owns.
        let a0 = vec4<f32>(Input[ob], Input[ob + 1u], Input[ob + 2u], Input[ob + 3u]);
        let a1 = vec4<f32>(Input[ob + 32u], Input[ob + 33u], Input[ob + 34u], Input[ob + 35u]);
        let a2 = vec4<f32>(Input[ob + 64u], Input[ob + 65u], Input[ob + 66u], Input[ob + 67u]);
        let a3 = vec4<f32>(Input[ob + 96u], Input[ob + 97u], Input[ob + 98u], Input[ob + 99u]);

        let base = row0 * words_per_row + blk * Q6K_WORDS;

        let w0 = q6k_unit_weights(base, i, jg);
        p0 += dot(a0, w0.a) + dot(a1, w0.b) + dot(a2, w0.c) + dot(a3, w0.d);
        if (row0 + 1u < info.N) {
            let w1 = q6k_unit_weights(base + words_per_row, i, jg);
            p1 += dot(a0, w1.a) + dot(a1, w1.b) + dot(a2, w1.c) + dot(a3, w1.d);
        }
        if (row0 + 2u < info.N) {
            let w2 = q6k_unit_weights(base + words_per_row * 2u, i, jg);
            p2 += dot(a0, w2.a) + dot(a1, w2.b) + dot(a2, w2.c) + dot(a3, w2.d);
        }
        if (row0 + 3u < info.N) {
            let w3 = q6k_unit_weights(base + words_per_row * 3u, i, jg);
            p3 += dot(a0, w3.a) + dot(a1, w3.b) + dot(a2, w3.c) + dot(a3, w3.d);
        }
    }

    wg_sum4[tid] = vec4<f32>(p0, p1, p2, p3);
    workgroupBarrier();
    for (var s = 32u; s > 0u; s >>= 1u) {
        if (tid < s) {
            wg_sum4[tid] += wg_sum4[tid + s];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        let r = wg_sum4[0];
        Output[row0] = r.x;
        if (row0 + 1u < info.N) { Output[row0 + 1u] = r.y; }
        if (row0 + 2u < info.N) { Output[row0 + 2u] = r.z; }
        if (row0 + 3u < info.N) { Output[row0 + 3u] = r.w; }
    }
}

// Batched Q6_K: grid is (n, m), one workgroup per output element.
@compute @workgroup_size(256)
fn bmm_q6k(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let n = wg_id.x;
    let row = wg_id.y;
    let tid = local_id.x;
    if (n >= info.N || row >= info.M) { return; }

    let blocks_per_row = info.K / 256u;
    let row_base = n * blocks_per_row * Q6K_WORDS;
    let total_units = blocks_per_row * 16u;
    let input_offset = row * info.K;

    var partial = 0.0;
    for (var u = tid; u < total_units; u = u + 256u) {
        let blk = u / 16u;
        let ub = u % 16u;
        let i = ub / 8u;
        let jg = ub % 8u;
        let ob = input_offset + blk * 256u + i * 128u + jg * 4u;

        let a0 = vec4<f32>(Input[ob], Input[ob + 1u], Input[ob + 2u], Input[ob + 3u]);
        let a1 = vec4<f32>(Input[ob + 32u], Input[ob + 33u], Input[ob + 34u], Input[ob + 35u]);
        let a2 = vec4<f32>(Input[ob + 64u], Input[ob + 65u], Input[ob + 66u], Input[ob + 67u]);
        let a3 = vec4<f32>(Input[ob + 96u], Input[ob + 97u], Input[ob + 98u], Input[ob + 99u]);

        let w = q6k_unit_weights(row_base + blk * Q6K_WORDS, i, jg);
        partial += dot(a0, w.a) + dot(a1, w.b) + dot(a2, w.c) + dot(a3, w.d);
    }

    wg_sum[tid] = partial;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            wg_sum[tid] += wg_sum[tid + s];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        Output[row * info.N + n] = wg_sum[0];
    }
}
