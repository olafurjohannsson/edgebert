//! Attention that never materialises the scores tensor.
//!
//! The straightforward path computes `[batch, heads, seq, seq]`, masks it, softmaxes
//! it, then multiplies by V. At a 4096 token prompt on Qwen2.5-0.5B that tensor is
//! 939 MB, and it crosses the memory bus four times per layer: written by `q @ kᵀ`,
//! read and rewritten by the softmax, read again by `scores @ v`. Measured on a
//! 13700, the softmax alone cost 0.8 s per layer, 19 s across the model.
//!
//! This keeps a running maximum and a running sum per query row, so a block of
//! scores can be consumed and discarded before the next block is computed. Each new
//! key block rescales the accumulator by `exp(m_old - m_new)` before adding its own
//! contribution, which is what makes normalising early legal.
//!
//! Two details matter as much as the algorithm, both found by measurement:
//! work is partitioned over (head, query block) rather than heads alone, because
//! 14 heads leave a third of a 24 core machine idle; and the inner loops run on
//! raw slices, because ndarray's `[[i, j]]` costs a bounds check and a stride
//! multiply per element in the hottest loop of the layer.

use ndarray::{Array4, ArrayView3};
use rayon::prelude::*;

/// Query rows per job. Anything from 32 to 256 measured within 8% of each other
/// once the tile fits in L2, so this is not a delicate number.
const BLOCK_Q: usize = 64;
/// Key columns held at once.
const BLOCK_K: usize = 256;

/// Streaming causal attention over a KV cache.
///
/// `q_heads` is `[batch, heads, seq, head_dim]`. `k_cache` and `v_cache` are
/// `[batch, total_len, kv_heads * head_dim]` and are read in place: grouped-query
/// models are handled by mapping query head `h` to kv head `h / n_rep` rather than
/// expanding the cache, which avoids a copy the materialising path has to make.
///
/// Returns `[batch, heads, seq, head_dim]`.
pub fn streaming_causal_attention(
    q_heads: &Array4<f32>,
    k_cache: &ArrayView3<f32>,
    v_cache: &ArrayView3<f32>,
    num_kv_heads: usize,
    scale: f32,
    start_write: usize,
) -> Array4<f32> {
    let (batch, heads, seq, hd) = q_heads.dim();
    let total_len = k_cache.shape()[1];
    let n_rep = heads / num_kv_heads;
    let kv_stride = num_kv_heads * hd;

    let mut out = Array4::<f32>::zeros((batch, heads, seq, hd));
    let nblocks = seq.div_ceil(BLOCK_Q);

    for b in 0..batch {
        let q_all = q_heads.slice(ndarray::s![b, .., .., ..]);
        let q_all = q_all.as_standard_layout();
        let qs = q_all.as_slice().expect("q contiguous");
        let k_b = k_cache.slice(ndarray::s![b, .., ..]);
        let k_b = k_b.as_standard_layout();
        let ks = k_b.as_slice().expect("k contiguous");
        let v_b = v_cache.slice(ndarray::s![b, .., ..]);
        let v_b = v_b.as_standard_layout();
        let vs = v_b.as_slice().expect("v contiguous");

        let jobs: Vec<(usize, usize)> = (0..heads)
            .flat_map(|h| (0..nblocks).map(move |blk| (h, blk)))
            .collect();

        let pieces: Vec<(usize, usize, Vec<f32>)> = jobs
            .into_par_iter()
            .map(|(h, blk)| {
                let qh = &qs[h * seq * hd..(h + 1) * seq * hd];
                let kv_off = (h / n_rep) * hd;

                let qstart = blk * BLOCK_Q;
                let qend = (qstart + BLOCK_Q).min(seq);
                let rows = qend - qstart;

                let mut m = vec![f32::NEG_INFINITY; rows];
                let mut l = vec![0f32; rows];
                let mut acc = vec![0f32; rows * hd];
                let mut s_blk = vec![0f32; rows * BLOCK_K];

                let mut kstart = 0usize;
                while kstart < total_len {
                    // A query at sequence position i sees keys 0..=start_write+i.
                    if kstart > start_write + qend - 1 {
                        break;
                    }
                    let kend = (kstart + BLOCK_K).min(total_len);
                    let kw = kend - kstart;

                    for i in 0..rows {
                        let qrow = &qh[(qstart + i) * hd..(qstart + i) * hd + hd];
                        let srow = &mut s_blk[i * BLOCK_K..i * BLOCK_K + kw];
                        let limit = start_write + qstart + i;
                        let visible = if limit >= kstart {
                            (limit - kstart + 1).min(kw)
                        } else {
                            0
                        };

                        // Four key columns per pass: one dot product at a time is a
                        // reduction that runs at FMA latency and reloads the q row
                        // for every column.
                        let quads = visible / 4;
                        for c in 0..quads {
                            let j = c * 4;
                            let base = (kstart + j) * kv_stride + kv_off;
                            let k0 = &ks[base..base + hd];
                            let k1 = &ks[base + kv_stride..base + kv_stride + hd];
                            let k2 = &ks[base + 2 * kv_stride..base + 2 * kv_stride + hd];
                            let k3 = &ks[base + 3 * kv_stride..base + 3 * kv_stride + hd];
                            let (mut d0, mut d1, mut d2, mut d3) = (0f32, 0f32, 0f32, 0f32);
                            for d in 0..hd {
                                let qv = qrow[d];
                                d0 += qv * k0[d];
                                d1 += qv * k1[d];
                                d2 += qv * k2[d];
                                d3 += qv * k3[d];
                            }
                            srow[j] = d0 * scale;
                            srow[j + 1] = d1 * scale;
                            srow[j + 2] = d2 * scale;
                            srow[j + 3] = d3 * scale;
                        }
                        for (j, sv) in srow.iter_mut().enumerate().take(visible).skip(quads * 4) {
                            let base = (kstart + j) * kv_stride + kv_off;
                            let krow = &ks[base..base + hd];
                            let mut dot = 0f32;
                            for (a, bb) in qrow.iter().zip(krow.iter()) {
                                dot += a * bb;
                            }
                            *sv = dot * scale;
                        }
                        for sv in srow[visible..].iter_mut() {
                            *sv = f32::NEG_INFINITY;
                        }
                    }

                    for i in 0..rows {
                        let srow = &mut s_blk[i * BLOCK_K..i * BLOCK_K + kw];
                        let mut bmax = f32::NEG_INFINITY;
                        for &x in srow.iter() {
                            if x > bmax {
                                bmax = x;
                            }
                        }
                        if bmax == f32::NEG_INFINITY {
                            continue;
                        }

                        let m_new = if m[i] > bmax { m[i] } else { bmax };
                        let corr = if m[i] == f32::NEG_INFINITY {
                            0.0
                        } else {
                            (m[i] - m_new).exp()
                        };

                        let mut sum = 0f32;
                        for x in srow.iter_mut() {
                            let p = if *x == f32::NEG_INFINITY {
                                0.0
                            } else {
                                (*x - m_new).exp()
                            };
                            *x = p;
                            sum += p;
                        }

                        let arow = &mut acc[i * hd..i * hd + hd];
                        if corr != 1.0 {
                            for a in arow.iter_mut() {
                                *a *= corr;
                            }
                        }
                        for (j, &p) in srow.iter().enumerate() {
                            if p == 0.0 {
                                continue;
                            }
                            let base = (kstart + j) * kv_stride + kv_off;
                            let vrow = &vs[base..base + hd];
                            for (a, &vv) in arow.iter_mut().zip(vrow.iter()) {
                                *a += p * vv;
                            }
                        }
                        l[i] = l[i] * corr + sum;
                        m[i] = m_new;
                    }
                    kstart = kend;
                }

                let mut piece = vec![0f32; rows * hd];
                for i in 0..rows {
                    let inv = if l[i] > 0.0 { 1.0 / l[i] } else { 0.0 };
                    for d in 0..hd {
                        piece[i * hd + d] = acc[i * hd + d] * inv;
                    }
                }
                (h, qstart, piece)
            })
            .collect();

        for (h, qstart, piece) in pieces {
            let rows = piece.len() / hd;
            for i in 0..rows {
                for d in 0..hd {
                    out[[b, h, qstart + i, d]] = piece[i * hd + d];
                }
            }
        }
    }

    out
}

/// Single-token decode attention, reading the KV cache in its natural layout.
///
/// The materialising path takes a transposed `[dim, cache_len]` view and walks `d`
/// with `t` fixed, so consecutive reads are `cache_len * 4` bytes apart — a 16 KB
/// stride at 4096 context. Every access touches a fresh cache line and uses 4 of
/// its 64 bytes. Measured on a 13700 that came to roughly 0.5 GB/s against a bus
/// that does forty times better.
///
/// The cache is `[batch, total_len, kv_heads * head_dim]`, so `k[t]` for one kv
/// head is `head_dim` contiguous floats. Walking `t` outermost reads whole lines.
/// Work is split over heads, because the batch axis is 1 during decode and
/// parallelising over it leaves the whole machine idle.
///
/// Returns `[batch, heads, 1, head_dim]`.
pub fn decode_attention(
    q_heads: &Array4<f32>,
    k_cache: &ArrayView3<f32>,
    v_cache: &ArrayView3<f32>,
    num_kv_heads: usize,
    scale: f32,
) -> Array4<f32> {
    let (batch, heads, _, hd) = q_heads.dim();
    let total_len = k_cache.shape()[1];
    let n_rep = heads / num_kv_heads;
    let kv_stride = num_kv_heads * hd;

    let mut out = Array4::<f32>::zeros((batch, heads, 1, hd));

    for b in 0..batch {
        let q_all = q_heads.slice(ndarray::s![b, .., 0, ..]);
        let q_all = q_all.as_standard_layout();
        let qs = q_all.as_slice().expect("q contiguous");
        let k_b = k_cache.slice(ndarray::s![b, .., ..]);
        let k_b = k_b.as_standard_layout();
        let ks = k_b.as_slice().expect("k contiguous");
        let v_b = v_cache.slice(ndarray::s![b, .., ..]);
        let v_b = v_b.as_standard_layout();
        let vs = v_b.as_slice().expect("v contiguous");

        let pieces: Vec<Vec<f32>> = (0..heads)
            .into_par_iter()
            .map(|h| {
                let qrow = &qs[h * hd..(h + 1) * hd];
                let kv_off = (h / n_rep) * hd;

                // scores over the whole cache, then a single softmax
                let mut scores = vec![0f32; total_len];
                let mut m = f32::NEG_INFINITY;
                for (t, sv) in scores.iter_mut().enumerate() {
                    let base = t * kv_stride + kv_off;
                    let krow = &ks[base..base + hd];
                    let mut dot = 0f32;
                    for (a, bb) in qrow.iter().zip(krow.iter()) {
                        dot += a * bb;
                    }
                    let x = dot * scale;
                    *sv = x;
                    if x > m {
                        m = x;
                    }
                }

                let mut sum = 0f32;
                for x in scores.iter_mut() {
                    let p = (*x - m).exp();
                    *x = p;
                    sum += p;
                }
                let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };

                let mut acc = vec![0f32; hd];
                for (t, &p) in scores.iter().enumerate() {
                    if p == 0.0 {
                        continue;
                    }
                    let w = p * inv;
                    let base = t * kv_stride + kv_off;
                    let vrow = &vs[base..base + hd];
                    for (a, &vv) in acc.iter_mut().zip(vrow.iter()) {
                        *a += w * vv;
                    }
                }
                acc
            })
            .collect();

        for (h, piece) in pieces.into_iter().enumerate() {
            for d in 0..hd {
                out[[b, h, 0, d]] = piece[d];
            }
        }
    }

    out
}
