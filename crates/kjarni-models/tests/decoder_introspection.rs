//! What the decoder computes layer by layer, checked against what it should.
//!
//! `CpuDecoder::forward_layers` takes a layer range, so an intermediate hidden state
//! can be projected through the same `lm_head` the final layer uses. That is the
//! logit lens, and it needs no changes to the engine.
//!
//! These guard two things. That the geometry accessors report the model's real shape:
//! `num_attention_heads`, `hidden_size` and `head_dim` on `LlamaCpuDecoder` all
//! returned a hardcoded 0 until a probe asked and got "0 heads, hidden 0" back for a
//! 24-layer model. Nothing in the engine consumed them, so the stubs were invisible.
//! And that the whole stack still answers a factual prompt correctly end to end.
//!
//! Ignored by default: they load a real model from the local cache.
//! Run with: cargo test --release -p kjarni-models --test logit_lens_probe -- --ignored --nocapture

use kjarni_models::models::qwen::QwenModel;
use kjarni_transformers::models::registry::ModelType;
use kjarni_transformers::{Device, LanguageModel};

#[tokio::test]
#[ignore = "loads a real model from the local cache"]
async fn logit_lens_across_depth() {
    let model = QwenModel::from_registry(
        ModelType::Qwen2_5_0_5B_Instruct,
        None,
        Device::Cpu,
        None,
        None,
    )
    .await
    .expect("load qwen2.5-0.5b");

    let pipe = model.pipeline();
    let dec = pipe.cpu_decoder().expect("cpu decoder");
    let tok = model.tokenizer();

    let prompt = "The capital of France is";
    let enc = tok.encode(prompt, false).expect("tokenize");
    let ids: Vec<u32> = enc.get_ids().to_vec();
    let seq = ids.len();
    println!("\nprompt   {prompt:?}");
    println!("tokens   {ids:?}");
    println!(
        "geometry {} layers, {} heads, hidden {}, vocab {}",
        dec.num_layers(),
        dec.num_attention_heads(),
        dec.hidden_size(),
        pipe.lm_head().vocab_size()
    );

    let token_arr = ndarray::Array2::from_shape_vec((1, seq), ids.clone()).unwrap();
    let mut hidden = pipe
        .embeddings()
        .embed_cpu(&token_arr, None, 0)
        .expect("embed");
    let mask = kjarni_transformers::utils::create_full_attention_mask(1, seq);

    // Step one layer at a time, projecting the running hidden state through the
    // same lm_head the final layer uses. That is the logit lens.
    println!(
        "\n{:<6} {:<28} {:>8} {:>9}",
        "layer", "top-5 after this layer", "top prob", "entropy"
    );
    let n = dec.num_layers();
    let mut json: Vec<String> = Vec::new();
    let mut final_top = String::new();
    for l in 0..n {
        hidden = dec
            .forward_layers(&hidden, &mask, 0, None, l, l + 1)
            .expect("forward one layer");

        let normed = dec.final_norm(&hidden).expect("final norm");
        let logits = pipe.lm_head().forward_cpu(&normed).expect("lm_head");

        // last position only: that is the one predicting the next token
        let row = logits.slice(ndarray::s![0, seq - 1, ..]).to_owned();
        let max = row.iter().cloned().fold(f32::MIN, f32::max);
        let exp: Vec<f32> = row.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        let probs: Vec<f32> = exp.iter().map(|v| v / sum).collect();

        let mut idx: Vec<usize> = (0..probs.len()).collect();
        idx.sort_by(|a, b| probs[*b].partial_cmp(&probs[*a]).unwrap());

        let entropy: f32 = -probs
            .iter()
            .filter(|p| **p > 0.0)
            .map(|p| p * p.log2())
            .sum::<f32>();
        let top: Vec<String> = idx[..5]
            .iter()
            .map(|i| {
                tok.decode(&[*i as u32], false)
                    .unwrap_or_default()
                    .trim()
                    .to_string()
            })
            .collect();

        println!(
            "{:<6} {:<28} {:>8.4} {:>9.2}",
            l + 1,
            top.join(" "),
            probs[idx[0]],
            entropy
        );
        final_top = top[0].clone();

        let row: Vec<String> = idx[..6]
            .iter()
            .map(|i| {
                let t = tok.decode(&[*i as u32], false).unwrap_or_default();
                format!("[{:?},{:.5}]", t, probs[*i])
            })
            .collect();
        json.push(format!(
            "{{\"layer\":{},\"entropy\":{:.4},\"top\":[{}]}}",
            l + 1,
            entropy,
            row.join(",")
        ));
    }

    println!("\nJSON_BEGIN");
    println!(
        "{{\"prompt\":{:?},\"layers\":[{}]}}",
        prompt,
        json.join(",")
    );
    println!("JSON_END");

    // End to end: the last layer must actually answer the question.
    assert!(
        final_top.contains("Paris"),
        "final layer predicted {final_top:?}, expected Paris"
    );
}

/// The same lens, but at every position rather than only the last.
///
/// If factual recall works the way the literature describes, the answer is written
/// into the residual stream at the *subject* position first and only copied to the
/// final position late. That predicts "Paris" appears over "France" before it
/// appears over "is".
#[tokio::test]
#[ignore = "loads a real model from the local cache"]
async fn where_the_fact_lives() {
    let model = QwenModel::from_registry(
        ModelType::Qwen2_5_0_5B_Instruct,
        None,
        Device::Cpu,
        None,
        None,
    )
    .await
    .expect("load qwen2.5-0.5b");

    let pipe = model.pipeline();
    let dec = pipe.cpu_decoder().expect("cpu decoder");
    let tok = model.tokenizer();

    let prompt = "The capital of France is";
    let ids: Vec<u32> = tok
        .encode(prompt, false)
        .expect("tokenize")
        .get_ids()
        .to_vec();
    let seq = ids.len();

    let names: Vec<String> = ids
        .iter()
        .map(|i| {
            tok.decode(&[*i], false)
                .unwrap_or_default()
                .trim()
                .to_string()
        })
        .collect();

    let token_arr = ndarray::Array2::from_shape_vec((1, seq), ids.clone()).unwrap();
    let mut hidden = pipe
        .embeddings()
        .embed_cpu(&token_arr, None, 0)
        .expect("embed");
    let mask = kjarni_transformers::utils::create_full_attention_mask(1, seq);

    let mut last_row: Vec<String> = Vec::new();
    println!("\ntop token at each position, after each layer");
    print!("{:<6}", "layer");
    for n in &names {
        print!("{:>14}", n);
    }
    println!();

    for l in 0..dec.num_layers() {
        hidden = dec
            .forward_layers(&hidden, &mask, 0, None, l, l + 1)
            .expect("fwd");
        let normed = dec.final_norm(&hidden).expect("norm");
        let logits = pipe.lm_head().forward_cpu(&normed).expect("lm_head");

        print!("{:<6}", l + 1);
        for pos in 0..seq {
            let row = logits.slice(ndarray::s![0, pos, ..]);
            let best = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            let t = tok.decode(&[best as u32], false).unwrap_or_default();
            let t = t.trim().replace('\n', "\\n");
            let t: String = t.chars().take(12).collect();
            let shown = if t.is_empty() { "_".to_string() } else { t };
            print!("{shown:>14}");
            if l + 1 == dec.num_layers() {
                last_row.push(shown);
            }
        }
        println!();
    }

    // A causal model predicts at every position. By the final layer each one should
    // name its true continuation, which is what makes the mid-stack noise a property
    // of the probe rather than of the model.
    assert_eq!(
        last_row.last().map(String::as_str),
        Some("Paris"),
        "final position should predict Paris, got {last_row:?}"
    );
    assert_eq!(
        last_row.get(3).map(String::as_str),
        Some("is"),
        "the France position should predict 'is', got {last_row:?}"
    );
}

/// Is the garbage a property of the model, or of the instrument?
///
/// `lm_head` is trained to read the *final* layer's representation. If early
/// hidden states simply live in a different basis, the noise says more about the
/// probe than about the model. Measuring each layer's cosine similarity to the
/// final hidden state separates the two: a late, sharp rise means the readout only
/// becomes valid at the end.
#[tokio::test]
#[ignore = "loads a real model from the local cache"]
async fn distance_to_final_representation() {
    let model = QwenModel::from_registry(
        ModelType::Qwen2_5_0_5B_Instruct,
        None,
        Device::Cpu,
        None,
        None,
    )
    .await
    .expect("load qwen2.5-0.5b");

    let pipe = model.pipeline();
    let dec = pipe.cpu_decoder().expect("cpu decoder");
    let tok = model.tokenizer();

    let ids: Vec<u32> = tok
        .encode("The capital of France is", false)
        .expect("tokenize")
        .get_ids()
        .to_vec();
    let seq = ids.len();
    let arr = ndarray::Array2::from_shape_vec((1, seq), ids).unwrap();
    let mut hidden = pipe.embeddings().embed_cpu(&arr, None, 0).expect("embed");
    let mask = kjarni_transformers::utils::create_full_attention_mask(1, seq);

    // Keep the last-position hidden state after every layer.
    let mut states: Vec<Vec<f32>> = Vec::new();
    for l in 0..dec.num_layers() {
        hidden = dec
            .forward_layers(&hidden, &mask, 0, None, l, l + 1)
            .expect("fwd");
        states.push(hidden.slice(ndarray::s![0, seq - 1, ..]).to_vec());
    }
    let final_state = states.last().unwrap().clone();

    let cos = |a: &[f32], b: &[f32]| {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb)
    };

    println!("\n{:<6} {:>10} {:>12}", "layer", "cos->final", "norm");
    let mut sims = Vec::new();
    let mut norms = Vec::new();
    for (i, s) in states.iter().enumerate() {
        let norm: f32 = s.iter().map(|x| x * x).sum::<f32>().sqrt();
        let sim = cos(s, &final_state);
        println!("{:<6} {:>10.4} {:>12.1}", i + 1, sim, norm);
        sims.push(sim);
        norms.push(norm);
    }

    // The last layer is trivially identical to itself.
    assert!(
        (sims[sims.len() - 1] - 1.0).abs() < 1e-4,
        "final layer must match itself"
    );
    // Early states sit close to orthogonal to their own final form, which is why
    // reading them through lm_head yields noise: the head expects the final basis.
    assert!(
        sims[0] < 0.4,
        "layer 1 alignment {} unexpectedly high",
        sims[0]
    );
    // Every layer adds to the residual stream rather than replacing it.
    assert!(
        norms[norms.len() - 2] > norms[0] * 5.0,
        "residual norm should accumulate: {} -> {}",
        norms[0],
        norms[norms.len() - 2]
    );
}

/// The streaming prefill must agree with the materialising one, and be faster.
#[tokio::test]
#[ignore = "loads a real model"]
async fn streaming_prefill_matches_and_is_faster() {
    use std::time::Instant;

    let model = QwenModel::from_registry(
        ModelType::Qwen2_5_0_5B_Instruct,
        None,
        Device::Cpu,
        None,
        None,
    )
    .await
    .expect("load");
    let pipe = model.pipeline();
    let dec = pipe.cpu_decoder().expect("dec");
    let tok = model.tokenizer();

    let para = "Retrieval augmented generation finds passages close to the question. ";
    let unit: Vec<u32> = tok.encode(para, false).unwrap().get_ids().to_vec();

    println!(
        "\n  {:>6}  {:>11}  {:>11}  {:>8}  {:>12}",
        "seq", "materialised", "streaming", "gain", "max |diff|"
    );

    for seq in [512usize, 1024, 2048, 4096] {
        let mut ids = Vec::new();
        while ids.len() < seq {
            ids.extend_from_slice(&unit);
        }
        ids.truncate(seq);
        let arr = ndarray::Array2::from_shape_vec((1, seq), ids).unwrap();
        let mask = kjarni_transformers::utils::create_full_attention_mask(1, seq);
        let h0 = pipe.embeddings().embed_cpu(&arr, None, 0).expect("embed");

        let run = |label: &str| -> (f64, ndarray::Array3<f32>) {
            // SAFETY: single-threaded test setup, before any worker touches it.
            unsafe {
                if label == "off" {
                    std::env::set_var("KJARNI_NO_STREAMING_ATTN", "1");
                } else {
                    std::env::remove_var("KJARNI_NO_STREAMING_ATTN");
                }
            }
            let mut best = f64::MAX;
            let mut out = None;
            for _ in 0..2 {
                let t = Instant::now();
                let o = dec
                    .forward_layers(&h0, &mask, 0, None, 0, dec.num_layers())
                    .expect("fwd");
                let e = t.elapsed().as_secs_f64();
                if e < best {
                    best = e;
                }
                out = Some(o);
            }
            (best, out.unwrap())
        };

        // the escape hatch is read once per process, so measure in two processes
        // worth of order: streaming first while the OnceLock is unset
        let (t_stream, o_stream) = run("on");
        let (t_whole, o_whole) = run("off");

        let diff = o_stream
            .iter()
            .zip(o_whole.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);

        println!(
            "  {:>6}  {:>9.2} s  {:>9.2} s  {:>7.2}x  {:>12.2e}",
            seq,
            t_whole,
            t_stream,
            t_whole / t_stream,
            diff
        );

        // The half of the name that is a correctness claim. Streaming attention
        // reassociates the softmax, so it is not bit-identical and must not be
        // asserted as such; what it must not do is change the answer.
        assert!(
            diff < 1e-2,
            "streaming attention changed the output at seq {seq}: worst element {diff:.3e}"
        );
    }
}

/// Can a prompt be prefilled in two pieces onto one cache, and give the same
/// answer as prefilling it whole? That is the whole question behind prefix
/// caching: if it holds, reusing a cache across turns is plumbing rather than
/// new maths.
#[tokio::test]
#[ignore = "loads a real model"]
async fn split_prefill_matches_whole_prefill() {
    use kjarni_transformers::decoder::traits::DecoderLanguageModel;
    use std::time::Instant;

    let model = QwenModel::from_registry(
        ModelType::Qwen2_5_0_5B_Instruct,
        None,
        Device::Cpu,
        None,
        None,
    )
    .await
    .expect("load");
    let pipe = model.pipeline();
    let dec = pipe.cpu_decoder().expect("dec");
    let ops = model.decoder_cpu_ops().expect("ops");
    let tok = model.tokenizer();

    let para = "Retrieval augmented generation finds passages close to the question. \
                The capital of France is Paris and the capital of Germany is Berlin. ";
    let unit: Vec<u32> = tok.encode(para, false).unwrap().get_ids().to_vec();

    let total = 2048usize;
    let prefix = 1984usize; // what a second turn would already have cached
    let suffix = total - prefix; // only the new question

    let mut ids = Vec::new();
    while ids.len() < total {
        ids.extend_from_slice(&unit);
    }
    ids.truncate(total);

    let top_of = |logits: &ndarray::Array3<f32>, pos: usize| -> (usize, f32) {
        let row = logits.slice(ndarray::s![0, pos, ..]);
        let b = row
            .iter()
            .enumerate()
            .max_by(|a, c| a.1.partial_cmp(c.1).unwrap())
            .unwrap();
        (b.0, *b.1)
    };

    // --- whole prompt in one pass ---
    let arr = ndarray::Array2::from_shape_vec((1, total), ids.clone()).unwrap();
    let mut c1 = model.new_cache(1, total + 8, 1).expect("cache");
    let t0 = Instant::now();
    let h = ops.embed(&arr, 0).expect("embed");
    let mask = ops.get_attention_mask(total, 0).expect("mask");
    let out = dec.forward(&h, &mask, 0, Some(c1.as_mut())).expect("whole");
    let t_whole = t0.elapsed().as_secs_f64();
    let l_whole = ops.project_to_logits(&out).expect("head");
    let (tok_whole, val_whole) = top_of(&l_whole, total - 1);

    // --- prefix first, then only the suffix onto the warm cache ---
    let pre = ndarray::Array2::from_shape_vec((1, prefix), ids[..prefix].to_vec()).unwrap();
    let mut c2 = model.new_cache(1, total + 8, 1).expect("cache");
    let hp = ops.embed(&pre, 0).expect("embed");
    let mp = ops.get_attention_mask(prefix, 0).expect("mask");
    let t_pre = Instant::now();
    let _ = dec.forward(&hp, &mp, 0, Some(c2.as_mut())).expect("prefix");
    let t_prefix = t_pre.elapsed().as_secs_f64();

    let suf = ndarray::Array2::from_shape_vec((1, suffix), ids[prefix..].to_vec()).unwrap();
    let hs = ops.embed(&suf, prefix).expect("embed suffix");
    let ms = ops.get_attention_mask(suffix, prefix).expect("mask suffix");
    let t1 = Instant::now();
    let out2 = dec
        .forward(&hs, &ms, prefix, Some(c2.as_mut()))
        .expect("suffix");
    let t_suffix = t1.elapsed().as_secs_f64();
    let l_split = ops.project_to_logits(&out2).expect("head");
    let (tok_split, val_split) = top_of(&l_split, suffix - 1);

    println!("\n  prompt {total} tokens, prefix {prefix} cached, suffix {suffix} new\n");
    println!("    whole prompt      {t_whole:>7.2}s   top={tok_whole} ({val_whole:.4})");
    println!("    prefix pass       {t_prefix:>7.2}s   (paid once, reused after)");
    println!("    suffix only       {t_suffix:>7.2}s   top={tok_split} ({val_split:.4})");
    println!(
        "    a second turn costs {:.2}s instead of {:.2}s  ->  {:.0}x",
        t_suffix,
        t_whole,
        t_whole / t_suffix
    );

    // The invariant prefix caching rests on: prefilling a prompt in two passes
    // must land on the same next token as prefilling it whole. Without this the
    // whole feature is silently wrong, and the name of this test claimed to check
    // it long before anything here actually did.
    assert_eq!(
        tok_whole, tok_split,
        "split prefill chose a different next token ({tok_whole} vs {tok_split}); \
         a cached prefix is not equivalent to prefilling the whole prompt"
    );
    assert!(
        (val_whole - val_split).abs() < 1e-2,
        "same token but the logit moved: {val_whole} vs {val_split}"
    );
}

/// What does 4-bit quantization actually change about the weights?
#[tokio::test]
#[ignore = "reads a local GGUF and safetensors of the same model"]
async fn quantization_error_against_bf16() {
    use kjarni_transformers::weights::ModelWeights;

    let gguf = ModelWeights::new(std::path::Path::new(
        "/home/olafurj/.cache/kjarni/llama-3.2-1b-instruct-q4_k_m/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    )).expect("gguf");
    let st = ModelWeights::new(std::path::Path::new(
        "/home/olafurj/.cache/kjarni/meta-llama_Llama-3.2-1B-Instruct",
    ))
    .expect("safetensors");

    // ffn_gate, not a q/k projection: GGUF row-permutes those for RoPE and the
    // rows would not line up.
    let q = gguf
        .get_typed_tensor("blk.0.ffn_gate.weight")
        .expect("q4k tensor");
    let raw = st
        .loader()
        .get_raw("model.layers.0.mlp.gate_proj.weight")
        .expect("bf16");
    #[allow(deprecated)]
    let orig = raw.to_ndarray_f32().expect("f32");

    // get_typed_tensor hands back the packed blocks; dequantizing is what puts
    // them back on the same footing as the bf16 original.
    let deq = match q {
        kjarni_transformers::tensor::CpuTensor::F32(a) => {
            a.into_dimensionality::<ndarray::Ix2>().expect("2d")
        }
        kjarni_transformers::tensor::CpuTensor::Q4_K(m) => m.dequantize().expect("dequantize"),
        other => panic!("unexpected dtype: {:?}", other.dtype()),
    };

    let a: Vec<f32> = orig.iter().copied().collect();
    let b: Vec<f32> = deq.iter().copied().collect();
    assert_eq!(
        a.len(),
        b.len(),
        "shape mismatch: {:?} vs {:?}",
        orig.shape(),
        deq.shape()
    );

    let n = a.len();
    let mut abs_err = 0f64;
    let mut max_err = 0f32;
    let mut sq = 0f64;
    let mut sig = 0f64;
    let mut exact = 0usize;
    let mut distinct: std::collections::HashSet<u32> = Default::default();
    for (x, y) in a.iter().zip(b.iter()) {
        let e = (x - y).abs();
        abs_err += e as f64;
        if e > max_err {
            max_err = e;
        }
        sq += (e as f64) * (e as f64);
        sig += (*x as f64) * (*x as f64);
        if e == 0.0 {
            exact += 1;
        }
        distinct.insert(y.to_bits());
    }

    println!("\n  layer 0 ffn_gate  {} weights", n);
    println!("    mean |error|      {:.6}", abs_err / n as f64);
    println!("    max  |error|      {:.6}", max_err);
    println!("    rms error         {:.6}", (sq / n as f64).sqrt());
    println!("    signal rms        {:.6}", (sig / n as f64).sqrt());
    println!("    error / signal    {:.2}%", (sq / sig).sqrt() * 100.0);
    println!(
        "    bit-exact         {} of {} ({:.3}%)",
        exact,
        n,
        exact as f64 / n as f64 * 100.0
    );
    println!(
        "    distinct values   {} after quantization",
        distinct.len()
    );
    println!("    (bf16 holds up to 65,536 distinct values per sign)");
}
