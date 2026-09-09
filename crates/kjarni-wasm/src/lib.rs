// This crate used to be `#![cfg(target_arch = "wasm32")]`, compiling to nothing on
// the host. Combined with being excluded from the workspace, that meant no
// `cargo test` invocation could reach it — which is how it accumulated a second copy
// of the encoder stack that silently drifted from the engine. The bindings are still
// wasm-only, but the crate now builds natively so its behaviour can be tested.

use anyhow::Result;
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::JsFuture;
#[cfg(target_arch = "wasm32")]
use web_sys::{Response, Window, WorkerGlobalScope};

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};

// The shared engine. Everything below that used to run on this crate's own copy of
// the encoder stack is being moved onto these.
use kjarni_models::{CrossEncoder, SentenceEncoder, SequenceClassifier};
use kjarni_transformers::Conversation;
use kjarni_transformers::decoder::generator::DecoderGenerator;
use kjarni_transformers::decoder::traits::DecoderLanguageModel;
use kjarni_transformers::pipeline::EncoderLoader;
use kjarni_transformers::traits::Device as EngineDevice;
use kjarni_transformers::weights::kjq;

// ─── Debug Logging ───────────────────────────────────────────────

static DEBUG_LOGGING: AtomicBool = AtomicBool::new(false);

#[wasm_bindgen]
pub fn set_debug_logging(enabled: bool) {
    DEBUG_LOGGING.store(enabled, Ordering::Relaxed);
}

macro_rules! klog {
    ($($arg:tt)*) => {
        if DEBUG_LOGGING.load(Ordering::Relaxed) {
            #[cfg(target_arch = "wasm32")]
            web_sys::console::log_1(&format!("[kjarni] {}", format!($($arg)*)).into());
            #[cfg(not(target_arch = "wasm32"))]
            println!("[kjarni] {}", format!($($arg)*));
        }
    };
}

#[cfg(target_arch = "wasm32")]
fn now_ms() -> f64 {
    js_sys::Date::now()
}

/// Host builds have no `Date.now()`; timings only feed debug logging.
#[cfg(not(target_arch = "wasm32"))]
fn now_ms() -> f64 {
    0.0
}

// The encoder stack that used to live here — Model, BertLayer, MultiHeadAttention,
// FeedForward, LayerNorm, Config and their matmul/softmax/pooling helpers — was a
// second implementation of what kjarni-transformers already does. It existed only
// because the engine did not build for wasm32. It does now, so the bindings below
// call the engine and this copy is gone.

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let dot: f32 = a[..n].iter().zip(&b[..n]).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b + 1e-8)
}

// ─── WASM Bindings ───────────────────────────────────────────────

use kjarni_rag::{SearchIndex, SplitterConfig, TextSplitter};

/// Load a `.kjq` container into the engine's sentence encoder.
///
/// Replaces this crate's own weight parsing and tokenizer, so the browser runs the
/// same code path as a native build.
fn load_kjq_encoder(model_data: &[u8]) -> Result<SentenceEncoder, JsValue> {
    let unpacked = kjq::unpack(model_data).map_err(|e| JsValue::from_str(&e.to_string()))?;
    EncoderLoader::load_from_bytes::<SentenceEncoder>(
        &unpacked.safetensors,
        &unpacked.config_json,
        unpacked.tokenizer_json.as_bytes(),
        EngineDevice::Cpu,
        None,
        None,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Encode a batch with L2 normalization, driving the engine's async API to
/// completion.
///
/// The wasm CPU encode path contains no `.await`, so the future is ready on its
/// first poll and this never parks the browser's main thread. See `WasmModel::encode`.
fn encode_all(model: &SentenceEncoder, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
    futures::executor::block_on(model.encode_batch(texts))
}

// ─── WasmIndexBuilder ────────────────────────────────────────────

#[wasm_bindgen]
pub struct WasmIndexBuilder {
    model: SentenceEncoder,
    index: SearchIndex,
    splitter: TextSplitter,
}

#[wasm_bindgen]
impl WasmIndexBuilder {
    #[wasm_bindgen]
    pub fn new(model_data: &[u8]) -> Result<WasmIndexBuilder, JsValue> {
        let t0 = now_ms();
        let model = load_kjq_encoder(model_data)?;

        klog!(
            "WasmIndexBuilder::new: loaded model in {:.1}ms",
            now_ms() - t0
        );

        Ok(WasmIndexBuilder {
            model,
            index: SearchIndex::with_dimension(384),
            splitter: TextSplitter::new(SplitterConfig::default()),
        })
    }
    #[wasm_bindgen]
    pub fn add_chunk(
        &mut self,
        text: String,
        embedding: Vec<f32>,
        source: String,
        chunk_index: usize,
    ) -> Result<(), JsValue> {
        let mut meta = std::collections::HashMap::new();
        meta.insert("source".to_string(), source);
        meta.insert("chunk_index".to_string(), chunk_index.to_string());

        self.index
            .add_document(text, embedding, Some(meta))
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(())
    }

    #[wasm_bindgen]
    pub fn add_file(&mut self, text: &str, source_path: &str) -> Result<usize, JsValue> {
        let t0 = now_ms();

        let chunks = self.splitter.split(text);
        if chunks.is_empty() {
            return Ok(0);
        }

        let t_enc = now_ms();
        let chunk_refs: Vec<&str> = chunks.iter().map(|s| s.as_str()).collect();
        let embeddings =
            encode_all(&self.model, &chunk_refs).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let enc_ms = now_ms() - t_enc;

        let t_idx = now_ms();
        for (i, (chunk, embedding)) in chunks.iter().zip(embeddings).enumerate() {
            let mut meta = std::collections::HashMap::new();
            meta.insert("source".to_string(), source_path.to_string());
            meta.insert("chunk_index".to_string(), i.to_string());

            self.index
                .add_document(chunk.clone(), embedding, Some(meta))
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        let idx_ms = now_ms() - t_idx;

        klog!(
            "add_file: {} chunks, encode={:.1}ms, index={:.1}ms, total={:.1}ms | {}",
            chunks.len(),
            enc_ms,
            idx_ms,
            now_ms() - t0,
            source_path
        );

        Ok(chunks.len())
    }

    #[wasm_bindgen]
    pub fn finish(&self) -> Result<Vec<u8>, JsValue> {
        let t0 = now_ms();
        let mut buf = Vec::new();
        self.index
            .save_binary(&mut buf)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        klog!(
            "finish: serialized {} docs to {} bytes in {:.1}ms",
            self.index.len(),
            buf.len(),
            now_ms() - t0
        );
        Ok(buf)
    }

    #[wasm_bindgen]
    pub fn doc_count(&self) -> usize {
        self.index.len()
    }
}

// ─── WasmSearch ──────────────────────────────────────────────────

#[wasm_bindgen]
pub struct WasmSearch {
    model: SentenceEncoder,
    index: SearchIndex,
    splitter: TextSplitter,
}

#[wasm_bindgen]
impl WasmSearch {
    #[wasm_bindgen]
    pub fn add_chunk(
        &mut self,
        text: String,
        embedding: Vec<f32>,
        source: String,
        chunk_index: usize,
    ) -> Result<(), JsValue> {
        let mut meta = std::collections::HashMap::new();
        meta.insert("source".to_string(), source);
        meta.insert("chunk_index".to_string(), chunk_index.to_string());

        self.index
            .add_document(text, embedding, Some(meta))
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(())
    }

    #[wasm_bindgen]
    pub fn load(model_data: &[u8], index_data: &[u8]) -> Result<WasmSearch, JsValue> {
        let t0 = now_ms();

        let t_model = now_ms();
        let model = load_kjq_encoder(model_data)?;
        let model_ms = now_ms() - t_model;

        let t_idx = now_ms();
        let cursor = std::io::Cursor::new(index_data);
        let index =
            SearchIndex::load_binary(cursor).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let idx_ms = now_ms() - t_idx;

        klog!(
            "WasmSearch::load: model={:.1}ms, index={:.1}ms ({} docs, {} bytes), total={:.1}ms",
            model_ms,
            idx_ms,
            index.len(),
            index_data.len(),
            now_ms() - t0
        );

        Ok(WasmSearch {
            model,
            index,
            splitter: TextSplitter::new(SplitterConfig::default()),
        })
    }

    #[wasm_bindgen]
    pub fn update_file(&mut self, text: &str, source_path: &str) -> Result<usize, JsValue> {
        let t0 = now_ms();

        let t_rm = now_ms();
        let removed = self.index.remove_by_source(source_path);
        let rm_ms = now_ms() - t_rm;

        let chunks = self.splitter.split(text);
        if chunks.is_empty() {
            klog!(
                "update_file: removed {} chunks, 0 new chunks, total={:.1}ms | {}",
                removed,
                now_ms() - t0,
                source_path
            );
            return Ok(0);
        }

        let t_enc = now_ms();
        let chunk_refs: Vec<&str> = chunks.iter().map(|s| s.as_str()).collect();
        let embeddings =
            encode_all(&self.model, &chunk_refs).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let enc_ms = now_ms() - t_enc;

        let t_idx = now_ms();
        for (i, (chunk, embedding)) in chunks.iter().zip(embeddings).enumerate() {
            let mut meta = std::collections::HashMap::new();
            meta.insert("source".to_string(), source_path.to_string());
            meta.insert("chunk_index".to_string(), i.to_string());

            self.index
                .add_document(chunk.clone(), embedding, Some(meta))
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        let idx_ms = now_ms() - t_idx;

        klog!(
            "update_file: removed={}, added={} chunks, remove={:.1}ms, encode={:.1}ms, index={:.1}ms, total={:.1}ms | {}",
            removed,
            chunks.len(),
            rm_ms,
            enc_ms,
            idx_ms,
            now_ms() - t0,
            source_path
        );

        Ok(chunks.len())
    }

    #[wasm_bindgen]
    pub fn remove_file(&mut self, source_path: &str) -> usize {
        let t0 = now_ms();
        let removed = self.index.remove_by_source(source_path);
        klog!(
            "remove_file: removed {} chunks in {:.1}ms | {}",
            removed,
            now_ms() - t0,
            source_path
        );
        removed
    }

    #[wasm_bindgen]
    pub fn save_index(&self) -> Result<Vec<u8>, JsValue> {
        let t0 = now_ms();
        let mut buf = Vec::new();
        self.index
            .save_binary(&mut buf)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        klog!(
            "save_index: {} docs, {} bytes in {:.1}ms",
            self.index.len(),
            buf.len(),
            now_ms() - t0
        );
        Ok(buf)
    }

    #[wasm_bindgen]
    pub fn search(&self, query: &str, limit: usize) -> Result<JsValue, JsValue> {
        let t0 = now_ms();

        let t_enc = now_ms();
        let embedding =
            encode_all(&self.model, &[query]).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let enc_ms = now_ms() - t_enc;

        let t_search = now_ms();
        let results = self.index.search_hybrid(query, &embedding[0], limit);
        let search_ms = now_ms() - t_search;

        klog!(
            "search: query=\"{}\" limit={}, encode={:.1}ms, hybrid_search={:.1}ms, results={}, total={:.1}ms",
            query,
            limit,
            enc_ms,
            search_ms,
            results.len(),
            now_ms() - t0
        );

        // Log top results
        for (i, r) in results.iter().take(5).enumerate() {
            let source = r.metadata.get("source").map(|s| s.as_str()).unwrap_or("?");
            let snippet: String = r.text.chars().take(60).collect();
            let snippet = snippet.replace('\n', " ");
            klog!(
                "  result[{}]: score={:.4} source=\"{}\" \"{}...\"",
                i,
                r.score,
                source,
                snippet
            );
        }

        serde_wasm_bindgen::to_value(&results).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen]
    pub fn search_semantic(&self, query: &str, limit: usize) -> Result<JsValue, JsValue> {
        let t0 = now_ms();
        let embedding =
            encode_all(&self.model, &[query]).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let enc_ms = now_ms() - t0;

        let t_search = now_ms();
        let results = self.index.search_semantic(&embedding[0], limit);
        let search_ms = now_ms() - t_search;

        klog!(
            "search_semantic: query=\"{}\" encode={:.1}ms, search={:.1}ms, results={}",
            query,
            enc_ms,
            search_ms,
            results.len()
        );

        serde_wasm_bindgen::to_value(&results).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen]
    pub fn search_keywords(&self, query: &str, limit: usize) -> Result<JsValue, JsValue> {
        let t0 = now_ms();
        let results = self.index.search_keywords(query, limit);
        klog!(
            "search_keywords: query=\"{}\" results={}, {:.1}ms",
            query,
            results.len(),
            now_ms() - t0
        );

        serde_wasm_bindgen::to_value(&results).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen]
    pub fn doc_count(&self) -> usize {
        self.index.len()
    }
}

// ─── WasmEncoder ─────────────────────────────────────────────────

#[wasm_bindgen]
pub struct WasmEncoder {
    model: SentenceEncoder,
    splitter: TextSplitter,
}

#[wasm_bindgen]
impl WasmEncoder {
    #[wasm_bindgen]
    pub fn new(model_data: &[u8]) -> Result<WasmEncoder, JsValue> {
        let t0 = now_ms();
        let model = load_kjq_encoder(model_data)?;

        klog!("WasmEncoder::new: loaded in {:.1}ms", now_ms() - t0);

        Ok(WasmEncoder {
            model,
            splitter: TextSplitter::new(SplitterConfig::default()),
        })
    }

    #[wasm_bindgen]
    pub fn encode_file(&self, text: &str, source_path: &str) -> Result<JsValue, JsValue> {
        let t0 = now_ms();

        let chunks = self.splitter.split(text);
        if chunks.is_empty() {
            return serde_wasm_bindgen::to_value::<Vec<EncodedChunk>>(&vec![])
                .map_err(|e| JsValue::from_str(&e.to_string()));
        }

        let mut results: Vec<EncodedChunk> = Vec::new();
        let max_batch = 1;
        let num_batches = chunks.len().div_ceil(max_batch);

        klog!(
            "encode_file: {} chunks, {} batches (max {}) | {}",
            chunks.len(),
            num_batches,
            max_batch,
            source_path
        );

        for (batch_idx, batch) in chunks.chunks(max_batch).enumerate() {
            let t_batch = now_ms();
            let chunk_refs: Vec<&str> = batch.iter().map(|s| s.as_str()).collect();
            let embeddings = encode_all(&self.model, &chunk_refs)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let batch_ms = now_ms() - t_batch;

            klog!(
                "  batch {}/{}: {} chunks, {:.1}ms ({:.1}ms/chunk)",
                batch_idx + 1,
                num_batches,
                batch.len(),
                batch_ms,
                batch_ms / batch.len() as f64
            );

            for (i, (text, embedding)) in batch.iter().zip(embeddings).enumerate() {
                results.push(EncodedChunk {
                    text: text.clone(),
                    embedding,
                    source: source_path.to_string(),
                    chunk_index: batch_idx * max_batch + i,
                });
            }
        }

        klog!(
            "encode_file: total={:.1}ms, {} chunks | {}",
            now_ms() - t0,
            results.len(),
            source_path
        );

        serde_wasm_bindgen::to_value(&results).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[derive(Serialize)]
struct EncodedChunk {
    text: String,
    embedding: Vec<f32>,
    source: String,
    chunk_index: usize,
}

// ─── WasmClassifier ──────────────────────────────────────────────

/// One label and its score.
/// One label and its score.
#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct ClassifyResult {
    /// The predicted label.
    pub label: String,
    /// Model confidence.
    pub score: f32,
    /// Index of the label in `labels()`.
    pub index: usize,
}

/// Sequence classification: sentiment, emotion, toxicity.
///
/// New to the browser. Classification was always in the engine but never had a
/// binding here, because this crate previously carried its own encoder and nobody
/// wrote a second classifier head for it.
#[wasm_bindgen]
pub struct WasmClassifier {
    inner: SequenceClassifier,
}

#[wasm_bindgen]
impl WasmClassifier {
    /// Load a classifier from a `.kjq` container.
    #[wasm_bindgen]
    pub fn load(data: &[u8]) -> Result<WasmClassifier, JsValue> {
        WasmClassifier::load_core(data).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Classify one text, returning every label with its score, highest first.
    ///
    /// Serializing to a `JsValue` panics outside wasm32, so the work lives in
    /// [`WasmClassifier::classify_core`] and this is only the conversion. That is
    /// what makes the behaviour testable on the host.
    #[wasm_bindgen]
    pub fn classify(&self, text: &str) -> Result<JsValue, JsValue> {
        let out = self
            .classify_core(text)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&out).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Classify a batch, returning the single best label for each input.
    ///
    /// One result per input, in input order, so the caller can line them up with
    /// whatever it passed in.
    #[wasm_bindgen]
    pub fn classify_batch(&self, texts: Vec<String>) -> Result<JsValue, JsValue> {
        let out = self
            .classify_batch_core(&texts)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        serde_wasm_bindgen::to_value(&out).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// The labels this model predicts, in index order.
    #[wasm_bindgen]
    pub fn labels(&self) -> Vec<String> {
        self.inner.labels().map(|l| l.to_vec()).unwrap_or_default()
    }

    /// How many labels the model predicts.
    #[wasm_bindgen]
    pub fn num_labels(&self) -> usize {
        self.inner.num_labels()
    }
}

/// The classifier's logic, in plain Rust.
///
/// Kept out of the `#[wasm_bindgen]` impl deliberately: those methods speak
/// `JsValue`, and both constructing and serializing one panics outside wasm32, so
/// nothing in them can be asserted from a normal test.
impl WasmClassifier {
    /// Load a classifier from a `.kjq` container, reporting failure as a plain
    /// Rust error so the failure path can be asserted on the host.
    pub fn load_core(data: &[u8]) -> anyhow::Result<WasmClassifier> {
        let t0 = now_ms();
        let unpacked = kjq::unpack(data)?;

        let inner = EncoderLoader::load_from_bytes::<SequenceClassifier>(
            &unpacked.safetensors,
            &unpacked.config_json,
            unpacked.tokenizer_json.as_bytes(),
            EngineDevice::Cpu,
            None,
            None,
        )?;

        klog!("WasmClassifier::load: {:.1}ms", now_ms() - t0);
        Ok(WasmClassifier { inner })
    }

    /// Every label with its score, highest first.
    pub fn classify_core(&self, text: &str) -> anyhow::Result<Vec<ClassifyResult>> {
        let k = self.inner.num_labels();
        let results = futures::executor::block_on(self.inner.classify_top_k(text, k))?;
        Ok(results
            .into_iter()
            .map(|r| ClassifyResult {
                label: r.label,
                score: r.score,
                index: r.index,
            })
            .collect())
    }

    /// The single best label for each input, in input order.
    pub fn classify_batch_core(&self, texts: &[String]) -> anyhow::Result<Vec<ClassifyResult>> {
        let refs: Vec<&str> = texts.iter().map(|t| t.as_str()).collect();
        let batches = futures::executor::block_on(self.inner.classify_batch(&refs, 1))?;
        let labels = self.inner.labels().map(|l| l.to_vec()).unwrap_or_default();

        Ok(batches
            .into_iter()
            .map(|mut ranked| {
                let (label, score) = if ranked.is_empty() {
                    (String::new(), 0.0)
                } else {
                    ranked.remove(0)
                };
                let index = labels.iter().position(|l| *l == label).unwrap_or(0);
                ClassifyResult {
                    label,
                    score,
                    index,
                }
            })
            .collect())
    }
}

// ─── WasmChat ────────────────────────────────────────────────────

/// A local language model running in the browser.
///
/// Text generation is synchronous here, and deliberately so. The engine's streaming
/// path builds a background task with `tokio::task::spawn_blocking`, and wasm has no
/// blocking pool for that to run on — it compiles but would fail at runtime. Calling
/// the generation loop directly and draining the channel afterwards avoids the
/// problem entirely.
///
/// The cost is that a call blocks until the whole response is ready, which for a few
/// hundred tokens is seconds, not milliseconds. Run it in a Web Worker; on the main
/// thread it will freeze the tab.
/// Tokens a browser prefix cache holds unless the caller picks a size.
///
/// Half the native default. KV is eagerly allocated f32, roughly 24KB per token on
/// Qwen2.5-0.5B, so this is about 48MB, and wasm32 has a hard address-space ceiling
/// that a model's weights are already eating into.
pub const DEFAULT_WASM_PREFIX_CACHE_TOKENS: usize = 2048;

#[wasm_bindgen]
pub struct WasmChat {
    generator: DecoderGenerator,
    /// Turns so far. `RefCell` because wasm_bindgen hands JS `&self`, and wasm is
    /// single-threaded, so there is no contention to guard against.
    conversation: RefCell<Conversation>,
}

#[wasm_bindgen]
impl WasmChat {
    /// Load a decoder model from a `.kjq` container.
    ///
    /// `model_id` selects the chat template, e.g. `qwen2.5-0.5b-instruct`. Without
    /// one the model still generates, but its turn markers will not be applied.
    #[wasm_bindgen]
    pub fn load(data: &[u8], model_id: Option<String>) -> Result<WasmChat, JsValue> {
        WasmChat::load_core(data, model_id.as_deref())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Reuse the KV cache across turns, so a conversation prefills only its newest
    /// message instead of replaying the whole transcript.
    ///
    /// Measured natively on Qwen2.5-0.5B with a long system prompt: turn one is
    /// unchanged, turns two and three go from about 2.2s to roughly 0.2s, for
    /// identical answers. A browser tab is one conversation, so there is no sharing
    /// hazard here.
    ///
    /// `tokens` is the longest transcript that can be reused, and what the memory
    /// cost scales with; pass 0 for [`DEFAULT_WASM_PREFIX_CACHE_TOKENS`]. Longer
    /// conversations keep working, they simply stop reusing once they outgrow it.
    #[wasm_bindgen(js_name = enablePrefixCache)]
    pub fn enable_prefix_cache(&mut self, tokens: usize) -> Result<(), JsValue> {
        let tokens = if tokens == 0 {
            DEFAULT_WASM_PREFIX_CACHE_TOKENS
        } else {
            tokens
        };
        self.generator
            .enable_prefix_cache(tokens)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Whether prefix reuse is on.
    #[wasm_bindgen(js_name = hasPrefixCache)]
    pub fn has_prefix_cache(&self) -> bool {
        self.generator.has_prefix_cache()
    }

    /// Generate a completion for `prompt`.
    #[wasm_bindgen]
    pub fn generate(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
    ) -> Result<String, JsValue> {
        self.generate_core(prompt, max_new_tokens, temperature)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Generate a completion, invoking `on_token` with each piece as it arrives.
    ///
    /// The callback runs on whichever thread called this, synchronously between
    /// decoding steps. On the main thread that still blocks the page, so this is
    /// meant to be driven from a Web Worker; the `kjarni-wasm` npm wrapper does
    /// exactly that and turns it into an async iterator.
    ///
    /// Returning `false` from the callback does not stop generation yet; the
    /// value is ignored.
    ///
    /// Browser-only: `js_sys::Function` has no native equivalent. The streaming
    /// itself is in `generate_stream_core`, which builds and is tested natively.
    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen]
    pub fn generate_stream(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
        on_token: &js_sys::Function,
    ) -> Result<String, JsValue> {
        let this = JsValue::NULL;
        self.generate_stream_core(prompt, max_new_tokens, temperature, |piece| {
            // A throwing callback must not abort generation midway and leave the
            // model in a half-stepped state, so the error is dropped here and the
            // completed text is still returned.
            let _ = on_token.call1(&this, &JsValue::from_str(piece));
        })
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Sends a message in an ongoing conversation, streaming the reply.
    ///
    /// Unlike `generate_stream`, which completes raw text, this keeps the turns so
    /// far and formats them with the model's own chat template, so an instruct
    /// model answers instead of continuing the sentence. The reply is appended to
    /// the history, so the next call sees it.
    ///
    /// Browser-only: `js_sys::Function` has no native equivalent. The work is in
    /// `send_stream_core`, which builds and is tested natively.
    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen]
    pub fn send_stream(
        &self,
        message: &str,
        max_new_tokens: usize,
        temperature: f32,
        on_token: &js_sys::Function,
    ) -> Result<String, JsValue> {
        let this = JsValue::NULL;
        self.send_stream_core(message, max_new_tokens, temperature, |piece| {
            let _ = on_token.call1(&this, &JsValue::from_str(piece));
        })
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Drops the conversation, keeping the system prompt.
    #[wasm_bindgen]
    pub fn clear_history(&self) {
        self.conversation.borrow_mut().clear(true);
    }

    /// Number of messages held, system prompt included.
    #[wasm_bindgen]
    pub fn history_len(&self) -> usize {
        self.conversation.borrow().len()
    }

    /// The model's context window, in tokens.
    #[wasm_bindgen]
    pub fn context_size(&self) -> usize {
        self.generator.model.context_size()
    }
}

impl WasmChat {
    /// Load a decoder model, reporting failure as a plain Rust error.
    pub fn load_core(data: &[u8], model_id: Option<&str>) -> anyhow::Result<WasmChat> {
        Self::load_core_on(data, model_id, None)
    }

    /// `load_core`, but onto a caller-supplied device. `None` means the CPU.
    pub fn load_core_on(
        data: &[u8],
        model_id: Option<&str>,
        context: Option<std::sync::Arc<kjarni_transformers::WgpuContext>>,
    ) -> anyhow::Result<WasmChat> {
        use kjarni_transformers::models::ModelType;
        use kjarni_transformers::pipeline::DecoderLoader;

        let t0 = now_ms();
        let unpacked = kjq::unpack(data)?;
        let model_type = model_id.and_then(ModelType::from_cli_name);

        // Architecture decides which concrete model type parses the weights. The
        // config's own `model_type` field is the source of truth, not the caller.
        let arch: String = serde_json::from_str::<serde_json::Value>(&unpacked.config_json)
            .ok()
            .and_then(|v| {
                v.get("model_type")
                    .and_then(|m| m.as_str().map(String::from))
            })
            .unwrap_or_default();

        let model: std::sync::Arc<dyn DecoderLanguageModel + Send + Sync> = match arch.as_str() {
            "qwen2" => std::sync::Arc::new(DecoderLoader::load_from_kjq_on::<
                kjarni_models::models::qwen::QwenModel,
            >(
                &unpacked, None, model_type, context.clone()
            )?),
            "llama" => std::sync::Arc::new(DecoderLoader::load_from_kjq_on::<
                kjarni_models::models::llama::LlamaModel,
            >(
                &unpacked, None, model_type, context.clone()
            )?),
            "mistral" => std::sync::Arc::new(DecoderLoader::load_from_kjq_on::<
                kjarni_models::models::mistral::MistralModel,
            >(
                &unpacked, None, model_type, context.clone()
            )?),
            other => {
                return Err(anyhow::anyhow!(
                    "unsupported architecture '{other}' for browser chat. \
                     Supported: llama, qwen2, mistral."
                ));
            }
        };

        let generator = DecoderGenerator::new(model)?;

        // Seed with the template's own system prompt, so a conversation starts the
        // way the model was trained to see one.
        let conversation = match generator
            .model
            .chat_template()
            .and_then(|t| t.default_system_prompt())
        {
            Some(system) => Conversation::with_system(system),
            None => Conversation::new(),
        };

        klog!("WasmChat::load: {:.1}ms", now_ms() - t0);
        Ok(WasmChat {
            generator,
            conversation: RefCell::new(conversation),
        })
    }

    /// Sends a message in the ongoing conversation, in plain Rust so it can be
    /// tested on the host.
    pub fn send_stream_core(
        &self,
        message: &str,
        max_new_tokens: usize,
        temperature: f32,
        mut on_token: impl FnMut(&str),
    ) -> anyhow::Result<String> {
        let (prompt, stops) = {
            let mut conv = self.conversation.borrow_mut();
            conv.push_user(message);
            match self.generator.model.chat_template() {
                Some(t) => (t.apply(&conv), t.stop_sequences()),
                None => (message.to_string(), Vec::new()),
            }
        };

        // A stop sequence can arrive split across tokens ("<|im", "_end|>"), so the
        // search runs over the accumulated text rather than each piece, and nothing
        // past it reaches the caller.
        let mut full = String::new();
        let mut cut: Option<usize> = None;
        let text = self.generate_stream_core(&prompt, max_new_tokens, temperature, |piece| {
            if cut.is_some() {
                return;
            }
            let start = full.len();
            full.push_str(piece);
            match stops.iter().filter_map(|s| full.find(s.as_str())).min() {
                Some(at) => {
                    cut = Some(at);
                    if at > start {
                        on_token(&full[start..at]);
                    }
                }
                None => on_token(piece),
            }
        })?;

        let reply = match cut {
            Some(at) => full[..at].trim().to_string(),
            None => text.trim().to_string(),
        };
        self.conversation.borrow_mut().push_assistant(&reply);
        Ok(reply)
    }

    /// Generate a completion, in plain Rust so it can be tested on the host.
    pub fn generate_core(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
    ) -> anyhow::Result<String> {
        self.generate_stream_core(prompt, max_new_tokens, temperature, |_| {})
    }

    /// Generates, calling `on_token` with each piece as it is produced.
    ///
    /// The engine has always streamed: `run_generation_loop` sends every token
    /// into a channel the moment it is decoded. The earlier version of this
    /// method threw that away, blocking on the whole loop and only then draining
    /// the channel, which is why it needed one large enough to hold the entire
    /// response.
    ///
    /// Joining the loop with its own drain fixes that on a single thread. There
    /// is no second thread to spare in wasm, but there does not need to be: the
    /// loop awaits on `send`, and at that point `block_on` polls the drain, which
    /// hands the token out before the loop resumes. Tokens surface as they are
    /// decoded, and the channel goes back to being a small buffer that applies
    /// real back-pressure.
    pub fn generate_stream_core(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
        mut on_token: impl FnMut(&str),
    ) -> anyhow::Result<String> {
        use kjarni_transformers::common::{
            DecodingStrategy, GenerationConfig, SamplingParams, TokenType,
        };
        use kjarni_transformers::decoder::generator::{
            run_generation_loop, run_generation_loop_with_cache,
        };

        let config = GenerationConfig {
            max_new_tokens: Some(max_new_tokens),
            strategy: if temperature <= 0.0 {
                DecodingStrategy::Greedy
            } else {
                DecodingStrategy::Sample(SamplingParams {
                    temperature,
                    top_k: Some(40),
                    top_p: Some(0.95),
                    min_p: None,
                })
            },
            ..Default::default()
        };

        let tokens = self.generator.encode(prompt, &config)?;
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);

        // A browser conversation re-sends its whole transcript every turn, so with
        // the cache on only the new turn is prefilled. One tab is one conversation,
        // which is exactly the shape this cache serves.
        let prefix = self.generator.prefix_cache_handle();
        let model = self.generator.model.clone();
        let backend = self.generator.backend();
        let generate = async {
            match prefix {
                Some(pc) => {
                    let mut guard = pc.lock().await;
                    run_generation_loop_with_cache(
                        model,
                        backend,
                        tokens,
                        config,
                        tx,
                        None,
                        Some(&mut guard),
                    )
                    .await
                }
                None => run_generation_loop(model, backend, tokens, config, tx, None).await,
            }
        };

        let drain = async {
            let mut out = String::new();
            // Ends when the loop returns and drops its sender.
            while let Some(item) = rx.recv().await {
                let token = item?;
                if token.token_type == TokenType::Generated {
                    on_token(&token.text);
                    out.push_str(&token.text);
                }
            }
            Ok::<String, anyhow::Error>(out)
        };

        let (generated, drained) =
            futures::executor::block_on(futures::future::join(generate, drain));
        generated?;
        drained
    }
}

// ─── WasmModel (standalone) ─────────────────────────────────────

#[wasm_bindgen]
pub struct WasmModel {
    inner: SentenceEncoder,
}

// Browser-only: panic hooks, `globalThis` and `fetch` have no host equivalent.
// Everything above and below this block builds natively so it can be tested.
#[cfg(target_arch = "wasm32")]
mod browser_only {
    use super::*;

    #[wasm_bindgen(start)]
    pub fn init() {
        console_error_panic_hook::set_once();
    }

    #[wasm_bindgen]
    pub enum WasmModelType {
        MiniLML6V2,
    }

    pub(crate) enum Global {
        Window(Window),
        Worker(WorkerGlobalScope),
    }

    pub(crate) fn get_global() -> Result<Global, String> {
        let g = js_sys::global();
        if let Ok(win) = g.clone().dyn_into::<Window>() {
            Ok(Global::Window(win))
        } else if let Ok(worker) = g.clone().dyn_into::<WorkerGlobalScope>() {
            Ok(Global::Worker(worker))
        } else {
            Err("Unknown global scope".to_string())
        }
    }

    pub(crate) async fn fetch_bytes(url: &str) -> Result<Vec<u8>, String> {
        let global = get_global()?;
        let resp_js = match global {
            Global::Window(win) => JsFuture::from(win.fetch_with_str(url)).await,
            Global::Worker(worker) => JsFuture::from(worker.fetch_with_str(url)).await,
        }
        .map_err(|e| format!("Fetch error: {:?}", e))?;

        let resp: Response = resp_js.dyn_into().map_err(|_| "Response cast failed")?;
        let array_buffer = JsFuture::from(resp.array_buffer().map_err(|_| "ArrayBuffer error")?)
            .await
            .map_err(|e| format!("ArrayBuffer await failed: {:?}", e))?;

        Ok(js_sys::Uint8Array::new(&array_buffer).to_vec())
    }

    pub(crate) async fn fetch_text(url: &str) -> Result<String, String> {
        let global = get_global()?;
        let resp_js = match global {
            Global::Window(win) => JsFuture::from(win.fetch_with_str(url)).await,
            Global::Worker(worker) => JsFuture::from(worker.fetch_with_str(url)).await,
        }
        .map_err(|e| format!("Fetch error: {:?}", e))?;

        let resp: Response = resp_js.dyn_into().map_err(|_| "Response cast failed")?;
        let text_js = JsFuture::from(resp.text().map_err(|_| "Text conversion failed")?)
            .await
            .map_err(|e| format!("Text await failed: {:?}", e))?;

        Ok(text_js.as_string().ok_or("Failed to convert text")?)
    }
}

#[cfg(target_arch = "wasm32")]
pub use browser_only::*;

#[wasm_bindgen]
impl WasmModel {
    /// Load from raw safetensors plus its config and tokenizer.
    #[wasm_bindgen(constructor)]
    pub fn new(
        weights_data: &[u8],
        config_json: &str,
        tokenizer_json: &str,
    ) -> Result<WasmModel, JsValue> {
        let inner = EncoderLoader::load_from_bytes::<SentenceEncoder>(
            weights_data,
            config_json,
            tokenizer_json.as_bytes(),
            EngineDevice::Cpu,
            None,
            None,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(WasmModel { inner })
    }

    /// Load from a `.kjq` container: config, tokenizer and int8 tensors in one file.
    #[wasm_bindgen]
    pub fn from_quantized(data: &[u8]) -> Result<WasmModel, JsValue> {
        let unpacked = kjq::unpack(data).map_err(|e| JsValue::from_str(&e.to_string()))?;
        WasmModel::new(
            &unpacked.safetensors,
            &unpacked.config_json,
            &unpacked.tokenizer_json,
        )
    }

    /// Fetch a model straight from Hugging Face. Browser-only: it uses `fetch`.
    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen]
    pub async fn from_type(model_type: WasmModelType) -> Result<WasmModel, JsValue> {
        let (weights_url, config_url, tokenizer_url) = match model_type {
            WasmModelType::MiniLML6V2 => (
                "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.safetensors",
                "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json",
                "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json",
            ),
        };

        let (weights, config, tokenizer) = futures::future::join3(
            fetch_bytes(weights_url),
            fetch_text(config_url),
            fetch_text(tokenizer_url),
        )
        .await;

        let weights = weights.map_err(|e| JsValue::from_str(&e))?;
        let config = config.map_err(|e| JsValue::from_str(&e))?;
        let tokenizer = tokenizer.map_err(|e| JsValue::from_str(&e))?;

        WasmModel::new(&weights, &config, &tokenizer)
    }

    /// Encode a batch, returning every vector concatenated into one flat array.
    ///
    /// The engine's encode is `async` because the GPU path awaits buffer readback.
    /// There is no GPU path here, and the wasm CPU implementation contains no await
    /// at all, so the future is always ready on its first poll and `block_on`
    /// returns without ever parking the browser's main thread.
    #[wasm_bindgen]
    pub fn encode(&self, texts: Vec<String>, normalize: bool) -> Result<Vec<f32>, JsValue> {
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

        // These two differ only in the normalize flag, and both take the pooling
        // strategy from the loaded pipeline rather than assuming one.
        let embeddings = futures::executor::block_on(async {
            if normalize {
                self.inner.encode_batch(&text_refs).await
            } else {
                self.inner.encode_batch_raw(&text_refs).await
            }
        })
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(embeddings.into_iter().flatten().collect())
    }
}

// ─── WasmReranker ────────────────────────────────────────────────

#[wasm_bindgen]
pub struct WasmReranker {
    inner: CrossEncoder,
}

#[derive(Serialize)]
struct RerankResult {
    index: usize,
    score: f32,
    text: String,
}

#[wasm_bindgen]
impl WasmReranker {
    #[wasm_bindgen]
    /// Load a cross-encoder from a `.kjq` container.
    ///
    /// The classifier head used to be pulled out of the weights by hand here; the
    /// engine's `CrossEncoder` loads it as part of the model.
    pub fn load(data: &[u8]) -> Result<WasmReranker, JsValue> {
        let t0 = now_ms();
        let unpacked = kjq::unpack(data).map_err(|e| JsValue::from_str(&e.to_string()))?;

        let inner = EncoderLoader::load_from_bytes::<CrossEncoder>(
            &unpacked.safetensors,
            &unpacked.config_json,
            unpacked.tokenizer_json.as_bytes(),
            EngineDevice::Cpu,
            None,
            None,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

        klog!("WasmReranker::load: {:.1}ms", now_ms() - t0);
        Ok(WasmReranker { inner })
    }

    #[wasm_bindgen]
    /// Score every document against the query and return the best `limit`,
    /// highest first. Scores are raw logits, so negative values are normal.
    pub fn rerank(
        &self,
        query: &str,
        documents: Vec<String>,
        limit: usize,
    ) -> Result<JsValue, JsValue> {
        let t0 = now_ms();

        if documents.is_empty() {
            return serde_wasm_bindgen::to_value::<Vec<RerankResult>>(&vec![])
                .map_err(|e| JsValue::from_str(&e.to_string()));
        }

        let doc_refs: Vec<&str> = documents.iter().map(|d| d.as_str()).collect();
        let ranked = futures::executor::block_on(self.inner.rerank_top_k(query, &doc_refs, limit))
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let results: Vec<RerankResult> = ranked
            .into_iter()
            .map(|(index, score)| RerankResult {
                index,
                score,
                text: documents[index].clone(),
            })
            .collect();

        klog!(
            "WasmReranker::rerank: {} documents in {:.1}ms",
            documents.len(),
            now_ms() - t0
        );

        serde_wasm_bindgen::to_value(&results).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen]
    /// Score a single query-document pair.
    pub fn score(&self, query: &str, document: &str) -> Result<f32, JsValue> {
        futures::executor::block_on(self.inner.predict_pair(query, document))
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

/// The adapter limits that decide whether the compute stack runs at all.
///
/// Shaders declaring more than `maxComputeInvocationsPerWorkgroup` threads fail
/// to create, and weights larger than `maxStorageBufferBindingSize` must be split
/// across bindings. WebGPU only guarantees 256 and 128MiB, well under what a
/// native Vulkan adapter hands out.
#[cfg(all(target_arch = "wasm32", feature = "wasm-gpu"))]
fn adapter_limits_of(context: &kjarni_transformers::WgpuContext) -> Result<JsValue, JsValue> {
    let l = context.adapter.limits();
    let out = serde_json::json!({
        "maxComputeInvocationsPerWorkgroup": l.max_compute_invocations_per_workgroup,
        "maxComputeWorkgroupSizeX": l.max_compute_workgroup_size_x,
        "maxStorageBufferBindingSize": l.max_storage_buffer_binding_size,
        "maxBufferSize": l.max_buffer_size,
    });
    serde_wasm_bindgen::to_value(&out).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Unpack a `.kjq` and load it onto a fresh WebGPU context.
///
/// Every GPU wrapper needs the same four steps, and the context has to outlive
/// the model, so it comes back alongside it rather than being dropped here.
#[cfg(all(target_arch = "wasm32", feature = "wasm-gpu"))]
async fn load_on_gpu<M: kjarni_transformers::pipeline::EncoderModelFactory>(
    model_data: &[u8],
) -> Result<
    (
        std::sync::Arc<M>,
        std::sync::Arc<kjarni_transformers::WgpuContext>,
    ),
    JsValue,
> {
    let unpacked = kjq::unpack(model_data).map_err(|e| JsValue::from_str(&e.to_string()))?;

    let context = kjarni_transformers::WgpuContext::new()
        .await
        .map_err(|e| JsValue::from_str(&format!("no WebGPU adapter: {e}")))?;

    let model = EncoderLoader::load_from_bytes_with_context::<M>(
        &unpacked.safetensors,
        &unpacked.config_json,
        unpacked.tokenizer_json.as_bytes(),
        Some(context.clone()),
        None,
        None,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok((std::sync::Arc::new(model), context))
}

// ─── GPU encoder (browser WebGPU) ────────────────────────────────

/// A sentence encoder running on the browser's WebGPU adapter.
///
/// Separate from [`WasmModel`] rather than a flag on it because the two have
/// different shapes: acquiring an adapter is asynchronous, and so is every encode,
/// since reading results back off the GPU is a real await rather than the CPU
/// path's already-ready future.
#[cfg(all(target_arch = "wasm32", feature = "wasm-gpu"))]
#[wasm_bindgen]
pub struct WasmGpuEncoder {
    model: std::sync::Arc<SentenceEncoder>,
    context: std::sync::Arc<kjarni_transformers::WgpuContext>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm-gpu"))]
#[wasm_bindgen]
impl WasmGpuEncoder {
    /// Load a `.kjq` container onto the GPU.
    ///
    /// Rejects when the browser has no WebGPU adapter, which is the caller's cue
    /// to fall back to [`WasmModel`].
    #[wasm_bindgen(js_name = fromQuantized)]
    pub async fn from_quantized(model_data: Vec<u8>) -> Result<WasmGpuEncoder, JsValue> {
        let (model, context) = load_on_gpu::<SentenceEncoder>(&model_data).await?;
        Ok(Self { model, context })
    }

    /// Embed one string. Resolves to an array of `dimension()` floats.
    pub fn embed(&self, text: String) -> js_sys::Promise {
        let model = self.model.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            let v = model
                .encode(&text)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            serde_wasm_bindgen::to_value(&v).map_err(|e| JsValue::from_str(&e.to_string()))
        })
    }

    /// Embed a batch. Resolves to an array of arrays.
    #[wasm_bindgen(js_name = embedBatch)]
    pub fn embed_batch(&self, texts: Vec<String>) -> js_sys::Promise {
        let model = self.model.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
            let v = model
                .encode_batch(&refs)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            serde_wasm_bindgen::to_value(&v).map_err(|e| JsValue::from_str(&e.to_string()))
        })
    }

    pub fn dimension(&self) -> usize {
        use kjarni_transformers::cpu::encoder::traits::EncoderLanguageModel;
        self.model.dimension()
    }

    /// What the browser's adapter actually grants.
    ///
    /// Two of these decide whether the compute stack runs here at all: shaders
    /// declaring more than `maxComputeInvocationsPerWorkgroup` threads fail to
    /// create, and weights larger than `maxStorageBufferBindingSize` have to be
    /// split across bindings. WebGPU only guarantees 256 and 128MiB.
    #[wasm_bindgen(js_name = adapterLimits)]
    pub fn adapter_limits(&self) -> Result<JsValue, JsValue> {
        adapter_limits_of(&self.context)
    }
}

// ─── GPU reranker ────────────────────────────────────────────────

/// A cross-encoder reranker running on the browser's WebGPU adapter.
#[cfg(all(target_arch = "wasm32", feature = "wasm-gpu"))]
#[wasm_bindgen]
pub struct WasmGpuReranker {
    model: std::sync::Arc<CrossEncoder>,
    context: std::sync::Arc<kjarni_transformers::WgpuContext>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm-gpu"))]
#[wasm_bindgen]
impl WasmGpuReranker {
    #[wasm_bindgen(js_name = fromQuantized)]
    pub async fn from_quantized(model_data: Vec<u8>) -> Result<WasmGpuReranker, JsValue> {
        let (model, context) = load_on_gpu::<CrossEncoder>(&model_data).await?;
        Ok(Self { model, context })
    }

    /// Rank `documents` against `query`, best first. Scores are raw logits, the
    /// same as the CPU path and as torch: pass them through `sigmoid` for a 0..1
    /// scale.
    pub fn rerank(&self, query: String, documents: Vec<String>, limit: usize) -> js_sys::Promise {
        let model = self.model.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            if documents.is_empty() {
                return serde_wasm_bindgen::to_value::<Vec<RerankResult>>(&vec![])
                    .map_err(|e| JsValue::from_str(&e.to_string()));
            }
            let doc_refs: Vec<&str> = documents.iter().map(|d| d.as_str()).collect();
            let ranked = model
                .rerank_top_k(&query, &doc_refs, limit)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            let results: Vec<RerankResult> = ranked
                .into_iter()
                .map(|(index, score)| RerankResult {
                    index,
                    score,
                    text: documents[index].clone(),
                })
                .collect();
            serde_wasm_bindgen::to_value(&results).map_err(|e| JsValue::from_str(&e.to_string()))
        })
    }

    /// Score a single query-document pair.
    pub fn score(&self, query: String, document: String) -> js_sys::Promise {
        let model = self.model.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            let s = model
                .predict_pair(&query, &document)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            Ok(JsValue::from_f64(s as f64))
        })
    }

    /// See [`WasmGpuEncoder::adapter_limits`].
    #[wasm_bindgen(js_name = adapterLimits)]
    pub fn adapter_limits(&self) -> Result<JsValue, JsValue> {
        adapter_limits_of(&self.context)
    }
}

// ─── GPU classifier ──────────────────────────────────────────────

/// A sequence classifier running on the browser's WebGPU adapter.
#[cfg(all(target_arch = "wasm32", feature = "wasm-gpu"))]
#[wasm_bindgen]
pub struct WasmGpuClassifier {
    model: std::sync::Arc<SequenceClassifier>,
    context: std::sync::Arc<kjarni_transformers::WgpuContext>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm-gpu"))]
#[wasm_bindgen]
impl WasmGpuClassifier {
    #[wasm_bindgen(js_name = fromQuantized)]
    pub async fn from_quantized(model_data: Vec<u8>) -> Result<WasmGpuClassifier, JsValue> {
        let (model, context) = load_on_gpu::<SequenceClassifier>(&model_data).await?;
        Ok(Self { model, context })
    }

    /// Every label with its score, highest first.
    pub fn classify(&self, text: String) -> js_sys::Promise {
        let model = self.model.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            let k = model.num_labels();
            let results = model
                .classify_top_k(&text, k)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out: Vec<ClassifyResult> = results
                .into_iter()
                .map(|r| ClassifyResult {
                    label: r.label,
                    score: r.score,
                    index: r.index,
                })
                .collect();
            serde_wasm_bindgen::to_value(&out).map_err(|e| JsValue::from_str(&e.to_string()))
        })
    }

    pub fn labels(&self) -> Vec<String> {
        self.model.labels().map(|l| l.to_vec()).unwrap_or_default()
    }

    /// See [`WasmGpuEncoder::adapter_limits`].
    #[wasm_bindgen(js_name = adapterLimits)]
    pub fn adapter_limits(&self) -> Result<JsValue, JsValue> {
        adapter_limits_of(&self.context)
    }
}

// ─── GPU chat (browser WebGPU) ───────────────────────────────────

/// A decoder running on the browser's WebGPU adapter.
///
/// Separate from [`WasmChat`] for the same reason the encoder is: acquiring an
/// adapter is asynchronous, and generation reads logits back off the GPU once per
/// token, so every call is a real await rather than the CPU path's already-ready
/// future.
///
/// Expect modest tokens per second. The per-dispatch cost the CPU path does not
/// pay is charged once per token here, and it has not been optimised.
#[cfg(all(target_arch = "wasm32", feature = "wasm-gpu"))]
#[wasm_bindgen]
pub struct WasmGpuChat {
    /// `Rc`, not `Arc`: `WasmChat` keeps its turns in a `RefCell` and so is not
    /// `Sync`, and wasm is single threaded, so there is nothing to share across.
    inner: std::rc::Rc<WasmChat>,
    context: std::sync::Arc<kjarni_transformers::WgpuContext>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm-gpu"))]
#[wasm_bindgen]
impl WasmGpuChat {
    /// Load a `.kjq` decoder onto the GPU.
    ///
    /// Rejects when there is no WebGPU adapter, or when the adapter cannot run
    /// kjarni's shaders, which is the caller's cue to fall back to [`WasmChat`].
    #[wasm_bindgen(js_name = fromQuantized)]
    pub async fn from_quantized(
        model_data: Vec<u8>,
        model_id: Option<String>,
    ) -> Result<WasmGpuChat, JsValue> {
        let context = kjarni_transformers::WgpuContext::new()
            .await
            .map_err(|e| JsValue::from_str(&format!("no usable WebGPU adapter: {e}")))?;

        let inner = WasmChat::load_core_on(&model_data, model_id.as_deref(), Some(context.clone()))
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(Self {
            inner: std::rc::Rc::new(inner),
            context,
        })
    }

    /// Generate a continuation. Resolves to the generated text.
    pub fn generate(&self, prompt: String, max_new_tokens: usize) -> js_sys::Promise {
        let chat = self.inner.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            let config = kjarni_transformers::common::GenerationConfig {
                max_new_tokens: Some(max_new_tokens),
                strategy: kjarni_transformers::common::DecodingStrategy::Greedy,
                ..Default::default()
            };
            let out = chat
                .generator
                .generate(&prompt, &config, None)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            Ok(JsValue::from_str(&out))
        })
    }

    /// See [`WasmGpuEncoder::adapter_limits`].
    #[wasm_bindgen(js_name = adapterLimits)]
    pub fn adapter_limits(&self) -> Result<JsValue, JsValue> {
        adapter_limits_of(&self.context)
    }
}
