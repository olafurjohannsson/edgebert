#[cfg(all(not(target_arch = "wasm32"), feature = "openblas"))]
extern crate blas_src;

#[cfg(not(target_arch = "wasm32"))]
use std::fs;

#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};

#[cfg(not(target_arch = "wasm32"))]
use dirs;

#[cfg(not(target_arch = "wasm32"))]
use reqwest;

use anyhow::Result;
use ndarray::{Array1, Array2, Array3, Array4, Axis, Zip, s};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
mod tokenizer;
mod weights;
use lru::LruCache;
use once_cell::sync::Lazy;
use std::sync::Mutex;
use weights::ModelWeights;

#[cfg(target_arch = "wasm32")]
pub use tokenizer::WordPieceTokenizer as ModelTokenizer;

#[cfg(not(target_arch = "wasm32"))]
use tokenizers::Tokenizer as ModelTokenizer;

#[derive(Debug, Clone, Copy)]
pub enum ModelType {
    MiniLML6V2,
}

#[wasm_bindgen]
pub struct Model {
    word_embeddings: Array2<f32>,
    position_embeddings: Array2<f32>,
    token_type_embeddings: Array2<f32>,
    layers: Vec<BertLayer>,
    layer_norm_final: LayerNorm,
    config: Config,
    tokenizer: ModelTokenizer,

    #[cfg(not(target_arch = "wasm32"))]
    cache: Lazy<Mutex<LruCache<String, Vec<f32>>>>,

    #[cfg(target_arch = "wasm32")]
    buffer_pool: BufferPool,
}

struct BertLayer {
    attention: MultiHeadAttention,
    intermediate: FeedForward,
    layer_norm1: LayerNorm,
    layer_norm2: LayerNorm,
}

struct MultiHeadAttention {
    query_weight_t: Array2<f32>,
    query_bias: Array1<f32>,
    key_weight_t: Array2<f32>,
    key_bias: Array1<f32>,
    value_weight_t: Array2<f32>,
    value_bias: Array1<f32>,
    output_weight_t: Array2<f32>,
    output_bias: Array1<f32>,
    num_heads: usize,
    head_dim: usize,
    scale_factor: f32,
}

struct FeedForward {
    dense1_weight_t: Array2<f32>,
    dense1_bias: Array1<f32>,
    dense2_weight_t: Array2<f32>,
    dense2_bias: Array1<f32>,
}

struct LayerNorm {
    weight: Array1<f32>,
    bias: Array1<f32>,
    eps: f32,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct Config {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub intermediate_size: usize,
    pub layer_norm_eps: f32,
    pub hidden_act: String,
    pub model_type: String,
}

#[cfg(target_arch = "wasm32")]
struct BufferPool {
    input_ids: Array2<f32>,
    attention_mask: Array2<f32>,
    max_capacity: (usize, usize), // (batch, seq_len)
}

#[cfg(target_arch = "wasm32")]
impl BufferPool {
    fn new() -> Self {
        Self {
            input_ids: Array2::zeros((8, 512)), // Pre-allocate
            attention_mask: Array2::zeros((8, 512)),
            max_capacity: (8, 512),
        }
    }

    fn get_buffers(
        &mut self,
        batch_size: usize,
        seq_len: usize,
    ) -> (&mut Array2<f32>, &mut Array2<f32>) {
        if batch_size > self.max_capacity.0 || seq_len > self.max_capacity.1 {
            let new_batch = batch_size.max(self.max_capacity.0);
            let new_seq = seq_len.max(self.max_capacity.1);

            self.input_ids = Array2::zeros((new_batch, new_seq));
            self.attention_mask = Array2::zeros((new_batch, new_seq));
            self.max_capacity = (new_batch, new_seq);
        }

        // Zero out the region we'll use
        self.input_ids
            .slice_mut(s![..batch_size, ..seq_len])
            .fill(0.0);
        self.attention_mask
            .slice_mut(s![..batch_size, ..seq_len])
            .fill(0.0);

        (&mut self.input_ids, &mut self.attention_mask)
    }
}

const SQRT_2_OVER_PI: f32 = 0.7978845608_f32;
const GELU_COEFF: f32 = 0.044715_f32;

impl Model {
    pub fn from_weights(
        weights: ModelWeights,
        tokenizer: ModelTokenizer,
        config: Config,
    ) -> Result<Self> {
        // Load embeddings
        let word_embeddings = weights.get_array2("embeddings.word_embeddings.weight")?;
        let position_embeddings = weights.get_array2("embeddings.position_embeddings.weight")?;
        let token_type_embeddings =
            weights.get_array2("embeddings.token_type_embeddings.weight")?;

        // Load layers
        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let prefix = format!("encoder.layer.{}", i);

            let attention = MultiHeadAttention {
                query_weight_t: weights
                    .get_array2(&format!("{}.attention.self.query.weight", prefix))?
                    .t()
                    .to_owned(),
                query_bias: weights.get_array1(&format!("{}.attention.self.query.bias", prefix))?,
                key_weight_t: weights
                    .get_array2(&format!("{}.attention.self.key.weight", prefix))?
                    .t()
                    .to_owned(),
                key_bias: weights.get_array1(&format!("{}.attention.self.key.bias", prefix))?,
                value_weight_t: weights
                    .get_array2(&format!("{}.attention.self.value.weight", prefix))?
                    .t()
                    .to_owned(),
                value_bias: weights.get_array1(&format!("{}.attention.self.value.bias", prefix))?,
                output_weight_t: weights
                    .get_array2(&format!("{}.attention.output.dense.weight", prefix))?
                    .t()
                    .to_owned(),
                output_bias: weights
                    .get_array1(&format!("{}.attention.output.dense.bias", prefix))?,
                num_heads: config.num_attention_heads,
                head_dim: config.hidden_size / config.num_attention_heads,
                // Pre-calculate the scale factor with reciprocal for multiplication
                scale_factor: 1.0
                    / ((config.hidden_size / config.num_attention_heads) as f32).sqrt(),
            };

            let intermediate = FeedForward {
                dense1_weight_t: weights
                    .get_array2(&format!("{}.intermediate.dense.weight", prefix))?
                    .t()
                    .to_owned(),
                dense1_bias: weights.get_array1(&format!("{}.intermediate.dense.bias", prefix))?,
                dense2_weight_t: weights
                    .get_array2(&format!("{}.output.dense.weight", prefix))?
                    .t()
                    .to_owned(),
                dense2_bias: weights.get_array1(&format!("{}.output.dense.bias", prefix))?,
            };

            let layer_norm1 = LayerNorm {
                weight: weights
                    .get_array1(&format!("{}.attention.output.LayerNorm.weight", prefix))?,
                bias: weights.get_array1(&format!("{}.attention.output.LayerNorm.bias", prefix))?,
                eps: config.layer_norm_eps,
            };

            let layer_norm2 = LayerNorm {
                weight: weights.get_array1(&format!("{}.output.LayerNorm.weight", prefix))?,
                bias: weights.get_array1(&format!("{}.output.LayerNorm.bias", prefix))?,
                eps: config.layer_norm_eps,
            };

            layers.push(BertLayer {
                attention,
                intermediate,
                layer_norm1,
                layer_norm2,
            });
        }

        let layer_norm_final = LayerNorm {
            weight: weights.get_array1("embeddings.LayerNorm.weight")?,
            bias: weights.get_array1("embeddings.LayerNorm.bias")?,
            eps: config.layer_norm_eps,
        };

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layers,
            layer_norm_final,
            config,
            tokenizer,
            #[cfg(not(target_arch = "wasm32"))]
            cache: Lazy::new(|| {
                Mutex::new(LruCache::new(
                    std::num::NonZeroUsize::new(1000).unwrap(), // Cache up to 1000 embeddings
                ))
            }),
            #[cfg(target_arch = "wasm32")]
            buffer_pool: BufferPool::new(),
        })
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_pretrained(model_type: ModelType) -> Result<Self> {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("edgebert");

        fs::create_dir_all(&cache_dir)?;

        let model_path = match model_type {
            ModelType::MiniLML6V2 => cache_dir.join("all-MiniLM-L6-v2"),
        };

        Self::ensure_model_files(model_type, &model_path)?;
        Self::from_pretrained_path(model_path.to_str().unwrap())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_pretrained_path(model_path: &str) -> Result<Self> {
        let model_path = Path::new(model_path);

        let weights = ModelWeights::load(model_path)?;
        let tokenizer_file = model_path.join("tokenizer.json");
        let config = weights.config.clone();

        let mut tokenizer = ModelTokenizer::from_file(tokenizer_file)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        if let Some(padding) = tokenizer.get_padding_mut() {
            padding.strategy = tokenizers::PaddingStrategy::BatchLongest;
        }
        if let Some(truncation) = tokenizer.get_truncation_mut() {
            truncation.max_length = 512;
        }

        Self::from_weights(weights, tokenizer, config)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn ensure_model_files(model_id: ModelType, model_path: &Path) -> Result<()> {
        let (weights_url, tokenizer_url, config_url, weights_file, tokenizer_file, config_file) =
            match model_id {
                ModelType::MiniLML6V2 => (
                    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.safetensors",
                    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json",
                    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json",
                    model_path.join("model.safetensors"),
                    model_path.join("tokenizer.json"),
                    model_path.join("config.json"),
                ),
            };

        fs::create_dir_all(model_path)?;

        let download = |url: &str, path: &Path| -> anyhow::Result<()> {
            if !path.exists() {
                let resp = reqwest::blocking::get(url)?;
                if !resp.status().is_success() {
                    anyhow::bail!("Failed to download {}: {}", url, resp.status());
                }
                let bytes = resp.bytes()?;
                fs::write(path, &bytes)?;
            }
            Ok(())
        };

        download(weights_url, &weights_file)?;
        download(tokenizer_url, &tokenizer_file)?;
        download(config_url, &config_file)?;

        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn encode_batch(
        &self,
        batches: Vec<Vec<&str>>,
        normalize_embeddings: bool,
    ) -> Result<Vec<Vec<Vec<f32>>>> {
        use rayon::prelude::*;

        batches
            .into_par_iter()
            .map(|batch| self.encode(batch, normalize_embeddings))
            .collect()
    }

    // WASM version - sequential, uses &mut self with buffer pooling
    #[cfg(target_arch = "wasm32")]
    pub fn encode_batch(
        &mut self,
        batches: Vec<Vec<&str>>,
        normalize_embeddings: bool,
    ) -> Result<Vec<Vec<Vec<f32>>>> {
        batches
            .into_iter()
            .map(|batch| self.encode(batch, normalize_embeddings))
            .collect()
    }

    #[cfg(target_arch = "wasm32")]
    pub fn encode(
        &mut self,
        texts: Vec<&str>,
        normalize_embeddings: bool,
    ) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        self.encode_uncached(texts, normalize_embeddings)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn encode(&self, texts: Vec<&str>, normalize_embeddings: bool) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let mut results = vec![None; texts.len()];
        let mut uncached_indices = Vec::new();
        let mut uncached_texts = Vec::new();

        // Check cache
        {
            let mut cache = self.cache.lock().unwrap();
            for (i, text) in texts.iter().enumerate() {
                let cache_key = format!("{}||{}", text, normalize_embeddings);
                if let Some(cached_embedding) = cache.get(&cache_key) {
                    results[i] = Some(cached_embedding.clone());
                } else {
                    uncached_indices.push(i);
                    uncached_texts.push(*text);
                }
            }
        }

        if uncached_texts.is_empty() {
            return Ok(results.into_iter().map(|r| r.unwrap()).collect());
        }

        let new_embeddings = self.encode_uncached(uncached_texts, normalize_embeddings)?;

        // Store in cache and results
        {
            let mut cache = self.cache.lock().unwrap();
            for (idx, embedding) in uncached_indices.iter().zip(new_embeddings) {
                let text = texts[*idx];
                let cache_key = format!("{}||{}", text, normalize_embeddings);
                cache.put(cache_key, embedding.clone());
                results[*idx] = Some(embedding);
            }
        }

        Ok(results.into_iter().map(|r| r.unwrap()).collect())
    }

    // WASM version with buffer pooling
    #[cfg(target_arch = "wasm32")]
    fn encode_uncached(
        &mut self,
        texts: Vec<&str>,
        normalize_embeddings: bool,
    ) -> Result<Vec<Vec<f32>>> {
        let encodings = self.tokenizer.encode_batch(texts, 512)?;

        let batch_size = encodings.len();
        if batch_size == 0 {
            return Ok(vec![]);
        }

        let max_len = encodings.iter().map(|e| e.len()).max().unwrap();

        // Get reusable buffers
        let (input_ids, attention_mask) = self.buffer_pool.get_buffers(batch_size, max_len);

        for (i, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();

            for (j, (&id, &m)) in ids.iter().zip(mask.iter()).enumerate() {
                input_ids[[i, j]] = id as f32;
                attention_mask[[i, j]] = m as f32;
            }
        }

        let input_ids_slice = input_ids.slice(s![..batch_size, ..max_len]).to_owned();
        let attention_mask_slice = attention_mask.slice(s![..batch_size, ..max_len]).to_owned();

        let embeddings = self.forward(&input_ids_slice, &attention_mask_slice)?;

        let final_embeddings = if normalize_embeddings {
            let norms = embeddings.mapv(|x| x.powi(2)).sum_axis(Axis(1));
            let norms = norms.mapv(|x| x.sqrt().max(1e-12));
            let norms_expanded = norms.insert_axis(Axis(1));
            embeddings / &norms_expanded
        } else {
            embeddings
        };

        let vector: Vec<Vec<f32>> = final_embeddings
            .outer_iter()
            .map(|row| row.to_vec())
            .collect();
        Ok(vector)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn encode_uncached(
        &self,
        texts: Vec<&str>,
        normalize_embeddings: bool,
    ) -> Result<Vec<Vec<f32>>> {
        let encodings = self
            .tokenizer
            .encode_batch(texts, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let batch_size = encodings.len();
        if batch_size == 0 {
            return Ok(vec![]);
        }

        let max_len = encodings.iter().map(|e| e.len()).max().unwrap();

        let mut input_ids = Array2::<f32>::zeros((batch_size, max_len));
        let mut attention_mask = Array2::<f32>::zeros((batch_size, max_len));

        for (i, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();

            for (j, (&id, &m)) in ids.iter().zip(mask.iter()).enumerate() {
                input_ids[[i, j]] = id as f32;
                attention_mask[[i, j]] = m as f32;
            }
        }

        let embeddings = self.forward(&input_ids, &attention_mask)?;

        let final_embeddings = if normalize_embeddings {
            let norms = embeddings.mapv(|x| x.powi(2)).sum_axis(Axis(1));
            let norms = norms.mapv(|x| x.sqrt().max(1e-12));
            let norms_expanded = norms.insert_axis(Axis(1));
            embeddings / &norms_expanded
        } else {
            embeddings
        };

        let vector: Vec<Vec<f32>> = final_embeddings
            .outer_iter()
            .map(|row| row.to_vec())
            .collect();
        Ok(vector)
    }

    fn forward(
        &self,
        input_ids: &Array2<f32>,
        attention_mask: &Array2<f32>,
    ) -> Result<Array2<f32>> {
        let (batch_size, seq_len) = input_ids.dim();

        let mut hidden = Array3::<f32>::zeros((batch_size, seq_len, self.config.hidden_size));
        // Parallelize over batch dimension
        #[cfg(not(target_arch = "wasm32"))]
        {
            use ndarray::parallel::prelude::*;
            hidden
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .zip(input_ids.axis_iter(Axis(0)))
                .for_each(|(mut hidden_slice, ids)| {
                    for (j, &token_id) in ids.iter().enumerate() {
                        let word_emb = self.word_embeddings.row(token_id as usize);
                        hidden_slice.slice_mut(s![j, ..]).assign(&word_emb);
                    }
                });
        }
        #[cfg(target_arch = "wasm32")]
        {
            for i in 0..batch_size {
                for j in 0..seq_len {
                    let token_id = input_ids[[i, j]] as usize;
                    let word_emb = self.word_embeddings.row(token_id);
                    hidden.slice_mut(s![i, j, ..]).assign(&word_emb);
                }
            }
        }

        let pos_embeddings = self.position_embeddings.slice(s![0..seq_len, ..]);
        hidden += &pos_embeddings;

        let type_embeddings = self.token_type_embeddings.row(0);
        hidden += &type_embeddings;

        let mut hidden = apply_layer_norm_3d(&hidden, &self.layer_norm_final);
        for layer in &self.layers {
            hidden = layer.forward(hidden, attention_mask)?;
        }
        mean_pool(&hidden, attention_mask)
    }
}

impl BertLayer {
    fn forward(&self, input: Array3<f32>, attention_mask: &Array2<f32>) -> Result<Array3<f32>> {
        // Self attention
        let mut attention_out = self.attention.forward(&input, attention_mask)?;
        attention_out += &input;
        let attention_out = apply_layer_norm_3d(&attention_out, &self.layer_norm1);

        // Feed forward
        let mut ff_out = self.intermediate.forward(&attention_out)?;
        ff_out += &attention_out;
        let output = apply_layer_norm_3d(&ff_out, &self.layer_norm2);

        Ok(output)
    }
}

impl MultiHeadAttention {
    fn forward(&self, hidden: &Array3<f32>, attention_mask: &Array2<f32>) -> Result<Array3<f32>> {
        let batch_size = hidden.shape()[0];
        let seq_len = hidden.shape()[1];

        // Linear projections WITH BIAS
        let mut q = matmul_3d_2d(hidden, &self.query_weight_t);
        q += &self.query_bias;

        let mut k = matmul_3d_2d(hidden, &self.key_weight_t);
        k += &self.key_bias;

        let mut v = matmul_3d_2d(hidden, &self.value_weight_t);
        v += &self.value_bias;

        let q = q
            .into_shape_with_order((batch_size, seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3]);

        let k = k
            .into_shape_with_order((batch_size, seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3]);

        let v = v
            .into_shape_with_order((batch_size, seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3]);

        // Compute attention scores
        let mut scores = matmul_4d(&q, &k.permuted_axes([0, 1, 3, 2]));
        scores *= self.scale_factor;

        // Apply mask
        let scores = apply_attention_mask(scores, attention_mask);

        // Softmax
        let weights = softmax(&scores);

        // Apply attention to values
        let context = matmul_4d(&weights, &v);

        // Reshape back - use to_shape again
        let context = context.permuted_axes([0, 2, 1, 3]);
        let context = context
            .as_standard_layout()
            .into_shape_with_order((batch_size, seq_len, self.num_heads * self.head_dim))?
            .to_owned();

        let mut output = matmul_3d_2d(&context, &self.output_weight_t);
        output += &self.output_bias;

        Ok(output)
    }
}

impl FeedForward {
    fn forward(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
        let mut intermediate = matmul_3d_2d(hidden, &self.dense1_weight_t);
        intermediate += &self.dense1_bias; // in place mutation, no allocation

        gelu(&mut intermediate);

        let mut output = matmul_3d_2d(&intermediate, &self.dense2_weight_t);
        output += &self.dense2_bias; // in place mutation, no allocation
        Ok(output)
    }
}

/*
**   Public function
**/

/// Computes cosine similarity between two vectors: dot(a,b) / (||a|| * ||b||).
///
/// Uses tree reduction pattern for better CPU instruction-level parallelism.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    const CHUNK: usize = 8;
    let n = a.len().min(b.len());

    // Split arrays into chunks of 8 for tree reduction
    let (chunks_a, tail_a) = a[..n].split_at(n - (n % CHUNK));
    let (chunks_b, tail_b) = b[..n].split_at(n - (n % CHUNK));

    // Tree-reduce dot product in chunks of 8
    let chunk_dot: f32 = chunks_a
        .chunks_exact(CHUNK)
        .zip(chunks_b.chunks_exact(CHUNK))
        .map(|(ca, cb)| {
            let d0 = ca[0] * cb[0] + ca[1] * cb[1];
            let d1 = ca[2] * cb[2] + ca[3] * cb[3];
            let d2 = ca[4] * cb[4] + ca[5] * cb[5];
            let d3 = ca[6] * cb[6] + ca[7] * cb[7];
            (d0 + d1) + (d2 + d3)
        })
        .sum();

    // Handle remaining elements that didn't fit in chunks
    let dot = chunk_dot + tail_a.iter().zip(tail_b).map(|(x, y)| x * y).sum::<f32>();

    // Compute L2 norms (TODO: could use tree reduction here)
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    // Add small epsilon to prevent division by zero
    dot / (norm_a * norm_b + 1e-8)
}

/*
**   Private functions, mainly Linear Algebra and Matrix multiplication dot product
**/

/// Performs batched matrix multiplication between a 3D and 2D tensor.
///
/// Multiplies each 2D slice of the 3D tensor with the same 2D matrix.
/// Input shapes: a=[batch, m, k], b=[k, n]
/// Output shape: [batch, m, n]
///
/// This is used for linear transformations in BERT where the same weight matrix
/// is applied to each example in the batch.
#[inline(always)]
fn matmul_3d_2d(a: &Array3<f32>, b: &Array2<f32>) -> Array3<f32> {
    let (batch, m, k) = a.dim();
    let n = b.shape()[1];
    assert_eq!(
        k,
        b.shape()[0],
        "Matrix dimensions are incompatible for multiplication"
    );

    // Ensure contiguous memory layout
    let a_cont = a.as_standard_layout();
    let b_cont = b.as_standard_layout();

    // allocate output tensor
    let mut c = Array3::<f32>::zeros((batch, m, n));

    // iter batch dimension, multiplying each slice with the weight matrix
    #[cfg(not(target_arch = "wasm32"))]
    {
        if batch <= 4 {
            // Small batches: sequential
            Zip::from(c.axis_iter_mut(Axis(0)))
                .and(a_cont.axis_iter(Axis(0)))
                .for_each(|mut c_slice, a_slice| {
                    c_slice.assign(&a_slice.dot(&b_cont));
                });
        } else {
            // Large batches: parallel
            Zip::from(c.axis_iter_mut(Axis(0)))
                .and(a_cont.axis_iter(Axis(0)))
                .par_for_each(|mut c_slice, a_slice| {
                    c_slice.assign(&a_slice.dot(&b_cont));
                });
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        Zip::from(c.axis_iter_mut(Axis(0)))
            .and(a_cont.axis_iter(Axis(0)))
            .for_each(|mut c_slice, a_slice| {
                c_slice.assign(&a_slice.dot(&b_cont));
            });
    }

    c
}

/// Computes softmax activation over the last dimension of a 4D tensor.
///
/// Takes attention scores of shape [batch, num_heads, seq_len, seq_len]
/// and normalizes them to probabilities that sum to 1.0 along the last axis.
///
/// Uses numerically stable softmax: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
#[inline(always)]
fn softmax(scores: &Array4<f32>) -> Array4<f32> {
    // Use parallel iteration for large batches
    let max_vals = scores.fold_axis(Axis(3), f32::NEG_INFINITY, |&acc, &x| acc.max(x));
    let max_expanded = max_vals.insert_axis(Axis(3));

    // Fuse operations to reduce memory access
    let mut result = scores - &max_expanded;
    result.mapv_inplace(f32::exp);

    let sum_exp = result.sum_axis(Axis(3)).insert_axis(Axis(3));
    result /= &sum_exp; // More efficient than multiplication by reciprocal for ndarray
    result
}

/// Applies attention mask to attention scores before softmax computation.
///
/// Takes a 4D tensor of attention scores [batch, num_heads, seq_len, seq_len]
/// and a 2D attention mask [batch, seq_len] where 1.0 = real token, 0.0 = padding.
///
/// Sets scores for padding tokens to negative infinity so they become ~0 after softmax,
/// preventing the model from attending to padding positions.
#[inline(always)]
fn apply_attention_mask(mut scores: Array4<f32>, mask: &Array2<f32>) -> Array4<f32> {
    let mask_expanded = mask
        .clone()
        .insert_axis(Axis(1)) // [batch, 1, seq]
        .insert_axis(Axis(2)); // [batch, 1, 1, seq]

    if let Some(broadcast_mask) = mask_expanded.broadcast(scores.dim()) {
        Zip::from(&mut scores)
            .and(&broadcast_mask)
            .for_each(|s, &m| {
                if m == 0.0 {
                    *s = f32::NEG_INFINITY;
                }
            });
    }
    scores
}

#[inline(always)]
fn matmul_4d(a: &Array4<f32>, b: &Array4<f32>) -> Array4<f32> {
    assert_eq!(a.shape()[0], b.shape()[0], "Batch size mismatch");
    assert_eq!(a.shape()[1], b.shape()[1], "Heads mismatch");
    assert_eq!(a.shape()[3], b.shape()[2], "Dimension mismatch");

    let (batch, heads, seq1, dim) = a.dim();
    let seq2 = b.shape()[3];

    let a_cont = a.as_standard_layout();
    let b_cont = b.as_standard_layout();

    let a_3d = a_cont
        .view()
        .into_shape_with_order((batch * heads, seq1, dim))
        .expect("Could not reshape A into 3D");
    let b_3d = b_cont
        .view()
        .into_shape_with_order((batch * heads, dim, seq2))
        .expect("Could not reshape B into 3D");

    let total_batches = batch * heads;
    let mut output_3d = Array3::<f32>::zeros((total_batches, seq1, seq2));

    #[cfg(all(not(target_arch = "wasm32")))]
    {
        if total_batches <= 8 {
            Zip::from(output_3d.axis_iter_mut(Axis(0)))
                .and(a_3d.axis_iter(Axis(0)))
                .and(b_3d.axis_iter(Axis(0)))
                .for_each(|mut c_slice, a_slice, b_slice| {
                    let result = a_slice.dot(&b_slice);
                    c_slice.assign(&result);
                });
        } else {
            Zip::from(output_3d.axis_iter_mut(Axis(0)))
                .and(a_3d.axis_iter(Axis(0)))
                .and(b_3d.axis_iter(Axis(0)))
                .par_for_each(|mut c_slice, a_slice, b_slice| {
                    let result = a_slice.dot(&b_slice);
                    c_slice.assign(&result);
                });
        }
    }
    #[cfg(any(target_arch = "wasm32"))]
    {
        Zip::from(output_3d.axis_iter_mut(Axis(0)))
            .and(a_3d.axis_iter(Axis(0)))
            .and(b_3d.axis_iter(Axis(0)))
            .for_each(|mut c_slice, a_slice, b_slice| {
                let result = a_slice.dot(&b_slice);
                c_slice.assign(&result);
            });
    }

    output_3d
        .into_shape_with_order((batch, heads, seq1, seq2))
        .unwrap()
}

#[inline(always)]
fn gelu(x: &mut Array3<f32>) {
    #[cfg(all(not(target_arch = "wasm32")))]
    {
        x.par_mapv_inplace(|val| {
            let val_squared = val * val;
            let val_cubed = val_squared * val;
            let inner = SQRT_2_OVER_PI * (val + GELU_COEFF * val_cubed);
            val * 0.5 * (1.0 + inner.tanh())
        });
    }
    #[cfg(target_arch = "wasm32")]
    {
        x.mapv_inplace(|val| {
            let val_squared = val * val;
            let val_cubed = val_squared * val;
            let inner = SQRT_2_OVER_PI * (val + GELU_COEFF * val_cubed);
            val * 0.5 * (1.0 + inner.tanh())
        });
    }
}

#[inline(always)]
fn apply_layer_norm_3d(hidden: &Array3<f32>, ln: &LayerNorm) -> Array3<f32> {
    let mean = hidden.mean_axis(Axis(2)).unwrap();
    // Calculate variance across the last dimension. Shape: [batch, seq]
    let var = hidden.var_axis(Axis(2), 0.0);
    let mean_expanded = mean.insert_axis(Axis(2));
    let var_expanded = var.insert_axis(Axis(2));

    // reciprocal of standard deviation
    let inv_std = (&var_expanded + ln.eps).mapv(|x| 1.0 / x.sqrt());
    (hidden - &mean_expanded) * &inv_std * &ln.weight + &ln.bias
}

/// Performs mean pooling over sequence dimension to convert token embeddings to sentence embeddings.
///
/// Takes a 3D tensor of shape [batch, sequence_length, hidden_size] containing token embeddings
/// and produces a 2D tensor of shape [batch, hidden_size] containing sentence embeddings.
///
/// The attention_mask indicates which tokens are real (1.0) vs padding (0.0).
/// Only real tokens contribute to the mean.
fn mean_pool(hidden: &Array3<f32>, attention_mask: &Array2<f32>) -> Result<Array2<f32>> {
    let mask_expanded = attention_mask.clone().insert_axis(Axis(2));
    let masked_hidden = hidden * &mask_expanded;
    let sum = masked_hidden.sum_axis(Axis(1));
    let count = attention_mask
        .sum_axis(Axis(1))
        .mapv(|x| x.max(1.0))
        .insert_axis(Axis(1));

    Ok(sum / &count)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmModel {
    inner: Model,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub enum WasmModelType {
    MiniLML6V2,
}

#[cfg(target_arch = "wasm32")]
impl From<WasmModelType> for ModelType {
    fn from(wasm_type: WasmModelType) -> Self {
        match wasm_type {
            WasmModelType::MiniLML6V2 => ModelType::MiniLML6V2,
        }
    }
}

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::JsFuture;
#[cfg(target_arch = "wasm32")]
use web_sys::{Response, Window, WorkerGlobalScope};

#[cfg(target_arch = "wasm32")]
fn get_global() -> Result<Global, String> {
    let g = js_sys::global();

    if let Ok(win) = g.clone().dyn_into::<Window>() {
        Ok(Global::Window(win))
    } else if let Ok(worker) = g.clone().dyn_into::<WorkerGlobalScope>() {
        Ok(Global::Worker(worker))
    } else {
        Err("Unknown global scope".to_string())
    }
}

#[cfg(target_arch = "wasm32")]
enum Global {
    Window(Window),
    Worker(WorkerGlobalScope),
}

#[cfg(target_arch = "wasm32")]
pub async fn fetch_bytes(url: &str) -> Result<Vec<u8>, String> {
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

#[cfg(target_arch = "wasm32")]
pub async fn fetch_text(url: &str) -> Result<String, String> {
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

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmModel {
    #[wasm_bindgen(constructor)]
    pub fn new(
        weights_data: &[u8],
        config_json: &str,
        tokenizer_json: &str,
    ) -> Result<WasmModel, JsValue> {
        use crate::tokenizer::WordPieceTokenizer;

        // Parse weights from bytes
        let weights = ModelWeights::from_bytes(weights_data, config_json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Create tokenizer
        let tokenizer = WordPieceTokenizer::from_json_str(tokenizer_json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Parse config
        let config =
            serde_json::from_str(config_json).map_err(|e| JsValue::from_str(&e.to_string()))?;

        let model = Model::from_weights(weights, tokenizer, config)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(WasmModel { inner: model })
    }
    #[wasm_bindgen]
    pub async fn from_type(model_type: WasmModelType) -> Result<WasmModel, JsValue> {
        let (weights_url, config_url, tokenizer_url) = match model_type {
            WasmModelType::MiniLML6V2 => (
                "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.safetensors",
                "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json",
                "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json",
            ),
            // add more models here
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

        let model = WasmModel::new(&weights, &config, &tokenizer)?;
        Ok(model)
    }

    #[wasm_bindgen]
    pub fn encode(&mut self, texts: Vec<String>, normalize: bool) -> Result<Vec<f32>, JsValue> {
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embeddings = self
            .inner
            .encode(text_refs, normalize)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let vector: Vec<f32> = embeddings.into_iter().flatten().collect();

        Ok(vector)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array4;

    #[test]
    fn test_softmax_simple() {
        let input: Array4<f32> = Array4::from_shape_vec(
            (1, 1, 1, 3), // shape: (batch, seq, heads, dim)
            vec![1.0, 2.0, 3.0],
        )
        .unwrap();
        let output = softmax(&input);
        let sum_last_axis = output.sum_axis(Axis(3));
        assert!((sum_last_axis[[0, 0, 0]] - 1.0).abs() < 1e-6);
        assert!(output.iter().all(|&x| x > 0.0));
    }
    #[test]
    fn test_model_encode_and_cosine_similarity() -> Result<()> {
        let model = Model::from_pretrained(ModelType::MiniLML6V2)?;
        let texts = vec!["Hello world", "Hello world", "Goodbye world"];
        let embeddings = model.encode(texts.clone(), true)?;
        let sim_00 = cosine_similarity(&embeddings[0], &embeddings[0]);
        let sim_01 = cosine_similarity(&embeddings[0], &embeddings[1]);
        let sim_02 = cosine_similarity(&embeddings[0], &embeddings[2]);
        assert!((sim_00 - 1.0).abs() < 1e-6, "Self similarity should be 1");
        assert!(
            sim_01 <= 1.0 && sim_01 >= -1.0,
            "Cosine similarity in [-1,1]"
        );
        assert!(
            sim_02 <= 1.0 && sim_02 >= -1.0,
            "Cosine similarity in [-1,1]"
        );
        let sim_10 = cosine_similarity(&embeddings[1], &embeddings[0]);
        assert!(
            (sim_01 - sim_10).abs() < 1e-6,
            "Cosine similarity should be symmetric"
        );

        Ok(())
    }
    #[test]
    fn test_layer_norm() {
        let input: Array3<f32> =
            Array3::from_shape_vec((1, 1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let ln = LayerNorm {
            weight: Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]),
            bias: Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]),
            eps: 1e-5,
        };
        let output = apply_layer_norm_3d(&input, &ln);
        let mean = output.mean_axis(Axis(2)).unwrap();
        let std = output.std_axis(Axis(2), 0.0);
        assert!((mean[[0, 0]]).abs() < 1e-5);
        assert!((std[[0, 0]] - 1.0).abs() < 1e-5);
    }
}
