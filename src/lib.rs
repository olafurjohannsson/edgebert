#[cfg(not(target_arch = "wasm32"))]
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
}

struct BertLayer {
    attention: MultiHeadAttention,
    intermediate: FeedForward,
    layer_norm1: LayerNorm,
    layer_norm2: LayerNorm,
}

struct MultiHeadAttention {
    query_weight: Array2<f32>,
    query_bias: Array1<f32>,
    key_weight: Array2<f32>,
    key_bias: Array1<f32>,
    value_weight: Array2<f32>,
    value_bias: Array1<f32>,
    output_weight: Array2<f32>,
    output_bias: Array1<f32>,
    num_heads: usize,
    head_dim: usize,
}

struct FeedForward {
    dense1_weight: Array2<f32>,
    dense1_bias: Array1<f32>,
    dense2_weight: Array2<f32>,
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
                query_weight: weights
                    .get_array2(&format!("{}.attention.self.query.weight", prefix))?,
                query_bias: weights.get_array1(&format!("{}.attention.self.query.bias", prefix))?,
                key_weight: weights.get_array2(&format!("{}.attention.self.key.weight", prefix))?,
                key_bias: weights.get_array1(&format!("{}.attention.self.key.bias", prefix))?,
                value_weight: weights
                    .get_array2(&format!("{}.attention.self.value.weight", prefix))?,
                value_bias: weights.get_array1(&format!("{}.attention.self.value.bias", prefix))?,
                output_weight: weights
                    .get_array2(&format!("{}.attention.output.dense.weight", prefix))?,
                output_bias: weights
                    .get_array1(&format!("{}.attention.output.dense.bias", prefix))?,
                num_heads: config.num_attention_heads,
                head_dim: config.hidden_size / config.num_attention_heads,
            };

            let intermediate = FeedForward {
                dense1_weight: weights
                    .get_array2(&format!("{}.intermediate.dense.weight", prefix))?,
                dense1_bias: weights.get_array1(&format!("{}.intermediate.dense.bias", prefix))?,
                dense2_weight: weights.get_array2(&format!("{}.output.dense.weight", prefix))?,
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
    pub fn encode(&self, texts: Vec<&str>, normalize: bool) -> Result<Vec<Vec<f32>>> {
        #[cfg(target_arch = "wasm32")]
        let encodings = self.tokenizer.encode_batch(texts, 512)?;

        #[cfg(not(target_arch = "wasm32"))]
        let encodings = self
            .tokenizer
            .encode_batch(texts, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let batch_size = encodings.len();

        if batch_size == 0 {
            return Ok(vec![]);
        }

        let max_len = encodings.iter().map(|e| e.len()).max().unwrap();

        // Prepare input tensors from actual tokenizer output
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
        let vector: Vec<Vec<f32>> = embeddings.outer_iter().map(|row| row.to_vec()).collect();

        if normalize {
            Ok(vector
                .into_iter()
                .map(|emb| {
                    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                    emb.into_iter().map(|x| x / norm.max(1e-12)).collect()
                })
                .collect())
        } else {
            Ok(vector)
        }
    }

    fn forward(
        &self,
        input_ids: &Array2<f32>,
        attention_mask: &Array2<f32>,
    ) -> Result<Array2<f32>> {
        let batch_size = input_ids.shape()[0];
        let seq_len = input_ids.shape()[1];

        // Embedding layer
        let mut hidden = Array3::<f32>::zeros((batch_size, seq_len, self.config.hidden_size));

        // Word embeddings
        for i in 0..batch_size {
            for j in 0..seq_len {
                let token_id = input_ids[[i, j]] as usize;
                let word_emb = self.word_embeddings.row(token_id);
                let pos_emb = self.position_embeddings.row(j);
                let type_emb = self.token_type_embeddings.row(0); // token_type_id = 0

                for k in 0..self.config.hidden_size {
                    hidden[[i, j, k]] = word_emb[k] + pos_emb[k] + type_emb[k];
                }
            }
        }

        // Apply embedding layer norm
        hidden = apply_layer_norm_3d(&hidden, &self.layer_norm_final);

        // Pass through layers
        for layer in &self.layers {
            hidden = layer.forward(hidden, attention_mask)?;
        }

        // Mean pooling
        mean_pool(&hidden, attention_mask)
    }
}

impl BertLayer {
    fn forward(&self, input: Array3<f32>, attention_mask: &Array2<f32>) -> Result<Array3<f32>> {
        // Self attention
        let attention_out = self.attention.forward(&input, attention_mask)?;
        let attention_out = &input + &attention_out;
        let attention_out = apply_layer_norm_3d(&attention_out, &self.layer_norm1);

        // Feed forward
        let ff_out = self.intermediate.forward(&attention_out)?;
        let output = &attention_out + &ff_out;
        let output = apply_layer_norm_3d(&output, &self.layer_norm2);

        Ok(output)
    }
}

impl MultiHeadAttention {
    fn forward(&self, hidden: &Array3<f32>, attention_mask: &Array2<f32>) -> Result<Array3<f32>> {
        let batch_size = hidden.shape()[0];
        let seq_len = hidden.shape()[1];

        // Linear projections WITH BIAS
        let q = matmul_3d_2d(hidden, &self.query_weight.t().to_owned());
        let q = q + &self
            .query_bias
            .view()
            .insert_axis(Axis(0))
            .insert_axis(Axis(0));

        let k = matmul_3d_2d(hidden, &self.key_weight.t().to_owned());
        let k = k + &self
            .key_bias
            .view()
            .insert_axis(Axis(0))
            .insert_axis(Axis(0));

        let v = matmul_3d_2d(hidden, &self.value_weight.t().to_owned());
        let v = v + &self
            .value_bias
            .view()
            .insert_axis(Axis(0))
            .insert_axis(Axis(0));

        let q = q
            .to_shape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .to_owned() // Create owned contiguous copy
            .permuted_axes([0, 2, 1, 3]);

        let k = k
            .to_shape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .to_owned()
            .permuted_axes([0, 2, 1, 3]);

        let v = v
            .to_shape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .to_owned()
            .permuted_axes([0, 2, 1, 3]);

        // Compute attention scores
        let scores = matmul_4d(&q, &k.permuted_axes([0, 1, 3, 2]));
        let scores = scores / (self.head_dim as f32).sqrt();

        // Apply mask
        let scores = apply_attention_mask_4d(scores, attention_mask);

        // Softmax
        let weights = softmax(&scores);

        // Apply attention to values
        let context = matmul_4d(&weights, &v);

        // Reshape back - use to_shape again
        let context = context.permuted_axes([0, 2, 1, 3]);
        let context = context
            .to_shape((batch_size, seq_len, self.num_heads * self.head_dim))?
            .to_owned();

        let output = matmul_3d_2d(&context, &self.output_weight.t().to_owned());
        Ok(output
            + &self
                .output_bias
                .view()
                .insert_axis(Axis(0))
                .insert_axis(Axis(0)))
    }
}

impl FeedForward {
    fn forward(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
        let intermediate = matmul_3d_2d(hidden, &self.dense1_weight.t().to_owned());
        let intermediate = intermediate
            + &self
                .dense1_bias
                .view()
                .insert_axis(Axis(0))
                .insert_axis(Axis(0));
        let intermediate = gelu(&intermediate);

        let output = matmul_3d_2d(&intermediate, &self.dense2_weight.t().to_owned());
        Ok(output
            + &self
                .dense2_bias
                .view()
                .insert_axis(Axis(0))
                .insert_axis(Axis(0)))
    }
}

// helpers
fn matmul_3d_2d(a: &Array3<f32>, b: &Array2<f32>) -> Array3<f32> {
    let (batch, m, k) = a.dim();
    let n = b.shape()[1];
    assert_eq!(
        k,
        b.shape()[0],
        "Matrix dimensions are incompatible for multiplication"
    );

    let a_cont = a.as_standard_layout();
    let b_cont = b.as_standard_layout();

    let mut c = Array3::<f32>::zeros((batch, m, n));

    Zip::from(c.axis_iter_mut(Axis(0)))
        .and(a_cont.axis_iter(Axis(0)))
        .for_each(|mut c_slice, a_slice| {
            let result = a_slice.dot(&b_cont);
            c_slice.assign(&result);
        });

    c
}
// just a helper, used for unit tests, can be improved
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x*x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x*x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b + 1e-8)
}
// todo: vectorize
fn softmax(scores: &Array4<f32>) -> Array4<f32> {
    let max_along_axis = scores.fold_axis(Axis(3), f32::NEG_INFINITY, |&acc, &elem| acc.max(elem));
    let max_expanded = max_along_axis.insert_axis(Axis(3));
    let mut stable_scores = scores.to_owned();
    stable_scores -= &max_expanded;

    // Exponentiate
    let exp_scores = stable_scores.mapv_into(f32::exp);

    // Sum along the axis and normalize
    let sum_exp = exp_scores.sum_axis(Axis(3));
    &exp_scores / &sum_exp.insert_axis(Axis(3)).mapv(|x| x.max(1e-9))
}
fn apply_attention_mask_4d(mut scores: Array4<f32>, mask: &Array2<f32>) -> Array4<f32> {
    let scores_shape = scores.dim();
    let mask_4d = mask.clone().insert_axis(Axis(1)).insert_axis(Axis(2));

    if let Some(broadcast_mask) = mask_4d.broadcast(scores_shape) {
        // Broadcast successful now vectorize
        Zip::from(&mut scores)
            .and(broadcast_mask)
            .for_each(|score_elem, mask_elem| {
                if *mask_elem == 0.0 {
                    *score_elem = -1e9;
                }
            });
    }

    scores
}

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

fn gelu(x: &Array3<f32>) -> Array3<f32> {
    x.mapv(|val| {
        let cdf = 0.5
            * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (val + 0.044715 * val.powi(3))).tanh());
        val * cdf
    })
}

// todo: very bad
fn apply_layer_norm_3d(hidden: &Array3<f32>, ln: &LayerNorm) -> Array3<f32> {
    let (batch, seq, hidden_size) = hidden.dim();
    let mut result = Array3::<f32>::zeros((batch, seq, hidden_size));

    for b in 0..batch {
        for s in 0..seq {
            let slice = hidden.slice(s![b, s, ..]);

            // Compute mean and variance manually
            let mean = slice.mean().unwrap();
            let var = slice.mapv(|x| (x - mean).powi(2)).mean().unwrap();
            let std = (var + ln.eps).sqrt();

            // Broadcast weight and bias for LayerNorm
            let normalized = (&slice - mean) / std;
            let scaled = &normalized * &ln.weight + &ln.bias;

            result.slice_mut(s![b, s, ..]).assign(&scaled);
        }
    }

    result
}

fn mean_pool(hidden: &Array3<f32>, attention_mask: &Array2<f32>) -> Result<Array2<f32>> {
    let (batch, _, hidden_size) = hidden.dim();
    let mut result = Array2::zeros((batch, hidden_size));

    for b in 0..batch {
        let mut sum = Array1::zeros(hidden_size);
        let mut count = 0.0;

        for s in 0..hidden.shape()[1] {
            if attention_mask[[b, s]] == 1.0 {
                sum = sum + hidden.slice(s![b, s, ..]);
                count += 1.0;
            }
        }

        if count > 0.0 {
            result.slice_mut(s![b, ..]).assign(&(sum / count));
        }
    }

    Ok(result)
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
            WasmModelType::MiniLML6V2 => ModelType::MiniLML6V2
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
    pub fn encode(&self, texts: Vec<String>, normalize: bool) -> Result<Vec<f32>, JsValue> {
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
    use ndarray::{Array4};

    #[test]
    fn test_softmax_simple() {
        let input: Array4<f32> = Array4::from_shape_vec(
            (1, 1, 1, 3),      // shape: (batch, seq, heads, dim)
            vec![1.0, 2.0, 3.0]
        ).unwrap();
        let output = softmax(&input);
        let sum_last_axis = output.sum_axis(Axis(3));
        assert!((sum_last_axis[[0,0,0]] - 1.0).abs() < 1e-6);
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
        assert!(sim_01 <= 1.0 && sim_01 >= -1.0, "Cosine similarity in [-1,1]");
        assert!(sim_02 <= 1.0 && sim_02 >= -1.0, "Cosine similarity in [-1,1]");
        let sim_10 = cosine_similarity(&embeddings[1], &embeddings[0]);
        assert!((sim_01 - sim_10).abs() < 1e-6, "Cosine similarity should be symmetric");

        Ok(())
    }
    #[test]
    fn test_layer_norm() {
        let input: Array3<f32> = Array3::from_shape_vec(
            (1, 1, 4),
            vec![1.0, 2.0, 3.0, 4.0]
        ).unwrap();

        let ln = LayerNorm {
            weight: Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]),
            bias: Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]),
            eps: 1e-5,
        };
        let output = apply_layer_norm_3d(&input, &ln);
        let mean = output.mean_axis(Axis(2)).unwrap();
        let std = output.std_axis(Axis(2), 0.0);
        assert!((mean[[0,0]]).abs() < 1e-5);
        assert!((std[[0,0]] - 1.0).abs() < 1e-5);
    }
}