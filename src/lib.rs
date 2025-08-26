use anyhow::Result;
use ndarray::{Zip, Array1, Array2, Array3, Array4, Axis, s, ArrayView3};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
mod tokenizer;
mod weights;

use weights::ModelWeights;

#[cfg(target_arch = "wasm32")]
pub use tokenizer::WordPieceTokenizer as ModelTokenizer;

#[cfg(not(target_arch = "wasm32"))]
use tokenizers::Tokenizer as ModelTokenizer;

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
    #[serde(default)]
    pub type_vocab_size: usize,
}

// #[wasm_bindgen]
impl Model {
    pub fn from_weights(weights: ModelWeights, tokenizer: ModelTokenizer, config: Config) -> Result<Self> {
        // Load embeddings
        let word_embeddings = weights.get_array2("embeddings.word_embeddings.weight")?;
        let position_embeddings = weights.get_array2("embeddings.position_embeddings.weight")?;
        let token_type_embeddings = weights.get_array2("embeddings.token_type_embeddings.weight")?;

        // Load layers
        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let prefix = format!("encoder.layer.{}", i);

            let attention = MultiHeadAttention {
                query_weight: weights.get_array2(&format!("{}.attention.self.query.weight", prefix))?,
                query_bias: weights.get_array1(&format!("{}.attention.self.query.bias", prefix))?,
                key_weight: weights.get_array2(&format!("{}.attention.self.key.weight", prefix))?,
                key_bias: weights.get_array1(&format!("{}.attention.self.key.bias", prefix))?,
                value_weight: weights.get_array2(&format!("{}.attention.self.value.weight", prefix))?,
                value_bias: weights.get_array1(&format!("{}.attention.self.value.bias", prefix))?,
                output_weight: weights.get_array2(&format!("{}.attention.output.dense.weight", prefix))?,
                output_bias: weights.get_array1(&format!("{}.attention.output.dense.bias", prefix))?,
                num_heads: config.num_attention_heads,
                head_dim: config.hidden_size / config.num_attention_heads,
            };

            let intermediate = FeedForward {
                dense1_weight: weights.get_array2(&format!("{}.intermediate.dense.weight", prefix))?,
                dense1_bias: weights.get_array1(&format!("{}.intermediate.dense.bias", prefix))?,
                dense2_weight: weights.get_array2(&format!("{}.output.dense.weight", prefix))?,
                dense2_bias: weights.get_array1(&format!("{}.output.dense.bias", prefix))?,
            };

            let layer_norm1 = LayerNorm {
                weight: weights.get_array1(&format!("{}.attention.output.LayerNorm.weight", prefix))?,
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
    pub fn from_pretrained(model_path: &str) -> Result<Self> {
        let weights = ModelWeights::load(model_path)?;
        let config = weights.config.clone();

        #[cfg(not(target_arch = "wasm32"))]
        {
            let tokenizer_file = &format!("{}_tokenizer.json", model_path);
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

        #[cfg(target_arch = "wasm32")]
        {
            let tokenizer_file = &format!("{}_tokenizer.json", model_path);
            let tokenizer = ModelTokenizer::from_file(tokenizer_file)?;
            Self::from_weights(weights, tokenizer, config)

        }
    }

    pub fn encode(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>> {
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

        Ok(embeddings.outer_iter().map(|row| row.to_vec()).collect())
    }

    pub fn encode_normalized(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>> {
        let embeddings = self.encode(texts)?;
        Ok(embeddings
            .into_iter()
            .map(|emb| {
                let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                emb.into_iter().map(|x| x / norm.max(1e-12)).collect()
            })
            .collect())
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
        let scores = matmul_4d_fixed(&q, &k.permuted_axes([0, 1, 3, 2]));
        let scores = scores / (self.head_dim as f32).sqrt();

        // Apply mask
        let scores = apply_attention_mask_4d(scores, attention_mask);

        // Softmax
        let weights = softmax_4d_fixed(&scores);

        // Apply attention to values
        let context = matmul_4d_fixed(&weights, &v);

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
    assert_eq!(k, b.shape()[0], "Matrix dimensions are incompatible for multiplication");

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

// todo: vectorize
fn softmax_4d_fixed(scores: &Array4<f32>) -> Array4<f32> {
    let mut result = scores.clone();
    let (batch, heads, seq1, seq2) = result.dim();

    for b in 0..batch {
        for h in 0..heads {
            for s1 in 0..seq1 {
                // Find max
                let mut max = f32::NEG_INFINITY;
                for s2 in 0..seq2 {
                    max = max.max(result[[b, h, s1, s2]]);
                }

                // Exp and sum
                let mut sum = 0.0;
                for s2 in 0..seq2 {
                    result[[b, h, s1, s2]] = (result[[b, h, s1, s2]] - max).exp();
                    sum += result[[b, h, s1, s2]];
                }
                // Clamp
                sum = sum.max(1e-9);

                // Normalize
                for s2 in 0..seq2 {
                    result[[b, h, s1, s2]] /= sum;
                }
            }
        }
    }
    result
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
fn matmul_4d_fixed(a: &Array4<f32>, b: &Array4<f32>) -> Array4<f32> {
    // 1. Get original dimensions
    let (batch, heads, seq1, dim) = a.dim();
    let seq2 = b.shape()[3];

    let a_cont = a.as_standard_layout();
    let b_cont = b.as_standard_layout();
    let a_3d = a_cont.view().into_shape_with_order((batch * heads, seq1, dim)).unwrap();
    let b_3d = b_cont.view().into_shape_with_order((batch * heads, dim, seq2)).unwrap();

    // 3d output
    let mut c_3d = Array3::<f32>::zeros((batch * heads, seq1, seq2));

    // batch matrix mul
    Zip::from(c_3d.axis_iter_mut(Axis(0)))
        .and(a_3d.axis_iter(Axis(0)))
        .and(b_3d.axis_iter(Axis(0)))
        .for_each(|mut c_slice, a_slice, b_slice| {
            // possibly use BLAS
            let result = a_slice.dot(&b_slice);
            c_slice.assign(&result);
        });

    // reshape 3d into 4d
    c_3d.into_shape_with_order((batch, heads, seq1, seq2)).unwrap()
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
use wasm_bindgen::prelude::*;


#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmModel {
    inner: Model,
}
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmModel {
    #[wasm_bindgen(constructor)]
    pub fn new(weights_data: &[u8], config_json: &str, tokenizer_json: &str) -> Result<WasmModel, JsValue> {
        use crate::tokenizer::WordPieceTokenizer;

        // Parse weights from bytes
        let weights = ModelWeights::from_bytes(weights_data, config_json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Create tokenizer
        let tokenizer = WordPieceTokenizer::from_json_str(tokenizer_json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Parse config
        let config = serde_json::from_str(config_json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let model = Model::from_weights(weights, tokenizer, config)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(WasmModel { inner: model })
    }

    #[wasm_bindgen]
    pub fn encode(&self, texts: Vec<String>) -> Result<Vec<f32>, JsValue> {
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embeddings = self.inner.encode(text_refs)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Flatten for JS
        Ok(embeddings.into_iter().flatten().collect())
    }
}

