//! High-level interface for loading model weights.

#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use ndarray::{Array1, Array2, Array3, ArrayD, IxDyn};
#[cfg(not(target_arch = "wasm32"))]
use serde_json::json;

use super::WeightLoader;
#[cfg(not(target_arch = "wasm32"))]
use super::gguf_loader::GgufLoader;
use super::safetensors_loader::SafeTensorsLoader;
use crate::tensor::raw_tensor::TensorView;
use crate::tensor::{CpuTensor, DType, QuantizedMatrix};
#[cfg(not(target_arch = "wasm32"))]
use crate::weights::raw_to_typed_gguf;

#[derive(Debug, Clone, Copy)]
pub struct AttentionLayout {
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

struct ModelWeightsInner {
    loader: Box<dyn WeightLoader + Send + Sync>,
    config_json: String,
    is_gguf: bool,
}

/// High-level interface for loading model weights.
pub struct ModelWeights {
    inner: Arc<ModelWeightsInner>,
}

impl Clone for ModelWeights {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl ModelWeights {
    /// Creates a new ModelWeights from a path.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(path: &Path) -> Result<Self> {
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("gguf") {
            return Self::from_gguf_file(path);
        }

        if path.is_dir() {
            if path.join("model.safetensors").exists()
                || path.join("model.safetensors.index.json").exists()
            {
                let loader =
                    Box::new(SafeTensorsLoader::new(path)?) as Box<dyn WeightLoader + Send + Sync>;
                let config_json = std::fs::read_to_string(path.join("config.json"))
                    .unwrap_or_else(|_| "{}".to_string());

                return Ok(Self {
                    inner: Arc::new(ModelWeightsInner {
                        loader,
                        config_json,
                        is_gguf: false,
                    }),
                });
            }

            let gguf_in_dir = std::fs::read_dir(path)?
                .filter_map(|e| e.ok())
                .find(|e| e.path().extension().and_then(|s| s.to_str()) == Some("gguf"));

            if let Some(entry) = gguf_in_dir {
                return Self::from_gguf_file(&entry.path());
            }
        }

        Err(anyhow!("no supported weight format found at {:?}", path))
    }

    /// Load from raw safetensors bytes + config JSON (for WASM)
    pub fn from_safetensors_bytes(data: &[u8], config_json: &str) -> Result<Self> {
        let loader = SafeTensorsLoader::from_bytes(data)?;
        Ok(Self {
            inner: Arc::new(ModelWeightsInner {
                loader: Box::new(loader),
                config_json: config_json.to_string(),
                is_gguf: false,
            }),
        })
    }

    /// Creates `ModelWeights` from an unpacked `.kjq` container.
    ///
    /// A `KJQ1` file arrives as f32 safetensors and takes the ordinary path. A
    /// `KJQ8` file keeps its `BlockQ8_0` tensors, which is the whole point of
    /// that encoding: expanding a 494M parameter decoder to f32 needs 1.98GB,
    /// and wasm32 caps a single allocation at 2GB.
    pub fn from_kjq(unpacked: &crate::weights::kjq::KjqUnpacked) -> Result<Self> {
        use crate::weights::kjq::KjqEncoding;

        let loader: Box<dyn WeightLoader + Send + Sync> = match unpacked.encoding {
            KjqEncoding::Kjq1 => Box::new(SafeTensorsLoader::from_bytes(&unpacked.safetensors)?),
            KjqEncoding::Kjq8 => Box::new(crate::weights::kjq_loader::KjqLoader::new(unpacked)?),
        };

        Ok(Self {
            inner: Arc::new(ModelWeightsInner {
                loader,
                config_json: unpacked.config_json.clone(),
                is_gguf: false,
            }),
        })
    }

    /// Creates ModelWeights from a specific file.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_file(path: &Path) -> Result<Self> {
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("gguf") {
            return Self::from_gguf_file(path);
        }

        if path.is_dir() {
            return Self::new(path);
        }

        if path.exists() {
            let loader =
                Box::new(SafeTensorsLoader::new(path)?) as Box<dyn WeightLoader + Send + Sync>;
            return Ok(Self {
                inner: Arc::new(ModelWeightsInner {
                    loader,
                    config_json: "{}".to_string(),
                    is_gguf: false,
                }),
            });
        }

        Self::new(path)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn from_gguf_file(path: &Path) -> Result<Self> {
        let loader = GgufLoader::new(path)?;

        if log::log_enabled!(log::Level::Debug) {
            loader.debug_print_tensors();
        }

        let config_json = Self::synthesize_config_from_gguf(&loader)?;

        Ok(Self {
            inner: Arc::new(ModelWeightsInner {
                loader: Box::new(loader) as Box<dyn WeightLoader + Send + Sync>,
                config_json,
                is_gguf: true,
            }),
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn synthesize_config_from_gguf(loader: &GgufLoader) -> Result<String> {
        let arch = loader.get_string("general.architecture").unwrap_or("llama");

        let rope_scaling = loader
            .get_string(&format!("{}.rope.scaling.type", arch))
            .map(|_| {
                json!({
                    "rope_type": loader.get_string(&format!("{}.rope.scaling.type", arch))
                        .unwrap_or("llama3"),
                    "factor": loader.get_f32(&format!("{}.rope.scaling.factor", arch))
                        .unwrap_or(32.0),
                    "low_freq_factor": loader.get_f32(&format!("{}.rope.scaling.low_freq_factor", arch))
                        .unwrap_or(1.0),
                    "high_freq_factor": loader.get_f32(&format!("{}.rope.scaling.high_freq_factor", arch))
                        .unwrap_or(4.0),
                    "original_max_position_embeddings": loader
                        .get_u32(&format!("{}.rope.scaling.orig_ctx_len", arch))
                        .unwrap_or(8192),
                })
            });

        let config = json!({
            "architecture": arch,
            "hidden_size": loader.get_u32(&format!("{}.embedding_length", arch)),
            "intermediate_size": loader.get_u32(&format!("{}.feed_forward_length", arch)),
            "num_attention_heads": loader.get_u32(&format!("{}.attention.head_count", arch)),
            "num_hidden_layers": loader.get_u32(&format!("{}.block_count", arch)),
            "num_key_value_heads": loader.get_u32(&format!("{}.attention.head_count_kv", arch)),
            "max_position_embeddings": loader.get_u32(&format!("{}.context_length", arch)),
            "rope_theta": loader.get_f32(&format!("{}.rope.freq_base", arch)).unwrap_or(10000.0),
            "rope_scaling": rope_scaling,
            "rms_norm_eps": loader
                .get_f32(&format!("{}.attention.layer_norm_rms_epsilon", arch))
                .unwrap_or(1e-5),
            "vocab_size": loader.get_u32(&format!("{}.vocab_size", arch)).unwrap_or(128256),
            "bos_token_id": loader.get_u32("tokenizer.ggml.bos_token_id").unwrap_or(128000),
            "eos_token_id": loader.get_u32("tokenizer.ggml.eos_token_id").unwrap_or(128001),
        });

        Ok(config.to_string())
    }

    /// Returns a reference to the underlying loader.
    pub fn loader(&self) -> &(dyn WeightLoader + Send + Sync) {
        &*self.inner.loader
    }

    /// Returns the model configuration JSON.
    pub fn config_json(&self) -> &str {
        &self.inner.config_json
    }

    /// Returns `true` if a tensor exists.
    pub fn contains(&self, name: &str) -> bool {
        self.inner.loader.contains(name)
    }

    /// Returns `true` if this is a GGUF file.
    pub fn is_gguf(&self) -> bool {
        self.inner.is_gguf
    }

    /// Returns the model_type from config.
    pub fn model_type(&self) -> Option<String> {
        let v: serde_json::Value = serde_json::from_str(&self.inner.config_json).ok()?;
        v.get("model_type")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }

    pub fn is_bert(&self) -> bool {
        matches!(self.model_type(), Some(mt) if mt.eq_ignore_ascii_case("bert"))
    }

    pub fn is_mpnet(&self) -> bool {
        matches!(self.model_type(), Some(mt) if mt.eq_ignore_ascii_case("mpnet"))
    }

    pub fn is_distilbert(&self) -> bool {
        matches!(self.model_type(), Some(mt) if mt.eq_ignore_ascii_case("distilbert"))
    }

    pub fn is_roberta(&self) -> bool {
        matches!(self.model_type(), Some(mt) if mt.eq_ignore_ascii_case("roberta"))
    }

    pub fn is_distilroberta(&self) -> bool {
        matches!(self.model_type(), Some(mt) if mt.eq_ignore_ascii_case("distilroberta"))
    }

    /// Processes a tensor's raw bytes through a callback.
    pub fn with_raw_tensor<R, F>(&self, name: &str, f: F) -> Result<R>
    where
        F: FnOnce(TensorView<'_>) -> Result<R>,
    {
        let view = self.inner.loader.get_raw(name)?;
        f(view)
    }

    /// Resolves the dtype of a tensor, optionally overriding.
    pub fn resolve_dtype(&self, tensor_name: &str, override_dtype: Option<DType>) -> Result<DType> {
        if let Some(dt) = override_dtype {
            return Ok(dt);
        }
        self.with_raw_tensor(tensor_name, |view| Ok(view.dtype))
    }

    /// Returns a typed CPU tensor. Prefer `with_raw_tensor` for large tensors.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn get_typed_tensor(&self, name: &str) -> Result<CpuTensor> {
        self.with_raw_tensor(name, |raw| {
            if self.inner.is_gguf {
                let attn = self
                    .inner
                    .loader
                    .as_any()
                    .downcast_ref::<GgufLoader>()
                    .and_then(|g| g.attention_layout());
                raw_to_typed_gguf(raw, attn)
            } else {
                raw_to_typed(raw)
            }
        })
    }

    /// Returns a typed CPU tensor. Prefer `with_raw_tensor` for large tensors.
    #[cfg(target_arch = "wasm32")]
    pub fn get_typed_tensor(&self, name: &str) -> Result<CpuTensor> {
        self.with_raw_tensor(name, raw_to_typed)
    }

    /// Returns a 1D f32 array. Only use for small tensors like biases.
    pub fn get_array1(&self, name: &str) -> Result<Array1<f32>> {
        #[allow(deprecated)]
        self.get_typed_tensor(name)?
            .to_array1_f32()
            .map_err(|e| anyhow!("failed to load '{}': {}", name, e))
    }

    /// Returns a 2D f32 array. Prefer `with_raw_tensor` for large matrices.
    pub fn get_array2(&self, name: &str) -> Result<Array2<f32>> {
        #[allow(deprecated)]
        self.get_typed_tensor(name)?
            .to_array2_f32()
            .map_err(|e| anyhow!("failed to load '{}': {}", name, e))
    }

    /// Returns a 3D f32 array. Prefer `with_raw_tensor` for large tensors.
    pub fn get_array3(&self, name: &str) -> Result<Array3<f32>> {
        #[allow(deprecated)]
        self.get_typed_tensor(name)?
            .to_array3_f32()
            .map_err(|e| anyhow!("failed to load '{}': {}", name, e))
    }

    /// Returns tensor shape without loading data.
    pub fn tensor_shape(&self, name: &str) -> Result<Vec<usize>> {
        self.with_raw_tensor(name, |view| Ok(view.shape.clone()))
    }

    /// Returns tensor dtype without loading data.
    pub fn tensor_dtype(&self, name: &str) -> Result<DType> {
        self.with_raw_tensor(name, |view| Ok(view.dtype))
    }

    /// Returns tensor size in bytes without loading data.
    pub fn tensor_size_bytes(&self, name: &str) -> Result<usize> {
        self.with_raw_tensor(name, |view| Ok(view.bytes.len()))
    }
}

/// Copies bytes to a typed vector, handling alignment.
pub fn cast_or_copy<T: bytemuck::Pod + bytemuck::Zeroable>(bytes: &[u8]) -> Vec<T> {
    if let Ok(slice) = bytemuck::try_cast_slice(bytes) {
        slice.to_vec()
    } else {
        let mut aligned = vec![0u8; bytes.len()];
        aligned.copy_from_slice(bytes);
        bytemuck::cast_slice(&aligned).to_vec()
    }
}

/// Converts a TensorView to a typed CpuTensor.
pub fn raw_to_typed(raw: TensorView<'_>) -> Result<CpuTensor> {
    match raw.dtype {
        DType::F32 => Ok(CpuTensor::F32(ArrayD::from_shape_vec(
            IxDyn(&raw.shape),
            cast_or_copy(&raw.bytes),
        )?)),

        DType::F16 => Ok(CpuTensor::F16(ArrayD::from_shape_vec(
            IxDyn(&raw.shape),
            cast_or_copy(&raw.bytes),
        )?)),

        DType::BF16 => Ok(CpuTensor::BF16(ArrayD::from_shape_vec(
            IxDyn(&raw.shape),
            cast_or_copy(&raw.bytes),
        )?)),

        DType::Q8_0 => Ok(CpuTensor::Q8_0(QuantizedMatrix {
            blocks: cast_or_copy(&raw.bytes),
            shape: [raw.shape[0], raw.shape[1]],
        })),

        DType::Q4_K => Ok(CpuTensor::Q4_K(QuantizedMatrix {
            blocks: cast_or_copy(&raw.bytes),
            shape: [raw.shape[0], raw.shape[1]],
        })),

        DType::Q6_K => Ok(CpuTensor::Q6_K(QuantizedMatrix {
            blocks: cast_or_copy(&raw.bytes),
            shape: [raw.shape[0], raw.shape[1]],
        })),

        _ => Err(anyhow!("unsupported dtype for conversion: {:?}", raw.dtype)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tempfile::TempDir;

    fn create_test_safetensors(
        dir: &TempDir,
        tensors: &[(&str, Vec<f32>, Vec<usize>)],
    ) -> Result<()> {
        use safetensors::tensor::{Dtype, TensorView as StTensorView};

        let stored: Vec<(String, Vec<usize>, Vec<u8>)> = tensors
            .iter()
            .map(|(name, values, shape)| {
                let bytes: Vec<u8> = values.iter().flat_map(|f| f.to_le_bytes()).collect();
                (name.to_string(), shape.clone(), bytes)
            })
            .collect();

        let mut tensor_map = HashMap::new();
        for (name, shape, bytes) in &stored {
            tensor_map.insert(
                name.clone(),
                StTensorView::new(Dtype::F32, shape.clone(), bytes)?,
            );
        }

        let file_path = dir.path().join("model.safetensors");
        safetensors::serialize_to_file(&tensor_map, &None, &file_path)?;

        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "bert", "hidden_size": 4}"#,
        )?;

        Ok(())
    }

    #[test]
    fn test_model_weights_new_safetensors() {
        let dir = tempfile::tempdir().unwrap();
        create_test_safetensors(
            &dir,
            &[("test.weight", vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])],
        )
        .unwrap();

        let weights = ModelWeights::new(dir.path()).unwrap();
        assert!(!weights.is_gguf());
        assert!(weights.contains("test.weight"));
        assert!(!weights.contains("nonexistent"));
    }

    #[test]
    fn test_with_raw_tensor() {
        let dir = tempfile::tempdir().unwrap();
        create_test_safetensors(
            &dir,
            &[("layer.weight", vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])],
        )
        .unwrap();

        let weights = ModelWeights::new(dir.path()).unwrap();

        let result = weights
            .with_raw_tensor("layer.weight", |view| {
                assert_eq!(view.shape, vec![2, 2]);
                assert_eq!(view.dtype, DType::F32);
                assert_eq!(view.bytes.len(), 16);
                Ok(view.shape.iter().product::<usize>())
            })
            .unwrap();

        assert_eq!(result, 4);
    }

    #[test]
    fn test_tensor_shape_without_loading() {
        let dir = tempfile::tempdir().unwrap();
        create_test_safetensors(&dir, &[("big.weight", vec![0.0; 1000], vec![10, 100])]).unwrap();

        let weights = ModelWeights::new(dir.path()).unwrap();
        let shape = weights.tensor_shape("big.weight").unwrap();
        assert_eq!(shape, vec![10, 100]);
    }

    #[test]
    fn test_tensor_dtype() {
        let dir = tempfile::tempdir().unwrap();
        create_test_safetensors(&dir, &[("test.weight", vec![1.0], vec![1])]).unwrap();

        let weights = ModelWeights::new(dir.path()).unwrap();
        let dtype = weights.tensor_dtype("test.weight").unwrap();
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_model_type_detection() {
        let dir = tempfile::tempdir().unwrap();
        create_test_safetensors(&dir, &[("test.weight", vec![1.0], vec![1])]).unwrap();

        let weights = ModelWeights::new(dir.path()).unwrap();
        assert!(weights.is_bert());
        assert!(!weights.is_distilbert());
    }

    #[test]
    fn test_resolve_dtype() {
        let dir = tempfile::tempdir().unwrap();
        create_test_safetensors(&dir, &[("test.weight", vec![1.0], vec![1])]).unwrap();

        let weights = ModelWeights::new(dir.path()).unwrap();

        let dtype = weights
            .resolve_dtype("test.weight", Some(DType::F16))
            .unwrap();
        assert_eq!(dtype, DType::F16);

        let dtype = weights.resolve_dtype("test.weight", None).unwrap();
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_missing_tensor_error() {
        let dir = tempfile::tempdir().unwrap();
        create_test_safetensors(&dir, &[("exists.weight", vec![1.0], vec![1])]).unwrap();

        let weights = ModelWeights::new(dir.path()).unwrap();

        let result = weights.with_raw_tensor("does_not_exist", |_| Ok(()));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn test_config_json_loaded() {
        let dir = tempfile::tempdir().unwrap();
        create_test_safetensors(&dir, &[("test.weight", vec![1.0], vec![1])]).unwrap();

        let weights = ModelWeights::new(dir.path()).unwrap();
        let config = weights.config_json();

        assert!(config.contains("bert"));
        assert!(config.contains("hidden_size"));
    }

    #[test]
    fn test_cast_or_copy_aligned() {
        let floats = vec![1.0f32, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();

        let result: Vec<f32> = cast_or_copy(&bytes);
        assert_eq!(result, floats);
    }

    #[test]
    fn test_cast_or_copy_misaligned() {
        let floats = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut bytes: Vec<u8> = vec![0];
        bytes.extend(floats.iter().flat_map(|f| f.to_le_bytes()));

        let result: Vec<f32> = cast_or_copy(&bytes[1..]);
        assert_eq!(result, floats);
    }
}
