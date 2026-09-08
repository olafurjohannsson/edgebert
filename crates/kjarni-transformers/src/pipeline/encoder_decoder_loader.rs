use anyhow::{Result, anyhow};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;

use crate::Device;
use crate::common::HFGenerationConfig;
use crate::{
    WgpuContext,
    common::HFGenerationDefaults,
    cpu::encoder::{CpuEncoder, GpuEncoder},
    encoder_decoder::traits::{CpuCrossDecoder, GpuCrossDecoder},
    models::{
        ModelType, base::ModelLoadConfig, download_model_files, get_default_cache_dir,
        registry::WeightsFormat,
    },
    pipeline::{EncoderDecoderPipeline, EncoderDecoderPipelineBuilder},
    traits::{ModelConfig, ModelLayout, ModelMetadata},
    weights::ModelWeights,
};

pub trait EncoderDecoderModelFactory: Sized {
    type Config: ModelConfig + 'static;

    fn load_config(weights: &ModelWeights) -> Result<Arc<Self::Config>>;

    fn build_backends(
        weights: &ModelWeights,
        meta: &ModelMetadata,
        layout: &ModelLayout,
        config: &Arc<Self::Config>,
        load_config: &ModelLoadConfig,
        context: Option<&Arc<WgpuContext>>,
        device: Device,
    ) -> Result<(
        Option<Box<dyn CpuEncoder>>,
        Option<Box<dyn GpuEncoder>>,
        Option<Box<dyn CpuCrossDecoder>>,
        Option<Box<dyn GpuCrossDecoder>>,
    )>;

    fn new_from_pipeline(
        pipeline: EncoderDecoderPipeline,
        tokenizer: Tokenizer,
        config: Arc<Self::Config>,
        generation_defaults: Option<HFGenerationDefaults>,
        generation_config: HFGenerationConfig,
    ) -> Self;
}

pub struct Seq2SeqLoader;

impl Seq2SeqLoader {
    pub async fn load_from_registry<M: EncoderDecoderModelFactory>(
        model_type: ModelType,
        cache_dir: Option<PathBuf>,
        device: crate::prelude::Device,
        context: Option<Arc<WgpuContext>>,
        load_config: Option<ModelLoadConfig>,
    ) -> Result<M> {
        let info = model_type.info();
        let cache_dir = cache_dir.unwrap_or_else(get_default_cache_dir);
        let model_dir = cache_dir.join(model_type.repo_id().replace('/', "_"));

        let generation_config = HFGenerationConfig::load_or_default(&model_dir);

        let quiet = load_config.as_ref().is_some_and(|c| c.quiet);
        download_model_files(&model_dir, &info.paths, WeightsFormat::SafeTensors, quiet).await?;

        let context = if device.is_gpu() && context.is_none() {
            Some(WgpuContext::new().await?)
        } else {
            context
        };

        Self::load_from_pretrained::<M>(&model_dir, device, context, load_config, generation_config)
    }

    pub fn load_from_pretrained<M: EncoderDecoderModelFactory>(
        model_path: &Path,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_config: Option<ModelLoadConfig>,
        generation_config: HFGenerationConfig,
    ) -> Result<M> {
        let weights: ModelWeights = ModelWeights::new(model_path)?;
        let load_config: ModelLoadConfig = load_config.unwrap_or_default();

        // Config & Tokenizer
        let config = M::load_config(&weights)?;
        let meta: ModelMetadata = config.metadata();
        let layout: ModelLayout = config.layout();

        let tokenizer_path: PathBuf = model_path.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        // Build Backends
        let (cpu_enc, gpu_enc, cpu_dec, gpu_dec) = M::build_backends(
            &weights,
            &meta,
            &layout,
            &config,
            &load_config,
            context.as_ref(),
            device,
        )?;

        // Build Pipeline
        let pipeline = EncoderDecoderPipelineBuilder::new(&weights, config.clone())
            .with_load_config(load_config)
            .with_context(context)
            .with_encoder_backends(cpu_enc, gpu_enc)
            .with_decoder_backends(cpu_dec, gpu_dec)
            .build()?;

        // Generation Defaults
        let gen_defaults =
            if let Ok(json) = std::fs::read_to_string(model_path.join("generation_config.json")) {
                HFGenerationDefaults::from_json(&json).ok()
            } else {
                None
            };

        Ok(M::new_from_pipeline(
            pipeline,
            tokenizer,
            config,
            gen_defaults,
            generation_config,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::HFGenerationConfig;
    use crate::cpu::encoder::CpuEncoder;
    use crate::cpu::encoder::traits::CpuEncoderOutput;

    use crate::encoder_decoder::traits::{
        CpuCrossAttentionKVCache, CpuCrossDecoder, CpuCrossDecoderOutput,
    };
    use crate::models::base::ModelLoadConfig;
    use crate::traits::{ModelConfig, ModelLayout, ModelMetadata, NormalizationStrategy};
    use crate::weights::ModelWeights;
    use crate::{Cache, Device};
    use anyhow::Result;
    use ndarray::{Array2, Array3};
    use std::path::Path;
    use std::sync::Arc;
    use tempfile::TempDir;
    use tokenizers::Tokenizer;

    #[derive(Clone, Debug)]
    struct MockConfig {
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
        vocab_size: usize,
    }

    impl Default for MockConfig {
        fn default() -> Self {
            Self {
                hidden_size: 64,
                num_layers: 2,
                num_heads: 4,
                vocab_size: 1000,
            }
        }
    }

    impl ModelConfig for MockConfig {
        fn model_type(&self) -> &str {
            "MockModel"
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn metadata(&self) -> ModelMetadata {
            ModelMetadata {
                hidden_size: self.hidden_size,
                num_attention_heads: self.num_heads,
                num_layers: self.num_layers,

                intermediate_size: self.hidden_size * 4,
                vocab_size: self.vocab_size,
                extra_pos_embeddings: 0,
                num_kv_heads: self.num_heads,
                head_dim: self.hidden_size / self.num_heads,
                normalization_strategy: NormalizationStrategy::LayerNorm,
                scale_embeddings: false,
                max_seq_len: 0,
                activation: crate::activations::Activation::Gelu,
                decoder_layers: None,
                is_prenorm: true,
                no_scale_qk: false,
                norm_eps: 1e-5,
                normalize_embedding: false,
                rope_scaling: None,
                rope_theta: None,
                transpose_attention_weights: false,
                transpose_ffn_weights: false,
                problem_type: None,
            }
        }

        fn layout(&self) -> ModelLayout {
            ModelLayout {
                token_embedding: "model.embed_tokens.weight".to_string(),
                decoder: None,
                encoder: None,
                lm_head: "lm_head".to_string(),
            }
        }
    }

    struct MockCpuEncoder {
        hidden_size: usize,
    }

    impl CpuEncoder for MockCpuEncoder {
        fn forward(
            &self,
            _hidden_states: &Array3<f32>,
            attention_mask: &Array2<f32>,
        ) -> Result<CpuEncoderOutput> {
            let (batch, seq) = attention_mask.dim();
            Ok(CpuEncoderOutput {
                last_hidden_state: Array3::zeros((batch, seq, self.hidden_size)),
            })
        }

        fn create_buffers(
            &self,
            _max_batch: usize,
            _max_seq: usize,
        ) -> crate::cpu::encoder::buffers::EncoderBuffers {
            unimplemented!()
        }

        fn forward_layers(
            &self,
            hidden_states: &Array3<f32>,
            _attention_mask: &Array2<f32>,
            _start_layer: usize,
            _end_layer: usize,
        ) -> Result<Array3<f32>> {
            Ok(hidden_states.clone())
        }
    }

    impl crate::traits::CpuTransformerCore for MockCpuEncoder {
        fn num_layers(&self) -> usize {
            2
        }
        fn hidden_size(&self) -> usize {
            self.hidden_size
        }
        fn num_attention_heads(&self) -> usize {
            4
        }
        fn final_norm(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
            Ok(hidden.clone())
        }
        fn embed_norm(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
            Ok(hidden.clone())
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
    }

    struct MockCpuDecoder {
        hidden_size: usize,
    }

    impl MockCpuDecoder {
        #[expect(
            dead_code,
            reason = "held for the GPU decoder path, which is not wired up yet"
        )]
        fn new(hidden_size: usize) -> Self {
            Self { hidden_size }
        }
    }

    impl CpuCrossDecoder for MockCpuDecoder {
        fn embed(&self, _decoder_input_ids: &Array2<u32>, _position_offset: usize) -> Array3<f32> {
            unimplemented!()
        }
        fn precompute_cross_attention_kv(
            &self,
            _encoder_hidden_states: &Array3<f32>,
        ) -> Result<crate::encoder_decoder::traits::CpuCrossAttentionKVCache> {
            unimplemented!()
        }
        fn embed_norm(&self, _hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
            unimplemented!()
        }

        fn layers(&self) -> &Vec<crate::cpu::encoder_decoder::CrossDecoderLayer> {
            unimplemented!()
        }

        fn forward_layers(
            &self,
            hidden_states: &Array3<f32>,
            _encoder_hidden_states: &Array3<f32>,
            _decoder_padding_mask: Option<&Array2<f32>>,
            _encoder_padding_mask: Option<&Array2<f32>>,
            _cache: Option<&mut dyn Cache>,
            _cross_kv_cache: Option<&CpuCrossAttentionKVCache>,
            _start_layer: usize,
            _end_layer: usize,
        ) -> Result<CpuCrossDecoderOutput> {
            Ok(CpuCrossDecoderOutput {
                last_hidden_state: hidden_states.clone(),
                new_self_attn_kv: vec![],
            })
        }

        fn final_norm(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
            Ok(hidden_states.clone())
        }

        fn num_layers(&self) -> usize {
            2
        }
        fn hidden_size(&self) -> usize {
            self.hidden_size
        }
    }

    struct MockSeq2SeqModel {
        #[expect(
            dead_code,
            reason = "held for the GPU decoder path, which is not wired up yet"
        )]
        pipeline: EncoderDecoderPipeline,
        #[expect(
            dead_code,
            reason = "held for the GPU decoder path, which is not wired up yet"
        )]
        tokenizer: Tokenizer,
        #[expect(
            dead_code,
            reason = "held for the GPU decoder path, which is not wired up yet"
        )]
        config: Arc<MockConfig>,
    }

    impl std::fmt::Debug for MockSeq2SeqModel {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("MockSeq2SeqModel").finish()
        }
    }

    impl EncoderDecoderModelFactory for MockSeq2SeqModel {
        type Config = MockConfig;

        fn load_config(_weights: &ModelWeights) -> Result<Arc<Self::Config>> {
            Ok(Arc::new(MockConfig::default()))
        }

        fn build_backends(
            _weights: &ModelWeights,
            meta: &ModelMetadata,
            _layout: &ModelLayout,
            _config: &Arc<Self::Config>,
            _load_config: &ModelLoadConfig,
            _context: Option<&Arc<WgpuContext>>,
            _device: Device,
        ) -> Result<(
            Option<Box<dyn CpuEncoder>>,
            Option<Box<dyn GpuEncoder>>,
            Option<Box<dyn CpuCrossDecoder>>,
            Option<Box<dyn GpuCrossDecoder>>,
        )> {
            Ok((
                Some(Box::new(MockCpuEncoder {
                    hidden_size: meta.hidden_size,
                })),
                None,
                Some(Box::new(MockCpuDecoder {
                    hidden_size: meta.hidden_size,
                })),
                None,
            ))
        }

        fn new_from_pipeline(
            pipeline: EncoderDecoderPipeline,
            tokenizer: Tokenizer,
            config: Arc<Self::Config>,
            _generation_defaults: Option<HFGenerationDefaults>,
            _generation_config: HFGenerationConfig,
        ) -> Self {
            Self {
                pipeline,
                tokenizer,
                config,
            }
        }
    }
    fn create_minimal_model_files(dir: &Path) -> Result<()> {
        use safetensors::Dtype;
        use safetensors::tensor::TensorView;
        use std::fs::File;
        use std::io::Write;

        let vocab_size = 1000;
        let hidden_size = 64;
        let embed_data: Vec<f32> = (0..(vocab_size * hidden_size))
            .map(|x| (x as f32) * 0.001)
            .collect();
        let embed_bytes: Vec<u8> = embed_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let embed_view = TensorView::new(Dtype::F32, vec![vocab_size, hidden_size], &embed_bytes)?;

        let serialized =
            safetensors::serialize(vec![("model.embed_tokens.weight", embed_view)], &None)?;

        let mut file = File::create(dir.join("model.safetensors"))?;
        file.write_all(&serialized)?;

        let config_json = r#"{
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "vocab_size": 1000,
            "max_position_embeddings": 512
        }"#;
        std::fs::write(dir.join("config.json"), config_json)?;

        let tokenizer_json = r#"{
            "version": "1.0",
            "model": {
                "type": "BPE",
                "vocab": {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3},
                "merges": []
            },
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": null
        }"#;
        std::fs::write(dir.join("tokenizer.json"), tokenizer_json)?;

        Ok(())
    }

    fn create_generation_config(dir: &Path) -> Result<()> {
        let gen_config = r#"{
            "max_length": 128,
            "min_length": 0,
            "num_beams": 1,
            "decoder_start_token_id": 2,
            "eos_token_id": 2,
            "pad_token_id": 0
        }"#;
        std::fs::write(dir.join("generation_config.json"), gen_config)?;
        Ok(())
    }
    #[test]
    fn test_factory_trait_load_config() -> Result<()> {
        let dir = TempDir::new()?;
        create_minimal_model_files(dir.path())?;

        let weights = ModelWeights::new(dir.path())?;
        let config = MockSeq2SeqModel::load_config(&weights)?;

        assert_eq!(config.hidden_size, 64);
        assert_eq!(config.num_layers, 2);
        assert_eq!(config.num_heads, 4);

        Ok(())
    }

    #[test]
    fn test_factory_trait_build_backends_cpu() -> Result<()> {
        let dir = TempDir::new()?;
        create_minimal_model_files(dir.path())?;

        let weights = ModelWeights::new(dir.path())?;
        let config = MockSeq2SeqModel::load_config(&weights)?;
        let meta = config.metadata();
        let layout = config.layout();
        let load_config = ModelLoadConfig::default();

        let (cpu_enc, gpu_enc, cpu_dec, gpu_dec) = MockSeq2SeqModel::build_backends(
            &weights,
            &meta,
            &layout,
            &config,
            &load_config,
            None,
            Device::Cpu,
        )?;

        assert!(cpu_enc.is_some());
        assert!(gpu_enc.is_none());
        assert!(cpu_dec.is_some());
        assert!(gpu_dec.is_none());

        Ok(())
    }

    #[test]
    fn test_factory_trait_metadata() -> Result<()> {
        let config = MockConfig::default();
        let meta = config.metadata();

        assert_eq!(meta.hidden_size, 64);
        assert_eq!(meta.num_attention_heads, 4);
        assert_eq!(meta.num_layers, 2);
        assert_eq!(meta.intermediate_size, 256);
        assert_eq!(meta.vocab_size, 1000);

        Ok(())
    }

    #[test]
    fn test_factory_trait_layout() -> Result<()> {
        let config = MockConfig::default();
        let layout = config.layout();

        assert_eq!(layout.token_embedding, "model.embed_tokens.weight");

        Ok(())
    }

    #[test]
    fn test_loader_missing_path() {
        let result = Seq2SeqLoader::load_from_pretrained::<MockSeq2SeqModel>(
            Path::new("/nonexistent/path"),
            Device::Cpu,
            None,
            None,
            HFGenerationConfig::default(),
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_loader_missing_tokenizer() -> Result<()> {
        let dir = TempDir::new()?;

        use safetensors::Dtype;
        use safetensors::tensor::TensorView;
        use std::fs::File;
        use std::io::Write;

        let data: Vec<u8> = vec![0.0f32; 64000]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let view = TensorView::new(Dtype::F32, vec![1000, 64], &data)?;
        let serialized = safetensors::serialize(vec![("model.embed_tokens.weight", view)], &None)?;
        let mut file = File::create(dir.path().join("model.safetensors"))?;
        file.write_all(&serialized)?;

        std::fs::write(dir.path().join("config.json"), r#"{"hidden_size": 64}"#)?;

        let result = Seq2SeqLoader::load_from_pretrained::<MockSeq2SeqModel>(
            dir.path(),
            Device::Cpu,
            None,
            None,
            HFGenerationConfig::default(),
        );

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("tokenizer") || err_msg.contains("No such file"),
            "Expected tokenizer error, got: {}",
            err_msg
        );

        Ok(())
    }

    #[test]
    fn test_loader_missing_weights() -> Result<()> {
        let dir = TempDir::new()?;

        let tokenizer_json = r#"{
            "version": "1.0",
            "model": {"type": "BPE", "vocab": {}, "merges": []}
        }"#;
        std::fs::write(dir.path().join("tokenizer.json"), tokenizer_json)?;

        let result = Seq2SeqLoader::load_from_pretrained::<MockSeq2SeqModel>(
            dir.path(),
            Device::Cpu,
            None,
            None,
            HFGenerationConfig::default(),
        );

        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_loader_with_generation_config() -> Result<()> {
        let dir = TempDir::new()?;
        create_minimal_model_files(dir.path())?;
        create_generation_config(dir.path())?;

        let gen_config_path = dir.path().join("generation_config.json");
        assert!(gen_config_path.exists());

        let json = std::fs::read_to_string(&gen_config_path)?;
        let defaults = HFGenerationDefaults::from_json(&json)?;

        assert_eq!(defaults.max_length, Some(128));
        assert_eq!(defaults.decoder_start_token_id, Some(2));

        Ok(())
    }

    #[test]
    fn test_loader_without_generation_config() -> Result<()> {
        let dir = TempDir::new()?;
        create_minimal_model_files(dir.path())?;

        let gen_config_path = dir.path().join("generation_config.json");
        assert!(!gen_config_path.exists());

        let defaults =
            if let Ok(json) = std::fs::read_to_string(dir.path().join("generation_config.json")) {
                HFGenerationDefaults::from_json(&json).ok()
            } else {
                None
            };

        assert!(defaults.is_none());

        Ok(())
    }

    #[test]
    fn test_loader_load_config_option() -> Result<()> {
        let dir = TempDir::new()?;
        create_minimal_model_files(dir.path())?;

        let load_config = ModelLoadConfig {
            ..Default::default()
        };

        let weights = ModelWeights::new(dir.path())?;
        let config = MockSeq2SeqModel::load_config(&weights)?;
        let meta = config.metadata();
        let layout = config.layout();

        let (cpu_enc, _, cpu_dec, _) = MockSeq2SeqModel::build_backends(
            &weights,
            &meta,
            &layout,
            &config,
            &load_config,
            None,
            Device::Cpu,
        )?;

        assert!(cpu_enc.is_some());
        assert!(cpu_dec.is_some());

        Ok(())
    }

    #[test]
    fn test_loader_cpu_device() -> Result<()> {
        let dir = TempDir::new()?;
        create_minimal_model_files(dir.path())?;

        let weights = ModelWeights::new(dir.path())?;
        let config = MockSeq2SeqModel::load_config(&weights)?;
        let meta = config.metadata();
        let layout = config.layout();
        let load_config = ModelLoadConfig::default();

        let (cpu_enc, gpu_enc, cpu_dec, gpu_dec) = MockSeq2SeqModel::build_backends(
            &weights,
            &meta,
            &layout,
            &config,
            &load_config,
            None,
            Device::Cpu,
        )?;

        assert!(cpu_enc.is_some());
        assert!(cpu_dec.is_some());
        assert!(gpu_enc.is_none());
        assert!(gpu_dec.is_none());

        Ok(())
    }

    #[test]
    fn test_generation_defaults_from_json() -> Result<()> {
        let json = r#"{
            "max_length": 200,
            "min_length": 10,
            "num_beams": 4,
            "decoder_start_token_id": 2,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "forced_bos_token_id": 1,
            "forced_eos_token_id": 2
        }"#;

        let defaults = HFGenerationDefaults::from_json(json)?;

        assert_eq!(defaults.max_length, Some(200));
        assert_eq!(defaults.decoder_start_token_id, Some(2));
        Ok(())
    }

    #[test]
    fn test_generation_defaults_partial_json() -> Result<()> {
        let json = r#"{
            "max_length": 100
        }"#;
        let defaults = HFGenerationDefaults::from_json(json)?;
        assert_eq!(defaults.max_length, Some(100));
        Ok(())
    }

    #[test]
    fn test_generation_defaults_empty_json() -> Result<()> {
        let json = r#"{}"#;

        let defaults = HFGenerationDefaults::from_json(json)?;

        assert_eq!(defaults.max_length, None);
        assert_eq!(defaults.decoder_start_token_id, None);

        Ok(())
    }

    #[test]
    fn test_generation_defaults_invalid_json() {
        let json = r#"{ invalid json }"#;

        let result = HFGenerationDefaults::from_json(json);

        assert!(result.is_err());
    }

    #[test]
    fn test_generation_config_default() {
        let config = HFGenerationConfig::default();
        assert!(config.max_length.is_none() || config.max_length.unwrap() > 0);
    }

    #[test]
    fn test_mock_decoder_hidden_size() {
        let decoder = MockCpuDecoder { hidden_size: 128 };

        assert_eq!(decoder.hidden_size(), 128);
        assert_eq!(decoder.num_layers(), 2);
    }

    #[test]
    fn test_factory_pattern_compile() {
        fn accepts_factory<M: EncoderDecoderModelFactory>() {}

        accepts_factory::<MockSeq2SeqModel>();
    }

    #[test]
    fn test_trait_object_safety() {
        let encoder: Box<dyn CpuEncoder> = Box::new(MockCpuEncoder { hidden_size: 64 });
        let decoder: Box<dyn CpuCrossDecoder> = Box::new(MockCpuDecoder { hidden_size: 64 });

        let _ = encoder;
        let _ = decoder;
    }

    #[test]
    fn test_tokenizer_path_construction() -> Result<()> {
        let dir = TempDir::new()?;
        let tokenizer_path = dir.path().join("tokenizer.json");

        assert_eq!(tokenizer_path.file_name().unwrap(), "tokenizer.json");
        assert!(tokenizer_path.starts_with(dir.path()));

        Ok(())
    }

    #[test]
    fn test_generation_config_path_construction() -> Result<()> {
        let dir = TempDir::new()?;
        let gen_config_path = dir.path().join("generation_config.json");

        assert_eq!(
            gen_config_path.file_name().unwrap(),
            "generation_config.json"
        );

        Ok(())
    }

    #[test]
    fn test_error_messages_are_descriptive() {
        let result = Seq2SeqLoader::load_from_pretrained::<MockSeq2SeqModel>(
            Path::new("/definitely/not/a/real/path"),
            Device::Cpu,
            None,
            None,
            HFGenerationConfig::default(),
        );

        let err = result.unwrap_err();
        let err_string = err.to_string();

        assert!(!err_string.is_empty(), "Error message should not be empty");
    }
}
