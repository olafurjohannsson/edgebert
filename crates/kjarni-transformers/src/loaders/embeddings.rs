//! Unified embedding loading for all model architectures.

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
use std::sync::Arc;

use anyhow::{Result, anyhow};
use ndarray::Array2;

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
use crate::WgpuContext;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
use crate::gpu::{GpuEmbeddingWeights, GpuEmbeddings};
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
use crate::gpu::{GpuTensor, GpuTensorPool};
use crate::linear_layer::LinearLayer;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
use crate::models::base::ModelInput;
use crate::tensor::DType;
use crate::weights::ModelWeights;
use crate::{EmbeddingData, Embeddings};

/// Configuration for embedding loading.
#[derive(Debug, Clone, Default)]
pub struct EmbeddingConfig {
    pub word_embedding: String,
    pub position_embedding: Option<String>,
    pub type_embedding: Option<String>,
    pub hidden_size: usize,
    pub position_offset: usize,
    pub scale_embeddings: bool,
}

impl EmbeddingConfig {
    pub fn new(word_embedding: impl Into<String>, hidden_size: usize) -> Self {
        Self {
            word_embedding: word_embedding.into(),
            hidden_size,
            ..Default::default()
        }
    }

    pub fn with_position_embedding(mut self, position_embedding: Option<String>) -> Self {
        self.position_embedding = position_embedding;
        self
    }

    pub fn with_type_embedding(mut self, type_embedding: Option<String>) -> Self {
        self.type_embedding = type_embedding;
        self
    }

    /// Creates a builder for more complex configurations.
    pub fn builder(
        word_embedding: impl Into<String>,
        hidden_size: usize,
    ) -> EmbeddingConfigBuilder {
        EmbeddingConfigBuilder {
            word_embedding: word_embedding.into(),
            hidden_size,
            position_embedding: None,
            type_embedding: None,
            position_offset: 0,
            scale_embeddings: false,
        }
    }
}

pub struct EmbeddingConfigBuilder {
    word_embedding: String,
    hidden_size: usize,
    position_embedding: Option<String>,
    type_embedding: Option<String>,
    position_offset: usize,
    scale_embeddings: bool,
}

impl EmbeddingConfigBuilder {
    pub fn position_embedding(mut self, name: impl Into<String>) -> Self {
        self.position_embedding = Some(name.into());
        self
    }

    pub fn with_token_type_embedding(mut self, name: Option<String>) -> Self {
        self.type_embedding = name;
        self
    }

    pub fn type_embedding(mut self, name: impl Into<String>) -> Self {
        self.type_embedding = Some(name.into());
        self
    }

    pub fn position_offset(mut self, offset: usize) -> Self {
        self.position_offset = offset;
        self
    }

    pub fn scale_embeddings(mut self, scale: bool) -> Self {
        self.scale_embeddings = scale;
        self
    }

    pub fn build(self) -> EmbeddingConfig {
        EmbeddingConfig {
            word_embedding: self.word_embedding,
            position_embedding: self.position_embedding,
            type_embedding: self.type_embedding,
            hidden_size: self.hidden_size,
            position_offset: self.position_offset,
            scale_embeddings: self.scale_embeddings,
        }
    }
}

pub enum EmbeddingInput<'a> {
    Cpu(&'a Array2<u32>),
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
    Gpu(&'a GpuTensor),
}

/// Loaded embeddings that can be on CPU, GPU, or hybrid.
pub struct LoadedEmbeddings {
    pub cpu: Option<Embeddings>,
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
    pub gpu_weights: Option<GpuEmbeddingWeights>,
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
    pub gpu_layer: Option<GpuEmbeddings>,
    pub config: EmbeddingConfig,
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
    context: Option<Arc<WgpuContext>>,
}

impl LoadedEmbeddings {
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
    pub fn new(
        ctx: Option<&Arc<WgpuContext>>,
        weights: &ModelWeights,
        config: EmbeddingConfig,
        load_cpu: bool,
        load_gpu: bool,
        target_dtype: Option<DType>,
    ) -> Result<Self> {
        let cpu = if load_cpu {
            log::info!("loading embeddings to CPU RAM");
            Some(Embeddings::from_weights(
                weights,
                &config.word_embedding,
                config.position_embedding.as_deref(),
                config.type_embedding.as_deref(),
                target_dtype,
            )?)
        } else {
            None
        };

        let (gpu_weights, gpu_layer) = if load_gpu {
            let ctx = ctx.ok_or_else(|| anyhow!("GPU embeddings require WgpuContext"))?;
            log::info!("loading embeddings to GPU VRAM");

            let gpu_weights = GpuEmbeddingWeights::new(
                ctx,
                weights,
                &config.word_embedding,
                config.position_embedding.as_deref(),
                config.type_embedding.as_deref(),
                target_dtype,
            )?;
            let gpu_layer = GpuEmbeddings::new(ctx)?;
            (Some(gpu_weights), Some(gpu_layer))
        } else {
            (None, None)
        };

        if cpu.is_none() && gpu_weights.is_none() {
            return Err(anyhow!("must load embeddings to at least one device"));
        }

        Ok(Self {
            cpu,
            gpu_weights,
            gpu_layer,
            config,
            context: ctx.cloned(),
        })
    }

    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-gpu")))]
    pub fn new(
        weights: &ModelWeights,
        config: EmbeddingConfig,
        target_dtype: Option<DType>,
    ) -> Result<Self> {
        log::info!("loading embeddings to CPU RAM");
        let cpu = Embeddings::from_weights(
            weights,
            &config.word_embedding,
            config.position_embedding.as_deref(),
            config.type_embedding.as_deref(),
            target_dtype,
        )?;

        Ok(Self {
            cpu: Some(cpu),
            config,
        })
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
    pub fn word_embeddings_gpu(&self) -> Option<GpuTensor> {
        self.gpu_weights.as_ref().map(|w| w.word_embeddings.clone())
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
    pub fn with_shared_words(
        ctx: Option<&Arc<WgpuContext>>,
        weights: &ModelWeights,
        config: EmbeddingConfig,
        shared_word_cpu: Option<EmbeddingData>,
        shared_word_gpu: Option<GpuTensor>,
        load_cpu: bool,
        load_gpu: bool,
        target_dtype: Option<DType>,
    ) -> Result<Self> {
        let cpu = if load_cpu {
            let shared = shared_word_cpu
                .ok_or_else(|| anyhow!("CPU word embeddings required but not provided"))?;

            let pos_emb = config
                .position_embedding
                .as_ref()
                .filter(|k| !k.is_empty())
                .map(|k| weights.get_array2(k))
                .transpose()?;

            let type_emb = config
                .type_embedding
                .as_ref()
                .filter(|k| !k.is_empty())
                .map(|k| weights.get_array2(k))
                .transpose()?;

            Some(Embeddings::new(shared, pos_emb, type_emb))
        } else {
            None
        };

        let (gpu_weights, gpu_layer) = if load_gpu {
            let ctx = ctx.ok_or_else(|| anyhow!("GPU embeddings require WgpuContext"))?;
            let shared = shared_word_gpu
                .ok_or_else(|| anyhow!("GPU word embeddings required but not provided"))?;

            let gpu_weights = GpuEmbeddingWeights::with_shared_words(
                ctx,
                weights,
                shared,
                config
                    .position_embedding
                    .as_deref()
                    .filter(|k| !k.is_empty()),
                config.type_embedding.as_deref().filter(|k| !k.is_empty()),
                target_dtype,
            )?;
            let gpu_layer = GpuEmbeddings::new(ctx)?;
            (Some(gpu_weights), Some(gpu_layer))
        } else {
            (None, None)
        };

        if cpu.is_none() && gpu_weights.is_none() {
            return Err(anyhow!("must load embeddings to at least one device"));
        }

        Ok(Self {
            cpu,
            gpu_weights,
            gpu_layer,
            config,
            context: ctx.cloned(),
        })
    }

    /// Returns the raw word embedding weights for CPU weight sharing.
    pub fn word_embeddings_cpu(&self) -> Option<LinearLayer> {
        self.cpu.as_ref().map(|e| match &e.word_embeddings {
            EmbeddingData::F32(arc_w) => LinearLayer::from_arc_f32(arc_w.clone(), None),
            EmbeddingData::BF16(arc_w) => LinearLayer::from_arc_bf16(arc_w.clone(), None),
            EmbeddingData::Q8_0(arc_q) => LinearLayer::from_arc_q8_0(arc_q.clone(), None),
        })
    }

    #[inline]
    pub fn is_cpu(&self) -> bool {
        self.cpu.is_some()
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
    #[inline]
    pub fn is_gpu(&self) -> bool {
        self.gpu_weights.is_some()
    }

    #[inline]
    pub fn config(&self) -> &EmbeddingConfig {
        &self.config
    }

    #[inline]
    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    pub fn is_cpu_loaded(&self) -> bool {
        self.cpu.is_some()
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
    pub fn is_gpu_loaded(&self) -> bool {
        self.gpu_weights.is_some()
    }

    /// CPU-native embedding lookup.
    pub fn embed_cpu(
        &self,
        token_ids: &Array2<u32>,
        token_type_ids: Option<&Array2<u32>>,
        _position_offset: usize,
    ) -> Result<ndarray::Array3<f32>> {
        let cpu_layer = self
            .cpu
            .as_ref()
            .ok_or_else(|| anyhow!("embeddings not loaded on CPU"))?;

        Ok(cpu_layer.forward(
            token_ids,
            token_type_ids,
            self.config.position_offset,
            self.config.scale_embeddings,
        ))
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
    pub fn embed(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        input: ModelInput<'_>,
        token_type_ids: Option<ModelInput<'_>>,
        position_offset: usize,
    ) -> Result<GpuTensor> {
        let ctx = self
            .context
            .as_ref()
            .ok_or_else(|| anyhow!("no GPU context for embeddings"))?;

        match input {
            ModelInput::HiddenGpu(t) => return Ok(t.clone()),
            ModelInput::HiddenCpu(view) => {
                return GpuTensor::from_ndarray(ctx, &view.as_standard_layout().to_owned());
            }
            _ => {}
        }

        match input {
            ModelInput::TokensCpu(ids) => {
                if let Some(cpu_layer) = &self.cpu {
                    let cpu_types_storage: Option<Array2<u32>> = match token_type_ids {
                        Some(ModelInput::TokensCpu(t)) => Some(t.to_owned()),
                        None => None,
                        _ => return Err(anyhow!("input is CPU but token types are GPU")),
                    };

                    let ids_owned = ids.to_owned();
                    let hidden = cpu_layer.forward(
                        &ids_owned,
                        cpu_types_storage.as_ref(),
                        position_offset + self.config.position_offset,
                        self.config.scale_embeddings,
                    );

                    return GpuTensor::from_ndarray(ctx, &hidden);
                }

                if let Some(gpu_layer) = &self.gpu_layer {
                    let ids_gpu = GpuTensor::from_ndarray(ctx, &ids.to_owned())?;

                    let types_gpu = match token_type_ids {
                        Some(ModelInput::TokensCpu(t)) => {
                            Some(GpuTensor::from_ndarray(ctx, &t.to_owned())?)
                        }
                        None => None,
                        _ => return Err(anyhow!("input is CPU but token types are GPU")),
                    };

                    return gpu_layer.encode(
                        encoder,
                        self.gpu_weights.as_ref().unwrap(),
                        &ids_gpu,
                        types_gpu.as_ref(),
                        position_offset,
                        self.config.hidden_size,
                        self.config.position_offset,
                        self.config.scale_embeddings,
                        pool,
                    );
                }
            }

            ModelInput::TokensGpu(ids) => {
                if let Some(gpu_layer) = &self.gpu_layer {
                    let types_gpu = match token_type_ids {
                        Some(ModelInput::TokensGpu(t)) => Some(t),
                        None => None,
                        _ => return Err(anyhow!("input is GPU but token types are CPU")),
                    };

                    return gpu_layer.encode(
                        encoder,
                        self.gpu_weights.as_ref().unwrap(),
                        ids,
                        types_gpu,
                        position_offset,
                        self.config.hidden_size,
                        self.config.position_offset,
                        self.config.scale_embeddings,
                        pool,
                    );
                } else {
                    return Err(anyhow!("tokens on GPU but embeddings are CPU-only"));
                }
            }
            _ => unreachable!(),
        }

        Err(anyhow!("no embeddings loaded"))
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        input: EmbeddingInput,
        token_type_ids: Option<EmbeddingInput>,
        position_offset: usize,
    ) -> Result<GpuTensor> {
        let ctx = self
            .context
            .as_ref()
            .ok_or_else(|| anyhow!("no GPU context"))?;

        if let EmbeddingInput::Cpu(cpu_ids) = input
            && let Some(cpu_layer) = &self.cpu
        {
            let cpu_types = match token_type_ids {
                Some(EmbeddingInput::Cpu(t)) => Some(t),
                None => None,
                _ => return Err(anyhow!("cannot mix CPU tokens with GPU token types")),
            };

            let hidden = cpu_layer.forward(
                cpu_ids,
                cpu_types,
                position_offset + self.config.position_offset,
                self.config.scale_embeddings,
            );

            return GpuTensor::from_ndarray(ctx, &hidden);
        }

        let ids_gpu = match input {
            EmbeddingInput::Cpu(t) => GpuTensor::from_ndarray(ctx, t)?,
            EmbeddingInput::Gpu(t) => t.clone(),
        };

        let types_gpu = match token_type_ids {
            Some(EmbeddingInput::Cpu(t)) => Some(GpuTensor::from_ndarray(ctx, t)?),
            Some(EmbeddingInput::Gpu(t)) => Some(t.clone()),
            None => None,
        };

        if let Some(gpu_layer) = &self.gpu_layer {
            let gpu_weights = self.gpu_weights.as_ref().unwrap();

            return gpu_layer.encode(
                encoder,
                gpu_weights,
                &ids_gpu,
                types_gpu.as_ref(),
                position_offset,
                self.config.hidden_size,
                self.config.position_offset,
                self.config.scale_embeddings,
                pool,
            );
        }

        Err(anyhow!("tokens on GPU but embeddings only on CPU"))
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
    pub fn encode_cpu(
        &self,
        context: &Arc<WgpuContext>,
        token_ids: &Array2<u32>,
        token_type_ids: Option<&Array2<u32>>,
        position_offset: usize,
    ) -> Result<GpuTensor> {
        let cpu_embs = self
            .cpu
            .as_ref()
            .ok_or_else(|| anyhow!("CPU embeddings not loaded"))?;

        let hidden = cpu_embs.forward(
            token_ids,
            token_type_ids,
            position_offset + self.config.position_offset,
            self.config.scale_embeddings,
        );

        GpuTensor::from_ndarray(context, &hidden)
    }

    /// Performs embedding lookup from CPU token IDs.
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
    pub fn forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        token_ids: &Array2<u32>,
        token_type_ids: Option<&Array2<u32>>,
        position_offset: usize,
    ) -> Result<GpuTensor> {
        if let Some(cpu_embs) = &self.cpu {
            let embeddings = cpu_embs.forward(
                token_ids,
                token_type_ids,
                position_offset + self.config.position_offset,
                self.config.scale_embeddings,
            );
            GpuTensor::from_ndarray(self.context.as_ref().unwrap(), &embeddings)
        } else {
            let gpu_layer = self.gpu_layer.as_ref().unwrap();
            let gpu_weights = self.gpu_weights.as_ref().unwrap();

            let ids_tensor = GpuTensor::from_ndarray(self.context.as_ref().unwrap(), token_ids)?;
            let type_tensor = token_type_ids
                .map(|t| GpuTensor::from_ndarray(self.context.as_ref().unwrap(), t))
                .transpose()?;

            gpu_layer.encode(
                encoder,
                gpu_weights,
                &ids_tensor,
                type_tensor.as_ref(),
                position_offset,
                self.config.hidden_size,
                self.config.position_offset,
                self.config.scale_embeddings,
                pool,
            )
        }
    }

    /// Performs embedding lookup from GPU token IDs.
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
    pub async fn forward_gpu(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        token_ids: &GpuTensor,
        token_type_ids: Option<&GpuTensor>,
        position_offset: usize,
    ) -> Result<GpuTensor> {
        if let Some(cpu_embs) = &self.cpu {
            let token_data: Vec<u32> =
                bytemuck::cast_slice(&token_ids.read_raw_data().await?).to_vec();
            let shape = token_ids.shape();
            let token_array = Array2::from_shape_vec((shape[0], shape[1]), token_data)?;

            let type_array = if let Some(type_ids) = token_type_ids {
                let type_data: Vec<u32> =
                    bytemuck::cast_slice(&type_ids.read_raw_data().await?).to_vec();
                Some(Array2::from_shape_vec((shape[0], shape[1]), type_data)?)
            } else {
                None
            };

            let embeddings = cpu_embs.forward(
                &token_array,
                type_array.as_ref(),
                position_offset + self.config.position_offset,
                self.config.scale_embeddings,
            );
            GpuTensor::from_ndarray(self.context.as_ref().unwrap(), &embeddings)
        } else {
            let gpu_layer = self.gpu_layer.as_ref().unwrap();
            let gpu_weights = self.gpu_weights.as_ref().unwrap();

            gpu_layer.encode(
                encoder,
                gpu_weights,
                token_ids,
                token_type_ids,
                position_offset,
                self.config.hidden_size,
                self.config.position_offset,
                self.config.scale_embeddings,
                pool,
            )
        }
    }
}
