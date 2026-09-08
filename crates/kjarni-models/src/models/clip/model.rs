//! CLIP vision tower: an image to a vector in the shared image/text space.
//!
//! The transformer here is the ordinary encoder stack this crate already runs
//! over tokens. Only the front differs. A text encoder looks a row up in an
//! embedding table; this projects a flattened patch of pixels through one matrix.
//! After that first step the two are the same computation, which is why almost
//! nothing below is new code.

use std::path::Path;

use anyhow::{Context, Result};
use kjarni_transformers::cpu::encoder::CpuTransformerEncoder;
use kjarni_transformers::cpu::encoder::traits::CpuEncoder;
use kjarni_transformers::cpu::normalization::LayerNorm;
use kjarni_transformers::models::base::ModelLoadConfig;
use kjarni_transformers::traits::{CpuTransformerCore, ModelConfig};
use kjarni_transformers::weights::ModelWeights;
use ndarray::{Array1, Array2, Array3, s};

use super::config::ClipConfig;
use super::preprocess::PreprocessorConfig;

pub struct ClipVisionModel {
    config: ClipConfig,
    preprocessor: PreprocessorConfig,
    encoder: CpuTransformerEncoder,

    /// `[hidden, patch_dim]`. The checkpoint stores this as a `[hidden, 3, p, p]`
    /// conv kernel; with stride equal to patch size the patches never overlap, so
    /// the convolution is exactly a matmul against flattened patches and is
    /// reshaped once here rather than convolved every call.
    patch_projection: Array2<f32>,

    /// Prepended to the patch sequence. Its output row is the image vector: it
    /// attends to every patch and belongs to none.
    class_embedding: Array1<f32>,

    /// `[num_positions, hidden]`, learned. A ViT has no geometric prior, so
    /// without these the patches are an unordered set.
    position_embedding: Array2<f32>,

    /// Applied to the pooled class token only, after the blocks.
    post_layernorm: LayerNorm,

    /// `[projection_dim, hidden]` into the space the text tower also lands in.
    visual_projection: Array2<f32>,
}

impl ClipVisionModel {
    /// Loads from a directory holding `config.json`, `preprocessor_config.json`
    /// and `model.safetensors`, as `kjarni model download` leaves it.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let config = ClipConfig::from_json(
            &std::fs::read_to_string(dir.join("config.json"))
                .with_context(|| format!("reading config.json in {}", dir.display()))?,
        )?;
        let preprocessor = PreprocessorConfig::from_json(
            &std::fs::read_to_string(dir.join("preprocessor_config.json")).with_context(|| {
                format!("reading preprocessor_config.json in {}", dir.display())
            })?,
        )?;
        let weights = ModelWeights::new(dir)?;
        Self::from_weights(config, preprocessor, &weights)
    }

    pub fn from_weights(
        config: ClipConfig,
        preprocessor: PreprocessorConfig,
        weights: &ModelWeights,
    ) -> Result<Self> {
        let meta = config.metadata();
        let layout = config.layout();
        let hidden = config.vision_config.hidden_size;
        let patch_dim = config.patch_dim();

        let encoder =
            CpuTransformerEncoder::new(weights, meta, layout, ModelLoadConfig::default())?;

        // [hidden, channels, p, p] -> [hidden, channels*p*p]. Row-major already
        // groups each output filter's weights contiguously, so this is a view
        // reinterpretation rather than a permutation.
        let name = "vision_model.embeddings.patch_embedding.weight";
        let k = weights.tensor_shape(name).context("CLIP patch_embedding")?;
        anyhow::ensure!(
            k.len() == 4 && k[0] == hidden && k[1] * k[2] * k[3] == patch_dim,
            "patch kernel is {k:?}, expected [{hidden}, c, p, p] flattening to {patch_dim}"
        );
        // Read rank-agnostically and fold the trailing three axes. Row-major
        // already stores each output filter's channel/row/column weights
        // contiguously, so this is a reinterpretation and not a permutation.
        let patch_projection = weights.with_raw_tensor(name, |view| {
            #[allow(deprecated)]
            let dense = view.to_ndarray_f32()?;
            dense
                .into_shape_with_order((hidden, patch_dim))
                .context("flattening the patch kernel")?
                .into_dimensionality::<ndarray::Ix2>()
                .context("patch kernel is not 2D after flattening")
        })?;

        let class_embedding = weights
            .get_array1("vision_model.embeddings.class_embedding")
            .context("CLIP class_embedding")?;
        let position_embedding = weights
            .get_array2("vision_model.embeddings.position_embedding.weight")
            .context("CLIP position_embedding")?;
        anyhow::ensure!(
            position_embedding.shape()[0] == config.num_positions(),
            "checkpoint has {} positions, config implies {}",
            position_embedding.shape()[0],
            config.num_positions()
        );

        let post_layernorm = LayerNorm::new(
            weights.get_array1("vision_model.post_layernorm.weight")?,
            weights.get_array1("vision_model.post_layernorm.bias")?,
            config.vision_config.layer_norm_eps,
        );
        let visual_projection = weights
            .get_array2("visual_projection.weight")
            .context("CLIP visual_projection")?;

        Ok(Self {
            config,
            preprocessor,
            encoder,
            patch_projection,
            class_embedding,
            position_embedding,
            post_layernorm,
            visual_projection,
        })
    }

    pub fn config(&self) -> &ClipConfig {
        &self.config
    }

    pub fn preprocessor(&self) -> &PreprocessorConfig {
        &self.preprocessor
    }

    /// Cuts the image into patches and projects each one, then prepends the class
    /// token and adds the learned positions.
    ///
    /// Returns `[1, num_positions, hidden]`.
    pub fn embed_pixels(&self, pixels: &Array3<f32>) -> Result<Array3<f32>> {
        let hidden = self.config.vision_config.hidden_size;
        let p = self.config.vision_config.patch_size;
        let grid = self.config.patches_per_side();
        let channels = self.config.num_channels();
        let expected = self.config.vision_config.image_size;

        let shape = pixels.shape();
        anyhow::ensure!(
            shape == [channels, expected, expected],
            "expected pixels shaped [{channels}, {expected}, {expected}], got {shape:?}"
        );

        let num_positions = self.config.num_positions();
        let mut out = Array3::<f32>::zeros((1, num_positions, hidden));

        // Row 0 is the class token; patches follow in raster order, which is the
        // order the position table was learned in.
        for (h, &v) in self.class_embedding.iter().enumerate() {
            out[[0, 0, h]] = v;
        }

        let mut patch = vec![0.0f32; self.config.patch_dim()];
        for gy in 0..grid {
            for gx in 0..grid {
                // Channel-major within a patch: c, then row, then column, which is
                // how the conv kernel's own memory is laid out.
                let mut i = 0;
                for c in 0..channels {
                    for y in 0..p {
                        for x in 0..p {
                            patch[i] = pixels[[c, gy * p + y, gx * p + x]];
                            i += 1;
                        }
                    }
                }

                let row = 1 + gy * grid + gx;
                for h in 0..hidden {
                    let w = self.patch_projection.row(h);
                    let mut acc = 0.0f32;
                    for (a, b) in w.iter().zip(patch.iter()) {
                        acc += a * b;
                    }
                    out[[0, row, h]] = acc;
                }
            }
        }

        // The patch projection has no bias in CLIP, so positions are added here
        // and nowhere else.
        for r in 0..num_positions {
            for h in 0..hidden {
                out[[0, r, h]] += self.position_embedding[[r, h]];
            }
        }

        Ok(out)
    }

    /// Preprocessed pixels to a projected, L2-normalized image vector.
    ///
    /// Normalized because the only thing these vectors are for is cosine
    /// similarity against text vectors, and doing it here means the index never
    /// has to remember to.
    pub fn embed_image_tensor(&self, pixels: &Array3<f32>) -> Result<Array1<f32>> {
        let embedded = self.embed_pixels(pixels)?;

        // `pre_layrnorm` runs on the assembled sequence before the blocks.
        let normalized = self.encoder.embed_norm(&embedded)?;
        let mask = Array2::<f32>::ones((1, normalized.shape()[1]));
        let hidden =
            self.encoder
                .forward_layers(&normalized, &mask, 0, self.encoder.num_layers())?;

        // Pool the class token, then norm it. CLIP norms after pooling, not
        // before: doing it the other way round changes the result.
        let pooled = hidden.slice(s![0..1, 0..1, ..]).to_owned();
        let pooled = self.post_layernorm.forward_3d(&pooled);

        let h = self.config.vision_config.hidden_size;
        let d = self.config.projection_dim;
        let mut projected = Array1::<f32>::zeros(d);
        for i in 0..d {
            let w = self.visual_projection.row(i);
            let mut acc = 0.0f32;
            for j in 0..h {
                acc += w[j] * pooled[[0, 0, j]];
            }
            projected[i] = acc;
        }

        let norm = projected.dot(&projected).sqrt();
        if norm > 0.0 {
            projected /= norm;
        }
        Ok(projected)
    }

    /// Raw RGB bytes to an image vector, preprocessing included.
    pub fn embed_rgb8(&self, pixels: &[u8], width: usize, height: usize) -> Result<Array1<f32>> {
        let tensor = self.preprocessor.preprocess_rgb8(pixels, width, height)?;
        self.embed_image_tensor(&tensor)
    }

    /// Encoded JPEG or PNG bytes to an image vector.
    #[cfg(feature = "image-io")]
    pub fn embed_image_bytes(&self, bytes: &[u8]) -> Result<Array1<f32>> {
        let img = super::image_io::decode(bytes)?;
        self.embed_rgb8(&img.pixels, img.width, img.height)
    }

    /// An image file on disk to an image vector.
    #[cfg(feature = "image-io")]
    pub fn embed_image_file(&self, path: &Path) -> Result<Array1<f32>> {
        let img = super::image_io::decode_file(path)?;
        self.embed_rgb8(&img.pixels, img.width, img.height)
    }
}
