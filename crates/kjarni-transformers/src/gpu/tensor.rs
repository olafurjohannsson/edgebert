//! GPU-backed tensor with reference-counted buffer and shape metadata.

use std::borrow::Cow;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Result, anyhow};
use ndarray::{Array, Array1, Array2, Array3, Array4, Dimension};
use wgpu::util::DeviceExt;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, CommandEncoder};

use crate::WgpuContext;
use crate::gpu_ops::primitives::layout::permute::GpuPermute;
use crate::gpu_ops::primitives::layout::slice::GpuSlice;
use crate::tensor::CpuTensor;
use crate::tensor::raw_tensor::TensorView;
use crate::weights::ModelWeights;

pub use crate::tensor::DType;

static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(0);

pub trait GpuDType: bytemuck::Pod {
    const DTYPE: DType;
}

impl GpuDType for f32 {
    const DTYPE: DType = DType::F32;
}

impl GpuDType for u32 {
    const DTYPE: DType = DType::U32;
}

/// A GPU-backed tensor with reference-counted buffer.
pub struct GpuTensor {
    buffer: Arc<Buffer>,
    shape: Vec<usize>,
    pub dtype: DType,
    context: Arc<WgpuContext>,
    id: u64,
}

impl Clone for GpuTensor {
    fn clone(&self) -> Self {
        Self {
            buffer: Arc::clone(&self.buffer),
            shape: self.shape.clone(),
            dtype: self.dtype,
            context: self.context.clone(),
            id: self.id,
        }
    }
}

impl fmt::Debug for GpuTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuTensor")
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("buffer_size", &self.buffer.size())
            .finish_non_exhaustive()
    }
}

impl GpuTensor {
    /// Returns `(in_features, out_features)` for a `[out, in]` weight matrix.
    pub fn linear_layer_dims(&self) -> (usize, usize) {
        (self.shape[1], self.shape[0])
    }

    /// Uploads a row range of a larger tensor.
    ///
    /// Slicing happens on the CPU before upload, and keeps the source dtype so a bf16
    /// checkpoint is not doubled on the way to the card.
    #[allow(clippy::too_many_arguments)]
    fn from_model_weight_rows(
        ctx: &Arc<WgpuContext>,
        weights: &ModelWeights,
        base_name: &str,
        start: usize,
        end: usize,
        target_dt: Option<DType>,
        label: &str,
    ) -> Result<Self> {
        use ndarray::s;

        let full = weights.get_typed_tensor(base_name)?;
        let rows = full.shape().first().copied().unwrap_or(0);
        anyhow::ensure!(
            end <= rows && start < end,
            "row range {start}..{end} does not fit '{base_name}' with {rows} rows"
        );

        // BF16 is the dtype the GPU kernels want anyway, so a BF16 checkpoint uploads
        // without conversion and an F32 one is converted once, here, rather than twice.
        let want_bf16 = target_dt.is_none_or(|t| t == DType::BF16);

        match full {
            CpuTensor::BF16(arr) if want_bf16 => {
                let two = arr.into_dimensionality::<ndarray::Ix2>()?;
                let slice = two.slice(s![start..end, ..]).to_owned();
                let layout = slice.as_standard_layout();
                let view = TensorView {
                    name: label.to_string(),
                    bytes: Cow::Borrowed(bytemuck::cast_slice(
                        layout
                            .as_slice()
                            .ok_or_else(|| anyhow!("slice not contiguous"))?,
                    )),
                    shape: vec![end - start, two.shape()[1]],
                    dtype: DType::BF16,
                };
                GpuTensor::from_raw(ctx, &view, label)
            }
            other => {
                let two = match other {
                    CpuTensor::F32(a) => a.into_dimensionality::<ndarray::Ix2>()?,
                    CpuTensor::BF16(a) => a
                        .mapv(|v| v.to_f32())
                        .into_dimensionality::<ndarray::Ix2>()?,
                    o => anyhow::bail!(
                        "cannot take a row range of '{base_name}': {:?} is not a plain 2D array",
                        o.dtype()
                    ),
                };
                let slice = two.slice(s![start..end, ..]).to_owned();
                GpuTensor::from_ndarray(ctx, &slice)
            }
        }
    }

    /// Loads a tensor from model weights, using zero-copy when possible.
    pub fn from_model_weights(
        ctx: &Arc<WgpuContext>,
        weights: &ModelWeights,
        name: &str,
        target_dt: Option<DType>,
        label: &str,
    ) -> Result<Self> {
        // A layout may name a row range of a fused tensor, as `name[start:end]`. Phi-3
        // fuses q, k and v into one `qkv_proj`, and gate and up into one `gate_up_proj`,
        // where other decoders ship them separately. The CPU factory understands the
        // same syntax; without it here the GPU path fails with "tensor not found" on a
        // name the CPU path loads happily.
        if let Some((base, start, end)) = crate::weights::parse_row_range(name) {
            return Self::from_model_weight_rows(ctx, weights, base, start, end, target_dt, label);
        }

        // Q4_K is the one quantised format the GPU can read as it sits: the linear
        // primitive and the fused SwiGLU both have kernels that unpack the 4-bit
        // blocks in the shader. Uploading it packed is what took a 3B model from
        // 8.6GB of VRAM to 4.1GB and its load from 24s to 8s, most of which was the
        // dequantise.
        //
        // It goes through `get_typed_tensor` rather than `with_raw_tensor`, because
        // GGUF stores the Q and K projections row-permuted for RoPE and the CPU
        // conversion undoes that. Uploading `raw.bytes` skips the permutation and
        // produces a model that still emits fluent text, just not the right text.
        if weights.tensor_dtype(name).ok() == Some(DType::Q4_K)
            && target_dt.is_none_or(|t| t == DType::Q4_K)
            && let CpuTensor::Q4_K(matrix) = weights.get_typed_tensor(name)?
        {
            let view = TensorView {
                name: name.to_string(),
                bytes: Cow::Borrowed(bytemuck::cast_slice(&matrix.blocks)),
                shape: matrix.shape.to_vec(),
                dtype: DType::Q4_K,
            };
            return GpuTensor::from_raw(ctx, &view, label);
        }

        // Q8_0 reads packed too, but its 34-byte block puts `qs` two bytes out of
        // word alignment, so a straight copy would make the shader reassemble every
        // value across a word boundary. Repacking to 9 words -- d alone, then the 32
        // values as eight whole words -- costs 5.9% of the buffer and lets the kernel
        // use one aligned `unpack4xI8` per word.
        if weights.tensor_dtype(name).ok() == Some(DType::Q8_0)
            && target_dt.is_none_or(|t| t == DType::Q8_0)
            && let CpuTensor::Q8_0(matrix) = weights.get_typed_tensor(name)?
        {
            let view = TensorView {
                name: name.to_string(),
                bytes: Cow::Owned(repack_q8_0(&matrix.blocks)),
                shape: matrix.shape.to_vec(),
                dtype: DType::Q8_0,
            };
            return GpuTensor::from_raw(ctx, &view, label);
        }

        // Q6_K likewise reads packed, but its 210-byte block is not a multiple of
        // four, so each block is padded to 212 on the way up. That costs 0.95% of the
        // buffer and lets the shader index whole words instead of reassembling values
        // that straddle a word boundary.
        //
        // Note this path is also how the embedding table arrives, and the lookup
        // shader cannot walk quantised blocks. `GpuEmbeddingWeights::new` forces a
        // non-quantised target for that tensor; without it, a Q6_K embedding reaches
        // the lookup packed and panics.
        if weights.tensor_dtype(name).ok() == Some(DType::Q6_K)
            && target_dt.is_none_or(|t| t == DType::Q6_K)
            && let CpuTensor::Q6_K(matrix) = weights.get_typed_tensor(name)?
        {
            const SRC: usize = 210;
            const DST: usize = 212;
            let src = bytemuck::cast_slice::<_, u8>(matrix.blocks.as_slice());
            let mut padded = vec![0u8; matrix.blocks.len() * DST];
            for (i, chunk) in src.chunks_exact(SRC).enumerate() {
                padded[i * DST..i * DST + SRC].copy_from_slice(chunk);
            }
            let view = TensorView {
                name: name.to_string(),
                bytes: Cow::Owned(padded),
                shape: matrix.shape.to_vec(),
                dtype: DType::Q6_K,
            };
            return GpuTensor::from_raw(ctx, &view, label);
        }

        let upload_attempt = weights.with_raw_tensor(name, |raw| {
            if raw.dtype.is_quantized() {
                // Everything else quantised still has to be expanded, because nothing
                // on the GPU can walk its blocks.
                let target = target_dt.unwrap_or(DType::F32);
                if !target.is_quantized() {
                    return Ok(None);
                }
            }

            if let Some(target) = target_dt
                && target != raw.dtype
            {
                return Ok(None);
            }

            let tensor = GpuTensor::from_raw(ctx, &raw, label)?;
            Ok(Some(tensor))
        })?;

        if let Some(tensor) = upload_attempt {
            return Ok(tensor);
        }

        // A tensor that was quantised on disk has to be expanded here, because the
        // GPU linear kernels read F32 and BF16 only and cannot walk Q4_K blocks.
        // Expand it to BF16 rather than F32: the values came from 4-bit blocks, so
        // BF16's mantissa loses nothing that matters, and F32 costs twice the VRAM.
        // A 3B model is 12.8GB expanded to F32 and 6.4GB to BF16, and on a 12GB card
        // the F32 copy spills to host memory, which drops decode from ~6 tok/s to 0.2.
        let source_was_quantised = weights
            .tensor_dtype(name)
            .map(|dt| dt.is_quantized())
            .unwrap_or(false);
        let target = target_dt.unwrap_or(if source_was_quantised {
            DType::BF16
        } else {
            DType::F32
        });

        log::debug!(
            "converting tensor '{}' to {:?} for GPU upload",
            label,
            target
        );

        let typed_cpu = weights.get_typed_tensor(name)?;
        let (converted_bytes, final_shape) = convert_cpu_tensor_to_bytes(typed_cpu, target)?;

        let converted_view = TensorView {
            name: name.to_string(),
            bytes: Cow::Owned(converted_bytes),
            shape: final_shape,
            dtype: target,
        };

        GpuTensor::from_raw(ctx, &converted_view, label)
    }

    /// Creates a tensor from f32 data.
    pub fn create(
        ctx: &Arc<WgpuContext>,
        data: &[f32],
        shape: Vec<usize>,
        label: &str,
    ) -> Result<GpuTensor> {
        GpuTensor::from_raw(
            ctx,
            &TensorView {
                bytes: Cow::Owned(bytemuck::cast_slice(data).to_vec()),
                shape,
                dtype: DType::F32,
                name: label.to_string(),
            },
            label,
        )
    }

    /// Creates a tensor from a raw tensor view.
    pub fn from_raw(ctx: &Arc<WgpuContext>, raw: &TensorView, label: &str) -> Result<Self> {
        log::info!(
            "uploading tensor '{}': dtype={:?}, shape={:?} ({:.2} MB)",
            label,
            raw.dtype,
            raw.shape,
            (raw.bytes.len() as f64) / 1024.0 / 1024.0
        );

        let buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: &raw.bytes,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            });

        Ok(Self::new_allocation(
            Arc::new(buffer),
            raw.shape.clone(),
            raw.dtype,
            ctx.clone(),
        ))
    }

    fn new_allocation(
        buffer: Arc<Buffer>,
        shape: Vec<usize>,
        dtype: DType,
        context: Arc<WgpuContext>,
    ) -> Self {
        if let Some(elem_size) = dtype.element_size() {
            let expected_size = shape.iter().product::<usize>() * elem_size;
            assert_eq!(
                buffer.size() as usize,
                expected_size,
                "buffer size does not match shape dimensions"
            );
        }

        Self {
            buffer,
            shape,
            dtype,
            context,
            id: NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// Creates a tensor from raw bytes with explicit dtype.
    pub fn from_bytes(
        context: &Arc<WgpuContext>,
        bytes: &[u8],
        shape: Vec<usize>,
        dtype: DType,
        label: &str,
    ) -> Result<Self> {
        let expected_bytes = dtype.buffer_size_for_shape(&shape)?;
        anyhow::ensure!(
            bytes.len() == expected_bytes,
            "byte count mismatch: got {}, expected {} for shape {:?} and dtype {:?}",
            bytes.len(),
            expected_bytes,
            shape,
            dtype
        );

        let buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            });

        Ok(Self::new_allocation(
            Arc::new(buffer),
            shape,
            dtype,
            context.clone(),
        ))
    }

    fn new_view(
        buffer: Arc<Buffer>,
        shape: Vec<usize>,
        dtype: DType,
        context: Arc<WgpuContext>,
        id: u64,
    ) -> Self {
        Self {
            buffer,
            shape,
            dtype,
            context,
            id,
        }
    }

    /// Returns the unique buffer ID.
    pub fn buffer_id(&self) -> u64 {
        self.id
    }

    /// Creates a deep copy with a new buffer.
    pub fn deep_clone(&self, label: &str) -> Self {
        let new_buffer = self.context.device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size: self.buffer.size(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("GpuTensor::deep_clone encoder"),
                });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &new_buffer, 0, self.buffer.size());
        self.context.queue.submit(Some(encoder.finish()));

        Self::new_allocation(
            Arc::new(new_buffer),
            self.shape.clone(),
            self.dtype,
            self.context.clone(),
        )
    }

    /// Creates a view with permuted axes (metadata only, no data copy).
    pub fn permute_axes(&self, axes: &[usize]) -> GpuTensor {
        assert_eq!(
            axes.len(),
            self.rank(),
            "permutation axes must match tensor rank"
        );

        let mut new_shape = vec![0; self.rank()];
        for (i, &axis) in axes.iter().enumerate() {
            new_shape[i] = self.shape[axis];
        }

        Self::new_view(
            self.buffer.clone(),
            new_shape,
            self.dtype,
            self.context.clone(),
            self.id,
        )
    }

    /// Creates a tensor initialized with zeros.
    pub fn zeros(
        context: &Arc<WgpuContext>,
        shape: Vec<usize>,
        dtype: DType,
        label: &str,
    ) -> Result<Self> {
        let size_in_bytes = dtype.buffer_size_for_shape(&shape)?;
        let zeros_data = vec![0u8; size_in_bytes];

        let buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: &zeros_data,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            });

        Ok(Self::new_allocation(
            Arc::new(buffer),
            shape,
            dtype,
            context.clone(),
        ))
    }

    /// Creates a view with a different shape (no data copy).
    pub fn view(&self, shape: Vec<usize>) -> Self {
        let new_num_elements: usize = shape.iter().product();
        assert_eq!(
            self.num_elements(),
            new_num_elements,
            "cannot view tensor of shape {:?} as {:?}; element count mismatch",
            self.shape(),
            shape
        );

        Self::new_view(
            Arc::clone(&self.buffer),
            shape,
            self.dtype,
            self.context.clone(),
            self.id,
        )
    }

    /// Views a 4D tensor as 3D.
    pub fn view_as_3d(&self, dim0: usize, dim1: usize, dim2: usize) -> Result<GpuTensor> {
        assert_eq!(self.rank(), 4, "can only view 4D tensors as 3D");
        let (b, h, s, d) = self.dims4();
        assert_eq!(dim0, b * h, "first dimension must equal batch * heads");
        assert_eq!(dim1, s, "second dimension must match");
        assert_eq!(dim2, d, "third dimension must match");

        Ok(Self::new_view(
            self.buffer.clone(),
            vec![dim0, dim1, dim2],
            self.dtype,
            self.context.clone(),
            self.id,
        ))
    }

    /// Views a 3D tensor as 4D.
    pub fn view_as_4d(&self, b: usize, h: usize, s: usize, d: usize) -> Result<GpuTensor> {
        assert_eq!(self.rank(), 3, "can only view 3D tensors as 4D");
        let (bh, s0, d0) = self.dims3();
        assert_eq!(bh, b * h, "first dimension mismatch (B*H != bh)");
        assert_eq!(s, s0, "second dimension mismatch");
        assert_eq!(d, d0, "third dimension mismatch");

        Ok(Self::new_view(
            self.buffer.clone(),
            vec![b, h, s, d],
            self.dtype,
            self.context.clone(),
            self.id,
        ))
    }

    /// Creates an uninitialized tensor.
    pub fn uninitialized(
        context: &Arc<WgpuContext>,
        shape: Vec<usize>,
        dtype: DType,
        label: &str,
    ) -> Self {
        let size = (shape.iter().product::<usize>() * dtype.size_of()) as u64;
        let buffer = context.device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self::new_allocation(Arc::new(buffer), shape, dtype, context.clone())
    }

    /// Creates a tensor from an ndarray.
    pub fn from_ndarray<A, D>(context: &Arc<WgpuContext>, arr: &Array<A, D>) -> Result<Self>
    where
        A: GpuDType,
        D: Dimension,
    {
        let shape = arr.shape().to_vec();
        let buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tensor from ndarray"),
                contents: bytemuck::cast_slice(
                    arr.as_standard_layout()
                        .as_slice()
                        .ok_or_else(|| anyhow!("failed to get slice from ndarray"))?,
                ),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            });

        Ok(Self::new_allocation(
            Arc::new(buffer),
            shape,
            A::DTYPE,
            context.clone(),
        ))
    }

    /// Permutes with actual data copy (creates new buffer).
    pub fn permute(
        &self,
        encoder: &mut CommandEncoder,
        permute_kernel: &GpuPermute,
        axes: &[usize],
    ) -> Self {
        assert_eq!(
            self.rank(),
            axes.len(),
            "permutation axes must match tensor rank"
        );

        let mut output_shape = self.shape().to_vec();
        for i in 0..self.rank() {
            output_shape[i] = self.shape()[axes[i]];
        }

        let output =
            GpuTensor::uninitialized(&self.context, output_shape, self.dtype, "permute output");

        permute_kernel.encode(encoder, self, &output, axes);
        output
    }

    pub fn dims2(&self) -> (usize, usize) {
        assert_eq!(self.rank(), 2, "tensor is not rank 2");
        (self.shape[0], self.shape[1])
    }

    pub fn dims3(&self) -> (usize, usize, usize) {
        assert_eq!(self.rank(), 3, "tensor is not rank 3");
        (self.shape[0], self.shape[1], self.shape[2])
    }

    pub fn dims4(&self) -> (usize, usize, usize, usize) {
        assert_eq!(self.rank(), 4, "tensor is not rank 4");
        (self.shape[0], self.shape[1], self.shape[2], self.shape[3])
    }

    /// Creates a sliced tensor using a GPU kernel.
    pub fn slice(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        slice_kernel: &GpuSlice,
        offset: &[usize],
        shape: &[usize],
    ) -> Result<GpuTensor> {
        assert_eq!(
            self.rank(),
            offset.len(),
            "offset dimensions must match tensor rank"
        );
        assert_eq!(
            self.rank(),
            shape.len(),
            "shape dimensions must match tensor rank"
        );
        for i in 0..self.rank() {
            assert!(
                offset[i] + shape[i] <= self.shape()[i],
                "slice exceeds source tensor dimensions"
            );
        }

        let dst_tensor =
            GpuTensor::uninitialized(&self.context, shape.to_vec(), self.dtype, "sliced tensor");

        slice_kernel.encode(encoder, self, &dst_tensor, offset);

        Ok(dst_tensor)
    }

    pub async fn to_ndarray_1d<A>(&self) -> Result<Array1<A>>
    where
        A: GpuDType + Copy,
    {
        anyhow::ensure!(self.rank() == 1, "tensor rank is not 1");
        let raw_data = self.read_raw_data().await?;
        let data_slice: &[A] = bytemuck::cast_slice(&raw_data);
        Ok(Array1::from_shape_vec(self.shape[0], data_slice.to_vec())?)
    }

    pub async fn to_ndarray_2d<A>(&self) -> Result<Array2<A>>
    where
        A: GpuDType + Copy,
    {
        anyhow::ensure!(self.rank() == 2, "tensor rank is not 2");
        let raw_data = self.read_raw_data().await?;
        let data_slice: &[A] = bytemuck::cast_slice(&raw_data);
        Ok(Array2::from_shape_vec(
            (self.shape[0], self.shape[1]),
            data_slice.to_vec(),
        )?)
    }

    pub async fn to_ndarray_3d<A>(&self) -> Result<Array3<A>>
    where
        A: GpuDType + Copy,
    {
        anyhow::ensure!(self.rank() == 3, "tensor rank is not 3");
        let raw_data = self.read_raw_data().await?;
        let data_slice: &[A] = bytemuck::cast_slice(&raw_data);
        Ok(Array3::from_shape_vec(
            (self.shape[0], self.shape[1], self.shape[2]),
            data_slice.to_vec(),
        )?)
    }

    pub async fn to_ndarray_4d<A>(&self) -> Result<Array4<A>>
    where
        A: GpuDType + Copy,
    {
        anyhow::ensure!(self.rank() == 4, "tensor rank is not 4");
        let raw_data = self.read_raw_data().await?;
        let data_slice: &[A] = bytemuck::cast_slice(&raw_data);
        Ok(Array4::from_shape_vec(
            (self.shape[0], self.shape[1], self.shape[2], self.shape[3]),
            data_slice.to_vec(),
        )?)
    }

    pub async fn read_raw_data(&self) -> Result<Vec<u8>> {
        let device = &self.context.device;
        let queue = &self.context.queue;
        let buffer_size = self.buffer.size();

        let staging_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("staging readback buffer"),
            size: buffer_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback encoder"),
        });
        encoder.copy_buffer_to_buffer(self.buffer(), 0, &staging_buffer, 0, buffer_size);

        queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        crate::gpu_ops::context::drain(device);

        rx.receive()
            .await
            .ok_or(anyhow!("GPU readback channel closed"))??;

        let data = buffer_slice.get_mapped_range().to_vec();
        staging_buffer.unmap();

        Ok(data)
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn context(&self) -> &Arc<WgpuContext> {
        &self.context
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn dtype_as_str(&self) -> &str {
        match self.dtype {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::U32 => "u32",
            DType::Q4_K => "q4_k",
            DType::Q8_0 => "q8_0",
            DType::Q6_K => "q6_k",
            DType::Q5_K => "q5_k",
        }
    }
}

/// Repack `BlockQ8_0` for the GPU: 34 packed bytes become 9 words.
///
/// The scale moves into a word of its own so the 32 quantised values start on a
/// word boundary. See the Q8_0 kernels in `linear.wgsl`, which index this layout
/// directly.
pub(crate) fn repack_q8_0(blocks: &[crate::cpu::kernels::q_common::BlockQ8_0]) -> Vec<u8> {
    const DST: usize = 36;
    let mut out = vec![0u8; blocks.len() * DST];
    for (i, b) in blocks.iter().enumerate() {
        let o = i * DST;
        out[o..o + 2].copy_from_slice(&b.d.to_bits().to_le_bytes());
        // Bytes 2 and 3 stay zero: the shader reads only the low half of word 0.
        out[o + 4..o + 36].copy_from_slice(bytemuck::cast_slice(&b.qs));
    }
    out
}

fn convert_cpu_tensor_to_bytes(
    tensor: CpuTensor,
    target_dtype: DType,
) -> Result<(Vec<u8>, Vec<usize>)> {
    let shape = tensor.shape().to_vec();

    let f32_data: Vec<f32> = match shape.len() {
        1 => tensor.to_array1_f32()?.to_vec(),
        2 => {
            let arr = tensor.to_array2_f32()?;
            arr.as_slice()
                .ok_or_else(|| anyhow!("non-contiguous 2D array"))?
                .to_vec()
        }
        3 => {
            let arr = tensor.to_array3_f32()?;
            arr.as_slice()
                .ok_or_else(|| anyhow!("non-contiguous 3D array"))?
                .to_vec()
        }
        rank => anyhow::bail!("unsupported tensor rank {} for GPU conversion", rank),
    };

    let bytes = match target_dtype {
        DType::F32 => bytemuck::cast_slice(&f32_data).to_vec(),
        DType::F16 => {
            let f16_data: Vec<u16> = f32_data
                .iter()
                .map(|&v| half::f16::from_f32(v).to_bits())
                .collect();
            bytemuck::cast_slice(&f16_data).to_vec()
        }
        DType::BF16 => {
            let bf16_data: Vec<u16> = f32_data
                .iter()
                .map(|&v| half::bf16::from_f32(v).to_bits())
                .collect();
            bytemuck::cast_slice(&bf16_data).to_vec()
        }
        other => anyhow::bail!(
            "unsupported target dtype {:?} for CPU->GPU conversion",
            other
        ),
    };

    Ok((bytes, shape))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    async fn setup_context() -> Arc<WgpuContext> {
        WgpuContext::new().await.unwrap()
    }

    #[ignore = "GPU required"]
    #[tokio::test]
    async fn test_tensor_creation() {
        let ctx = setup_context().await;
        let tensor = GpuTensor::zeros(&ctx, vec![2, 3, 4], DType::F32, "test").unwrap();

        assert_eq!(tensor.shape(), &[2, 3, 4]);
        assert_eq!(tensor.rank(), 3);
        assert_eq!(tensor.num_elements(), 24);
    }

    #[ignore = "GPU required"]
    #[tokio::test]
    async fn test_from_ndarray() {
        let ctx = setup_context().await;
        let arr = Array3::<f32>::ones((2, 3, 4));
        let tensor = GpuTensor::from_ndarray(&ctx, &arr).unwrap();

        assert_eq!(tensor.shape(), &[2, 3, 4]);

        let result = tensor.to_ndarray_3d::<f32>().await.unwrap();
        assert_eq!(result, arr);
    }

    #[ignore = "GPU required"]
    #[tokio::test]
    async fn test_view_operations() {
        let ctx = setup_context().await;
        let tensor = GpuTensor::zeros(&ctx, vec![2, 3, 4], DType::F32, "test").unwrap();

        let viewed = tensor.view(vec![6, 4]);
        assert_eq!(viewed.shape(), &[6, 4]);
        assert_eq!(viewed.num_elements(), 24);
    }

    #[ignore = "GPU required"]
    #[tokio::test]
    async fn test_view_4d_to_3d() {
        let ctx = setup_context().await;
        let tensor = GpuTensor::zeros(&ctx, vec![2, 4, 8, 64], DType::F32, "test").unwrap();

        let viewed = tensor.view_as_3d(8, 8, 64).unwrap();
        assert_eq!(viewed.shape(), &[8, 8, 64]);
    }

    #[ignore = "GPU required"]
    #[tokio::test]
    async fn test_view_3d_to_4d() {
        let ctx = setup_context().await;
        let tensor = GpuTensor::zeros(&ctx, vec![8, 10, 64], DType::F32, "test").unwrap();

        let viewed = tensor.view_as_4d(2, 4, 10, 64).unwrap();
        assert_eq!(viewed.shape(), &[2, 4, 10, 64]);
    }

    #[ignore = "GPU required"]
    #[tokio::test]
    async fn test_dims_helpers() {
        let ctx = setup_context().await;

        let t2d = GpuTensor::zeros(&ctx, vec![5, 10], DType::F32, "test").unwrap();
        assert_eq!(t2d.dims2(), (5, 10));

        let t3d = GpuTensor::zeros(&ctx, vec![2, 3, 4], DType::F32, "test").unwrap();
        assert_eq!(t3d.dims3(), (2, 3, 4));

        let t4d = GpuTensor::zeros(&ctx, vec![1, 2, 3, 4], DType::F32, "test").unwrap();
        assert_eq!(t4d.dims4(), (1, 2, 3, 4));
    }

    #[ignore = "GPU required"]
    #[tokio::test]
    async fn test_roundtrip_cpu_gpu() {
        let ctx = setup_context().await;
        let original =
            Array3::<f32>::from_shape_fn((2, 3, 4), |(i, j, k)| (i * 12 + j * 4 + k) as f32);

        let gpu_tensor = GpuTensor::from_ndarray(&ctx, &original).unwrap();
        let result = gpu_tensor.to_ndarray_3d::<f32>().await.unwrap();

        assert_eq!(result, original);
    }

    #[tokio::test]
    #[ignore = "GPU required"]
    #[should_panic(expected = "cannot view tensor")]
    async fn test_invalid_view() {
        let ctx = setup_context().await;
        let tensor = GpuTensor::zeros(&ctx, vec![2, 3, 4], DType::F32, "test").unwrap();
        tensor.view(vec![2, 3, 5]);
    }

    #[tokio::test]
    async fn test_dtype_size() {
        assert_eq!(DType::F32.size_of(), 4);
        assert_eq!(DType::U32.size_of(), 4);
    }

    #[ignore = "GPU required"]
    #[tokio::test]
    async fn test_buffer_id_uniqueness() {
        let ctx = setup_context().await;

        let t1 = GpuTensor::zeros(&ctx, vec![2, 2], DType::F32, "t1").unwrap();
        let t2 = GpuTensor::zeros(&ctx, vec![2, 2], DType::F32, "t2").unwrap();

        assert_ne!(t1.buffer_id(), t2.buffer_id());
    }

    #[ignore = "GPU required"]
    #[tokio::test]
    async fn test_view_keeps_buffer_id() {
        let ctx = setup_context().await;

        let original = GpuTensor::zeros(&ctx, vec![2, 3, 4], DType::F32, "original").unwrap();
        let viewed = original.view(vec![6, 4]);

        assert_eq!(original.buffer_id(), viewed.buffer_id());
    }

    #[ignore = "GPU required"]
    #[tokio::test]
    async fn test_deep_clone_gets_new_id() {
        let ctx = setup_context().await;

        let original = GpuTensor::zeros(&ctx, vec![2, 2], DType::F32, "original").unwrap();
        let cloned = original.deep_clone("cloned");

        assert_ne!(original.buffer_id(), cloned.buffer_id());
        assert_eq!(original.shape(), cloned.shape());
        assert_eq!(original.dtype(), cloned.dtype());
    }

    #[ignore = "GPU required"]
    #[tokio::test]
    async fn test_clone_keeps_buffer_id() {
        let ctx = setup_context().await;

        let original = GpuTensor::zeros(&ctx, vec![2, 2], DType::F32, "original").unwrap();
        let shallow = original.clone();

        assert_eq!(original.buffer_id(), shallow.buffer_id());
    }

    #[ignore = "GPU required"]
    #[tokio::test]
    async fn test_from_bytes_validation() {
        let ctx = setup_context().await;

        let bytes = vec![0u8; 16];
        let result = GpuTensor::from_bytes(&ctx, &bytes, vec![2, 2], DType::F32, "test");
        assert!(result.is_ok());

        let wrong_bytes = vec![0u8; 12];
        let result = GpuTensor::from_bytes(&ctx, &wrong_bytes, vec![2, 2], DType::F32, "test");
        assert!(result.is_err());
    }

    #[ignore = "GPU required"]
    #[tokio::test]
    async fn test_linear_layer_dims() {
        let ctx = setup_context().await;

        let weight = GpuTensor::zeros(&ctx, vec![768, 512], DType::F32, "weight").unwrap();

        let (in_feat, out_feat) = weight.linear_layer_dims();
        assert_eq!(in_feat, 512);
        assert_eq!(out_feat, 768);
    }

    #[ignore = "GPU required"]
    #[tokio::test]
    async fn test_dtype_as_str() {
        let ctx = setup_context().await;

        let f32_tensor = GpuTensor::zeros(&ctx, vec![2], DType::F32, "f32").unwrap();
        assert_eq!(f32_tensor.dtype_as_str(), "f32");
    }

    #[ignore = "GPU required"]
    #[tokio::test]
    async fn test_permute_axes_metadata() {
        let ctx = setup_context().await;

        let tensor = GpuTensor::zeros(&ctx, vec![2, 3, 4], DType::F32, "test").unwrap();
        let permuted = tensor.permute_axes(&[2, 0, 1]);

        assert_eq!(permuted.shape(), &[4, 2, 3]);
        assert_eq!(tensor.buffer_id(), permuted.buffer_id());
    }

    #[ignore = "GPU required"]
    #[tokio::test]
    async fn test_uninitialized_tensor() {
        let ctx = setup_context().await;

        let tensor = GpuTensor::uninitialized(&ctx, vec![100, 100], DType::F32, "uninit");

        assert_eq!(tensor.shape(), &[100, 100]);
        assert_eq!(tensor.num_elements(), 10000);
    }

    #[ignore = "GPU required"]
    #[tokio::test]
    async fn test_debug_format() {
        let ctx = setup_context().await;

        let tensor = GpuTensor::zeros(&ctx, vec![2, 3], DType::F32, "test").unwrap();
        let debug_str = format!("{:?}", tensor);

        assert!(debug_str.contains("GpuTensor"));
        assert!(debug_str.contains("[2, 3]"));
        assert!(debug_str.contains("F32"));
    }
}
