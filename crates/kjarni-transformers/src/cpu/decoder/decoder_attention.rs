use crate::linear_layer::LinearLayer;
use crate::rope::RoPE;
use crate::utils::MASK_VALUE;
use anyhow::Result;
use ndarray::{Array2, Array3, Array4, s};

/// Self-attention for decoder-only transformer models.
pub struct DecoderAttention {
    pub q_proj: LinearLayer,
    pub k_proj: LinearLayer,
    pub v_proj: LinearLayer,
    pub o_proj: LinearLayer,

    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub scale_factor: f32,
}

impl DecoderAttention {
    /// Creates a new decoder attention layer with separate Q/K/V/O projections.
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        q: LinearLayer,
        k: LinearLayer,
        v: LinearLayer,
        o: LinearLayer,
        num_kv_heads: Option<usize>,
    ) -> Self {
        let num_kv_heads = num_kv_heads.unwrap_or(num_heads);
        let head_dim = hidden_size / num_heads;

        Self {
            q_proj: q,
            k_proj: k,
            v_proj: v,
            o_proj: o,
            num_heads,
            num_kv_heads,
            head_dim,
            scale_factor: 1.0 / (head_dim as f32).sqrt(),
        }
    }

    /// Computes self-attention with in-place KV cache updates.
    pub fn forward(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: Option<&Array2<f32>>,
        mut k_cache: ndarray::ArrayViewMut3<f32>,
        mut v_cache: ndarray::ArrayViewMut3<f32>,
        position_offset: usize,
        rope: Option<&RoPE>,
    ) -> Result<Array3<f32>> {
        let (batch, seq_len, _) = hidden_states.dim();
        let total_len = k_cache.shape()[1];
        let is_decode = seq_len == 1;

        let hidden_2d = hidden_states
            .view()
            .into_shape_with_order((batch * seq_len, self.num_heads * self.head_dim))?;

        let q = self.q_proj.matmul(&hidden_2d);
        let mut q_3d = q.into_shape_with_order((batch, seq_len, self.num_heads * self.head_dim))?;
        let start_write = total_len - seq_len;
        let k_new = self.k_proj.matmul(&hidden_2d);
        k_cache
            .slice_mut(s![.., start_write.., ..])
            .assign(&k_new.into_shape_with_order((
                batch,
                seq_len,
                self.num_kv_heads * self.head_dim,
            ))?);
        let v_new = self.v_proj.matmul(&hidden_2d);
        v_cache
            .slice_mut(s![.., start_write.., ..])
            .assign(&v_new.into_shape_with_order((
                batch,
                seq_len,
                self.num_kv_heads * self.head_dim,
            ))?);

        // Apply RoPE
        if let Some(r) = rope {
            let k_temp_owned = k_cache.slice(s![.., start_write.., ..]).to_owned();

            let (q_rot, k_rot) = r.apply_3d(
                &q_3d,
                &k_temp_owned,
                self.num_heads,
                self.num_kv_heads,
                position_offset,
            )?;
            q_3d.assign(&q_rot);
            // Write K back into the cache
            k_cache.slice_mut(s![.., start_write.., ..]).assign(&k_rot);
        }
        let q_heads = q_3d
            .into_shape_with_order((batch, seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3])
            .to_owned(); // [Batch, Heads, Seq, Dim]

        let n_rep = self.num_heads / self.num_kv_heads;

        // Prefill without a padding mask is the case where materialising
        // [batch, heads, seq, seq] is both unnecessary and expensive: at 4096
        // tokens that tensor is 939 MB and crosses the bus four times per layer.
        // The streaming path computes the same thing a block at a time. It is
        // gated on there being no external mask because it only knows how to
        // apply causality, and on seq_len being large enough for the block
        // machinery to pay for itself.
        let mask_is_noop = attention_mask.is_none_or(|m| m.iter().all(|&x| x == 1.0));
        if !is_decode && seq_len >= 128 && mask_is_noop && !streaming_disabled() {
            let context = super::streaming_attention::streaming_causal_attention(
                &q_heads,
                &k_cache.view(),
                &v_cache.view(),
                self.num_kv_heads,
                self.scale_factor,
                start_write,
            );
            let context_flat = context
                .permuted_axes([0, 2, 1, 3])
                .as_standard_layout()
                .into_shape_with_order((batch * seq_len, self.num_heads * self.head_dim))?
                .to_owned();
            let output = self.o_proj.matmul(&context_flat.view());
            return Ok(output.into_shape_with_order((
                batch,
                seq_len,
                self.num_heads * self.head_dim,
            ))?);
        }

        // Decode reads the whole KV cache for one query row, so how it walks that
        // cache is the entire cost. See the note on `decode_attention`.
        if is_decode && !streaming_disabled() {
            let context = super::streaming_attention::decode_attention(
                &q_heads,
                &k_cache.view(),
                &v_cache.view(),
                self.num_kv_heads,
                self.scale_factor,
            );
            let context_flat = context
                .permuted_axes([0, 2, 1, 3])
                .as_standard_layout()
                .into_shape_with_order((batch * seq_len, self.num_heads * self.head_dim))?
                .to_owned();
            let output = self.o_proj.matmul(&context_flat.view());
            return Ok(output.into_shape_with_order((
                batch,
                seq_len,
                self.num_heads * self.head_dim,
            ))?);
        }

        let mut scores = if is_decode {
            let k_view = k_cache
                .view()
                .into_shape_with_order((batch, total_len, self.num_kv_heads, self.head_dim))?
                .permuted_axes([0, 2, 3, 1]);

            crate::utils::linear_algebra::matmul_4d_decode_gqa(&q_heads, &k_view, n_rep)
        } else {
            let k_heads = self.prepare_kv_heads(&k_cache.view().into_owned(), batch)?;
            let k_t = k_heads.permuted_axes([0, 1, 3, 2]);
            crate::utils::linear_algebra::matmul_4d(&q_heads, &k_t)
        };

        // Scale & Softmax
        scores.mapv_inplace(|x| x * self.scale_factor);

        if let Some(mask) = attention_mask {
            scores = crate::utils::linear_algebra::apply_attention_mask(scores, mask);
        }

        self.apply_causal_mask(&mut scores, start_write);

        crate::activations::softmax_4d_inplace(&mut scores);
        let context = if is_decode {
            let v_view = v_cache
                .view()
                .into_shape_with_order((batch, total_len, self.num_kv_heads, self.head_dim))?
                .permuted_axes([0, 2, 1, 3]);

            crate::utils::linear_algebra::matmul_4d_context_gqa(&scores, &v_view, n_rep)
        } else {
            let v_heads = self.prepare_kv_heads(&v_cache.view().into_owned(), batch)?;
            crate::utils::linear_algebra::matmul_4d(&scores, &v_heads)
        };

        // Output Projection
        let context_flat = context
            .permuted_axes([0, 2, 1, 3])
            .as_standard_layout()
            .into_shape_with_order((batch * seq_len, self.num_heads * self.head_dim))?
            .to_owned();

        let output = self.o_proj.matmul(&context_flat.view());
        Ok(output.into_shape_with_order((batch, seq_len, self.num_heads * self.head_dim))?)
    }

    fn prepare_kv_heads(&self, kv: &Array3<f32>, batch: usize) -> Result<Array4<f32>> {
        let total_seq = kv.shape()[1];
        let kv_heads = kv
            .view()
            .into_shape_with_order((batch, total_seq, self.num_kv_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3]); // [B, KV_H, T, D]

        if self.num_heads == self.num_kv_heads {
            Ok(kv_heads.to_owned())
        } else {
            // Repeat Logic for GQA
            let groups = self.num_heads / self.num_kv_heads;
            let mut out = Array4::zeros((batch, self.num_heads, total_seq, self.head_dim));
            for i in 0..self.num_kv_heads {
                for g in 0..groups {
                    out.slice_mut(s![.., i * groups + g, .., ..])
                        .assign(&kv_heads.slice(s![.., i, .., ..]));
                }
            }
            Ok(out)
        }
    }

    /// Apply causality to scores
    fn apply_causal_mask(&self, scores: &mut Array4<f32>, cache_len: usize) {
        let (_, _, seq_q, _) = scores.dim();
        for i in 0..seq_q {
            let query_pos = cache_len + i;
            for j in 0..scores.shape()[3] {
                if j > query_pos {
                    scores.slice_mut(s![.., .., i, j]).fill(MASK_VALUE);
                }
            }
        }
    }
}

/// Escape hatch for the streaming path.
///
/// Set `KJARNI_NO_STREAMING_ATTN=1` to force the materialising path. Kept so the
/// two can be compared against each other in one build, and so a bug here can be
/// ruled out without a rebuild.
fn streaming_disabled() -> bool {
    use std::sync::OnceLock;
    static OFF: OnceLock<bool> = OnceLock::new();
    *OFF.get_or_init(|| {
        std::env::var("KJARNI_NO_STREAMING_ATTN")
            .map(|v| v != "0")
            .unwrap_or(false)
    })
}

#[cfg(test)]
mod decoder_attention_test {
    use super::*;
    use crate::linear_layer::LinearLayer;
    use anyhow::Result;
    use ndarray::{Array2, Array3, Array4, s};

    fn load_linear(rows: usize, cols: usize, data: Vec<f32>) -> LinearLayer {
        let weight = Array2::from_shape_vec((rows, cols), data).unwrap();
        LinearLayer::new_f32(weight, None)
    }
    fn get_weights() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        // weight_q Shape: [16, 16]
        let weight_q_data = vec![
            0.152908, 0.166002, -0.046854, 0.183722, -0.043821, 0.040358, -0.097371, 0.117457,
            0.176309, -0.146726, 0.173839, 0.037432, 0.147762, 0.027086, 0.096438, -0.028238,
            0.154177, 0.029562, -0.093368, 0.050980, -0.092147, -0.023455, -0.081232, 0.132674,
            -0.157874, -0.092202, -0.056475, -0.120254, 0.018877, -0.197536, 0.180622, -0.169894,
            0.154405, 0.033284, -0.064941, 0.123590, 0.031170, 0.161593, 0.021864, -0.063075,
            0.053737, -0.054236, 0.084172, 0.178564, 0.115612, -0.087435, 0.115453, 0.035785,
            0.101567, -0.121901, -0.197982, -0.077272, -0.153405, 0.164108, 0.057606, 0.082843,
            0.063252, -0.003479, 0.156522, -0.142103, 0.012593, -0.136508, 0.061670, -0.068876,
            0.061283, -0.041668, 0.165878, -0.118540, -0.119280, -0.119287, 0.179889, 0.066650,
            0.192450, -0.165055, -0.198375, -0.156473, -0.134538, 0.081008, 0.071615, 0.166185,
            -0.103285, -0.136342, 0.106116, -0.080841, 0.121385, -0.047460, 0.114409, -0.155394,
            -0.100930, 0.060975, 0.042282, -0.050992, 0.119214, 0.135962, -0.145035, -0.106774,
            0.183132, -0.067487, -0.070903, -0.193519, -0.114534, 0.049961, -0.026399, -0.145177,
            0.004691, -0.136616, -0.169679, -0.110133, -0.175042, -0.127348, 0.199922, 0.037775,
            0.061632, -0.186537, -0.131355, -0.066571, 0.031274, -0.175984, -0.086175, -0.119734,
            0.000554, -0.074421, -0.013859, -0.135526, -0.137279, -0.116680, -0.068459, -0.157856,
            0.167694, -0.039693, 0.172079, 0.062316, -0.169359, 0.138407, -0.055029, -0.076665,
            -0.166014, -0.198832, 0.057222, -0.043689, 0.077865, -0.164133, 0.148486, -0.146811,
            -0.034535, 0.041774, 0.103250, 0.161462, 0.182192, -0.158584, 0.050333, -0.086025,
            -0.021917, -0.149698, 0.182172, -0.146790, 0.106890, 0.070288, 0.064991, -0.108129,
            0.181790, 0.043950, 0.025728, -0.176251, 0.083958, -0.030004, -0.091625, 0.171789,
            0.044590, -0.110655, -0.101228, -0.009551, 0.111672, -0.051107, -0.114115, -0.068489,
            -0.149415, 0.071326, 0.154808, -0.188288, 0.046450, 0.103318, 0.036266, -0.071225,
            0.104388, 0.105103, 0.074785, -0.035144, -0.052960, 0.021396, -0.035331, -0.059600,
            0.127841, 0.171880, -0.019799, -0.044779, 0.002918, -0.011942, 0.048082, 0.056047,
            -0.181651, -0.073808, 0.168426, 0.077911, -0.009948, -0.120581, -0.122361, -0.179153,
            -0.065192, 0.067541, 0.127524, 0.092339, -0.176789, -0.120272, -0.031563, 0.193470,
            0.028931, -0.051794, 0.082743, -0.076176, -0.129451, 0.145977, -0.090940, -0.040093,
            -0.198961, 0.133854, 0.151527, 0.072890, -0.139455, -0.197388, -0.162436, 0.149140,
            0.096021, 0.168301, 0.104774, 0.050618, -0.001959, -0.152101, -0.171354, -0.187070,
            0.081872, -0.098194, -0.040251, -0.115101, -0.036445, -0.140767, -0.130683, 0.066342,
            -0.059439, 0.123469, -0.064162, -0.146713, -0.035288, -0.096949, -0.061188, -0.190399,
        ];

        // weight_k Shape: [8, 16]
        let weight_k_data = vec![
            0.111898, -0.139241, 0.100524, 0.090757, 0.142887, -0.153410, 0.143839, -0.094550,
            0.074214, 0.187823, -0.028206, -0.001547, -0.046046, -0.166997, 0.095981, -0.198544,
            0.124160, 0.149645, 0.189141, -0.047176, -0.164328, 0.044966, 0.110485, -0.199062,
            -0.045397, -0.119891, -0.017493, -0.098443, -0.081754, -0.063492, -0.190061, 0.164102,
            0.167666, -0.031374, -0.022776, -0.081624, -0.180613, -0.194629, 0.074332, -0.109809,
            -0.128575, -0.015605, -0.066602, -0.064704, 0.006426, -0.042422, -0.068862, -0.095761,
            -0.162765, 0.167701, -0.080037, 0.052996, -0.069393, 0.016252, 0.186460, 0.092145,
            -0.173320, 0.079381, 0.189849, 0.052617, 0.134085, 0.197178, -0.030646, 0.041511,
            -0.139007, -0.041215, 0.148117, 0.102529, -0.126558, -0.160377, -0.136673, -0.197376,
            -0.154328, -0.049459, 0.134975, 0.033476, -0.152121, -0.160445, 0.099495, -0.148768,
            -0.024625, 0.095941, -0.092562, -0.021808, -0.017409, -0.047317, -0.101406, -0.178288,
            -0.161671, -0.107092, 0.193168, -0.096603, -0.134306, 0.048479, 0.055122, 0.109582,
            0.152024, 0.111375, -0.198300, 0.017738, 0.121151, -0.018485, -0.117856, 0.190668,
            -0.074806, -0.113869, -0.180311, 0.008934, 0.088627, 0.044273, 0.039550, -0.151677,
            -0.186777, 0.003522, 0.182367, 0.115384, -0.116447, -0.025962, -0.147437, -0.096485,
            0.036220, 0.108908, 0.165674, -0.183621, 0.133723, -0.141058, 0.074893, 0.169249,
        ];

        // weight_v Shape: [8, 16]
        let weight_v_data = vec![
            0.002808, 0.181962, -0.170410, -0.076392, 0.116651, -0.043574, -0.040940, -0.083358,
            0.137861, 0.098101, 0.064090, -0.112393, -0.162350, 0.021632, 0.059256, -0.092342,
            -0.055960, 0.135074, 0.015932, 0.009024, -0.049220, -0.181118, -0.188051, -0.095603,
            -0.101664, 0.062311, -0.058222, -0.078244, 0.190686, 0.069665, 0.142580, -0.096823,
            -0.081693, 0.073508, -0.133255, -0.130741, -0.009660, -0.073152, -0.149932, 0.118632,
            0.160833, 0.032445, -0.034823, -0.185255, -0.072848, 0.050917, 0.094306, -0.025284,
            -0.079071, 0.111445, -0.159280, 0.126404, -0.077591, 0.003061, -0.039523, 0.024248,
            -0.060440, 0.145425, -0.005199, 0.156120, 0.192296, -0.097438, -0.145902, 0.160460,
            0.156723, -0.152709, -0.015461, -0.197225, -0.163720, 0.038629, 0.053207, 0.042396,
            -0.054433, 0.184516, 0.028596, -0.118017, -0.011323, 0.048029, 0.070039, -0.141416,
            0.074958, -0.102176, -0.166188, -0.109241, 0.192882, 0.170972, 0.179097, 0.117402,
            0.151090, -0.026770, -0.110046, 0.099931, -0.103637, -0.134973, -0.063867, 0.041972,
            0.102959, -0.077682, -0.117713, 0.026979, -0.117887, -0.130212, 0.104250, -0.033597,
            0.182757, 0.194557, 0.059821, 0.068832, 0.046057, 0.003132, -0.014546, 0.002749,
            0.074685, 0.185954, -0.051832, -0.084543, -0.048433, -0.096625, 0.034008, 0.149290,
            0.156395, 0.091825, -0.147186, -0.107341, -0.043942, -0.036865, 0.016450, -0.183594,
        ];

        // weight_o Shape: [16, 16]
        let weight_o_data = vec![
            0.062249, -0.152574, -0.126549, -0.166276, 0.174264, -0.189388, 0.150873, -0.006723,
            -0.023260, 0.125096, -0.018486, 0.125431, 0.144603, -0.173642, 0.076957, 0.037756,
            0.043002, 0.029198, 0.054706, -0.096213, -0.025588, 0.190024, 0.134370, -0.007514,
            -0.188106, 0.008766, -0.136195, 0.162638, -0.121417, -0.014440, -0.044389, 0.035591,
            0.188206, 0.019004, 0.115833, 0.155244, 0.161462, -0.069070, -0.044731, 0.096388,
            -0.054574, 0.093653, -0.043694, -0.135650, 0.081409, 0.030664, 0.089170, 0.198697,
            0.136547, 0.189596, 0.010705, -0.172043, -0.140305, -0.124235, -0.176250, -0.100250,
            -0.184114, -0.184523, -0.119511, -0.197167, -0.122762, 0.076262, 0.166811, -0.059493,
            -0.058176, 0.106791, -0.098674, -0.094567, 0.123226, -0.174260, 0.024455, 0.176676,
            0.034297, 0.054389, -0.116478, -0.002759, 0.010997, 0.049087, 0.077709, 0.173786,
            -0.152660, 0.005995, -0.099927, -0.158213, -0.016016, -0.176047, 0.139580, 0.023163,
            -0.107792, 0.104516, -0.189286, -0.077360, -0.038964, -0.169950, -0.127178, -0.032644,
            0.151753, 0.193131, 0.127252, -0.119425, -0.130844, 0.174539, 0.070743, 0.005314,
            0.027066, -0.160739, -0.066777, 0.192524, -0.049326, -0.010033, -0.166066, -0.111881,
            -0.004084, -0.124232, -0.024802, 0.081392, -0.195635, 0.059402, -0.132242, -0.097610,
            0.076782, 0.159024, -0.054662, -0.082114, -0.180847, -0.103132, -0.175127, -0.045774,
            0.040811, -0.187375, 0.174623, 0.125478, -0.195789, -0.095527, 0.065231, -0.041092,
            -0.021795, -0.090303, 0.160644, -0.111800, 0.165855, 0.012904, 0.040204, 0.156026,
            -0.032953, -0.113869, -0.032347, 0.162211, -0.148397, 0.045396, -0.196558, 0.104860,
            0.073894, 0.008478, 0.085839, 0.000225, 0.110671, -0.158324, -0.029370, 0.088723,
            0.199163, 0.101878, -0.145435, 0.153819, -0.044600, -0.042703, -0.181782, -0.031483,
            0.141465, 0.027889, -0.116491, 0.061562, -0.064129, 0.182599, -0.173591, -0.063175,
            -0.193115, -0.078766, 0.063050, 0.192523, 0.033589, 0.196072, 0.039130, 0.115507,
            0.160332, 0.167185, -0.111945, 0.183878, 0.121153, -0.093516, -0.095441, -0.167749,
            0.050227, -0.162110, 0.084485, 0.063160, -0.173760, 0.054500, -0.016266, 0.091364,
            0.114758, -0.198829, 0.183420, 0.167728, 0.079577, -0.182792, -0.071444, -0.057962,
            -0.051400, 0.112785, 0.072714, 0.158436, -0.074905, 0.067308, 0.071159, -0.166518,
            -0.194004, -0.103778, 0.136910, -0.188292, -0.174087, 0.112040, 0.107906, 0.164479,
            -0.150988, -0.146380, 0.102597, 0.173926, 0.119668, 0.031330, 0.065915, 0.189825,
            -0.129041, -0.090802, 0.139893, -0.136848, -0.110283, 0.145998, 0.063104, 0.064614,
            -0.084768, -0.002760, 0.183047, -0.120044, 0.001572, 0.095120, -0.138071, 0.194235,
            -0.099922, -0.048027, -0.054114, -0.130332, -0.196253, 0.112770, 0.053135, -0.187320,
        ];
        (weight_k_data, weight_o_data, weight_q_data, weight_v_data)
    }
    #[test]
    fn test_decoder_attention_golden_gqa() -> Result<()> {
        let hidden = 16;
        let heads = 4;
        let kv_heads = 2;
        let head_dim = hidden / heads; // 4

        let (weight_k_data, weight_o_data, weight_q_data, weight_v_data) = get_weights();

        let q = load_linear(hidden, hidden, weight_q_data);
        let k = load_linear(kv_heads * head_dim, hidden, weight_k_data);
        let v = load_linear(kv_heads * head_dim, hidden, weight_v_data);
        let o = load_linear(hidden, hidden, weight_o_data);

        let attn = DecoderAttention::new(hidden, heads, q, k, v, o, Some(kv_heads));

        let attn_input_hidden_data = vec![
            0.344363, -3.101606, -1.458723, -1.431826, -0.607127, -0.259738, -0.719019, -0.385831,
            0.523353, -0.821177, -0.470869, 0.601642, -0.282511, 0.769268, -0.766892, -0.949487,
        ];
        let hidden_in = Array3::from_shape_vec((1, 1, 16), attn_input_hidden_data)?;

        let total_len = 3;
        let cache_dim = kv_heads * head_dim;
        let mut k_cache = Array3::zeros((1, total_len, cache_dim));
        let mut v_cache = Array3::zeros((1, total_len, cache_dim));

        let attn_history_k_data = vec![
            0.016917, 0.080277, 0.744841, 1.345485, -0.734670, 0.044657, -1.521120, 0.347838,
            0.126822, -2.452072, 0.415976, 1.902536, 0.740177, 1.416200, 0.683398, -0.138252,
        ];
        let attn_history_v_data = vec![
            0.921300, 0.528244, -0.008228, -1.449332, -0.414599, 1.455870, 0.331653, -1.000101,
            -0.605182, -0.179245, 0.199559, -1.246195, -0.691952, -0.471991, -1.289434, 1.076281,
        ];
        let history_k = Array3::from_shape_vec((1, 2, 8), attn_history_k_data)?;
        let history_v = Array3::from_shape_vec((1, 2, 8), attn_history_v_data)?;

        k_cache.slice_mut(s![.., ..2, ..]).assign(&history_k);
        v_cache.slice_mut(s![.., ..2, ..]).assign(&history_v);

        let output = attn.forward(
            &hidden_in,
            None,
            k_cache.view_mut(),
            v_cache.view_mut(),
            2,
            None, // No RoPE
        )?;

        let attn_output_data = vec![
            0.060655, 0.152719, -0.276075, 0.318217, -0.113340, 0.063543, 0.101776, 0.033563,
            -0.264849, -0.573791, 0.028018, -0.277676, -0.501199, 0.055808, -0.315817, 0.134697,
        ];
        let golden_output = Array3::from_shape_vec((1, 1, 16), attn_output_data)?;

        let diff = (&output - &golden_output).mapv(|x| x.abs());
        let max_diff = diff.fold(0.0f32, |a, &b| a.max(b));
        println!("GQA Output Max Diff: {:.6}", max_diff);
        assert!(max_diff < 1e-4, "GQA Output mismatch");

        let k_updated_slot = k_cache.slice(s![.., 2, ..]);
        let v_updated_slot = v_cache.slice(s![.., 2, ..]);

        let attn_update_k_data = vec![
            -0.023620, -0.556388, 0.501513, -0.782774, -0.089049, -0.192072, 0.180540, -0.909185,
        ];
        let golden_k_update = Array2::from_shape_vec((1, 8), attn_update_k_data)?; // Sliced to 2D

        let k_diff = (&k_updated_slot - &golden_k_update).mapv(|x| x.abs());
        let max_k_diff = k_diff.fold(0.0f32, |a, &b| a.max(b));
        assert!(max_k_diff < 1e-4, "Cache K update mismatch");

        let attn_update_v_data = vec![
            -0.204815, -0.367078, 0.186254, -0.480577, 0.723541, 0.553549, 0.399943, -0.224999,
        ];
        let golden_v_update = Array2::from_shape_vec((1, 8), attn_update_v_data)?;

        let v_diff = (&v_updated_slot - &golden_v_update).mapv(|x| x.abs());
        let max_v_diff = v_diff.fold(0.0f32, |a, &b| a.max(b));
        assert!(max_v_diff < 1e-4, "Cache V update mismatch");

        Ok(())
    }
    #[test]
    fn test_decoder_attention_masking() -> Result<()> {
        let hidden = 8;
        let heads = 2;
        let eye = Array2::eye(hidden);
        let q = LinearLayer::new_f32(eye.clone(), None);
        let k = LinearLayer::new_f32(eye.clone(), None);
        let v = LinearLayer::new_f32(eye.clone(), None);
        let o = LinearLayer::new_f32(eye.clone(), None);

        let attn = DecoderAttention::new(hidden, heads, q, k, v, o, None);

        let hidden_in = Array3::from_elem((1, 1, hidden), 1.0);

        let total_len = 3;
        let mut k_cache = Array3::ones((1, total_len, hidden));
        let mut v_cache = Array3::zeros((1, total_len, hidden));

        v_cache.slice_mut(s![.., 0..2, ..]).fill(5.0);

        let out_unmasked = attn.forward(
            &hidden_in,
            None,
            k_cache.view_mut(),
            v_cache.view_mut(),
            2,
            None,
        )?;

        v_cache.slice_mut(s![.., 2, ..]).fill(0.0);

        let mask = Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 1.0]).unwrap();

        let out_masked = attn.forward(
            &hidden_in,
            Some(&mask),
            k_cache.view_mut(),
            v_cache.view_mut(),
            2,
            None,
        )?;

        println!(
            "Unmasked Mean (Expect ~3.66): {:.4}",
            out_unmasked.mean().unwrap()
        );
        println!(
            "Masked Mean   (Expect ~1.00): {:.4}",
            out_masked.mean().unwrap()
        );

        let diff = (&out_unmasked - &out_masked).mapv(|x| x.abs());
        let max_diff = diff.fold(0.0f32, |a, &b| a.max(b));

        assert!(
            max_diff > 1.0,
            "Masking failed: Output didn't change enough!"
        );

        Ok(())
    }

    fn create_dummy_attn(hidden: usize, heads: usize, kv_heads: usize) -> DecoderAttention {
        let eye = Array2::eye(hidden);
        let dummy_linear1 = LinearLayer::new_f32(eye.clone(), None);
        let dummy_linear2 = LinearLayer::new_f32(eye.clone(), None);
        let dummy_linear3 = LinearLayer::new_f32(eye.clone(), None);
        let dummy_linear4 = LinearLayer::new_f32(eye.clone(), None);

        DecoderAttention::new(
            hidden,
            heads,
            dummy_linear1,
            dummy_linear2,
            dummy_linear3,
            dummy_linear4,
            Some(kv_heads),
        )
    }

    #[test]
    fn test_prepare_kv_heads_gqa_expansion() {
        let hidden = 4;
        let heads = 4;
        let kv_heads = 2;
        let attn = create_dummy_attn(hidden, heads, kv_heads);

        let input_data = vec![10.0, 20.0];
        let kv_input = Array3::from_shape_vec((1, 1, 2), input_data).unwrap();

        let out = attn
            .prepare_kv_heads(&kv_input, 1)
            .expect("prepare_kv_heads failed");

        assert_eq!(out.shape(), &[1, 4, 1, 1]);

        assert_eq!(out[[0, 0, 0, 0]], 10.0, "Q Head 0 should match KV Head 0");
        assert_eq!(out[[0, 1, 0, 0]], 10.0, "Q Head 1 should match KV Head 0");

        assert_eq!(out[[0, 2, 0, 0]], 20.0, "Q Head 2 should match KV Head 1");
        assert_eq!(out[[0, 3, 0, 0]], 20.0, "Q Head 3 should match KV Head 1");
    }

    #[test]
    fn test_prepare_kv_heads_mha_no_expansion() {
        let hidden = 2;
        let heads = 2;
        let kv_heads = 2;
        let attn = create_dummy_attn(hidden, heads, kv_heads);

        let input_data = vec![10.0, 20.0];
        let kv_input = Array3::from_shape_vec((1, 1, 2), input_data).unwrap();

        let out = attn.prepare_kv_heads(&kv_input, 1).unwrap();

        assert_eq!(out[[0, 0, 0, 0]], 10.0);
        assert_eq!(out[[0, 1, 0, 0]], 20.0);
    }

    #[test]
    fn test_apply_causal_mask_structure() {
        let hidden = 4;
        let attn = create_dummy_attn(hidden, 1, 1);

        let mut scores = Array4::<f32>::zeros((1, 1, 3, 3));

        attn.apply_causal_mask(&mut scores, 0);

        assert_eq!(scores[[0, 0, 0, 0]], 0.0, "Pos 0 attends to 0");
        assert_eq!(scores[[0, 0, 0, 1]], MASK_VALUE, "Pos 0 CANNOT attend to 1");
        assert_eq!(scores[[0, 0, 0, 2]], MASK_VALUE, "Pos 0 CANNOT attend to 2");

        assert_eq!(scores[[0, 0, 1, 0]], 0.0);
        assert_eq!(scores[[0, 0, 1, 1]], 0.0);
        assert_eq!(scores[[0, 0, 1, 2]], MASK_VALUE);

        assert_eq!(scores[[0, 0, 2, 0]], 0.0);
        assert_eq!(scores[[0, 0, 2, 1]], 0.0);
        assert_eq!(scores[[0, 0, 2, 2]], 0.0);
    }

    #[test]
    fn test_apply_causal_mask_with_cache_offset() {
        let hidden = 4;
        let attn = create_dummy_attn(hidden, 1, 1);

        let mut scores = Array4::<f32>::zeros((1, 1, 2, 4));

        attn.apply_causal_mask(&mut scores, 2);

        assert_eq!(scores[[0, 0, 0, 0]], 0.0); // History
        assert_eq!(scores[[0, 0, 0, 1]], 0.0); // History
        assert_eq!(scores[[0, 0, 0, 2]], 0.0); // Self
        assert_eq!(scores[[0, 0, 0, 3]], MASK_VALUE, "Future");

        assert_eq!(scores[[0, 0, 1, 0]], 0.0);
        assert_eq!(scores[[0, 0, 1, 1]], 0.0);
        assert_eq!(scores[[0, 0, 1, 2]], 0.0);
        assert_eq!(scores[[0, 0, 1, 3]], 0.0);
    }

    fn create_test_layer(out: usize, inp: usize, seed: usize) -> LinearLayer {
        let weights = ndarray::Array2::from_shape_fn((out, inp), |(i, j)| {
            ((i * 17 + j * 13 + seed) % 1000) as f32 * 0.002 - 1.0
        });
        LinearLayer::new_f32(weights, None)
    }

    #[test]
    fn test_decoder_attention_shapes() {
        let hidden_size = 256;
        let num_heads = 8;
        let num_kv_heads = 2;
        let head_dim = hidden_size / num_heads;

        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let q = create_test_layer(q_dim, hidden_size, 1);
        let k = create_test_layer(kv_dim, hidden_size, 2);
        let v = create_test_layer(kv_dim, hidden_size, 3);
        let o = create_test_layer(hidden_size, hidden_size, 4);

        let attn = DecoderAttention::new(hidden_size, num_heads, q, k, v, o, Some(num_kv_heads));

        assert_eq!(attn.num_heads, 8);
        assert_eq!(attn.num_kv_heads, 2);
        assert_eq!(attn.head_dim, 32);
    }

    #[test]
    fn test_decoder_attention_forward() {
        let hidden_size = 128;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = hidden_size / num_heads;
        let kv_dim = num_kv_heads * head_dim;

        let q = create_test_layer(hidden_size, hidden_size, 1);
        let k = create_test_layer(kv_dim, hidden_size, 2);
        let v = create_test_layer(kv_dim, hidden_size, 3);
        let o = create_test_layer(hidden_size, hidden_size, 4);

        let attn = DecoderAttention::new(hidden_size, num_heads, q, k, v, o, Some(num_kv_heads));

        let batch = 1;
        let seq_len = 4;
        let total_len = 8;

        let hidden = Array3::from_shape_fn((batch, seq_len, hidden_size), |(b, s, h)| {
            ((b + s + h) % 100) as f32 * 0.01 - 0.5
        });

        let mut k_cache = Array3::zeros((batch, total_len, kv_dim));
        let mut v_cache = Array3::zeros((batch, total_len, kv_dim));

        let output = attn
            .forward(
                &hidden,
                None,
                k_cache.view_mut(),
                v_cache.view_mut(),
                0,
                None,
            )
            .expect("forward should succeed");

        assert_eq!(output.shape(), &[batch, seq_len, hidden_size]);
    }
}
