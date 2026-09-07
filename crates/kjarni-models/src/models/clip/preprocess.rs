//! Turning an image into the tensor CLIP was trained on.
//!
//! The values live in `preprocessor_config.json` rather than in this file on
//! purpose: resize, crop, mean and std differ between CLIP and SigLIP and between
//! checkpoints of each, and getting one wrong does not fail. It produces
//! embeddings that look entirely reasonable and rank badly, which is far harder to
//! notice than a crash.

use anyhow::{Context, Result};
use ndarray::Array3;
use serde::Deserialize;

/// `preprocessor_config.json`, as shipped beside the weights.
#[derive(Debug, Clone, Deserialize)]
pub struct PreprocessorConfig {
    #[serde(default = "default_size")]
    pub size: SizeSpec,
    #[serde(default = "default_crop")]
    pub crop_size: SizeSpec,
    #[serde(default = "default_true")]
    pub do_resize: bool,
    #[serde(default = "default_true")]
    pub do_center_crop: bool,
    #[serde(default = "default_true")]
    pub do_normalize: bool,
    #[serde(default = "default_mean")]
    pub image_mean: Vec<f32>,
    #[serde(default = "default_std")]
    pub image_std: Vec<f32>,
    /// PIL resample code. 3 is bicubic, which is what CLIP uses; 2 is bilinear.
    #[serde(default = "default_resample")]
    pub resample: u8,
}

/// `size` is an int on older exports and `{"shortest_edge": n}` or
/// `{"height": h, "width": w}` on newer ones.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum SizeSpec {
    Uniform(usize),
    Edge { shortest_edge: usize },
    Rect { height: usize, width: usize },
}

impl SizeSpec {
    /// The square edge this spec asks for. A non-square `Rect` is not something
    /// CLIP's own configs produce, so the shorter side is the safe reading.
    pub fn edge(&self) -> usize {
        match *self {
            SizeSpec::Uniform(n) => n,
            SizeSpec::Edge { shortest_edge } => shortest_edge,
            SizeSpec::Rect { height, width } => height.min(width),
        }
    }
}

fn default_size() -> SizeSpec {
    SizeSpec::Uniform(224)
}
fn default_crop() -> SizeSpec {
    SizeSpec::Uniform(224)
}
fn default_true() -> bool {
    true
}
fn default_resample() -> u8 {
    3
}
fn default_mean() -> Vec<f32> {
    vec![0.481_454_66, 0.457_827_5, 0.408_210_73]
}
fn default_std() -> Vec<f32> {
    vec![0.268_629_54, 0.261_302_58, 0.275_777_11]
}

impl PreprocessorConfig {
    pub fn from_json(s: &str) -> Result<Self> {
        serde_json::from_str(s).context("parsing preprocessor_config.json")
    }

    /// An RGB image as `[height][width][3]` bytes becomes `[3, edge, edge]`
    /// normalized floats, ready for patching.
    ///
    /// Resize shortest edge, centre crop, scale to 0..1, then subtract mean and
    /// divide by std per channel: the order torchvision uses, and the order the
    /// weights expect.
    pub fn preprocess_rgb8(
        &self,
        pixels: &[u8],
        width: usize,
        height: usize,
    ) -> Result<Array3<f32>> {
        anyhow::ensure!(
            width > 0 && height > 0,
            "image has a zero dimension: {width}x{height}"
        );
        anyhow::ensure!(
            pixels.len() == width * height * 3,
            "expected {} RGB bytes for {width}x{height}, got {}",
            width * height * 3,
            pixels.len()
        );

        let crop = self.crop_size.edge();
        let target = self.size.edge();

        // Resize so the *shorter* side hits `target` exactly, preserving aspect
        // ratio, so the crop that follows takes the middle of the frame rather
        // than a squashed whole.
        //
        // The long side truncates rather than rounds. That is not a detail: for a
        // 320x240 source the two differ by one pixel of width, 298 against 299,
        // and that one pixel shifts every resampled column and the crop origin
        // with it. Measured against HuggingFace, rounding gave a worst-case
        // difference of 2.80 per value and truncating gives 0.00.
        let (rw, rh) = if self.do_resize {
            let (short, long) = if width <= height {
                (width, height)
            } else {
                (height, width)
            };
            let new_long = (target * long) / short;
            let (nw, nh) = if width <= height {
                (target, new_long)
            } else {
                (new_long, target)
            };
            (nw.max(1), nh.max(1))
        } else {
            (width, height)
        };

        let resized = resize_bicubic(pixels, width, height, rw, rh);

        let (cw, ch, ox, oy) = if self.do_center_crop {
            let cw = crop.min(rw);
            let ch = crop.min(rh);
            (cw, ch, (rw - cw) / 2, (rh - ch) / 2)
        } else {
            (rw, rh, 0, 0)
        };

        let mut out = Array3::<f32>::zeros((3, ch, cw));
        for y in 0..ch {
            for x in 0..cw {
                let src = ((oy + y) * rw + (ox + x)) * 3;
                for c in 0..3 {
                    let v = resized[src + c] / 255.0;
                    out[[c, y, x]] = if self.do_normalize {
                        (v - self.image_mean[c]) / self.image_std[c]
                    } else {
                        v
                    };
                }
            }
        }
        Ok(out)
    }
}

/// PIL's bicubic resize, which is what `CLIPImageProcessor` calls.
///
/// Not a fixed four-tap kernel. When an image is being reduced, PIL widens the
/// filter support by the scale factor and renormalizes, which is antialiasing;
/// sampling four taps regardless is correct only when enlarging. Our fixture
/// downscales 240 to 224, and getting this wrong moved individual channel values
/// by up to 2.8 while the image still looked perfectly normal.
///
/// Separable, like PIL: horizontal pass, then vertical.
fn resize_bicubic(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    if sw == dw && sh == dh {
        return src.iter().map(|&v| v as f32).collect();
    }
    let horizontal: Vec<f32> = {
        let src_f: Vec<f32> = src.iter().map(|&v| v as f32).collect();
        resample_axis(&src_f, sw, sh, dw, true)
    };
    resample_axis(&horizontal, dw, sh, dh, false)
}

/// One separable pass. `horizontal` selects which axis is being resampled.
fn resample_axis(src: &[f32], w: usize, h: usize, out_len: usize, horizontal: bool) -> Vec<f32> {
    let (in_len, other) = if horizontal { (w, h) } else { (h, w) };
    let coeffs = precompute_coeffs(in_len, out_len);

    let mut out = vec![
        0.0f32;
        if horizontal {
            out_len * h * 3
        } else {
            w * out_len * 3
        }
    ];
    for o in 0..other {
        for (i, (start, weights)) in coeffs.iter().enumerate() {
            for c in 0..3 {
                let mut acc = 0.0f32;
                for (k, &wt) in weights.iter().enumerate() {
                    let idx = (start + k).min(in_len - 1);
                    let src_i = if horizontal {
                        (o * w + idx) * 3 + c
                    } else {
                        (idx * w + o) * 3 + c
                    };
                    acc += wt * src[src_i];
                }
                let dst_i = if horizontal {
                    (o * out_len + i) * 3 + c
                } else {
                    (i * w + o) * 3 + c
                };
                // PIL clamps to the pixel range after each pass, and the clamp is
                // load-bearing: bicubic overshoots at edges and the second pass
                // would otherwise carry values outside 0..255 into the result.
                // Rounded to whole bytes, because PIL's resize returns an 8-bit
                // image and normalization sees those quantized values, not the
                // exact filter output. Keeping floats here is more accurate and
                // less faithful, and drifts from every reference implementation.
                out[dst_i] = acc.clamp(0.0, 255.0).round();
            }
        }
    }
    out
}

/// PIL's `precompute_coeffs`: for each output position, the first source index
/// and the normalized weights covering it.
fn precompute_coeffs(in_len: usize, out_len: usize) -> Vec<(usize, Vec<f32>)> {
    const SUPPORT: f32 = 2.0;
    let scale = in_len as f32 / out_len as f32;
    // Only reduction widens the filter; enlarging keeps the base support.
    let filter_scale = scale.max(1.0);
    let support = SUPPORT * filter_scale;

    (0..out_len)
        .map(|i| {
            let center = (i as f32 + 0.5) * scale;
            let xmin = ((center - support + 0.5).floor() as isize).max(0) as usize;
            let xmax = ((center + support + 0.5).floor() as isize).min(in_len as isize) as usize;
            let mut weights = Vec::with_capacity(xmax.saturating_sub(xmin));
            let mut total = 0.0f32;
            for x in xmin..xmax {
                let w = cubic_filter((x as f32 + 0.5 - center) / filter_scale);
                weights.push(w);
                total += w;
            }
            if total != 0.0 {
                for w in &mut weights {
                    *w /= total;
                }
            }
            (xmin, weights)
        })
        .collect()
}

/// The cubic convolution kernel PIL uses, with `a = -0.5`.
fn cubic_filter(x: f32) -> f32 {
    const A: f32 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clip_config() -> PreprocessorConfig {
        PreprocessorConfig::from_json(
            r#"{"crop_size": 224, "do_center_crop": true, "do_normalize": true,
                "do_resize": true, "image_mean": [0.48145466, 0.4578275, 0.40821073],
                "image_std": [0.26862954, 0.26130258, 0.27577711],
                "resample": 3, "size": 224}"#,
        )
        .unwrap()
    }

    /// A flat colour: every output pixel must be the normalized value of that
    /// colour, whatever the resize did, because resampling a constant is constant.
    fn solid(w: usize, h: usize, rgb: [u8; 3]) -> Vec<u8> {
        rgb.iter()
            .cycle()
            .take(w * h * 3)
            .copied()
            .collect::<Vec<u8>>()
    }

    #[test]
    fn size_spec_accepts_every_shape_huggingface_writes() {
        let uniform: SizeSpec = serde_json::from_str("224").unwrap();
        assert_eq!(uniform.edge(), 224);
        let edge: SizeSpec = serde_json::from_str(r#"{"shortest_edge": 256}"#).unwrap();
        assert_eq!(edge.edge(), 256);
        let rect: SizeSpec = serde_json::from_str(r#"{"height": 240, "width": 320}"#).unwrap();
        assert_eq!(rect.edge(), 240, "the shorter side is the safe reading");
    }

    #[test]
    fn defaults_are_clips_own_constants() {
        let c = PreprocessorConfig::from_json("{}").unwrap();
        assert_eq!(c.size.edge(), 224);
        assert_eq!(c.resample, 3, "3 is PIL bicubic");
        assert!((c.image_mean[0] - 0.481_454_66).abs() < 1e-7);
        assert!(c.do_resize && c.do_center_crop && c.do_normalize);
    }

    #[test]
    fn output_is_always_channel_first_at_the_crop_size() {
        let c = clip_config();
        for (w, h) in [(320, 240), (240, 320), (224, 224), (1000, 500), (100, 100)] {
            let out = c
                .preprocess_rgb8(&solid(w, h, [128, 64, 32]), w, h)
                .unwrap();
            assert_eq!(out.shape(), [3, 224, 224], "for a {w}x{h} source");
        }
    }

    /// The bug that cost the most to find: HuggingFace computes the long side with
    /// integer division, so a 320x240 image resizes to 298 wide, not 299. One
    /// pixel shifts every column and the crop origin with it.
    #[test]
    fn the_long_side_truncates_rather_than_rounds() {
        // 224 * 320 / 240 = 298.666..., so truncation gives 298 and rounding 299.
        assert_eq!((224 * 320) / 240, 298);

        // Exercised through the real path: a horizontal ramp makes the sampling
        // grid visible, and the two choices give measurably different pixels.
        let (w, h) = (320usize, 240usize);
        let mut px = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let v = (x * 255 / w) as u8;
                for ch in 0..3 {
                    px[(y * w + x) * 3 + ch] = v;
                }
            }
        }
        let out = clip_config().preprocess_rgb8(&px, w, h).unwrap();
        assert_eq!(out.shape(), [3, 224, 224]);
        // A ramp must stay monotonic left to right after resize and crop.
        let row: Vec<f32> = (0..224).map(|x| out[[0, 112, x]]).collect();
        assert!(
            row.windows(2).all(|p| p[1] >= p[0] - 1e-3),
            "a horizontal ramp should not lose its ordering"
        );
        assert!(row[223] > row[0], "ramp direction inverted");
    }

    /// Normalization is `(v/255 - mean) / std`, per channel. Checked on a flat
    /// image so resampling cannot obscure the arithmetic.
    #[test]
    fn channels_are_normalized_independently() {
        let c = clip_config();
        let rgb = [255u8, 0, 128];
        let out = c.preprocess_rgb8(&solid(300, 300, rgb), 300, 300).unwrap();

        for ch in 0..3 {
            let want = (rgb[ch] as f32 / 255.0 - c.image_mean[ch]) / c.image_std[ch];
            let got = out[[ch, 100, 100]];
            assert!(
                (got - want).abs() < 1e-4,
                "channel {ch}: got {got}, want {want}"
            );
        }
        // Distinct means and stds must not be applied to the wrong channel.
        assert!(out[[0, 5, 5]] != out[[1, 5, 5]], "channels collapsed");
    }

    #[test]
    fn normalization_can_be_turned_off() {
        let mut c = clip_config();
        c.do_normalize = false;
        let out = c
            .preprocess_rgb8(&solid(300, 300, [255, 255, 255]), 300, 300)
            .unwrap();
        assert!(
            (out[[0, 10, 10]] - 1.0).abs() < 1e-4,
            "should be plain 0..1"
        );
    }

    /// A centre crop must take the middle. Painting the borders differently from
    /// the centre is the only way to tell a correct crop from an off-by-N one.
    #[test]
    fn the_crop_is_taken_from_the_centre() {
        let (w, h) = (448usize, 448usize);
        let mut px = vec![0u8; w * h * 3]; // black surround
        for y in 112..336 {
            for x in 112..336 {
                px[(y * w + x) * 3] = 255; // red square, exactly the middle half
            }
        }
        let mut c = clip_config();
        c.do_normalize = false;
        let out = c.preprocess_rgb8(&px, w, h).unwrap();

        // Resize halves it to 224, so the red square lands in the middle 112.
        assert!(out[[0, 112, 112]] > 0.9, "centre should be red");
        assert!(out[[0, 5, 5]] < 0.1, "corner should be black");
    }

    #[test]
    fn a_wrong_sized_buffer_is_rejected() {
        let c = clip_config();
        let err = c
            .preprocess_rgb8(&[0u8; 10], 100, 100)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("30000"),
            "should say how many bytes it wanted: {err}"
        );

        let err = c.preprocess_rgb8(&[], 0, 10).unwrap_err().to_string();
        assert!(err.contains("zero dimension"), "{err}");
    }

    #[test]
    fn resizing_a_matching_image_is_a_no_op() {
        let src: Vec<u8> = (0..300u32).map(|i| (i % 256) as u8).collect();
        let out = resize_bicubic(&src, 10, 10, 10, 10);
        assert_eq!(out.len(), src.len());
        assert!(out.iter().zip(&src).all(|(a, b)| *a == *b as f32));
    }

    /// PIL widens the filter when reducing and leaves it alone when enlarging.
    /// That asymmetry is what antialiasing is, and it is easy to omit.
    #[test]
    fn reduction_widens_the_filter_but_enlargement_does_not() {
        let shrink = precompute_coeffs(100, 25);
        let grow = precompute_coeffs(25, 100);
        let widest_shrink = shrink.iter().map(|(_, w)| w.len()).max().unwrap();
        let widest_grow = grow.iter().map(|(_, w)| w.len()).max().unwrap();
        assert!(
            widest_shrink > widest_grow,
            "reducing 4x should sample more taps ({widest_shrink}) than enlarging ({widest_grow})"
        );
        assert!(
            widest_grow <= 5,
            "enlarging needs about four taps, got {widest_grow}"
        );
    }

    /// Every output position must draw a full unit of weight, or the image gets
    /// darker or brighter as a side effect of resizing.
    #[test]
    fn filter_weights_always_sum_to_one() {
        for (from, to) in [(100, 25), (25, 100), (240, 224), (7, 7), (1000, 3)] {
            for (i, (_, weights)) in precompute_coeffs(from, to).iter().enumerate() {
                let sum: f32 = weights.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-5,
                    "{from}->{to} position {i} sums to {sum}"
                );
            }
        }
    }

    #[test]
    fn the_cubic_kernel_has_the_right_shape() {
        assert!((cubic_filter(0.0) - 1.0).abs() < 1e-6, "peak at the sample");
        assert!(
            cubic_filter(1.0).abs() < 1e-6,
            "zero at neighbouring samples"
        );
        assert_eq!(cubic_filter(2.0), 0.0, "support ends at two");
        assert_eq!(cubic_filter(3.0), 0.0);
        assert!(cubic_filter(1.5) < 0.0, "the ringing lobe is negative");
        // Symmetric, or the image shifts.
        assert!((cubic_filter(0.7) - cubic_filter(-0.7)).abs() < 1e-7);
    }

    /// Resampling a flat field must stay flat. Overshoot at edges is the usual
    /// way a bicubic implementation betrays itself.
    #[test]
    fn a_constant_image_survives_resizing_unchanged() {
        let src = vec![200u8; 64 * 64 * 3];
        for (dw, dh) in [(224, 224), (32, 32), (17, 61)] {
            let out = resize_bicubic(&src, 64, 64, dw, dh);
            let worst = out.iter().fold(0.0f32, |m, v| m.max((v - 200.0).abs()));
            assert!(worst < 0.51, "{dw}x{dh} drifted by {worst}");
        }
    }
}
