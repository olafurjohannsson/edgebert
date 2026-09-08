//! Decoding JPEG and PNG to the raw RGB the vision tower expects.
//!
//! Behind the `image-io` feature, and deliberately so. An FFI or WASM host
//! usually already holds decoded pixels, and making every caller compile a JPEG
//! decoder to reach `embed_rgb8` would be a poor trade. What this buys is the
//! path from a directory of photos to an index without the caller writing a
//! decoder themselves.
//!
//! `zune-jpeg` and `png` rather than `image`: both are pure Rust with no C
//! dependency, so the single-binary promise survives.

use std::path::Path;

use anyhow::{Context, Result, bail};

/// Extensions this module can decode, lowercase and without the dot.
pub const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png"];

/// True if the path looks like something [`decode`] can read.
pub fn is_supported(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .is_some_and(|e| IMAGE_EXTENSIONS.contains(&e.as_str()))
}

/// An image decoded to 8-bit RGB, row major, three bytes per pixel.
pub struct RgbImage {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
}

/// Prints the dimensions and the buffer length, never the pixels: a `Debug` that
/// dumped a megabyte into an error message would be worse than none.
impl std::fmt::Debug for RgbImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RgbImage")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("bytes", &self.pixels.len())
            .finish()
    }
}

impl RgbImage {
    fn check(self) -> Result<Self> {
        let expected = self.width * self.height * 3;
        if self.pixels.len() != expected {
            bail!(
                "decoder produced {} bytes for a {}x{} image, expected {expected}",
                self.pixels.len(),
                self.width,
                self.height
            );
        }
        if self.width == 0 || self.height == 0 {
            bail!("image has a zero dimension: {}x{}", self.width, self.height);
        }
        Ok(self)
    }
}

/// Decodes by content, not by extension.
///
/// A `.jpg` that is really a PNG is common enough in photo libraries that
/// trusting the extension would fail on real data. The magic bytes decide.
pub fn decode(bytes: &[u8]) -> Result<RgbImage> {
    if bytes.starts_with(&[0x89, b'P', b'N', b'G']) {
        decode_png(bytes)
    } else if bytes.starts_with(&[0xFF, 0xD8]) {
        decode_jpeg(bytes)
    } else {
        bail!("unrecognised image format: not PNG or JPEG")
    }
}

pub fn decode_file(path: &Path) -> Result<RgbImage> {
    let bytes = std::fs::read(path).with_context(|| format!("reading image {}", path.display()))?;
    decode(&bytes).with_context(|| format!("decoding {}", path.display()))
}

fn decode_jpeg(bytes: &[u8]) -> Result<RgbImage> {
    use zune_jpeg::JpegDecoder;
    use zune_jpeg::zune_core::colorspace::ColorSpace;
    use zune_jpeg::zune_core::options::DecoderOptions;

    // Forcing RGB out means greyscale and CMYK sources are converted for us
    // rather than silently producing the wrong channel count.
    let options = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
    let mut decoder = JpegDecoder::new_with_options(bytes, options);
    let pixels = decoder.decode().context("jpeg decode failed")?;
    let info = decoder.info().context("jpeg has no frame header")?;

    RgbImage {
        width: info.width as usize,
        height: info.height as usize,
        pixels,
    }
    .check()
}

fn decode_png(bytes: &[u8]) -> Result<RgbImage> {
    let decoder = png::Decoder::new(bytes);
    let mut reader = decoder.read_info().context("png header")?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).context("png decode failed")?;
    buf.truncate(info.buffer_size());

    let (w, h) = (info.width as usize, info.height as usize);

    // 16-bit PNGs come back big-endian; taking the high byte of each sample is
    // the same truncation every 8-bit pipeline applies.
    let step = match info.bit_depth {
        png::BitDepth::Sixteen => 2,
        _ => 1,
    };
    let sample = |raw: &[u8], i: usize| raw[i * step];

    let channels = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Indexed => {
            bail!("indexed PNGs need a palette expansion this decoder does not do")
        }
    };

    let mut pixels = Vec::with_capacity(w * h * 3);
    for i in 0..w * h {
        let base = i * channels;
        match channels {
            3 | 4 => {
                pixels.push(sample(&buf, base));
                pixels.push(sample(&buf, base + 1));
                pixels.push(sample(&buf, base + 2));
            }
            // Grey to RGB by replication, which is what every image library does
            // and what CLIP's own preprocessing assumes.
            1 | 2 => {
                let g = sample(&buf, base);
                pixels.extend_from_slice(&[g, g, g]);
            }
            _ => unreachable!("channel count is constrained above"),
        }
    }

    RgbImage {
        width: w,
        height: h,
        pixels,
    }
    .check()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encodes a real PNG in memory, so these tests need no fixture files.
    fn png_bytes(w: u32, h: u32, color: png::ColorType, data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut out, w, h);
            enc.set_color(color);
            enc.set_depth(png::BitDepth::Eight);
            let mut writer = enc.write_header().expect("png header");
            writer.write_image_data(data).expect("png data");
        }
        out
    }

    #[test]
    fn extensions_are_matched_case_insensitively() {
        for name in ["a.jpg", "a.JPG", "a.jpeg", "a.PNG", "a.Png"] {
            assert!(is_supported(Path::new(name)), "{name} should be supported");
        }
        for name in ["a.gif", "a.webp", "a.txt", "a.heic", "noextension"] {
            assert!(!is_supported(Path::new(name)), "{name} should not be");
        }
    }

    #[test]
    fn rgb_png_round_trips_exactly() {
        let pixels: Vec<u8> = (0..4 * 3 * 3).map(|i| (i * 7 % 256) as u8).collect();
        let img = decode(&png_bytes(4, 3, png::ColorType::Rgb, &pixels)).unwrap();
        assert_eq!((img.width, img.height), (4, 3));
        assert_eq!(img.pixels, pixels, "PNG is lossless");
    }

    /// Alpha is dropped rather than composited. CLIP has no alpha channel, and
    /// silently multiplying by it would darken transparent regions.
    #[test]
    fn rgba_loses_its_alpha_and_keeps_the_colours() {
        let mut rgba = Vec::new();
        for i in 0..6 {
            rgba.extend_from_slice(&[10 + i, 20 + i, 30 + i, 7]);
        }
        let img = decode(&png_bytes(3, 2, png::ColorType::Rgba, &rgba)).unwrap();
        assert_eq!(img.pixels.len(), 3 * 2 * 3);
        assert_eq!(&img.pixels[0..3], &[10, 20, 30]);
        assert_eq!(&img.pixels[3..6], &[11, 21, 31]);
    }

    /// A greyscale photo must not become a one-channel tensor the tower cannot
    /// read. Replication is what every reference pipeline does.
    #[test]
    fn grayscale_expands_by_replication() {
        let grey: Vec<u8> = vec![0, 64, 128, 255];
        let img = decode(&png_bytes(4, 1, png::ColorType::Grayscale, &grey)).unwrap();
        assert_eq!(img.pixels.len(), 4 * 3);
        for (i, g) in grey.iter().enumerate() {
            assert_eq!(&img.pixels[i * 3..i * 3 + 3], &[*g, *g, *g]);
        }
    }

    /// Format comes from the magic bytes. Photo libraries are full of files whose
    /// extension lies, and trusting it would fail on real data.
    #[test]
    fn format_is_decided_by_content() {
        let png = png_bytes(2, 2, png::ColorType::Rgb, &[0u8; 12]);
        assert!(png.starts_with(&[0x89, b'P', b'N', b'G']));
        // Decoded despite nothing saying "png" but the bytes themselves.
        assert!(decode(&png).is_ok());
    }

    #[test]
    fn unrecognised_bytes_are_an_error_not_a_panic() {
        for junk in [
            &b"not an image at all"[..],
            &[][..],
            &[0xFF][..],
            &[0x89, b'P'][..],
        ] {
            let err = decode(junk).unwrap_err().to_string();
            assert!(
                err.contains("unrecognised") || err.contains("png") || err.contains("jpeg"),
                "unhelpful error for {junk:?}: {err}"
            );
        }
    }

    #[test]
    fn a_truncated_png_fails_cleanly() {
        let mut png = png_bytes(8, 8, png::ColorType::Rgb, &[128u8; 8 * 8 * 3]);
        png.truncate(png.len() / 2);
        assert!(decode(&png).is_err(), "half a PNG must not decode");
    }

    #[test]
    fn a_missing_file_names_itself_in_the_error() {
        let err = decode_file(Path::new("/nonexistent/photo.png"))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("photo.png"),
            "error should name the file: {err}"
        );
    }
}
