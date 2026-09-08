//! CLIP: images and text embedded into one space.
//!
//! Both towers are here. The vision tower embeds a photo, the text tower embeds a
//! caption, and because they were trained together the two vectors are comparable
//! by cosine, which is what makes searching a photo library by description work.

pub mod config;
#[cfg(feature = "image-io")]
pub mod image_io;
#[cfg(feature = "image-io")]
pub mod index;
pub mod model;
mod nfc_table;
pub mod preprocess;
pub mod text;
pub mod tokenizer;

pub use config::{ClipConfig, ClipVisionConfig};
pub use model::ClipVisionModel;
pub use preprocess::PreprocessorConfig;
pub use text::ClipTextModel;
pub use tokenizer::ClipTokenizer;

#[cfg(feature = "image-io")]
pub use image_io::{
    IMAGE_EXTENSIONS, RgbImage, decode as decode_image, decode_file as decode_image_file,
};
#[cfg(feature = "image-io")]
pub use index::{ImageHit, ImageIndex, IndexedImage, ScanReport};
