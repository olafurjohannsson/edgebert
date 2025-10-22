//! # EdgeGPT
//!
//! A fast, dependency-free, and cross-platform toolkit for running modern AI models
//! on the edge and in the browser.
//! Works cross-platform and on native and IoT.


#[cfg(feature = "bert-cpu")]
pub mod bert {
    pub use edgebert::{BertModel, BertCrossEncoder, ModelType, cosine_similarity};
}

#[cfg(feature = "search")]
pub mod search {
    pub use edgesearch::{Index, SearchResult};
}

#[cfg(feature = "wgpu")]
pub mod gpu {
    // pub use edgetransformers::wgpu_context::WgpuContext;
}