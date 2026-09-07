//! kjarni-rag: RAG (Retrieval Augmented Generation) utilities

mod config;
mod index_reader;
mod index_writer;
mod loader;
mod progress;
mod search_index;
mod segment;
mod splitter;

pub use config::IndexConfig;
pub use index_reader::{IndexReader, MetadataFilter};
pub use index_writer::IndexWriter;
pub use kjarni_search::{Chunk, ChunkMetadata, SearchMode, SearchResult};
pub use loader::{
    DocumentLoader, IMAGE_EXTENSIONS, KIND_IMAGE, KIND_KEY, LoaderConfig, TEXT_EXTENSIONS,
};
pub use progress::{CancelToken, Progress, ProgressCallback, ProgressReporter, ProgressStage};
pub use search_index::SearchIndex;
pub use segment::{Segment, SegmentBuilder, SegmentMeta};
pub use splitter::{SplitterConfig, TextSplitter};

// Re-export from kjarni-search
pub use kjarni_search::bm25::Bm25Index;

#[cfg(test)]
mod tests;
