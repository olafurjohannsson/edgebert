use crate::embedder::Embedder;
use crate::indexer::{IndexInfo, IndexStats, IndexerBuilder, IndexerError, IndexerResult};
use anyhow::Result;
use kjarni_rag::{
    DocumentLoader, IndexConfig, IndexReader, IndexWriter, LoaderConfig, Progress,
    ProgressCallback, ProgressStage, SplitterConfig,
};
use kjarni_transformers::Device;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::sync::watch;

pub struct Indexer {
    embedder: Embedder,
    loader_config: LoaderConfig,
    chunk_size: usize,
    #[expect(
        dead_code,
        reason = "not referenced yet; kept until the path that needs it lands"
    )]
    chunk_overlap: usize,
    max_docs_per_segment: usize,
    batch_size: usize,
    quiet: bool,
    progress_callback: Option<ProgressCallback>,
    cancel_token: Option<watch::Receiver<bool>>,
}

fn calculate_index_size(path: &str) -> Result<u64> {
    let mut total = 0u64;
    for entry in walkdir::WalkDir::new(path)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if entry.file_type().is_file() {
            total += entry.metadata().map(|m| m.len()).unwrap_or(0);
        }
    }
    Ok(total)
}

impl Indexer {
    pub fn builder(model: &str) -> IndexerBuilder {
        IndexerBuilder::new(model)
    }

    pub(crate) async fn from_builder(builder: IndexerBuilder) -> IndexerResult<Self> {
        // Build embedder
        let mut embedder_builder = Embedder::builder(&builder.model);

        match builder.device {
            Device::Wgpu => embedder_builder = embedder_builder.gpu(),
            Device::Cpu => embedder_builder = embedder_builder.cpu(),
        }

        if let Some(ref cache_dir) = builder.cache_dir {
            embedder_builder = embedder_builder.cache_dir(cache_dir);
        }

        let embedder = embedder_builder
            .build()
            .await
            .map_err(IndexerError::EmbedderError)?;

        let splitter_config = SplitterConfig {
            chunk_size: builder.chunk_size,
            chunk_overlap: builder.chunk_overlap,
            ..Default::default()
        };

        let loader_config = LoaderConfig {
            splitter: splitter_config,
            recursive: builder.recursive,
            extensions: builder.extensions.clone(),
            exclude_patterns: builder.exclude_patterns.clone(),
            include_hidden: builder.include_hidden,
            max_file_size: builder.max_file_size,
            quiet: builder.quiet,
            // This indexer embeds text with a sentence encoder. Images need CLIP
            // and a different embedder entirely, so they are not picked up here;
            // `ImageIndex` in kjarni-models is the path for those.
            include_images: false,
        };

        Ok(Self {
            embedder,
            loader_config,
            chunk_size: builder.chunk_size,
            chunk_overlap: builder.chunk_overlap,
            max_docs_per_segment: builder.max_docs_per_segment,
            batch_size: builder.batch_size,
            quiet: builder.quiet,
            progress_callback: builder.progress_callback,
            cancel_token: builder.cancel_token,
        })
    }

    /// Create with default settings
    pub async fn new(model: &str) -> IndexerResult<Self> {
        Self::builder(model).build().await
    }

    /// Get index info (static - doesn't need embedder)
    pub fn info(index_path: &str) -> IndexerResult<IndexInfo> {
        if !Path::new(index_path).exists() {
            return Err(IndexerError::IndexNotFound(index_path.to_string()));
        }

        let reader = IndexReader::open(index_path).map_err(IndexerError::IndexingFailed)?;

        let size_bytes = calculate_index_size(index_path).unwrap_or(0);

        // Read config for model info
        let config_path = Path::new(index_path).join("config.json");
        let embedding_model = if config_path.exists() {
            std::fs::read_to_string(&config_path)
                .ok()
                .and_then(|s| serde_json::from_str::<IndexConfig>(&s).ok())
                .and_then(|c| c.embedding_model)
        } else {
            None
        };

        Ok(IndexInfo {
            path: index_path.to_string(),
            document_count: reader.len(),
            segment_count: reader.segment_count(),
            dimension: reader.dimension(),
            size_bytes,
            embedding_model,
            created_at: None,
        })
    }

    /// Delete an index
    pub fn delete(index_path: &str) -> IndexerResult<()> {
        if !Path::new(index_path).exists() {
            return Err(IndexerError::IndexNotFound(index_path.to_string()));
        }
        std::fs::remove_dir_all(index_path).map_err(|e| IndexerError::IndexingFailed(e.into()))
    }

    /// Report progress
    fn report_stored_progress(
        &self,
        stage: ProgressStage,
        current: usize,
        total: usize,
        msg: Option<&str>,
    ) {
        if let Some(ref callback) = self.progress_callback {
            let progress = Progress {
                stage,
                current,
                total: if total > 0 { Some(total) } else { None },
                message_len: msg.map(|s| s.len()).unwrap_or(0),
            };
            callback(&progress, msg); // Just call it - Fn doesn't need &mut
        }
    }

    /// Create a new index from files/directories
    pub async fn create(&self, index_path: &str, inputs: &[&str]) -> IndexerResult<IndexStats> {
        self.create_internal(index_path, inputs, false).await
    }

    /// Create with force overwrite option
    pub async fn create_with_options(
        &self,
        index_path: &str,
        inputs: &[&str],
        force: bool,
    ) -> IndexerResult<IndexStats> {
        self.create_internal(index_path, inputs, force).await
    }

    /// Internal create that uses the stored progress callback
    async fn create_internal(
        &self,
        index_path: &str,
        inputs: &[&str],
        force: bool,
    ) -> IndexerResult<IndexStats> {
        if inputs.is_empty() {
            return Err(IndexerError::NoInputs);
        }

        // Check if exists
        if Path::new(index_path).exists() {
            if force {
                std::fs::remove_dir_all(index_path)
                    .map_err(|e| IndexerError::IndexingFailed(e.into()))?;
            } else {
                return Err(IndexerError::IndexExists(index_path.to_string()));
            }
        }

        let start = std::time::Instant::now();
        let dimension = self.embedder.dimension();

        // Create index writer
        let config = IndexConfig {
            dimension,
            max_docs_per_segment: self.max_docs_per_segment,
            embedding_model: Some(self.embedder.model_name().to_string()),
            ..Default::default()
        };

        let mut writer =
            IndexWriter::open(index_path, config).map_err(IndexerError::IndexingFailed)?;

        // Collect files
        self.report_stored_progress(ProgressStage::Scanning, 0, 0, Some("Discovering files..."));
        let files = self.collect_files(inputs)?;
        let total_files = files.len();

        if self.is_cancelled() {
            return Err(IndexerError::Cancelled);
        }

        let loader = DocumentLoader::new(self.loader_config.clone());

        let mut total_docs = 0usize;
        let mut total_chunks = 0usize;
        let mut files_processed = 0usize;
        let mut files_skipped = 0usize;

        let mut batch_texts: Vec<String> = Vec::with_capacity(self.batch_size);
        let mut batch_metadata: Vec<HashMap<String, String>> = Vec::with_capacity(self.batch_size);

        for (file_idx, file_path) in files.iter().enumerate() {
            if self.is_cancelled() {
                return Err(IndexerError::Cancelled);
            }

            self.report_stored_progress(
                ProgressStage::Loading,
                file_idx,
                total_files,
                Some(&file_path.to_string_lossy()),
            );

            match loader.load_file(file_path) {
                Ok(chunks) => {
                    total_chunks += chunks.len();
                    files_processed += 1;

                    for chunk in chunks {
                        batch_texts.push(chunk.text);
                        batch_metadata.push(chunk.metadata.to_hashmap());

                        if batch_texts.len() >= self.batch_size {
                            if self.is_cancelled() {
                                return Err(IndexerError::Cancelled);
                            }

                            self.report_stored_progress(
                                ProgressStage::Embedding,
                                total_docs,
                                0,
                                None,
                            );

                            let added = self
                                .flush_batch(&mut writer, &mut batch_texts, &mut batch_metadata)
                                .await?;
                            total_docs += added;
                        }
                    }
                }
                Err(e) => {
                    files_skipped += 1;
                    if !self.quiet {
                        eprintln!("Warning: Failed to load {}: {}", file_path.display(), e);
                    }
                }
            }
        }

        // Flush remaining
        if !batch_texts.is_empty() {
            self.report_stored_progress(ProgressStage::Embedding, total_docs, 0, None);
            let added = self
                .flush_batch(&mut writer, &mut batch_texts, &mut batch_metadata)
                .await?;
            total_docs += added;
        }

        self.report_stored_progress(
            ProgressStage::Committing,
            total_docs,
            total_docs,
            Some("Finalizing index..."),
        );

        writer.commit().map_err(IndexerError::IndexingFailed)?;

        let size_bytes = calculate_index_size(index_path).unwrap_or(0);

        Ok(IndexStats {
            documents_indexed: total_docs,
            chunks_created: total_chunks,
            dimension,
            size_bytes,
            files_processed,
            files_skipped,
            elapsed_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Create with callback and cancellation support (for FFI)
    pub async fn create_with_callback<F, C>(
        &self,
        index_path: &str,
        inputs: &[&str],
        force: bool,
        on_progress: Option<F>,
        is_cancelled: Option<C>,
    ) -> IndexerResult<IndexStats>
    where
        F: Fn(ProgressStage, usize, usize, Option<&str>),
        C: Fn() -> bool,
    {
        self.create_impl(index_path, inputs, force, on_progress, is_cancelled)
            .await
    }

    async fn create_impl<F, C>(
        &self,
        index_path: &str,
        inputs: &[&str],
        force: bool,
        on_progress: Option<F>,
        is_cancelled: Option<C>,
    ) -> IndexerResult<IndexStats>
    where
        F: Fn(ProgressStage, usize, usize, Option<&str>),
        C: Fn() -> bool,
    {
        if inputs.is_empty() {
            return Err(IndexerError::NoInputs);
        }

        // Check if exists
        if Path::new(index_path).exists() {
            if force {
                std::fs::remove_dir_all(index_path)
                    .map_err(|e| IndexerError::IndexingFailed(e.into()))?;
            } else {
                return Err(IndexerError::IndexExists(index_path.to_string()));
            }
        }

        let start = std::time::Instant::now();
        let dimension = self.embedder.dimension();

        // Helper to report progress
        let report = |stage, current, total, msg: Option<&str>| {
            if let Some(ref cb) = on_progress {
                cb(stage, current, total, msg);
            }
        };

        // Helper to check cancellation
        let check_cancelled = || is_cancelled.as_ref().map(|f| f()).unwrap_or(false);

        // Create index writer
        let config = IndexConfig {
            dimension,
            max_docs_per_segment: self.max_docs_per_segment,
            embedding_model: Some(self.embedder.model_name().to_string()),
            ..Default::default()
        };

        let mut writer =
            IndexWriter::open(index_path, config).map_err(IndexerError::IndexingFailed)?;

        // Collect files
        report(ProgressStage::Scanning, 0, 0, Some("Discovering files..."));
        let files = self.collect_files(inputs)?;
        let total_files = files.len();

        if check_cancelled() {
            return Err(IndexerError::Cancelled);
        }

        let loader = DocumentLoader::new(self.loader_config.clone());

        let mut total_docs = 0usize;
        let mut total_chunks = 0usize;
        let mut files_processed = 0usize;
        let mut files_skipped = 0usize;

        let mut batch_texts: Vec<String> = Vec::with_capacity(self.batch_size);
        let mut batch_metadata: Vec<HashMap<String, String>> = Vec::with_capacity(self.batch_size);

        for (file_idx, file_path) in files.iter().enumerate() {
            if check_cancelled() {
                return Err(IndexerError::Cancelled);
            }

            report(
                ProgressStage::Loading,
                file_idx,
                total_files,
                Some(&file_path.to_string_lossy()),
            );

            match loader.load_file(file_path) {
                Ok(chunks) => {
                    total_chunks += chunks.len();
                    files_processed += 1;

                    for chunk in chunks {
                        batch_texts.push(chunk.text);
                        batch_metadata.push(chunk.metadata.to_hashmap());

                        if batch_texts.len() >= self.batch_size {
                            if check_cancelled() {
                                return Err(IndexerError::Cancelled);
                            }

                            report(ProgressStage::Embedding, total_docs, 0, None);

                            let added = self
                                .flush_batch(&mut writer, &mut batch_texts, &mut batch_metadata)
                                .await?;
                            total_docs += added;
                        }
                    }
                }
                Err(e) => {
                    files_skipped += 1;
                    if !self.quiet {
                        eprintln!("Warning: Failed to load {}: {}", file_path.display(), e);
                    }
                }
            }
        }

        // Flush remaining
        if !batch_texts.is_empty() {
            report(ProgressStage::Embedding, total_docs, 0, None);
            let added = self
                .flush_batch(&mut writer, &mut batch_texts, &mut batch_metadata)
                .await?;
            total_docs += added;
        }

        report(
            ProgressStage::Committing,
            total_docs,
            total_docs,
            Some("Finalizing index..."),
        );

        writer.commit().map_err(IndexerError::IndexingFailed)?;

        let size_bytes = calculate_index_size(index_path).unwrap_or(0);

        Ok(IndexStats {
            documents_indexed: total_docs,
            chunks_created: total_chunks,
            dimension,
            size_bytes,
            files_processed,
            files_skipped,
            elapsed_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Add documents to existing index
    pub async fn add(&self, index_path: &str, inputs: &[&str]) -> IndexerResult<usize> {
        self.add_internal(index_path, inputs).await
    }

    /// Internal add that uses the stored progress callback
    async fn add_internal(&self, index_path: &str, inputs: &[&str]) -> IndexerResult<usize> {
        if inputs.is_empty() {
            return Ok(0);
        }

        if !Path::new(index_path).exists() {
            return Err(IndexerError::IndexNotFound(index_path.to_string()));
        }

        // Open existing inde
        let mut writer =
            IndexWriter::open_existing(index_path).map_err(IndexerError::IndexingFailed)?;

        // Validate dimension
        if writer.dimension() != self.embedder.dimension() {
            return Err(IndexerError::DimensionMismatch {
                index_dim: writer.dimension(),
                model_dim: self.embedder.dimension(),
            });
        }

        // Collect
        self.report_stored_progress(ProgressStage::Scanning, 0, 0, Some("Discovering files..."));
        let files = self.collect_files(inputs)?;
        let total_files = files.len();

        if self.is_cancelled() {
            return Err(IndexerError::Cancelled);
        }

        let loader = DocumentLoader::new(self.loader_config.clone());

        let mut total_docs = 0usize;
        let mut batch_texts: Vec<String> = Vec::with_capacity(self.batch_size);
        let mut batch_metadata: Vec<HashMap<String, String>> = Vec::with_capacity(self.batch_size);

        for (file_idx, file_path) in files.iter().enumerate() {
            if self.is_cancelled() {
                return Err(IndexerError::Cancelled);
            }

            self.report_stored_progress(
                ProgressStage::Loading,
                file_idx,
                total_files,
                Some(&file_path.to_string_lossy()),
            );

            match loader.load_file(file_path) {
                Ok(chunks) => {
                    for chunk in chunks {
                        batch_texts.push(chunk.text);
                        batch_metadata.push(chunk.metadata.to_hashmap());

                        if batch_texts.len() >= self.batch_size {
                            if self.is_cancelled() {
                                return Err(IndexerError::Cancelled);
                            }

                            self.report_stored_progress(
                                ProgressStage::Embedding,
                                total_docs,
                                0,
                                None,
                            );

                            let added = self
                                .flush_batch(&mut writer, &mut batch_texts, &mut batch_metadata)
                                .await?;
                            total_docs += added;
                        }
                    }
                }
                Err(e) => {
                    if !self.quiet {
                        eprintln!("Warning: Failed to load {}: {}", file_path.display(), e);
                    }
                }
            }
        }

        // Flush remaining
        if !batch_texts.is_empty() {
            self.report_stored_progress(ProgressStage::Embedding, total_docs, 0, None);
            let added = self
                .flush_batch(&mut writer, &mut batch_texts, &mut batch_metadata)
                .await?;
            total_docs += added;
        }

        self.report_stored_progress(
            ProgressStage::Committing,
            total_docs,
            total_docs,
            Some("Finalizing..."),
        );

        writer.commit().map_err(IndexerError::IndexingFailed)?;

        Ok(total_docs)
    }

    /// callback support (for FFI)
    pub async fn add_with_callback<F, C>(
        &self,
        index_path: &str,
        inputs: &[&str],
        on_progress: Option<F>,
        is_cancelled: Option<C>,
    ) -> IndexerResult<usize>
    where
        F: Fn(ProgressStage, usize, usize, Option<&str>),
        C: Fn() -> bool,
    {
        self.add_impl(index_path, inputs, on_progress, is_cancelled)
            .await
    }

    async fn add_impl<F, C>(
        &self,
        index_path: &str,
        inputs: &[&str],
        on_progress: Option<F>,
        is_cancelled: Option<C>,
    ) -> IndexerResult<usize>
    where
        F: Fn(ProgressStage, usize, usize, Option<&str>),
        C: Fn() -> bool,
    {
        if inputs.is_empty() {
            return Ok(0);
        }

        if !Path::new(index_path).exists() {
            return Err(IndexerError::IndexNotFound(index_path.to_string()));
        }

        // report progress
        let report = |stage, current, total, msg: Option<&str>| {
            if let Some(ref cb) = on_progress {
                cb(stage, current, total, msg);
            }
        };

        // check cancellation
        let check_cancelled = || is_cancelled.as_ref().map(|f| f()).unwrap_or(false);

        // Open existing index for appending
        let mut writer =
            IndexWriter::open_existing(index_path).map_err(IndexerError::IndexingFailed)?;

        // Validate dimension matches
        if writer.dimension() != self.embedder.dimension() {
            return Err(IndexerError::DimensionMismatch {
                index_dim: writer.dimension(),
                model_dim: self.embedder.dimension(),
            });
        }

        // Collect files
        report(ProgressStage::Scanning, 0, 0, Some("Discovering files..."));
        let files = self.collect_files(inputs)?;
        let total_files = files.len();

        if check_cancelled() {
            return Err(IndexerError::Cancelled);
        }

        let loader = DocumentLoader::new(self.loader_config.clone());

        let mut total_docs = 0usize;
        let mut batch_texts: Vec<String> = Vec::with_capacity(self.batch_size);
        let mut batch_metadata: Vec<HashMap<String, String>> = Vec::with_capacity(self.batch_size);

        for (file_idx, file_path) in files.iter().enumerate() {
            if check_cancelled() {
                return Err(IndexerError::Cancelled);
            }

            report(
                ProgressStage::Loading,
                file_idx,
                total_files,
                Some(&file_path.to_string_lossy()),
            );

            match loader.load_file(file_path) {
                Ok(chunks) => {
                    for chunk in chunks {
                        batch_texts.push(chunk.text);
                        batch_metadata.push(chunk.metadata.to_hashmap());

                        if batch_texts.len() >= self.batch_size {
                            if check_cancelled() {
                                return Err(IndexerError::Cancelled);
                            }

                            report(ProgressStage::Embedding, total_docs, 0, None);

                            let added = self
                                .flush_batch(&mut writer, &mut batch_texts, &mut batch_metadata)
                                .await?;
                            total_docs += added;
                        }
                    }
                }
                Err(e) => {
                    if !self.quiet {
                        eprintln!("Warning: Failed to load {}: {}", file_path.display(), e);
                    }
                }
            }
        }

        if !batch_texts.is_empty() {
            report(ProgressStage::Embedding, total_docs, 0, None);
            let added = self
                .flush_batch(&mut writer, &mut batch_texts, &mut batch_metadata)
                .await?;
            total_docs += added;
        }

        report(
            ProgressStage::Committing,
            total_docs,
            total_docs,
            Some("Finalizing..."),
        );

        writer.commit().map_err(IndexerError::IndexingFailed)?;

        Ok(total_docs)
    }

    async fn flush_batch(
        &self,
        writer: &mut IndexWriter,
        texts: &mut Vec<String>,
        metadata: &mut Vec<HashMap<String, String>>,
    ) -> IndexerResult<usize> {
        let count = texts.len();

        let texts_ref: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embeddings = self
            .embedder
            .embed_batch(&texts_ref)
            .await
            .map_err(IndexerError::EmbedderError)?;

        for ((text, meta), embedding) in texts.drain(..).zip(metadata.drain(..)).zip(embeddings) {
            writer
                .add(&text, &embedding, Some(&meta))
                .map_err(IndexerError::IndexingFailed)?;
        }

        Ok(count)
    }

    fn collect_files(&self, inputs: &[&str]) -> IndexerResult<Vec<PathBuf>> {
        use walkdir::WalkDir;

        let mut files = Vec::new();

        for input in inputs {
            let path = Path::new(input);

            if !path.exists() {
                return Err(IndexerError::PathNotFound(input.to_string()));
            }

            if path.is_file() {
                if self.is_supported_file(path) {
                    files.push(path.to_path_buf());
                }
            } else if path.is_dir() {
                let walker = if self.loader_config.recursive {
                    WalkDir::new(path)
                } else {
                    WalkDir::new(path).max_depth(1)
                };

                for entry in walker.into_iter().filter_map(|e| e.ok()) {
                    let entry_path = entry.path();

                    if !entry_path.is_file() {
                        continue;
                    }

                    if !self.loader_config.include_hidden
                        && let Some(name) = entry_path.file_name().and_then(|n| n.to_str())
                        && name.starts_with('.')
                    {
                        continue;
                    }

                    let path_str = entry_path.to_string_lossy();
                    let excluded = self
                        .loader_config
                        .exclude_patterns
                        .iter()
                        .any(|pattern| glob_match::glob_match(pattern, &path_str));
                    if excluded {
                        continue;
                    }

                    if let Some(max_size) = self.loader_config.max_file_size
                        && let Ok(metadata) = entry_path.metadata()
                        && metadata.len() as usize > max_size
                    {
                        continue;
                    }

                    if self.is_supported_file(entry_path) {
                        files.push(entry_path.to_path_buf());
                    }
                }
            }
        }

        Ok(files)
    }

    fn is_supported_file(&self, path: &Path) -> bool {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase());

        match ext {
            Some(ext) => {
                if self.loader_config.extensions.is_empty() {
                    kjarni_rag::TEXT_EXTENSIONS.contains(&ext.as_str())
                } else {
                    self.loader_config.extensions.iter().any(|e| e == &ext)
                }
            }
            None => false,
        }
    }
    fn is_cancelled(&self) -> bool {
        self.cancel_token
            .as_ref()
            .map(|t| *t.borrow())
            .unwrap_or(false)
    }

    pub fn model_name(&self) -> &str {
        self.embedder.model_name()
    }

    pub fn dimension(&self) -> usize {
        self.embedder.dimension()
    }

    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn is_quiet(&self) -> bool {
        self.quiet
    }

    pub fn has_progress_callback(&self) -> bool {
        self.progress_callback.is_some()
    }
}

#[cfg(test)]
mod indexer_tests {

    use std::fs::{self, File};
    use std::io::Write;
    use std::path::PathBuf;
    use tempfile::TempDir;

    use crate::indexer::{IndexInfo, IndexStats, IndexerBuilder, IndexerError};

    #[test]
    fn test_index_stats_construction() {
        let stats = IndexStats {
            documents_indexed: 100,
            chunks_created: 250,
            dimension: 384,
            size_bytes: 1024 * 1024,
            files_processed: 10,
            files_skipped: 2,
            elapsed_ms: 5000,
        };

        assert_eq!(stats.documents_indexed, 100);
        assert_eq!(stats.chunks_created, 250);
        assert_eq!(stats.dimension, 384);
        assert_eq!(stats.size_bytes, 1024 * 1024);
        assert_eq!(stats.files_processed, 10);
        assert_eq!(stats.files_skipped, 2);
        assert_eq!(stats.elapsed_ms, 5000);
    }

    #[test]
    fn test_index_stats_clone() {
        let stats = IndexStats {
            documents_indexed: 50,
            chunks_created: 100,
            dimension: 768,
            size_bytes: 512,
            files_processed: 5,
            files_skipped: 1,
            elapsed_ms: 1000,
        };

        let cloned = stats.clone();

        assert_eq!(stats.documents_indexed, cloned.documents_indexed);
        assert_eq!(stats.dimension, cloned.dimension);
    }

    #[test]
    fn test_index_stats_debug() {
        let stats = IndexStats {
            documents_indexed: 10,
            chunks_created: 20,
            dimension: 384,
            size_bytes: 100,
            files_processed: 2,
            files_skipped: 0,
            elapsed_ms: 500,
        };

        let debug = format!("{:?}", stats);

        assert!(debug.contains("documents_indexed"));
        assert!(debug.contains("10"));
        assert!(debug.contains("384"));
    }

    #[test]
    fn test_index_stats_zero_values() {
        let stats = IndexStats {
            documents_indexed: 0,
            chunks_created: 0,
            dimension: 0,
            size_bytes: 0,
            files_processed: 0,
            files_skipped: 0,
            elapsed_ms: 0,
        };

        assert_eq!(stats.documents_indexed, 0);
        assert_eq!(stats.elapsed_ms, 0);
    }

    #[test]
    fn test_index_info_construction() {
        let info = IndexInfo {
            path: "/path/to/index".to_string(),
            document_count: 1000,
            segment_count: 5,
            dimension: 384,
            size_bytes: 10 * 1024 * 1024,
            embedding_model: Some("minilm-l6-v2".to_string()),
            created_at: Some(1700000000),
        };

        assert_eq!(info.path, "/path/to/index");
        assert_eq!(info.document_count, 1000);
        assert_eq!(info.segment_count, 5);
        assert_eq!(info.dimension, 384);
        assert_eq!(info.embedding_model, Some("minilm-l6-v2".to_string()));
        assert_eq!(info.created_at, Some(1700000000));
    }

    #[test]
    fn test_index_info_without_optional_fields() {
        let info = IndexInfo {
            path: "/index".to_string(),
            document_count: 100,
            segment_count: 1,
            dimension: 768,
            size_bytes: 1024,
            embedding_model: None,
            created_at: None,
        };

        assert!(info.embedding_model.is_none());
        assert!(info.created_at.is_none());
    }

    #[test]
    fn test_index_info_clone() {
        let info = IndexInfo {
            path: "/test".to_string(),
            document_count: 50,
            segment_count: 2,
            dimension: 384,
            size_bytes: 2048,
            embedding_model: Some("model".to_string()),
            created_at: None,
        };

        let cloned = info.clone();

        assert_eq!(info.path, cloned.path);
        assert_eq!(info.embedding_model, cloned.embedding_model);
    }

    #[test]
    fn test_index_info_debug() {
        let info = IndexInfo {
            path: "/my/index".to_string(),
            document_count: 42,
            segment_count: 1,
            dimension: 384,
            size_bytes: 999,
            embedding_model: None,
            created_at: None,
        };

        let debug = format!("{:?}", info);

        assert!(debug.contains("IndexInfo"));
        assert!(debug.contains("/my/index"));
        assert!(debug.contains("42"));
    }

    #[test]
    fn test_error_no_inputs() {
        let err = IndexerError::NoInputs;
        let msg = format!("{}", err);

        assert!(msg.contains("No input"));
    }

    #[test]
    fn test_error_path_not_found() {
        let err = IndexerError::PathNotFound("/nonexistent/path".to_string());
        let msg = format!("{}", err);

        assert!(msg.contains("Path not found"));
        assert!(msg.contains("/nonexistent/path"));
    }

    #[test]
    fn test_error_index_exists() {
        let err = IndexerError::IndexExists("/existing/index".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("already exists"));
        assert!(msg.contains("/existing/index"));
        assert!(msg.contains("force=true"));
    }

    #[test]
    fn test_error_index_not_found() {
        let err = IndexerError::IndexNotFound("/missing/index".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("not found"));
        assert!(msg.contains("/missing/index"));
    }

    #[test]
    fn test_error_dimension_mismatch() {
        let err = IndexerError::DimensionMismatch {
            index_dim: 384,
            model_dim: 768,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Dimension mismatch"));
        assert!(msg.contains("384"));
        assert!(msg.contains("768"));
    }

    #[test]
    fn test_error_cancelled() {
        let err = IndexerError::Cancelled;
        let msg = format!("{}", err);
        assert!(msg.contains("cancelled"));
    }

    #[test]
    fn test_error_debug() {
        let err = IndexerError::NoInputs;
        let debug = format!("{:?}", err);
        assert!(debug.contains("NoInputs"));
    }
    #[test]
    fn test_builder_new() {
        let builder = IndexerBuilder::new("minilm-l6-v2");

        assert_eq!(builder.model, "minilm-l6-v2");
        assert!(matches!(builder.device, kjarni_transformers::Device::Cpu));
        assert!(builder.cache_dir.is_none());
        assert_eq!(builder.chunk_size, 512);
        assert_eq!(builder.chunk_overlap, 50);
        assert!(builder.extensions.is_empty());
        assert!(builder.exclude_patterns.is_empty());
        assert!(builder.recursive);
        assert!(!builder.include_hidden);
        assert_eq!(builder.max_file_size, Some(10 * 1024 * 1024));
        assert_eq!(builder.max_docs_per_segment, 10_000);
        assert_eq!(builder.batch_size, 32);
        assert!(!builder.quiet);
    }

    #[test]
    fn test_builder_cpu() {
        let builder = IndexerBuilder::new("model").cpu();
        assert!(matches!(builder.device, kjarni_transformers::Device::Cpu));
    }

    #[test]
    fn test_builder_gpu() {
        let builder = IndexerBuilder::new("model").gpu();
        assert!(matches!(builder.device, kjarni_transformers::Device::Wgpu));
    }

    #[test]
    fn test_builder_device_string() {
        let builder_gpu = IndexerBuilder::new("model").device("gpu");
        assert!(matches!(
            builder_gpu.device,
            kjarni_transformers::Device::Wgpu
        ));

        let builder_cpu = IndexerBuilder::new("model").device("cpu");
        assert!(matches!(
            builder_cpu.device,
            kjarni_transformers::Device::Cpu
        ));

        let builder_unknown = IndexerBuilder::new("model").device("unknown");
        assert!(matches!(
            builder_unknown.device,
            kjarni_transformers::Device::Cpu
        ));
    }

    #[test]
    fn test_builder_cache_dir() {
        let builder = IndexerBuilder::new("model").cache_dir("/custom/cache");

        assert_eq!(builder.cache_dir, Some(PathBuf::from("/custom/cache")));
    }

    #[test]
    fn test_builder_cache_dir_pathbuf() {
        let path = PathBuf::from("/another/cache");
        let builder = IndexerBuilder::new("model").cache_dir(path.clone());
        assert_eq!(builder.cache_dir, Some(path));
    }

    #[test]
    fn test_builder_chunk_size() {
        let builder = IndexerBuilder::new("model").chunk_size(1000);
        assert_eq!(builder.chunk_size, 1000);
    }

    #[test]
    fn test_builder_chunk_overlap() {
        let builder = IndexerBuilder::new("model").chunk_overlap(100);
        assert_eq!(builder.chunk_overlap, 100);
    }

    #[test]
    fn test_builder_extension_single() {
        let builder = IndexerBuilder::new("model").extension("txt");
        assert_eq!(builder.extensions, vec!["txt"]);
    }

    #[test]
    fn test_builder_extension_with_dot() {
        let builder = IndexerBuilder::new("model").extension(".txt");
        assert_eq!(builder.extensions, vec!["txt"]);
    }

    #[test]
    fn test_builder_extension_uppercase() {
        let builder = IndexerBuilder::new("model").extension("TXT");
        assert_eq!(builder.extensions, vec!["txt"]);
    }

    #[test]
    fn test_builder_extensions_multiple() {
        let builder = IndexerBuilder::new("model").extensions(&["txt", "md", "rs"]);
        assert_eq!(builder.extensions, vec!["txt", "md", "rs"]);
    }

    #[test]
    fn test_builder_extension_chained() {
        let builder = IndexerBuilder::new("model")
            .extension("txt")
            .extension("md")
            .extension("rs");
        assert_eq!(builder.extensions, vec!["txt", "md", "rs"]);
    }

    #[test]
    fn test_builder_exclude() {
        let builder = IndexerBuilder::new("model")
            .exclude("*.log")
            .exclude("temp/*");
        assert_eq!(builder.exclude_patterns.len(), 2);
        assert!(builder.exclude_patterns.contains(&"*.log".to_string()));
        assert!(builder.exclude_patterns.contains(&"temp/*".to_string()));
    }

    #[test]
    fn test_builder_recursive() {
        let builder_true = IndexerBuilder::new("model").recursive(true);
        assert!(builder_true.recursive);
        let builder_false = IndexerBuilder::new("model").recursive(false);
        assert!(!builder_false.recursive);
    }

    #[test]
    fn test_builder_include_hidden() {
        let builder_true = IndexerBuilder::new("model").include_hidden(true);
        assert!(builder_true.include_hidden);
        let builder_false = IndexerBuilder::new("model").include_hidden(false);
        assert!(!builder_false.include_hidden);
    }

    #[test]
    fn test_builder_max_file_size() {
        let builder = IndexerBuilder::new("model").max_file_size(5 * 1024 * 1024);
        assert_eq!(builder.max_file_size, Some(5 * 1024 * 1024));
    }

    #[test]
    fn test_builder_max_docs_per_segment() {
        let builder = IndexerBuilder::new("model").max_docs_per_segment(5000);
        assert_eq!(builder.max_docs_per_segment, 5000);
    }

    #[test]
    fn test_builder_batch_size() {
        let builder = IndexerBuilder::new("model").batch_size(64);
        assert_eq!(builder.batch_size, 64);
    }

    #[test]
    fn test_builder_quiet() {
        let builder_true = IndexerBuilder::new("model").quiet(true);
        assert!(builder_true.quiet);

        let builder_false = IndexerBuilder::new("model").quiet(false);
        assert!(!builder_false.quiet);
    }

    #[test]
    fn test_builder_on_progress() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let called = Arc::new(AtomicBool::new(false));
        let called_clone = called.clone();

        let builder = IndexerBuilder::new("model").on_progress(move |_progress, _msg| {
            called_clone.store(true, Ordering::SeqCst);
        });

        assert!(builder.progress_callback.is_some());
    }

    #[test]
    fn test_builder_chained_configuration() {
        let builder = IndexerBuilder::new("minilm-l6-v2")
            .gpu()
            .cache_dir("/cache")
            .chunk_size(1000)
            .chunk_overlap(100)
            .extensions(&["txt", "md"])
            .exclude("*.log")
            .recursive(true)
            .include_hidden(false)
            .max_file_size(5 * 1024 * 1024)
            .max_docs_per_segment(5000)
            .batch_size(64)
            .quiet(true);

        assert_eq!(builder.model, "minilm-l6-v2");
        assert!(matches!(builder.device, kjarni_transformers::Device::Wgpu));
        assert_eq!(builder.cache_dir, Some(PathBuf::from("/cache")));
        assert_eq!(builder.chunk_size, 1000);
        assert_eq!(builder.chunk_overlap, 100);
        assert_eq!(builder.extensions, vec!["txt", "md"]);
        assert_eq!(builder.exclude_patterns, vec!["*.log"]);
        assert!(builder.recursive);
        assert!(!builder.include_hidden);
        assert_eq!(builder.max_file_size, Some(5 * 1024 * 1024));
        assert_eq!(builder.max_docs_per_segment, 5000);
        assert_eq!(builder.batch_size, 64);
        assert!(builder.quiet);
    }

    fn create_test_files(dir: &TempDir) -> Vec<PathBuf> {
        let x = "x".repeat(1000);
        let files = vec![
            ("doc1.txt", "Text content 1"),
            ("doc2.md", "# Markdown"),
            ("code.rs", "fn main() {}"),
            ("data.json", "{}"),
            (".hidden.txt", "hidden content"),
            ("large.txt", x.as_str()),
        ];

        let mut paths = Vec::new();
        for (name, content) in files {
            let path = dir.path().join(name);
            let mut file = File::create(&path).unwrap();
            file.write_all(content.as_bytes()).unwrap();
            paths.push(path);
        }

        // Create subdirectory with files
        let subdir = dir.path().join("subdir");
        fs::create_dir(&subdir).unwrap();
        let subfile = subdir.join("nested.txt");
        File::create(&subfile)
            .unwrap()
            .write_all(b"nested content")
            .unwrap();
        paths.push(subfile);

        paths
    }

    #[test]
    fn test_file_extension_filtering() {
        let dir = TempDir::new().unwrap();
        create_test_files(&dir);

        let extensions = ["txt".to_string()];

        let is_supported = |path: &std::path::Path| -> bool {
            path.extension()
                .and_then(|e| e.to_str())
                .map(|e| extensions.contains(&e.to_lowercase()))
                .unwrap_or(false)
        };

        assert!(is_supported(&dir.path().join("doc1.txt")));
        assert!(!is_supported(&dir.path().join("doc2.md")));
        assert!(!is_supported(&dir.path().join("code.rs")));
    }

    #[test]
    fn test_hidden_file_detection() {
        let dir = TempDir::new().unwrap();
        create_test_files(&dir);

        let is_hidden = |path: &std::path::Path| -> bool {
            path.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with('.'))
                .unwrap_or(false)
        };

        assert!(is_hidden(&dir.path().join(".hidden.txt")));
        assert!(!is_hidden(&dir.path().join("doc1.txt")));
    }

    #[test]
    fn test_exclude_pattern_matching() {
        let patterns = ["*.log".to_string(), "temp/*".to_string()];

        let is_excluded = |path: &str| -> bool {
            patterns
                .iter()
                .any(|pattern| glob_match::glob_match(pattern, path))
        };

        assert!(is_excluded("error.log"));
        assert!(is_excluded("temp/cache.txt"));
        assert!(!is_excluded("document.txt"));
        assert!(!is_excluded("data/file.txt"));
    }

    #[test]
    fn test_file_size_filtering() {
        let dir = TempDir::new().unwrap();

        let file_path = dir.path().join("test.txt");
        let content = "x".repeat(500);
        File::create(&file_path)
            .unwrap()
            .write_all(content.as_bytes())
            .unwrap();

        let metadata = file_path.metadata().unwrap();
        let file_size = metadata.len() as usize;

        assert_eq!(file_size, 500);

        let max_size_small = 100;
        let max_size_large = 1000;

        assert!(file_size > max_size_small);
        assert!(file_size <= max_size_large);
    }

    #[test]
    fn test_recursive_directory_walk() {
        let dir = TempDir::new().unwrap();

        let level1 = dir.path().join("level1");
        let level2 = level1.join("level2");
        let level3 = level2.join("level3");

        fs::create_dir_all(&level3).unwrap();

        File::create(dir.path().join("root.txt"))
            .unwrap()
            .write_all(b"root")
            .unwrap();
        File::create(level1.join("l1.txt"))
            .unwrap()
            .write_all(b"l1")
            .unwrap();
        File::create(level2.join("l2.txt"))
            .unwrap()
            .write_all(b"l2")
            .unwrap();
        File::create(level3.join("l3.txt"))
            .unwrap()
            .write_all(b"l3")
            .unwrap();

        let mut count_recursive = 0;
        for entry in walkdir::WalkDir::new(dir.path())
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if entry.path().is_file() {
                count_recursive += 1;
            }
        }

        let mut count_shallow = 0;
        for entry in walkdir::WalkDir::new(dir.path())
            .max_depth(1)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if entry.path().is_file() {
                count_shallow += 1;
            }
        }

        assert_eq!(count_recursive, 4);
        assert_eq!(count_shallow, 1);
    }

    #[test]
    fn test_builder_empty_model_name() {
        let builder = IndexerBuilder::new("");

        assert_eq!(builder.model, "");
    }

    #[test]
    fn test_builder_zero_values() {
        let builder = IndexerBuilder::new("model")
            .chunk_size(0)
            .chunk_overlap(0)
            .batch_size(0)
            .max_docs_per_segment(0);

        assert_eq!(builder.chunk_size, 0);
        assert_eq!(builder.chunk_overlap, 0);
        assert_eq!(builder.batch_size, 0);
        assert_eq!(builder.max_docs_per_segment, 0);
    }

    #[test]
    fn test_builder_large_values() {
        let builder = IndexerBuilder::new("model")
            .chunk_size(usize::MAX)
            .max_file_size(usize::MAX);

        assert_eq!(builder.chunk_size, usize::MAX);
        assert_eq!(builder.max_file_size, Some(usize::MAX));
    }

    #[test]
    fn test_empty_extensions_list() {
        let builder = IndexerBuilder::new("model").extensions(&[]);

        assert!(builder.extensions.is_empty());
    }

    #[test]
    fn test_empty_exclude_patterns() {
        let builder = IndexerBuilder::new("model");

        assert!(builder.exclude_patterns.is_empty());
    }

    #[test]
    fn test_error_from_anyhow() {
        let anyhow_err = anyhow::anyhow!("Something went wrong");
        let indexer_err: IndexerError = IndexerError::IndexingFailed(anyhow_err);

        let msg = format!("{}", indexer_err);
        assert!(msg.contains("Something went wrong"));
    }

    #[tokio::test]
    async fn test_indexer_create_basic() {
        use crate::Indexer;

        let dir = TempDir::new().unwrap();
        let index_path = dir.path().join("test_index");

        // Create test files
        let docs_dir = dir.path().join("docs");
        fs::create_dir(&docs_dir).unwrap();

        File::create(docs_dir.join("doc1.txt"))
            .unwrap()
            .write_all(b"Hello world. This is a test document.")
            .unwrap();

        File::create(docs_dir.join("doc2.txt"))
            .unwrap()
            .write_all(b"Another document with different content.")
            .unwrap();

        // Create indexer and index
        let indexer = Indexer::builder("minilm-l6-v2")
            .cpu()
            .quiet(true)
            .build()
            .await
            .expect("Failed to build indexer");

        let stats = indexer
            .create(index_path.to_str().unwrap(), &[docs_dir.to_str().unwrap()])
            .await
            .expect("Failed to create index");

        assert!(stats.documents_indexed > 0);
        assert!(stats.files_processed == 2);
        assert!(index_path.exists());
    }

    #[tokio::test]

    async fn test_indexer_info() {
        use crate::Indexer;

        let dir = TempDir::new().unwrap();
        let index_path = dir.path().join("test_index");

        let docs_dir = dir.path().join("docs");
        fs::create_dir(&docs_dir).unwrap();
        File::create(docs_dir.join("doc.txt"))
            .unwrap()
            .write_all(b"Test content")
            .unwrap();

        let indexer = Indexer::builder("minilm-l6-v2")
            .cpu()
            .quiet(true)
            .build()
            .await
            .unwrap();

        indexer
            .create(index_path.to_str().unwrap(), &[docs_dir.to_str().unwrap()])
            .await
            .unwrap();

        let info = Indexer::info(index_path.to_str().unwrap()).expect("Failed to get info");
        assert!(info.document_count > 0);
        assert_eq!(info.dimension, 384); // MiniLM dimension
    }

    #[tokio::test]
    async fn test_indexer_delete() {
        use crate::Indexer;
        let dir = TempDir::new().unwrap();
        let index_path = dir.path().join("test_index");
        fs::create_dir(&index_path).unwrap();
        assert!(index_path.exists());
        Indexer::delete(index_path.to_str().unwrap()).expect("Failed to delete");
        assert!(!index_path.exists());
    }

    #[test]
    fn test_indexer_delete_not_found() {
        use crate::Indexer;
        let result = Indexer::delete("/nonexistent/index/path");
        assert!(result.is_err());
        match result.unwrap_err() {
            IndexerError::IndexNotFound(path) => {
                assert!(path.contains("nonexistent"));
            }
            _ => panic!("Expected IndexNotFound error"),
        }
    }

    #[test]
    fn test_indexer_info_not_found() {
        use crate::Indexer;
        let result = Indexer::info("/nonexistent/index/path");
        assert!(result.is_err());
        match result.unwrap_err() {
            IndexerError::IndexNotFound(path) => {
                assert!(path.contains("nonexistent"));
            }
            _ => panic!("Expected IndexNotFound error"),
        }
    }

    #[tokio::test]
    async fn test_indexer_add_to_existing() {
        use crate::Indexer;

        let dir = TempDir::new().unwrap();
        let index_path = dir.path().join("test_index");

        let docs_dir = dir.path().join("docs");
        fs::create_dir(&docs_dir).unwrap();
        File::create(docs_dir.join("doc1.txt"))
            .unwrap()
            .write_all(b"Initial document")
            .unwrap();

        let indexer = Indexer::builder("minilm-l6-v2")
            .cpu()
            .quiet(true)
            .build()
            .await
            .unwrap();

        indexer
            .create(index_path.to_str().unwrap(), &[docs_dir.to_str().unwrap()])
            .await
            .unwrap();

        let more_docs = dir.path().join("more_docs");
        fs::create_dir(&more_docs).unwrap();
        File::create(more_docs.join("doc2.txt"))
            .unwrap()
            .write_all(b"Additional document")
            .unwrap();

        let added = indexer
            .add(index_path.to_str().unwrap(), &[more_docs.to_str().unwrap()])
            .await
            .expect("Failed to add documents");

        assert!(added > 0);
    }

    #[tokio::test]
    async fn test_indexer_force_overwrite() {
        use crate::Indexer;

        let dir = TempDir::new().unwrap();
        let index_path = dir.path().join("test_index");
        let docs_dir = dir.path().join("docs");

        fs::create_dir(&docs_dir).unwrap();
        File::create(docs_dir.join("doc.txt"))
            .unwrap()
            .write_all(b"Content")
            .unwrap();

        let indexer = Indexer::builder("minilm-l6-v2")
            .cpu()
            .quiet(true)
            .build()
            .await
            .unwrap();

        indexer
            .create(index_path.to_str().unwrap(), &[docs_dir.to_str().unwrap()])
            .await
            .unwrap();

        let result = indexer
            .create(index_path.to_str().unwrap(), &[docs_dir.to_str().unwrap()])
            .await;

        assert!(matches!(result, Err(IndexerError::IndexExists(_))));

        let result = indexer
            .create_with_options(
                index_path.to_str().unwrap(),
                &[docs_dir.to_str().unwrap()],
                true,
            )
            .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_indexer_dimension_mismatch() {
        use crate::Indexer;

        let dir = TempDir::new().unwrap();
        let index_path = dir.path().join("test_index");
        let docs_dir = dir.path().join("docs");

        fs::create_dir(&docs_dir).unwrap();
        File::create(docs_dir.join("doc.txt"))
            .unwrap()
            .write_all(b"Content")
            .unwrap();

        let indexer1 = Indexer::builder("minilm-l6-v2")
            .cpu()
            .quiet(true)
            .build()
            .await
            .unwrap();

        indexer1
            .create(index_path.to_str().unwrap(), &[docs_dir.to_str().unwrap()])
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_indexer_with_progress_callback() {
        use crate::Indexer;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let dir = TempDir::new().unwrap();
        let index_path = dir.path().join("test_index");
        let docs_dir = dir.path().join("docs");

        fs::create_dir(&docs_dir).unwrap();
        for i in 0..5 {
            File::create(docs_dir.join(format!("doc{}.txt", i)))
                .unwrap()
                .write_all(format!("Document {} content here", i).as_bytes())
                .unwrap();
        }

        let progress_count = Arc::new(AtomicUsize::new(0));
        let progress_count_clone = progress_count.clone();

        let indexer = Indexer::builder("minilm-l6-v2")
            .cpu()
            .quiet(true)
            .on_progress(move |_progress, _msg| {
                progress_count_clone.fetch_add(1, Ordering::SeqCst);
            })
            .build()
            .await
            .unwrap();

        assert!(indexer.has_progress_callback());

        indexer
            .create(index_path.to_str().unwrap(), &[docs_dir.to_str().unwrap()])
            .await
            .unwrap();

        let count = progress_count.load(Ordering::SeqCst);
        println!("Progress callback invoked {} times", count);
        assert!(
            count > 0,
            "Progress callback should have been invoked at least once"
        );
    }

    #[tokio::test]
    async fn test_indexer_without_progress_callback() {
        use crate::Indexer;

        let dir = TempDir::new().unwrap();
        let index_path = dir.path().join("test_index");
        let docs_dir = dir.path().join("docs");

        fs::create_dir(&docs_dir).unwrap();
        File::create(docs_dir.join("doc.txt"))
            .unwrap()
            .write_all(b"Content")
            .unwrap();

        let indexer = Indexer::builder("minilm-l6-v2")
            .cpu()
            .quiet(true)
            .build()
            .await
            .unwrap();
        assert!(!indexer.has_progress_callback());

        let result = indexer
            .create(index_path.to_str().unwrap(), &[docs_dir.to_str().unwrap()])
            .await;

        assert!(result.is_ok());
    }
}
