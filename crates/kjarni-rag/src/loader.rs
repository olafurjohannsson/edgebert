//! Document loading and chunking utilities

use crate::{Chunk, ChunkMetadata, SplitterConfig, TextSplitter};
use anyhow::Result;
use std::fs;
use std::path::Path;

/// Supported file extensions for text loading
pub const TEXT_EXTENSIONS: &[&str] = &[
    // Documents
    "txt", "md", "markdown", "rst", "org", // Data
    "json", "yaml", "yml", "toml", "xml", "csv", // Web
    "html", "htm", "css", // Code
    "rs", "py", "js", "ts", "go", "java", "c", "cpp", "h", "hpp", "cs", "rb", "sh", "bash", "zsh",
    "fish", "ps1", "sql", "r", "scala", "kt", "swift", "m", "mm", "lua", "pl", "php", "ex", "exs",
    "clj", "hs",
];

/// Image extensions the loader will index when `include_images` is set.
///
/// Deliberately narrow. These are what a photo library is actually made of, and
/// every one is something the decoder behind the `image-io` feature can read; a
/// longer list would only produce chunks that fail later.
pub const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png"];

/// Metadata key marking a chunk whose content is a file rather than its text.
///
/// The indexer reads this to decide which embedder to use: text chunks go through
/// the sentence encoder, image chunks through CLIP.
pub const KIND_KEY: &str = "kind";
pub const KIND_IMAGE: &str = "image";

#[derive(Debug, Clone)]
pub struct LoaderConfig {
    pub splitter: SplitterConfig,
    pub recursive: bool,
    pub extensions: Vec<String>,
    pub exclude_patterns: Vec<String>,
    pub include_hidden: bool,
    pub max_file_size: Option<usize>,
    pub quiet: bool,
    /// Walk images too, emitting one chunk per file instead of splitting text.
    pub include_images: bool,
}

impl LoaderConfig {
    /// Add extension to include
    pub fn with_extension(mut self, ext: &str) -> Self {
        self.extensions.push(ext.to_lowercase());
        self
    }

    /// Add multiple extensions
    pub fn with_extensions(mut self, exts: &[&str]) -> Self {
        for ext in exts {
            self.extensions.push(ext.to_lowercase());
        }
        self
    }

    /// Index images as well as text.
    ///
    /// An image chunk holds no extracted text: its vector comes from the pixels.
    /// What it carries instead is the filename, which is genuinely useful, since
    /// "beach-2019.jpg" is a real signal a keyword search can match.
    pub fn with_images(mut self) -> Self {
        self.include_images = true;
        self
    }

    /// Exclude files matching pattern (e.g., "*.min.js", "node_modules/**")
    pub fn exclude(mut self, pattern: &str) -> Self {
        self.exclude_patterns.push(pattern.to_string());
        self
    }
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self {
            splitter: SplitterConfig::default(),
            recursive: true,
            extensions: vec![],
            exclude_patterns: vec![],
            include_hidden: false,
            max_file_size: None,
            include_images: false,
            quiet: false,
        }
    }
}

/// Load and chunk documents from files/directories
pub struct DocumentLoader {
    config: LoaderConfig,
    splitter: TextSplitter,
}

impl DocumentLoader {
    pub fn new(config: LoaderConfig) -> Self {
        let splitter = TextSplitter::new(config.splitter.clone());
        Self { config, splitter }
    }

    pub fn with_defaults() -> Self {
        Self::new(LoaderConfig::default())
    }

    /// True if this path is an image the loader was asked to include.
    pub fn is_image(&self, path: &Path) -> bool {
        self.config.include_images
            && path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.to_lowercase())
                .is_some_and(|e| IMAGE_EXTENSIONS.contains(&e.as_str()))
    }

    /// One chunk for one image.
    ///
    /// Never split: an image is a single thing to retrieve, and there is no text
    /// to chunk. The file is not opened here, so a directory scan stays cheap and
    /// decoding failures surface at embedding time against the specific file.
    fn load_image(&self, path: &Path) -> Result<Vec<Chunk>> {
        let metadata = ChunkMetadata {
            source: Some(path.display().to_string()),
            chunk_index: Some(0),
            total_chunks: Some(1),
            custom: std::collections::HashMap::from([(
                KIND_KEY.to_string(),
                KIND_IMAGE.to_string(),
            )]),
            ..Default::default()
        };
        Ok(vec![
            Chunk::new(filename_as_text(path)).with_metadata(metadata),
        ])
    }

    /// Load chunks from a single file
    pub fn load_file(&self, path: &Path) -> Result<Vec<Chunk>> {
        if self.is_image(path) {
            return self.load_image(path);
        }
        let content = fs::read_to_string(path)?;
        if !self.config.quiet {
            eprintln!(
                "Splitting file: {} (Size: {} bytes)",
                path.display(),
                content.len()
            );
        }
        let texts = self.splitter.split(&content);
        if !self.config.quiet {
            eprintln!("  -> Generated {} chunks", texts.len());
        }
        let total = texts.len();

        let chunks: Vec<Chunk> = texts
            .into_iter()
            .enumerate()
            .map(|(i, text)| {
                let metadata = ChunkMetadata {
                    source: Some(path.display().to_string()),
                    chunk_index: Some(i),
                    total_chunks: Some(total),
                    ..Default::default()
                };
                Chunk::new(text).with_metadata(metadata)
            })
            .collect();

        Ok(chunks)
    }

    /// Load chunks from a directory
    pub fn load_directory(&self, dir: &Path) -> Result<Vec<Chunk>> {
        let mut all_chunks = Vec::new();

        let walker = if self.config.recursive {
            walkdir::WalkDir::new(dir)
        } else {
            walkdir::WalkDir::new(dir).max_depth(1)
        };

        for entry in walker.into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();

            // Skip directories
            if !path.is_file() {
                continue;
            }

            // Skip hidden files
            if !self.config.include_hidden
                && let Some(name) = path.file_name().and_then(|n| n.to_str())
                && name.starts_with('.')
            {
                continue;
            }

            // Check extension
            if !self.is_supported_extension(path) {
                continue;
            }

            // Load file
            match self.load_file(path) {
                Ok(chunks) => all_chunks.extend(chunks),
                Err(e) => {
                    log::warn!("Failed to load {}: {}", path.display(), e);
                }
            }
        }

        Ok(all_chunks)
    }

    /// Load chunks from multiple paths (files or directories)
    pub fn load_paths(&self, paths: &[&Path]) -> Result<Vec<Chunk>> {
        let mut all_chunks = Vec::new();

        for path in paths {
            if path.is_dir() {
                all_chunks.extend(self.load_directory(path)?);
            } else if path.is_file() {
                all_chunks.extend(self.load_file(path)?);
            } else {
                log::warn!("Path not found: {}", path.display());
            }
        }

        Ok(all_chunks)
    }

    /// Check if a file has a supported extension
    pub fn is_supported_extension(&self, path: &Path) -> bool {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase());

        match ext {
            Some(ext) => {
                let image = self.config.include_images && IMAGE_EXTENSIONS.contains(&ext.as_str());
                if self.config.extensions.is_empty() {
                    image || TEXT_EXTENSIONS.contains(&ext.as_str())
                } else {
                    image || self.config.extensions.iter().any(|e| e == &ext)
                }
            }
            None => false,
        }
    }
}

/// A filename turned into something a keyword search can match.
///
/// `IMG_2024-08-14_beach_sunset.jpg` becomes `IMG 2024 08 14 beach sunset`. This
/// is not a caption and is not pretending to be one: the image's meaning comes
/// from its vector. It is there because filenames often carry the only words a
/// photo has, and dropping them would throw away free signal.
fn filename_as_text(path: &Path) -> String {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    let words: Vec<&str> = stem
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    if words.is_empty() {
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_string()
    } else {
        words.join(" ")
    }
}

#[expect(
    dead_code,
    reason = "not referenced yet; kept until the path that needs it lands"
)]
/// Convenience function to load and chunk from paths
pub fn load_documents(paths: &[&str], config: Option<LoaderConfig>) -> Result<Vec<Chunk>> {
    let loader = DocumentLoader::new(config.unwrap_or_default());
    let path_refs: Vec<&Path> = paths.iter().map(|p| Path::new(*p)).collect();
    loader.load_paths(&path_refs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_loader_default() {
        let loader = DocumentLoader::with_defaults();
        assert!(loader.config.recursive);
    }

    #[test]
    fn test_is_supported_extension() {
        let loader = DocumentLoader::with_defaults();

        assert!(loader.is_supported_extension(Path::new("file.txt")));
        assert!(loader.is_supported_extension(Path::new("file.md")));
        assert!(loader.is_supported_extension(Path::new("file.rs")));
        assert!(loader.is_supported_extension(Path::new("file.py")));

        assert!(!loader.is_supported_extension(Path::new("file.pdf")));
        assert!(!loader.is_supported_extension(Path::new("file.docx")));
        assert!(!loader.is_supported_extension(Path::new("file.exe")));
        assert!(!loader.is_supported_extension(Path::new("file")));
    }

    #[test]
    fn test_load_file() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("test.txt");

        let mut file = fs::File::create(&file_path).unwrap();
        writeln!(file, "First paragraph.\n\nSecond paragraph.").unwrap();

        let loader = DocumentLoader::with_defaults();
        let chunks = loader.load_file(&file_path).unwrap();

        assert!(!chunks.is_empty());
        assert_eq!(
            chunks[0].metadata.source,
            Some(file_path.display().to_string())
        );
    }

    #[test]
    fn test_load_directory() {
        let dir = TempDir::new().unwrap();

        // Create test files
        fs::write(dir.path().join("a.txt"), "Content A").unwrap();
        fs::write(dir.path().join("b.md"), "Content B").unwrap();
        fs::write(dir.path().join("c.pdf"), "PDF content").unwrap(); // Should be skipped

        let loader = DocumentLoader::with_defaults();
        let chunks = loader.load_directory(dir.path()).unwrap();

        // Should have chunks from a.txt and b.md, not c.pdf
        let sources: Vec<_> = chunks
            .iter()
            .filter_map(|c| c.metadata.source.as_ref())
            .collect();

        assert!(sources.iter().any(|s| s.contains("a.txt")));
        assert!(sources.iter().any(|s| s.contains("b.md")));
        assert!(!sources.iter().any(|s| s.contains("c.pdf")));
    }

    #[test]
    fn test_skip_hidden_files() {
        let dir = TempDir::new().unwrap();

        fs::write(dir.path().join("visible.txt"), "Visible").unwrap();
        fs::write(dir.path().join(".hidden.txt"), "Hidden").unwrap();

        let loader = DocumentLoader::with_defaults();
        let chunks = loader.load_directory(dir.path()).unwrap();

        let sources: Vec<_> = chunks
            .iter()
            .filter_map(|c| c.metadata.source.as_ref())
            .collect();

        assert!(sources.iter().any(|s| s.contains("visible.txt")));
        assert!(!sources.iter().any(|s| s.contains(".hidden.txt")));
    }

    #[test]
    fn test_custom_extensions() {
        let dir = TempDir::new().unwrap();

        fs::write(dir.path().join("a.txt"), "Text").unwrap();
        fs::write(dir.path().join("b.custom"), "Custom").unwrap();

        let config = LoaderConfig {
            extensions: vec!["custom".to_string()],
            ..Default::default()
        };

        let loader = DocumentLoader::new(config);
        let chunks = loader.load_directory(dir.path()).unwrap();

        let sources: Vec<_> = chunks
            .iter()
            .filter_map(|c| c.metadata.source.as_ref())
            .collect();

        assert!(!sources.iter().any(|s| s.contains("a.txt")));
        assert!(sources.iter().any(|s| s.contains("b.custom")));
    }
}

#[cfg(test)]
mod image_loading_tests {
    use super::*;
    use tempfile::TempDir;

    fn touch(dir: &TempDir, name: &str, body: &[u8]) -> std::path::PathBuf {
        let p = dir.path().join(name);
        std::fs::write(&p, body).expect("write fixture");
        p
    }

    #[test]
    fn images_are_ignored_unless_asked_for() {
        let dir = TempDir::new().unwrap();
        touch(&dir, "photo.jpg", b"\xff\xd8not-really-a-jpeg");
        touch(&dir, "notes.txt", b"hello world");

        let chunks = DocumentLoader::with_defaults()
            .load_directory(dir.path())
            .expect("load");
        assert_eq!(chunks.len(), 1, "default config must skip images");
        assert!(chunks[0].text.contains("hello"));
    }

    #[test]
    fn an_image_becomes_exactly_one_chunk() {
        let dir = TempDir::new().unwrap();
        // Body is never read: the loader records the path and leaves decoding to
        // the embedder, so a scan does not pay to open every file.
        touch(&dir, "beach.png", b"not a real png");

        let loader = DocumentLoader::new(LoaderConfig::default().with_images());
        let chunks = loader.load_directory(dir.path()).expect("load");

        assert_eq!(chunks.len(), 1);
        let c = &chunks[0];
        assert_eq!(
            c.metadata.custom.get(KIND_KEY).map(String::as_str),
            Some(KIND_IMAGE)
        );
        assert!(c.metadata.source.as_ref().unwrap().ends_with("beach.png"));
        assert_eq!(c.metadata.total_chunks, Some(1), "images are never split");
    }

    #[test]
    fn filenames_become_searchable_words() {
        assert_eq!(
            filename_as_text(Path::new("/p/IMG_2024-08-14_beach_sunset.jpg")),
            "IMG 2024 08 14 beach sunset"
        );
        assert_eq!(filename_as_text(Path::new("/p/holiday.png")), "holiday");
        // A name with nothing word-like still has to yield something indexable.
        assert_eq!(filename_as_text(Path::new("/p/___.png")), "___.png");
    }

    #[test]
    fn text_and_images_can_be_indexed_together() {
        let dir = TempDir::new().unwrap();
        touch(&dir, "a.png", b"x");
        touch(&dir, "b.jpg", b"x");
        touch(&dir, "readme.md", b"# hello");
        touch(&dir, "ignored.bin", b"x");

        let loader = DocumentLoader::new(LoaderConfig::default().with_images());
        let chunks = loader.load_directory(dir.path()).expect("load");

        let images = chunks
            .iter()
            .filter(|c| c.metadata.custom.get(KIND_KEY).map(String::as_str) == Some(KIND_IMAGE))
            .count();
        assert_eq!(images, 2, "both images indexed");
        assert_eq!(chunks.len() - images, 1, "markdown indexed, .bin skipped");
    }
}
