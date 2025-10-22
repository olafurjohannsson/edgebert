use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    pub content: ChunkContent,
    pub metadata: ChunkMetadata,
    #[serde(default)]
    pub enrichment: Option<ChunkEnrichment>,
}

impl Chunk {
    pub fn as_text(&self) -> String {
        match &self.content {
            ChunkContent::Text { text } => text.text.clone(),
            ChunkContent::Table { table } => table.text.clone(),
            ChunkContent::Image { image } => {
                let mut combined_text = Vec::new();
                if let Some(caption) = &image.caption {
                    combined_text.push(caption.clone());
                }
                if let Some(ocr) = &image.ocr_text {
                    combined_text.push(ocr.clone());
                }
                combined_text.join("\n\n")
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ChunkContent {
    Text { text: TextChunk },
    Table { table: TableChunk },
    Image { image: ImageChunk },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextChunk {
    pub id: String,
    pub html: Option<String>,
    pub markdown: Option<String>,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableChunk {
    pub id: String,
    pub html: Option<String>,
    pub markdown: Option<String>,
    pub text: String,
    pub bounding_box: Option<[f32; 4]>,
    pub headers: Option<Vec<String>>,
    pub rows: Option<Vec<Vec<String>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageChunk {
    pub id: String,
    pub bounding_box: [f32; 4],
    pub image_path: Option<String>,
    pub caption: Option<String>,
    pub ocr_text: Option<String>,
}

/// Searchable metadata for each chunk.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChunkMetadata {
    pub source_file: Option<String>,
    pub document_title: Option<String>,
    pub document_type: Option<String>,
    pub document_subtype: Option<String>,
    pub document_status: Option<String>,
    
    pub date: Option<String>,
    pub year: Option<u16>,
    pub valid_from: Option<String>,
    pub valid_to: Option<String>,
    
    pub page_number: u32,
    pub source_page_image_id: Option<String>,
    
    #[serde(default)]
    pub section_hierarchy: Vec<String>,
    pub preceding_chunk_id: Option<String>,
    pub following_chunk_id: Option<String>,
    
    #[serde(default)]
    pub relations: Vec<Relation>,
    
    #[serde(default)]
    pub custom_metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    pub target_chunk_id: String,
    pub relation_type: RelationType,
    pub label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RelationType {
    Mentions,
    Summarizes,
    ExpandsOn,
    References,
    Contradicts,
}


#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChunkEnrichment {
}


/// This is the structure that your `search` function will return.
/// It now contains the rich `Chunk` type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub score: f32,
    pub chunk: Chunk,
    pub search_type: SearchType,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SearchType {
    Keyword,
    Semantic,
    Hybrid,
}