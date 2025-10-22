//! Hybrid RAG with BM25 and semantic search

use anyhow::Result;
use edgebert::{BertModel, ModelType as BertModelType};
use edgegpt::{GPTModel, ModelType as GPTModelType, GenerationConfig};
use std::collections::HashMap;

// Import from your edgerag crate
use edgerag::{Bm25Index, Chunk, ChunkContent, TextChunk, ChunkMetadata};

struct HybridRAG {
    bert_model: BertModel,
    gpt_model: GPTModel,
    bm25_index: Bm25Index,
    chunks: Vec<Chunk>,
    embeddings: Vec<Vec<f32>>,
}

impl HybridRAG {
    fn new(chunks: Vec<Chunk>) -> Result<Self> {
        println!("Loading models...");
        let bert_model = BertModel::from_pretrained(BertModelType::MiniLML6V2BiEncoder)?;
        let gpt_model = GPTModel::from_pretrained(GPTModelType::DistilGPT2)?;
        
        // Build BM25 index
        println!("Building BM25 index...");
        let mut bm25_index = Bm25Index::new();
        let texts: Vec<String> = chunks.iter().map(|c| c.as_text()).collect();
        
        // bm25_index.build(&texts);        TODO: add build method in Bm25Index

        // Generate embeddings
        println!("Generating embeddings...");
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embeddings = bert_model.encode(text_refs, true)?;
        
        Ok(Self {
            bert_model,
            gpt_model,
            bm25_index,
            chunks,
            embeddings,
        })
    }
    
    fn hybrid_search(&self, query: &str, limit: usize) -> Vec<(usize, f32)> {
        // Semantic search
        let query_emb = self.bert_model.encode(vec![query], true).unwrap();
        let query_emb = &query_emb[0];
        
        let mut semantic_scores = Vec::new();
        for (idx, doc_emb) in self.embeddings.iter().enumerate() {
            let sim = edgebert::cosine_similarity(query_emb, doc_emb);
            semantic_scores.push((idx, sim));
        }
        semantic_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // BM25 search
        let bm25_scores = self.bm25_index.search(query, limit * 2);
        
        // Combine scores using Reciprocal Rank Fusion
        let mut combined_scores: HashMap<usize, f32> = HashMap::new();
        let k = 60.0;
        
        for (rank, (idx, _)) in semantic_scores.iter().enumerate().take(limit * 2) {
            let score = 1.0 / (k + (rank + 1) as f32);
            combined_scores.entry(*idx)
                .and_modify(|s| *s += score)
                .or_insert(score);
        }
        
        for (rank, (idx, _)) in bm25_scores.iter().enumerate() {
            let score = 1.0 / (k + (rank + 1) as f32);
            combined_scores.entry(*idx)
                .and_modify(|s| *s += score)
                .or_insert(score);
        }
        
        let mut results: Vec<(usize, f32)> = combined_scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(limit);
        results
    }
    
    fn generate_answer(&self, query: &str, chunks: &[&Chunk]) -> Result<String> {
        let context = chunks.iter()
            .map(|c| c.as_text())
            .collect::<Vec<_>>()
            .join("\n");
        
        let prompt = format!(
            "Based on the following information:\n{}\n\nQuestion: {}\nAnswer:",
            context, query
        );
        
        let config = GenerationConfig {
            max_new_tokens: 100,
            temperature: 0.7,
            repetition_penalty: 1.1,
            ..Default::default()
        };
        
        self.gpt_model.generate(&prompt, &config)
    }
}

fn main() -> Result<()> {
    // Create sample chunks
    let chunks = vec![
        Chunk {
            id: "1".to_string(),
            content: ChunkContent::Text {
                text: TextChunk {
                    id: "1".to_string(),
                    text: "Rust is a systems programming language focused on safety and performance.".to_string(),
                    html: None,
                    markdown: None,
                },
            },
            metadata: ChunkMetadata {
                page_number: 1,
                document_title: Some("Rust Guide".to_string()),
                ..Default::default()
            },
            enrichment: None,
        },
        Chunk {
            id: "2".to_string(),
            content: ChunkContent::Text {
                text: TextChunk {
                    id: "2".to_string(),
                    text: "Memory safety in Rust is achieved through ownership and borrowing rules.".to_string(),
                    html: None,
                    markdown: None,
                },
            },
            metadata: ChunkMetadata {
                page_number: 2,
                document_title: Some("Rust Guide".to_string()),
                ..Default::default()
            },
            enrichment: None,
        },
        Chunk {
            id: "3".to_string(),
            content: ChunkContent::Text {
                text: TextChunk {
                    id: "3".to_string(),
                    text: "Python is known for its simple syntax and extensive library ecosystem.".to_string(),
                    html: None,
                    markdown: None,
                },
            },
            metadata: ChunkMetadata {
                page_number: 1,
                document_title: Some("Python Intro".to_string()),
                ..Default::default()
            },
            enrichment: None,
        },
    ];
    
    let rag = HybridRAG::new(chunks.clone())?;
    
    let query = "How does Rust achieve memory safety?";
    println!("Query: {}", query);
    
    // Hybrid search
    let results = rag.hybrid_search(query, 2);
    println!("\nTop results (hybrid search):");
    for (idx, score) in &results {
        if let Some(chunk) = rag.chunks.get(*idx) {
            println!("  [Score: {:.3}] {}", score, chunk.as_text());
        }
    }
    
    // Generate answer
    let context_chunks: Vec<&Chunk> = results.iter()
        .filter_map(|(idx, _)| rag.chunks.get(*idx))
        .collect();
    
    let answer = rag.generate_answer(query, &context_chunks)?;
    println!("\nGenerated Answer: {}", answer);
    
    Ok(())
}