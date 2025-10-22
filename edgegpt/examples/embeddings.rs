//! Using GPT for embeddings

use anyhow::Result;
use edgegpt::{GenerativeModel, ModelType};

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    dot / (norm_a.sqrt() * norm_b.sqrt() + 1e-8)
}

fn main() -> Result<()> {
    println!("Loading DistilGPT2 model...");
    let model = match GenerativeModel::from_pretrained(ModelType::DistilGPT2)? {
        GenerativeModel::DistilGPT2(m) => m,
        _ => panic!("Expected DistilGPT2"),
    };
    
    let texts = vec![
        "Artificial intelligence is transforming the world",
        "Machine learning models are getting better",
        "The weather today is sunny and warm",
        "I love eating pizza for dinner",
    ];
    
    println!("\nGenerating embeddings for {} texts...", texts.len());
    
    let mut embeddings = Vec::new();
    for text in &texts {
        let emb = model.get_embeddings(text)?;
        println!("Text: {} | Embedding dim: {}", text, emb.len());
        embeddings.push(emb);
    }
    
    println!("\nSimilarity matrix:");
    for i in 0..texts.len() {
        for j in 0..texts.len() {
            let sim = cosine_similarity(&embeddings[i], &embeddings[j]);
            print!("{:.3} ", sim);
        }
        println!();
    }
    
    Ok(())
}