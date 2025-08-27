use anyhow::Result;
use edgebert::{Model, ModelType, cosine_similarity};

fn main() -> Result<()> {
    let model = Model::from_pretrained(ModelType::MiniLML6V2)?;

    let texts = vec!["Hello world", "How are you?", "Goodbye world"];
    let embeddings = model.encode(texts.clone(), false)?; // normalize=false

    for (i, embedding) in embeddings.iter().enumerate() {
        let norm: f32 = embedding.iter().map(|x| x*x).sum::<f32>().sqrt();
        let n = embedding.len().min(10);
        println!("Text: {} | Norm: {:.4} | First 10 dims: {:?}", texts[i], norm, &embedding[0..n]);
    }

    for i in 0..embeddings.len() {
        for j in (i+1)..embeddings.len() {
            let sim = cosine_similarity(&embeddings[i], &embeddings[j]);
            println!("Cosine similarity ({} <-> {}) = {:.4}", texts[i], texts[j], sim);
        }
    }

    Ok(())
}