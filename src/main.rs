use anyhow::Result;
use edgebert::Model;

fn main() -> Result<()> {
    // Optimized cosine similarity using dot product for normalized vectors
    let cosine_sim = |a: &[f32], b: &[f32]| -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            dot / (norm_a * norm_b)
    };

    println!("Starting vectorizer...");

    let vectorizer = Model::from_pretrained("minilm-l6-v2")?;
    println!("Model loaded successfully!\n");

    let texts = vec!["Hello world", "How are you?"];

    // Test regular embeddings
    println!("=== Regular Embeddings ===");
    println!("Encoding texts: {:?}", texts);
    let embeddings = vectorizer.encode(texts.clone())?;
    println!("Embeddings shape: [{}, {}]", embeddings.len(), embeddings[0].len());

    println!("\nCosine similarities:");
    for i in 0..texts.len() {
        for j in i + 1..texts.len() {
            let sim = cosine_sim(&embeddings[i], &embeddings[j]);
            println!("'{}' vs '{}': {:.3}", texts[i], texts[j], sim);
        }
    }

    let first_emb = &embeddings[0];
    let norm = first_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("\nFirst embedding norm: {:.6}", norm);
    println!("First 10 values: {:?}", &embeddings[0][..10]);

    // Test normalized embeddings
    println!("\n=== Normalized Embeddings ===");
    let normalized_embeddings = vectorizer.encode_normalized(texts.clone())?;

    let norm_check = normalized_embeddings[0].iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("Normalized first embedding norm: {:.6} (should be ~1.0)", norm_check);
    println!("Normalized first 5 values: {:?}", &normalized_embeddings[0][..5]);

    // For normalized vectors, cosine similarity = dot product
    println!("\nCosine similarities (normalized):");
    for i in 0..texts.len() {
        for j in i + 1..texts.len() {
            // Just dot product for normalized vectors
            let sim: f32 = normalized_embeddings[i].iter()
                .zip(normalized_embeddings[j].iter())
                .map(|(x, y)| x * y)
                .sum();
            println!("'{}' vs '{}': {:.3}", texts[i], texts[j], sim);
        }
    }

    Ok(())
}
