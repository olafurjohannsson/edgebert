use std::time::Instant;
use anyhow::Result;
use edgebert::{BertModel, ModelType};
fn main() -> Result<()> {
    let model = BertModel::from_pretrained(ModelType::MiniLML6V2BiEncoder)?;
    let texts: Vec<String> = (0..100).map(|i| format!("Hello world {}", i)).collect();
    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    // Warm-up
    let _ = model.encode(vec!["warmup"], true)?;
    println!("Warm up complete, running 100 iterations");
    let iterations = 100;
    let mut total_duration = 0.0;
    for _ in 0..iterations {
        let start = Instant::now();
        let _ = model.encode(text_refs.clone(), true)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0; // ms
        total_duration += elapsed;
    }
    let avg_total = total_duration / iterations as f64;
    let avg_per_sentence = avg_total / texts.len() as f64;
    println!("Average total time per run (100 sentences): {:.3} ms", avg_total);
    println!("Average per sentence: {:.3} ms", avg_per_sentence);
    Ok(())
}
