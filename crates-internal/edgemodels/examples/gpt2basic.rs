//! Basic text generation example
use anyhow::Result;
use edgemodels::{ModelType, GenerativeModel};

fn main() -> Result<()> {
    let model = GenerativeModel::from_pretrained(ModelType::DistilGPT2)?;
    println!("DistilGPT2");
    println!("Input text: {}", "Once upon a time in a land far, far away,");
    model.generate(
        "Once upon a time in a land far, far away,",
        &edgemodels::generation::GenerationConfig {
            max_new_tokens: 100,
            temperature: 1.4,
            ..Default::default()
        },
    ).map(|output| {
        println!("Generated Text:\n{}", output);
    })?;


    
    Ok(())
}