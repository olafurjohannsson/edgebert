//! Basic text generation example

use anyhow::Result;
use edgegpt::{GenerativeModel, ModelType, GenerationConfig, SamplingStrategy};

fn main() -> Result<()> {
    println!("Loading DistilGPT2 model...");
    let model = GenerativeModel::from_pretrained(ModelType::DistilGPT2)?;
    
    // Basic generation with default config
    let prompt = "Once upon a time";
    println!("\nPrompt: {}", prompt);
    
    let config = GenerationConfig::default();
    let generated = model.generate(prompt, &config)?;
    println!("Generated: {}", generated);
    
    // Try different sampling strategies
    let strategies = vec![
        (SamplingStrategy::Greedy, "Greedy"),
        (SamplingStrategy::TopK, "Top-K"),
        (SamplingStrategy::TopP, "Top-P"),
        (SamplingStrategy::Temperature, "Temperature"),
    ];
    
    for (strategy, name) in strategies {
        let mut config = GenerationConfig {
            max_new_tokens: 30,
            temperature: 0.8,
            top_k: Some(40),
            top_p: Some(0.9),
            sampling_strategy: strategy,
            ..Default::default()
        };
        
        println!("\n{} Sampling:", name);
        let generated = model.generate(prompt, &config)?;
        println!("{}", generated);
    }
    
    Ok(())
}