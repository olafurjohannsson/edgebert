//! Batch text generation example

use anyhow::Result;
use edgegpt::{GenerativeModel, ModelType, GenerationConfig};

fn main() -> Result<()> {
    println!("Loading GPT-2 model...");
    let model = GenerativeModel::from_pretrained(ModelType::GPT2)?;
    
    let prompts = vec![
        "The meaning of life is",
        "In the year 2050,",
        "The best way to learn programming is",
        "Climate change will",
    ];
    
    let config = GenerationConfig {
        max_new_tokens: 40,
        temperature: 0.7,
        top_k: Some(40),
        repetition_penalty: 1.2,
        ..Default::default()
    };
    
    println!("Generating completions for {} prompts...\n", prompts.len());
    
    for prompt in prompts {
        println!("Prompt: {}", prompt);
        let generated = model.generate(prompt, &config)?;
        println!("Generated: {}\n", generated);
    }
    
    Ok(())
}