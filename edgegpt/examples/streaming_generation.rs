//! Streaming text generation example

use anyhow::Result;
use edgegpt::{GenerativeModel, ModelType, GenerationConfig};
use edgegpt::generation::{generate_text_streaming};
use std::io::{self, Write};

fn main() -> Result<()> {
    println!("Loading DistilGPT2 model...");
    let model = match GenerativeModel::from_pretrained(ModelType::DistilGPT2)? {
        GenerativeModel::DistilGPT2(m) => m,
        _ => panic!("Expected DistilGPT2"),
    };
    
    let prompt = "The future of artificial intelligence";
    println!("\nPrompt: {}", prompt);
    print!("Generated: ");
    io::stdout().flush()?;
    
    let config = GenerationConfig {
        max_new_tokens: 100,
        temperature: 0.8,
        top_k: Some(50),
        top_p: Some(0.95),
        repetition_penalty: 1.1,
        ..Default::default()
    };
    
    // Stream tokens as they're generated
    let result = generate_text_streaming(
        &model.base,
        &model.tokenizer,
        prompt,
        &config,
        Box::new(|_token_id, token_text| {
            print!("{}", token_text);
            io::stdout().flush().unwrap();
            true // Continue generation
        }),
    )?;
    
    println!("\n\nFull result: {}", result);
    
    Ok(())
}