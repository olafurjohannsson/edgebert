use anyhow::{Result, anyhow};
use edgemodels::gpt2::GPT2Model;
use edgetransformers::prelude::Device;
use edgetransformers::wgpu_context::WgpuContext;
use edgetransformers::CpuKVCache;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

const USE_GPU: bool = true;  // Start with CPU for testing
const MAX_NEW_TOKENS: usize = 50;

/// Helper function to ensure model files are available
async fn ensure_model_files(repo_id: &str, local_dir: &Path) -> Result<()> {
    if !local_dir.exists() {
        println!("-> Creating cache directory: {:?}", local_dir);
        tokio::fs::create_dir_all(local_dir).await?;
    }

    let files_to_check = ["model.safetensors", "config.json", "tokenizer.json"];
    for filename in files_to_check {
        let local_path = local_dir.join(filename);
        if !local_path.exists() {
            println!("-> Downloading {}...", filename);
            let download_url = format!(
                "https://huggingface.co/{}/resolve/main/{}",
                repo_id, filename
            );

            let response = reqwest::get(&download_url).await?.error_for_status()?;
            let content = response.bytes().await?;
            tokio::fs::write(&local_path, &content).await?;
            println!("   ... downloaded to {:?}", local_path);
        }
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let device_type_str = if USE_GPU { "GPU (WGPU)" } else { "CPU" };
    println!("Starting GPT-2 generation example...");

    // --- 1. Define Model and Ensure Files are Present ---
    let model_repo = "distilgpt2";  // Smaller, faster version of GPT-2
    
    let cache_dir = dirs::cache_dir()
        .ok_or_else(|| anyhow!("Failed to get cache directory"))?
        .join("edgegpt");

    let model_dir = cache_dir.join(model_repo.replace('/', "_"));

    ensure_model_files(model_repo, &model_dir).await?;
    println!("Model files are available in: {:?}", model_dir);

    // --- 2. Initialize Model ---
    let wgpu_context: Option<Arc<WgpuContext>> = if USE_GPU {
        println!("\nInitializing WGPU context...");
        let context = WgpuContext::new().await;
        println!("WGPU context initialized successfully.");
        Some(Arc::new(context))
    } else {
        None
    };
    
    println!("\nInitializing GPT-2 Model");
    let device = if USE_GPU { Device::Wgpu } else { Device::Cpu };
    let model = GPT2Model::from_pretrained(&model_dir, device, wgpu_context)?;
    println!("Model initialized successfully.");

    // --- 3. Test Prompts ---
    let prompts = vec![
        "Once upon a time",
        // "The future of AI is",
        // "In a world where technology",
        // "The quick brown fox",
    ];

    println!("\n=== Generation Examples ===\n");

    for prompt in prompts {
        println!("Prompt: \"{}\"", prompt);
        
        let start = Instant::now();
        let generated = model.generate(
            prompt,
            MAX_NEW_TOKENS,
            Some(1.0),  // temperature
            Some(50),   // top_k
        ).await?;
        let elapsed = start.elapsed();
        
        println!("Generated: {}", generated);
        println!("Time: {:.2}s ({:.1} tokens/sec)\n", 
            elapsed.as_secs_f32(),
            MAX_NEW_TOKENS as f32 / elapsed.as_secs_f32()
        );
    }

    // --- 4. Benchmark Generation Speed ---
    println!("\n=== Speed Benchmark ===");
    println!("Generating {} tokens with caching...\n", MAX_NEW_TOKENS);

    let prompt = "The future of artificial intelligence";
    let mut total_time = 0.0;
    let num_runs = 3;

    for run in 1..=num_runs {
        let start = Instant::now();
        let _ = model.generate(prompt, MAX_NEW_TOKENS, Some(1.0), Some(50)).await?;
        let elapsed = start.elapsed();
        total_time += elapsed.as_secs_f32();
        
        println!("Run {}/{}: {:.2}s ({:.1} tokens/sec)", 
            run, num_runs,
            elapsed.as_secs_f32(),
            MAX_NEW_TOKENS as f32 / elapsed.as_secs_f32()
        );
    }

    let avg_time = total_time / num_runs as f32;
    let avg_tokens_per_sec = MAX_NEW_TOKENS as f32 / avg_time;

    println!("\n--- Benchmark Results for {} ---", device_type_str);
    println!("Average time: {:.2}s", avg_time);
    println!("Average speed: {:.1} tokens/sec", avg_tokens_per_sec);
    println!("------------------------------------");

    Ok(())
}