use anyhow::{Result, anyhow};
use edgemodels::bert::BertBiEncoder;
use edgetransformers::prelude::Device;
use edgetransformers::wgpu_context::WgpuContext;
use std::fs;
use std::path::Path;
use std::sync::Arc;

const USE_GPU: bool = false;

/// ASYNC helper function to ensure model files are available, downloading them if necessary.
/// This logic lives in the example, NOT in the main library.
async fn ensure_model_files(repo_id: &str, local_dir: &Path) -> Result<()> {
    if !local_dir.exists() {
        println!("-> Creating cache directory: {:?}", local_dir);
        // Use tokio's async version of create_dir_all
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

            // Use the ASYNC version of reqwest and .await it.
            let response = reqwest::get(&download_url).await?.error_for_status()?;

            let content = response.bytes().await?;

            // Use tokio's async version of write
            tokio::fs::write(&local_path, &content).await?;
            println!("   ... downloaded to {:?}", local_path);
        }
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("Starting BERT embedding example...");

    // --- 1. Define Model and Ensure Files are Present ---
    let model_repo = "sentence-transformers/all-MiniLM-L6-v2";

    let cache_dir = dirs::cache_dir()
        .ok_or_else(|| anyhow!("Failed to get cache directory"))?
        .join("edgegpt");

    let model_dir = cache_dir.join(model_repo.replace('/', "_"));

    // We now .await the async helper function.
    ensure_model_files(model_repo, &model_dir).await?;
    println!("Model files are available in: {:?}", model_dir);

    let wgpu_context: Option<Arc<WgpuContext>> = if USE_GPU {
        println!("\nInitializing WGPU context...");
        let context = WgpuContext::new().await;
        println!("WGPU context initialized successfully.");
        Some(Arc::new(context))
    } else {
        None
    };
    println!("\nInitializing BertBiEncoder on CPU...");
    let device = if USE_GPU { Device::Wgpu } else { Device::Cpu };
    let bi_encoder = BertBiEncoder::from_pretrained(&model_dir, device, wgpu_context)?;
    println!("Model initialized successfully.");

    // --- 3. Prepare Input Texts ---
    let texts_to_encode = vec![
        "This is an example sentence.",
        "Each sentence is converted to a vector.",
    ];
    println!("\nEncoding {} sentences...", texts_to_encode.len());

    // --- 4. Call the `encode` method ---
    let embeddings = bi_encoder.encode(texts_to_encode.clone(), true).await?;
    println!("Encoding complete.");

    // --- 5. Print the Results ---
    println!("\n--- Embedding Results ---");
    for (i, embedding) in embeddings.iter().enumerate() {
        let norm: f32 = embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let preview: Vec<f32> = embedding.iter().take(5).cloned().collect();
        println!(
            "Text: \"{}\"\n  - Embedding (first 5 dims): {:?}\n  - L2 Norm: {:.4}\n",
            texts_to_encode[i], preview, norm
        );
    }

    Ok(())
}
