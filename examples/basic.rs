
use anyhow::Result;
use edgebert::{Model, ModelType};
fn main() -> Result<()> {
    let model = Model::from_pretrained(ModelType::MiniLML6V2)?;

    let texts = vec!["Hello world", "How are you"];
    let embeddings = model.encode(texts.clone(), true)?;

    for (i, embedding) in embeddings.iter().enumerate() {
        let n = embedding.len().min(10);
        println!("Text: {} == {:?}...", texts[i], &embedding[0..n]);
    }
    Ok(())
}