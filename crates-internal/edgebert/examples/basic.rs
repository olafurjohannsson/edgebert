
use anyhow::Result;
use edgebert::{BertModel, ModelType};
fn main() -> Result<()> {
    let model = BertModel::from_pretrained(ModelType::MiniLML6V2BiEncoder)?;

    let texts = vec!["Hello world", "How are you"];
    let embeddings = model.encode(texts.clone(), true)?;

    for (i, embedding) in embeddings.iter().enumerate() {
        let n = embedding.len().min(10);
        println!("Text: {} == {:?}...", texts[i], &embedding[0..n]);
    }
    Ok(())
}