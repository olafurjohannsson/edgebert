
use anyhow::Result;
use edgemodels::model::{BertModel, BertModelType};
fn main() -> Result<()> {
    let model = BertModel::from_pretrained(BertModelType::MiniLML6V2BiEncoder)?;

    let texts = vec!["Hello world", "How are you"];
    let embeddings = model.encode(texts.clone(), true)?;

    for (i, embedding) in embeddings.iter().enumerate() {
        let n = embedding.len().min(10);
        println!("Text: {} == {:?}...", texts[i], &embedding[0..n]);
    }
    Ok(())
}