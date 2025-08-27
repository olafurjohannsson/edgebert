use criterion::{black_box, criterion_group, criterion_main, Criterion};
use edgebert::{Model, ModelType};

fn benchmark_edgebert(c: &mut Criterion) {
    let model = Model::from_pretrained(ModelType::MiniLML6V2).unwrap();
    let texts: Vec<&str> = vec!["Hello world"; 100];

    c.bench_function("edgebert_100_texts", |b| {
        b.iter(|| model.encode(black_box(texts.clone()), true))
    });
}

criterion_group!(benches, benchmark_edgebert);
criterion_main!(benches);