wasm-pack build --release --target web -- --no-default-features
rm -rf ./examples/pkg
mv ./pkg ./examples
