# CLIP parity fixtures

Reference values produced by PyTorch and HuggingFace `transformers`, checked in so
the CLIP tests can run anywhere without either installed.

That is the point of committing them. Generating these in CI would mean a PyTorch
install on every run, and it would also mean the reference could shift underneath
us when `transformers` changes: a parity test whose expected values move is not a
parity test. Pinned here, a change in either implementation shows up as a failure
rather than as agreement with a new answer.

Regenerate only when the model or the preprocessing deliberately changes:

    python bench/clip_reference.py tests/fixtures/clip

| File | What |
|---|---|
| `fixture.png` / `.jpg` / `_gray.png` / `_rgba.png` | one synthetic image in four encodings |
| `fixture.rgb8` | the same image as raw RGB, so decoding can be checked losslessly |
| `fixture_jpeg_pil.rgb8` | PIL's decode of `fixture.jpg`, for comparing against a different IDCT |
| `pixel_values.f32` | HuggingFace's preprocessed tensor, `[3, 224, 224]` |
| `image_embeds.f32` | the image vector, 512-d, L2 normalized |
| `text_*.f32`, `text_reference.json` | caption vectors and the token ids they came from |
| `tokenizer_reference.json` | 13 tokenizer cases, chosen where BPE implementations diverge |
| `reference.json` | embedding dimension, model id, and the png/jpeg cosine |
| `photos/` | four crude scenes for the end-to-end image search test |

The image is a gradient with a checkerboard and two blocks rather than a
photograph: structure is what a resample-kernel mismatch distorts, and noise would
average the error away.
