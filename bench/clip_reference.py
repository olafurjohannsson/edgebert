#!/usr/bin/env python3
"""Reference fixtures for the CLIP vision parity test.

Writes a fixed image plus HuggingFace's own preprocessed tensor, image embedding,
and a handful of captions with their token ids and text embeddings, so the Rust
side can be checked against an implementation that shares none of its code.

    pip install torch transformers pillow numpy
    python bench/clip_reference.py /tmp/clip-fixtures
    KJARNI_CLIP_FIXTURES=/tmp/clip-fixtures \
        cargo test -p kjarni-models --test clip_vision_parity --release
    KJARNI_CLIP_FIXTURES=/tmp/clip-fixtures KJARNI_CLIP_PHOTOS=/tmp/clip-fixtures/photos \
        cargo test -p kjarni-models --features image-io --release

The image is structured rather than noise on purpose: a resample-kernel mismatch
averages out over noise and shows up plainly on edges and a checkerboard.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizerFast

MODEL = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"


def build_image(out: Path) -> Image.Image:
    rng = np.random.RandomState(1234)
    w, h = 320, 240  # deliberately not square, and reduced by the 224 resize
    img = np.zeros((h, w, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    img[..., 0] = (xx * 255 // w).astype(np.uint8)
    img[..., 1] = (yy * 255 // h).astype(np.uint8)
    img[..., 2] = (((xx // 16 + yy // 16) % 2) * 255).astype(np.uint8)
    img[60:120, 80:200] = [240, 30, 90]
    img[150:200, 40:120] = [10, 200, 220]
    img = (img.astype(np.int16) + rng.randint(-12, 13, img.shape)).clip(0, 255).astype(np.uint8)

    Image.fromarray(img).save(out / "fixture.png")
    img.tofile(out / "fixture.rgb8")
    json.dump({"width": w, "height": h}, open(out / "fixture.json", "w"))

    # Variants the image-io tests need: a lossy JPEG, and the two channel layouts
    # a real photo directory actually contains.
    pil = Image.open(out / "fixture.png").convert("RGB")
    pil.save(out / "fixture.jpg", quality=95)
    pil.convert("L").save(out / "fixture_gray.png")
    pil.convert("RGBA").save(out / "fixture_rgba.png")
    # PIL's own decode of that JPEG, so the Rust decoder can be compared against a
    # different IDCT implementation rather than against itself.
    np.asarray(Image.open(out / "fixture.jpg").convert("RGB"), dtype=np.uint8).tofile(
        out / "fixture_jpeg_pil.rgb8"
    )
    return pil


def build_photos(out: Path) -> int:
    """A tiny photo library for the image-search test.

    Crude renders rather than photographs, because the test has to run anywhere
    without shipping copyrighted images. CLIP is trained on natural pictures, so
    these are far from its comfort zone; the colour and layout cues are still
    strong enough that the intended image ranks first, which is what is asserted.
    """
    from PIL import ImageDraw

    out.mkdir(parents=True, exist_ok=True)

    beach = Image.new("RGB", (320, 240))
    d = ImageDraw.Draw(beach)
    for y in range(240):
        d.line([(0, y), (320, y)],
               fill=(90 + y // 3, 150 + y // 4, 235) if y < 150 else (222, 200, 150))
    d.ellipse([250, 20, 300, 70], fill=(255, 240, 120))
    beach.save(out / "beach_holiday.png")

    forest = Image.new("RGB", (320, 240), (34, 90, 40))
    d = ImageDraw.Draw(forest)
    for x in range(10, 320, 40):
        d.rectangle([x, 90, x + 14, 240], fill=(80, 55, 30))
        d.ellipse([x - 18, 40, x + 32, 110], fill=(20, 110, 45))
    forest.save(out / "deep_forest.png")

    car = Image.new("RGB", (320, 240), (200, 200, 205))
    d = ImageDraw.Draw(car)
    d.rectangle([60, 120, 260, 180], fill=(200, 25, 30))
    d.rectangle([100, 85, 220, 125], fill=(190, 30, 35))
    d.ellipse([80, 165, 125, 205], fill=(25, 25, 25))
    d.ellipse([195, 165, 240, 205], fill=(25, 25, 25))
    car.save(out / "red_car.png")

    snow = Image.new("RGB", (320, 240), (225, 235, 245))
    d = ImageDraw.Draw(snow)
    d.polygon([(60, 220), (160, 60), (260, 220)], fill=(245, 248, 252))
    d.polygon([(120, 130), (160, 60), (200, 130)], fill=(255, 255, 255))
    snow.save(out / "snowy_mountain.png")

    return 4


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/clip-fixtures")
    out.mkdir(parents=True, exist_ok=True)
    model_id = sys.argv[2] if len(sys.argv) > 2 else MODEL

    image = build_image(out)
    model = CLIPModel.from_pretrained(model_id).eval()
    processor = CLIPImageProcessor.from_pretrained(model_id)

    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"]
    np.asarray(pixel_values[0].numpy(), dtype=np.float32).tofile(out / "pixel_values.f32")

    with torch.no_grad():
        feats = model.get_image_features(pixel_values=pixel_values)[0]
        feats = feats / feats.norm()
    np.asarray(feats.numpy(), dtype=np.float32).tofile(out / "image_embeds.f32")

    # Text side. The ids are written out with the embeddings so the Rust test can
    # exercise the transformer on HuggingFace's own tokenization: a failure then
    # means the tower is wrong, not the tokenizer.
    tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
    captions = [
        "a photo of a dog",
        "a red sports car on a mountain road",
        "sunset over the ocean",
    ]
    records = []
    for i, caption in enumerate(captions):
        encoded = tokenizer(caption, return_tensors="pt")
        ids = encoded["input_ids"][0].tolist()
        with torch.no_grad():
            text_feats = model.get_text_features(**encoded)[0]
            text_feats = text_feats / text_feats.norm()
        np.asarray(text_feats.numpy(), dtype=np.float32).tofile(out / f"text_{i}.f32")
        # CLIP pools the end-of-text position, which under a causal mask is the
        # only one that has seen the whole caption.
        records.append({"text": caption, "ids": ids, "eos_index": len(ids) - 1})
    json.dump(records, open(out / "text_reference.json", "w"), indent=1)

    # Tokenizer cases, kept separate from the embeddings above. These are chosen
    # for the places BPE implementations actually diverge: casing, whitespace
    # runs, contractions, digits (which CLIP splits one at a time), punctuation
    # runs, decomposed accents, non-Latin script, and overflow past 77 tokens.
    tokenizer_cases = [
        "a photo of a dog",
        "A PHOTO OF A DOG",
        "  a   photo   of  a dog  ",
        "sunset over the ocean",
        "a red sports car on a mountain road",
        "it's a dog's life, isn't it?",
        "2024 was a year",
        "hello, world! (test) -- 42%",
        "cafe\u0301 nai\u0308ve re\u0301sume\u0301",
        "a" * 200,
        "\u732b \u3068 \u72ac",
        "e-mail: someone@example.com",
        "",
    ]
    json.dump(
        [{"text": c, "ids": tokenizer.encode(c)} for c in tokenizer_cases],
        open(out / "tokenizer_reference.json", "w"),
        indent=1,
    )

    # JPEG is lossy, so a JPEG and its source PNG do not embed identically. The
    # figure is recorded rather than assumed: on this deliberately hostile fixture
    # (checkerboard plus noise) it is far lower than a photograph would give, and
    # what the Rust test checks is that it reproduces this number, not that the
    # number is high.
    jpeg_px = processor(
        images=Image.open(out / "fixture.jpg").convert("RGB"), return_tensors="pt"
    )["pixel_values"]
    with torch.no_grad():
        jf = model.get_image_features(pixel_values=jpeg_px)[0]
        jf = jf / jf.norm()
    png_vs_jpeg = float((feats * jf).sum())
    json.dump(
        {"dim": int(feats.shape[0]), "model": model_id, "png_vs_jpeg_cosine": png_vs_jpeg},
        open(out / "reference.json", "w"),
    )

    photos = build_photos(out / "photos")

    print(
        f"wrote fixtures to {out} (embedding dim {feats.shape[0]}, "
        f"{len(records)} captions, png/jpeg cosine {png_vs_jpeg:.6f}, "
        f"{photos} photos in {out / 'photos'})"
    )


if __name__ == "__main__":
    main()
