// Runs the wasm-gpu bundle against a real WebGPU adapter.
//
// Separate from run.mjs because it needs a different bundle (built with
// --features wasm-gpu) and because it must be allowed to skip: CI runners have no
// GPU, and failing there would only teach people to ignore this stage. The
// compile checks in test-everything.sh cover the code on every run; this covers
// the thing only a browser can answer, which is whether the shaders agree with
// the CPU.
//
//   KJARNI_KJQ_DIR=/tmp/kjq KJARNI_PKG_DIR=../../pkg-gpu node run-gpu.mjs

import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { join, extname, basename } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const here = fileURLToPath(new URL(".", import.meta.url));
const pkgDir = process.env.KJARNI_PKG_DIR ?? join(here, "../../pkg-gpu");
const modelDir = process.env.KJARNI_KJQ_DIR;

if (!modelDir) {
  console.error("KJARNI_KJQ_DIR must point at a directory of .kjq fixtures.");
  process.exit(2);
}

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".wasm": "application/wasm",
  ".kjq": "application/octet-stream",
};

const server = createServer(async (req, res) => {
  const url = new URL(req.url, "http://localhost");
  let file;
  if (url.pathname.startsWith("/pkg/")) file = join(pkgDir, basename(url.pathname));
  else if (url.pathname.startsWith("/models/")) file = join(modelDir, basename(url.pathname));
  else file = join(here, "harness-gpu.html");

  try {
    const body = await readFile(file);
    res.writeHead(200, { "content-type": MIME[extname(file)] ?? "application/octet-stream" });
    res.end(body);
  } catch (e) {
    res.writeHead(404).end(String(e));
  }
});

await new Promise((r) => server.listen(0, "127.0.0.1", r));
const port = server.address().port;

let failures = 0;
const check = (ok, what, detail = "") => {
  if (ok) console.log(`  ok    ${what}`);
  else {
    failures++;
    console.log(`  FAIL  ${what}${detail ? `  (${detail})` : ""}`);
  }
};

// Headless Chromium keeps WebGPU behind flags, and on a machine with no GPU it
// falls back to SwiftShader, which is slow but answers the correctness question
// just as well.
const browser = await chromium.launch({
  args: [
    "--enable-unsafe-webgpu",
    "--enable-features=Vulkan,UseSkiaRenderer",
    "--use-angle=vulkan",
    "--use-vulkan=swiftshader",
    "--enable-features=WebGPUService",
  ],
});
const page = await browser.newPage();

const pageErrors = [];
page.on("pageerror", (e) => pageErrors.push(e.message));
page.on("console", (m) => m.type() === "error" && pageErrors.push(m.text()));

let skipped = null;
try {
  await page.goto(`http://127.0.0.1:${port}/harness-gpu.html`);
  const r = await page.evaluate(() => window.__run());

  check(pageErrors.length === 0, "page loads without errors", pageErrors.join("; "));
  check(r.missing.length === 0, "every GPU class is exported", `missing: ${r.missing}`);

  if (r.skipped) {
    skipped = r.skipped;
  } else if (r.gpuLoadError) {
    // Refusing an adapter that cannot run the shaders is correct behaviour, not a
    // failure: five of them declare more than the 256 invocations per workgroup
    // WebGPU guarantees, and running anyway yields NaN. SwiftShader lands here.
    const need = /invocations per workgroup/.test(r.gpuLoadError);
    check(need, "underpowered adapter is refused, not silently wrong", r.gpuLoadError);
    if (need) skipped = `adapter below required limits (${JSON.stringify(r.limits)})`;
  } else {
    console.log(`  info  adapter limits: ${JSON.stringify(r.limits)}`);
    check(r.dim === 384, "MiniLM reports 384 dimensions", `got ${r.dim}`);
    check(
      r.parity > 0.9999,
      "CPU and GPU embeddings agree",
      `cosine ${r.parity.toFixed(8)}`,
    );
    check(
      r.batchLen === 3 && r.batchParity > 0.9999,
      "batched encode matches the single-text path",
      `len ${r.batchLen}, cosine ${r.batchParity.toFixed(8)}`,
    );
    check(r.topIndex === 1, "reranker puts Reykjavik first", `index ${r.topIndex}`);
    check(r.ordered, "reranker returns descending scores");
  }
} catch (e) {
  failures++;
  console.log(`  FAIL  harness threw: ${e.message}`);
  if (pageErrors.length) console.log(`        page errors: ${pageErrors.join("; ")}`);
} finally {
  await browser.close();
  server.close();
}

if (skipped) {
  console.log(`\nskipped: no WebGPU adapter (${skipped})`);
  process.exit(0);
}
console.log(`\n${failures === 0 ? "all checks passed" : `${failures} failed`}`);
process.exit(failures === 0 ? 0 : 1);
