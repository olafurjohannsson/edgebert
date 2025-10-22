use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Response, Window, WorkerGlobalScope};

async fn fetch(url: &str) -> Result<Response, JsValue> {
    let global = js_sys::global();
    let resp_js = if let Ok(win) = global.dyn_into::<Window>() {
        JsFuture::from(win.fetch_with_str(url)).await?
    } else if let Ok(worker) = global.dyn_into::<WorkerGlobalScope>() {
        JsFuture::from(worker.fetch_with_str(url)).await?
    } else {
        return Err(JsValue::from_str("Unknown global execution scope"));
    };
    resp_js.dyn_into::<Response>().map_err(|_| "Could not cast to Response".into())
}

pub async fn fetch_bytes(url: &str) -> Result<Vec<u8>, JsValue> {
    let resp = fetch(url).await?;
    let array_buffer = JsFuture::from(resp.array_buffer()?).await?;
    Ok(js_sys::Uint8Array::new(&array_buffer).to_vec())
}

pub async fn fetch_text(url: &str) -> Result<String, JsValue> {
    let resp = fetch(url).await?;
    let text_js = JsFuture::from(resp.text()?).await?;
    text_js.as_string().ok_or_else(|| JsValue::from_str("Failed to convert response to text"))
}