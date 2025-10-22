use wasm_bindgen::prelude::*;

pub struct WasmError(anyhow::Error);

impl From<anyhow::Error> for WasmError {
    fn from(error: anyhow::Error) -> Self {
        WasmError(error)
    }
}

impl From<WasmError> for JsValue {
    fn from(error: WasmError) -> Self {
        JsValue::from_str(&error.0.to_string())
    }
}

// Your new result type
pub type WasmResult<T> = Result<T, WasmError>;