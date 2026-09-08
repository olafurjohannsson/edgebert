//! Reranker FFI bindings

use crate::error::set_last_error;
use crate::{KjarniDevice, KjarniErrorCode, get_runtime};
use kjarni::Reranker;
use kjarni::reranker::RerankerError;
use std::ffi::{CStr, c_char, c_float};
use std::ptr;

/// Puts a cross-encoder logit on a 0..1 scale.
///
/// Exported rather than reimplemented per binding. It is one line of arithmetic,
/// but the branch is not decoration: `exp(-x)` overflows to infinity for a large
/// negative x and `inf / inf` is NaN, which sorts unpredictably. A binding author
/// writing the obvious one-liner gets a function that passes every value they try
/// and returns NaN in production, so there is one implementation and it lives
/// here.
///
/// Monotonic, so ranking is identical either way; only the numbers move. Scores
/// stay logits by default because that is what PyTorch returns.
#[unsafe(no_mangle)]
pub extern "C" fn kjarni_sigmoid(x: c_float) -> c_float {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Single rerank result
#[repr(C)]
pub struct KjarniRerankResult {
    pub index: usize,
    pub score: c_float,
}

/// Array of rerank results
#[repr(C)]
pub struct KjarniRerankResults {
    pub results: *mut KjarniRerankResult,
    pub len: usize,
}

impl KjarniRerankResults {
    pub fn empty() -> Self {
        Self {
            results: ptr::null_mut(),
            len: 0,
        }
    }

    pub fn from_results(results: Vec<(usize, f32)>) -> Self {
        if results.is_empty() {
            return Self::empty();
        }

        let len = results.len();
        let mut c_results: Vec<KjarniRerankResult> = results
            .into_iter()
            .map(|(index, score)| KjarniRerankResult { index, score })
            .collect();

        let ptr = c_results.as_mut_ptr();
        std::mem::forget(c_results);

        Self { results: ptr, len }
    }
}

/// Free rerank results
///
/// # Safety
///
/// - `results` must be a value returned by a rerank call, passed here exactly once. Its
///   buffers are invalid to read afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_rerank_results_free(results: KjarniRerankResults) {
    crate::panic::guard("kjarni_rerank_results_free", (), || unsafe {
        if !results.results.is_null() && results.len > 0 {
            let _ = Vec::from_raw_parts(results.results, results.len, results.len);
        }
    })
}

/// Configuration for Reranker
#[repr(C)]
pub struct KjarniRerankerConfig {
    pub device: KjarniDevice,
    pub cache_dir: *const c_char,
    pub model_name: *const c_char,
    pub model_path: *const c_char,
    pub quiet: i32,
}

/// Get default reranker configuration
#[unsafe(no_mangle)]
pub extern "C" fn kjarni_reranker_config_default() -> KjarniRerankerConfig {
    KjarniRerankerConfig {
        device: KjarniDevice::Cpu,
        cache_dir: ptr::null(),
        model_name: ptr::null(),
        model_path: ptr::null(),
        quiet: 0,
    }
}

/// Opaque handle to a Reranker
pub struct KjarniReranker {
    inner: Reranker,
}

/// Create a new Reranker
///
/// # Safety
///
/// - `out` must be null, or a live handle returned by `kjarni_reranker_new` that has not been
///   freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_reranker_new(
    config: *const KjarniRerankerConfig,
    out: *mut *mut KjarniReranker,
) -> KjarniErrorCode {
    crate::panic::guard(
        "kjarni_reranker_new",
        KjarniErrorCode::Panic,
        || -> KjarniErrorCode {
            unsafe {
                if out.is_null() {
                    return KjarniErrorCode::NullPointer;
                }

                let default_config = kjarni_reranker_config_default();
                let config = if config.is_null() {
                    &default_config
                } else {
                    &*config
                };

                let result = get_runtime().block_on(async {
                    // Default model name
                    let model_name = if !config.model_name.is_null() {
                        match CStr::from_ptr(config.model_name).to_str() {
                            Ok(s) => s,
                            Err(_) => return Err(KjarniErrorCode::InvalidUtf8),
                        }
                    } else {
                        "minilm-l6-v2-cross-encoder" // Default cross-encoder
                    };

                    let mut builder = Reranker::builder(model_name);

                    // Device
                    match config.device {
                        KjarniDevice::Gpu => builder = builder.gpu(),
                        KjarniDevice::Cpu => builder = builder.cpu(),
                    }

                    // Cache dir
                    if !config.cache_dir.is_null() {
                        match CStr::from_ptr(config.cache_dir).to_str() {
                            Ok(s) => builder = builder.cache_dir(s),
                            Err(_) => return Err(KjarniErrorCode::InvalidUtf8),
                        }
                    }

                    // Model path
                    if !config.model_path.is_null() {
                        match CStr::from_ptr(config.model_path).to_str() {
                            Ok(s) => builder = builder.model_path(s),
                            Err(_) => return Err(KjarniErrorCode::InvalidUtf8),
                        }
                    }

                    builder = builder.quiet(config.quiet != 0);

                    builder.build().await.map_err(|e| {
                        set_last_error(e.to_string());
                        match &e {
                            RerankerError::UnknownModel(_) => KjarniErrorCode::ModelNotFound,
                            RerankerError::ModelNotDownloaded(_) => KjarniErrorCode::ModelNotFound,
                            RerankerError::GpuUnavailable => KjarniErrorCode::GpuUnavailable,
                            RerankerError::InvalidConfig(_) => KjarniErrorCode::InvalidConfig,
                            _ => KjarniErrorCode::LoadFailed,
                        }
                    })
                });

                match result {
                    Ok(reranker) => {
                        let handle = Box::new(KjarniReranker { inner: reranker });
                        *out = Box::into_raw(handle);
                        KjarniErrorCode::Ok
                    }
                    Err(e) => e,
                }
            }
        },
    )
}

/// Free a Reranker
///
/// # Safety
///
/// - `reranker` must be null, or a handle returned by `kjarni_reranker_new` that has not
///   already been passed to this function. It is invalid to use afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_reranker_free(reranker: *mut KjarniReranker) {
    crate::panic::guard("kjarni_reranker_free", (), || unsafe {
        if !reranker.is_null() {
            let _ = Box::from_raw(reranker);
        }
    })
}

/// Score a single query-document pair
///
/// # Safety
///
/// - `reranker` must be null, or a live handle returned by `kjarni_reranker_new` that has not
///   been freed.
/// - `query` must be null, or a valid pointer to a nul-terminated C string.
/// - `document` must be null, or a valid pointer to a nul-terminated C string.
/// - `out` must be a valid, writable pointer to a `c_float`. It is written on both success and
///   failure, and any buffers it receives are owned by the caller.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_reranker_score(
    reranker: *mut KjarniReranker,
    query: *const c_char,
    document: *const c_char,
    out: *mut c_float,
) -> KjarniErrorCode {
    crate::panic::guard(
        "kjarni_reranker_score",
        KjarniErrorCode::Panic,
        || -> KjarniErrorCode {
            unsafe {
                if reranker.is_null() || query.is_null() || document.is_null() || out.is_null() {
                    return KjarniErrorCode::NullPointer;
                }

                let reranker_ref = &(*reranker).inner;

                let query = match CStr::from_ptr(query).to_str() {
                    Ok(s) => s,
                    Err(_) => return KjarniErrorCode::InvalidUtf8,
                };

                let document = match CStr::from_ptr(document).to_str() {
                    Ok(s) => s,
                    Err(_) => return KjarniErrorCode::InvalidUtf8,
                };

                let result =
                    get_runtime().block_on(async { reranker_ref.score(query, document).await });

                match result {
                    Ok(score) => {
                        *out = score;
                        KjarniErrorCode::Ok
                    }
                    Err(e) => {
                        set_last_error(e.to_string());
                        *out = 0.0;
                        KjarniErrorCode::InferenceFailed
                    }
                }
            }
        },
    )
}

/// Rerank documents by relevance to query
///
/// # Safety
///
/// - `reranker` must be null, or a live handle returned by `kjarni_reranker_new` that has not
///   been freed.
/// - `query` must be null, or a valid pointer to a nul-terminated C string.
/// - `documents` must point to `num_docs` readable pointers, each null or a valid
///   nul-terminated C string that stays valid for the call.
/// - `out` must be a valid, writable pointer to a `KjarniRerankResults`. On `KJARNI_OK` it
///   receives a value the caller owns and must release with the matching `*_free`; on
///   failure it is either cleared or left unmodified, and must not be read.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_reranker_rerank(
    reranker: *mut KjarniReranker,
    query: *const c_char,
    documents: *const *const c_char,
    num_docs: usize,
    out: *mut KjarniRerankResults,
) -> KjarniErrorCode {
    crate::panic::guard(
        "kjarni_reranker_rerank",
        KjarniErrorCode::Panic,
        || -> KjarniErrorCode {
            unsafe {
                if reranker.is_null() || query.is_null() || documents.is_null() || out.is_null() {
                    return KjarniErrorCode::NullPointer;
                }

                if num_docs == 0 {
                    *out = KjarniRerankResults::empty();
                    return KjarniErrorCode::Ok;
                }

                let reranker_ref = &(*reranker).inner;

                let query = match CStr::from_ptr(query).to_str() {
                    Ok(s) => s,
                    Err(_) => return KjarniErrorCode::InvalidUtf8,
                };

                // Convert documents
                let mut doc_vec = Vec::with_capacity(num_docs);
                for i in 0..num_docs {
                    let doc_ptr = *documents.add(i);
                    if doc_ptr.is_null() {
                        return KjarniErrorCode::NullPointer;
                    }
                    match CStr::from_ptr(doc_ptr).to_str() {
                        Ok(s) => doc_vec.push(s.to_string()),
                        Err(_) => return KjarniErrorCode::InvalidUtf8,
                    }
                }
                let doc_refs: Vec<&str> = doc_vec.iter().map(|s| s.as_str()).collect();

                let result =
                    get_runtime().block_on(async { reranker_ref.rerank(query, &doc_refs).await });

                match result {
                    Ok(results) => {
                        let ranked: Vec<(usize, f32)> =
                            results.into_iter().map(|r| (r.index, r.score)).collect();
                        *out = KjarniRerankResults::from_results(ranked);
                        KjarniErrorCode::Ok
                    }
                    Err(e) => {
                        set_last_error(e.to_string());
                        *out = KjarniRerankResults::empty();
                        KjarniErrorCode::InferenceFailed
                    }
                }
            }
        },
    )
}

/// Rerank and return top-k results
///
/// # Safety
///
/// - `reranker` must be null, or a live handle returned by `kjarni_reranker_new` that has not
///   been freed.
/// - `query` must be null, or a valid pointer to a nul-terminated C string.
/// - `documents` must point to `num_docs` readable pointers, each null or a valid
///   nul-terminated C string that stays valid for the call.
/// - `out` must be a valid, writable pointer to a `KjarniRerankResults`. On `KJARNI_OK` it
///   receives a value the caller owns and must release with the matching `*_free`; on
///   failure it is either cleared or left unmodified, and must not be read.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_reranker_rerank_top_k(
    reranker: *mut KjarniReranker,
    query: *const c_char,
    documents: *const *const c_char,
    num_docs: usize,
    top_k: usize,
    out: *mut KjarniRerankResults,
) -> KjarniErrorCode {
    crate::panic::guard(
        "kjarni_reranker_rerank_top_k",
        KjarniErrorCode::Panic,
        || -> KjarniErrorCode {
            unsafe {
                if reranker.is_null() || query.is_null() || documents.is_null() || out.is_null() {
                    return KjarniErrorCode::NullPointer;
                }

                if num_docs == 0 {
                    *out = KjarniRerankResults::empty();
                    return KjarniErrorCode::Ok;
                }

                let reranker_ref = &(*reranker).inner;

                let query = match CStr::from_ptr(query).to_str() {
                    Ok(s) => s,
                    Err(_) => return KjarniErrorCode::InvalidUtf8,
                };

                // Convert documents
                let mut doc_vec = Vec::with_capacity(num_docs);
                for i in 0..num_docs {
                    let doc_ptr = *documents.add(i);
                    if doc_ptr.is_null() {
                        return KjarniErrorCode::NullPointer;
                    }
                    match CStr::from_ptr(doc_ptr).to_str() {
                        Ok(s) => doc_vec.push(s.to_string()),
                        Err(_) => return KjarniErrorCode::InvalidUtf8,
                    }
                }
                let doc_refs: Vec<&str> = doc_vec.iter().map(|s| s.as_str()).collect();

                let result = get_runtime()
                    .block_on(async { reranker_ref.rerank_top_k(query, &doc_refs, top_k).await });

                match result {
                    Ok(results) => {
                        let ranked: Vec<(usize, f32)> =
                            results.into_iter().map(|r| (r.index, r.score)).collect();
                        *out = KjarniRerankResults::from_results(ranked);
                        KjarniErrorCode::Ok
                    }
                    Err(e) => {
                        set_last_error(e.to_string());
                        *out = KjarniRerankResults::empty();
                        KjarniErrorCode::InferenceFailed
                    }
                }
            }
        },
    )
}
