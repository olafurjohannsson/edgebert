//! CLIP image search over the C ABI.
//!
//! Behind the `image-io` feature, because indexing a directory means decoding
//! JPEG and PNG. A host that already holds decoded pixels needs none of that and
//! can reach the vision tower directly.
//!
//! One handle owns both towers and the vectors. It is **not** safe to share
//! across threads: the search path takes `&self`, but `add_*` and `load` mutate,
//! and nothing here serialises them. One handle per thread, or guard it yourself.

use crate::error::set_last_error;
use crate::{KjarniErrorCode, get_runtime};
use kjarni::ImageIndex;
use std::ffi::{CStr, CString, c_char, c_float};
use std::path::Path;
use std::ptr;

/// An image index with its models loaded.
pub struct KjarniImageIndex {
    inner: ImageIndex,
}

/// One ranked image.
///
/// `path` is owned by this struct and freed by [`kjarni_image_results_free`].
#[repr(C)]
pub struct KjarniImageHit {
    pub path: *mut c_char,
    pub score: c_float,
}

/// A ranked list, and how many images could not be read.
#[repr(C)]
pub struct KjarniImageResults {
    pub hits: *mut KjarniImageHit,
    pub len: usize,
}

impl KjarniImageResults {
    fn empty() -> Self {
        Self {
            hits: ptr::null_mut(),
            len: 0,
        }
    }
}

/// What a scan did. Files that could not be decoded are counted, not fatal: one
/// truncated download should not abandon a scan of ten thousand photos.
#[repr(C)]
pub struct KjarniScanReport {
    pub added: usize,
    pub skipped: usize,
}

/// Loads CLIP from a model directory, as `kjarni model download` leaves it.
///
/// # Safety
/// `model_dir` must be a valid NUL-terminated path. `out` must be a valid pointer
/// to write the handle into.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_image_index_new(
    model_dir: *const c_char,
    out: *mut *mut KjarniImageIndex,
) -> KjarniErrorCode {
    crate::panic::guard(
        "kjarni_image_index_new",
        KjarniErrorCode::Panic,
        || -> KjarniErrorCode {
            unsafe {
                if out.is_null() || model_dir.is_null() {
                    return KjarniErrorCode::NullPointer;
                }
                let dir = match CStr::from_ptr(model_dir).to_str() {
                    Ok(s) => s,
                    Err(_) => return KjarniErrorCode::InvalidUtf8,
                };

                match ImageIndex::load(Path::new(dir)) {
                    Ok(inner) => {
                        *out = Box::into_raw(Box::new(KjarniImageIndex { inner }));
                        KjarniErrorCode::Ok
                    }
                    Err(e) => {
                        set_last_error(format!("{e:#}"));
                        KjarniErrorCode::LoadFailed
                    }
                }
            }
        },
    )
}

/// Frees an image index.
///
/// # Safety
/// `index` must be null, or a handle from `kjarni_image_index_new` that has not
/// already been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_image_index_free(index: *mut KjarniImageIndex) {
    crate::panic::guard("kjarni_image_index_free", (), || unsafe {
        if !index.is_null() {
            let _ = Box::from_raw(index);
        }
    })
}

/// Embeds one image file and adds or replaces it.
///
/// # Safety
/// Both pointers must be valid; `path` must be NUL-terminated.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_image_index_add(
    index: *mut KjarniImageIndex,
    path: *const c_char,
) -> KjarniErrorCode {
    crate::panic::guard(
        "kjarni_image_index_add",
        KjarniErrorCode::Panic,
        || -> KjarniErrorCode {
            unsafe {
                if index.is_null() || path.is_null() {
                    return KjarniErrorCode::NullPointer;
                }
                let p = match CStr::from_ptr(path).to_str() {
                    Ok(s) => s,
                    Err(_) => return KjarniErrorCode::InvalidUtf8,
                };
                match (*index).inner.add_image(Path::new(p)) {
                    Ok(()) => KjarniErrorCode::Ok,
                    Err(e) => {
                        set_last_error(format!("{e:#}"));
                        KjarniErrorCode::InferenceFailed
                    }
                }
            }
        },
    )
}

/// Walks a directory and indexes every image in it.
///
/// # Safety
/// `index` and `dir` must be valid; `dir` must be NUL-terminated. `report` may be
/// null if the counts are not wanted.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_image_index_add_directory(
    index: *mut KjarniImageIndex,
    dir: *const c_char,
    report: *mut KjarniScanReport,
) -> KjarniErrorCode {
    crate::panic::guard(
        "kjarni_image_index_add_directory",
        KjarniErrorCode::Panic,
        || -> KjarniErrorCode {
            unsafe {
                if index.is_null() || dir.is_null() {
                    return KjarniErrorCode::NullPointer;
                }
                let d = match CStr::from_ptr(dir).to_str() {
                    Ok(s) => s,
                    Err(_) => return KjarniErrorCode::InvalidUtf8,
                };
                match (*index).inner.add_directory(Path::new(d)) {
                    Ok(r) => {
                        if !report.is_null() {
                            *report = KjarniScanReport {
                                added: r.added,
                                skipped: r.skipped,
                            };
                        }
                        KjarniErrorCode::Ok
                    }
                    Err(e) => {
                        set_last_error(format!("{e:#}"));
                        KjarniErrorCode::InferenceFailed
                    }
                }
            }
        },
    )
}

/// Ranks indexed images against a description.
///
/// Scores are cosine similarities and are small in absolute terms even for a good
/// match, typically 0.2 to 0.35. Only the ordering is meaningful.
///
/// # Safety
/// All pointers must be valid; `query` must be NUL-terminated. The result must be
/// released with [`kjarni_image_results_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_image_index_search(
    index: *const KjarniImageIndex,
    query: *const c_char,
    top_k: usize,
    out: *mut KjarniImageResults,
) -> KjarniErrorCode {
    crate::panic::guard(
        "kjarni_image_index_search",
        KjarniErrorCode::Panic,
        || -> KjarniErrorCode {
            unsafe {
                if index.is_null() || query.is_null() || out.is_null() {
                    return KjarniErrorCode::NullPointer;
                }
                *out = KjarniImageResults::empty();

                let q = match CStr::from_ptr(query).to_str() {
                    Ok(s) => s,
                    Err(_) => return KjarniErrorCode::InvalidUtf8,
                };

                let hits = match (*index).inner.search(q, top_k) {
                    Ok(h) => h,
                    Err(e) => {
                        set_last_error(format!("{e:#}"));
                        return KjarniErrorCode::InferenceFailed;
                    }
                };
                if hits.is_empty() {
                    return KjarniErrorCode::Ok;
                }

                let mut c_hits = Vec::with_capacity(hits.len());
                for h in hits {
                    // A path that is not valid C string data cannot cross the
                    // boundary; skipping it is better than truncating at a NUL.
                    let Ok(path) = CString::new(h.path.to_string_lossy().as_bytes()) else {
                        continue;
                    };
                    c_hits.push(KjarniImageHit {
                        path: path.into_raw(),
                        score: h.score,
                    });
                }

                let len = c_hits.len();
                let boxed = c_hits.into_boxed_slice();
                *out = KjarniImageResults {
                    hits: Box::into_raw(boxed) as *mut KjarniImageHit,
                    len,
                };
                KjarniErrorCode::Ok
            }
        },
    )
}

/// Frees a result list and every path in it.
///
/// # Safety
/// `results` must come from `kjarni_image_index_search` and be freed once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_image_results_free(results: KjarniImageResults) {
    crate::panic::guard("kjarni_image_results_free", (), || unsafe {
        if results.hits.is_null() || results.len == 0 {
            return;
        }
        let slice = std::slice::from_raw_parts_mut(results.hits, results.len);
        for hit in slice.iter() {
            if !hit.path.is_null() {
                let _ = CString::from_raw(hit.path);
            }
        }
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(
            results.hits,
            results.len,
        ));
    })
}

/// Writes the vectors to `path`. The images stay where they are.
///
/// # Safety
/// Both pointers must be valid; `path` must be NUL-terminated.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_image_index_save(
    index: *const KjarniImageIndex,
    path: *const c_char,
) -> KjarniErrorCode {
    crate::panic::guard(
        "kjarni_image_index_save",
        KjarniErrorCode::Panic,
        || -> KjarniErrorCode {
            unsafe {
                if index.is_null() || path.is_null() {
                    return KjarniErrorCode::NullPointer;
                }
                let p = match CStr::from_ptr(path).to_str() {
                    Ok(s) => s,
                    Err(_) => return KjarniErrorCode::InvalidUtf8,
                };
                match (*index).inner.save(Path::new(p)) {
                    Ok(()) => KjarniErrorCode::Ok,
                    Err(e) => {
                        set_last_error(format!("{e:#}"));
                        KjarniErrorCode::InferenceFailed
                    }
                }
            }
        },
    )
}

/// Loads previously computed vectors, leaving the models as they are.
///
/// Refuses an index whose vectors are the wrong width, which means it was built
/// with a different checkpoint and cannot be compared against this one.
///
/// # Safety
/// Both pointers must be valid; `path` must be NUL-terminated. `count` may be
/// null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_image_index_load(
    index: *mut KjarniImageIndex,
    path: *const c_char,
    count: *mut usize,
) -> KjarniErrorCode {
    crate::panic::guard(
        "kjarni_image_index_load",
        KjarniErrorCode::Panic,
        || -> KjarniErrorCode {
            unsafe {
                if index.is_null() || path.is_null() {
                    return KjarniErrorCode::NullPointer;
                }
                let p = match CStr::from_ptr(path).to_str() {
                    Ok(s) => s,
                    Err(_) => return KjarniErrorCode::InvalidUtf8,
                };
                match (*index).inner.load_entries(Path::new(p)) {
                    Ok(n) => {
                        if !count.is_null() {
                            *count = n;
                        }
                        KjarniErrorCode::Ok
                    }
                    Err(e) => {
                        set_last_error(format!("{e:#}"));
                        KjarniErrorCode::InferenceFailed
                    }
                }
            }
        },
    )
}

/// How many images are indexed.
///
/// # Safety
/// `index` must be null or a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_image_index_len(index: *const KjarniImageIndex) -> usize {
    crate::panic::guard("kjarni_image_index_len", 0, || unsafe {
        if index.is_null() {
            0
        } else {
            (*index).inner.len()
        }
    })
}

// `get_runtime` is imported for symmetry with the other modules; the image path
// is synchronous because CLIP loading and embedding do no async work.
const _: fn() -> &'static tokio::runtime::Runtime = get_runtime;
