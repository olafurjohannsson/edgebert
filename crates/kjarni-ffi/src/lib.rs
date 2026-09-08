//! Kjarni FFI - C bindings for Kjarni
mod callback;
mod chat;
mod classifier;
mod embedder;
mod error;
#[cfg(feature = "image-io")]
mod image_index;
mod indexer;
mod panic;
mod reranker;
mod searcher;

pub use callback::*;
pub use classifier::*;
pub use embedder::*;
pub use error::*;
pub use indexer::*;
use kjarni::cosine_similarity;
pub use reranker::*;
pub use searcher::*;

use std::sync::OnceLock;
use tokio::runtime::Runtime;

/// Global tokio runtime for async operations
static RUNTIME: OnceLock<Runtime> = OnceLock::new();

/// Get or initialize the global runtime
pub(crate) fn get_runtime() -> &'static Runtime {
    RUNTIME.get_or_init(|| Runtime::new().expect("Failed to create tokio runtime"))
}

/// Initialize the Kjarni runtime. Optional - auto-initialized on first use.
#[unsafe(no_mangle)]
pub extern "C" fn kjarni_init() -> KjarniErrorCode {
    crate::panic::guard(
        "kjarni_init",
        KjarniErrorCode::Panic,
        || -> KjarniErrorCode {
            let physical_cores = num_cpus::get_physical();
            let _ = rayon::ThreadPoolBuilder::new()
                .num_threads(physical_cores)
                .build_global();

            let _ = get_runtime();

            KjarniErrorCode::Ok
        },
    )
}
/// Shutdown and cleanup. Call before process exit.
#[unsafe(no_mangle)]
pub extern "C" fn kjarni_shutdown() {
    // Runtime will be dropped when process exits
}

/// Get the library version string.
#[unsafe(no_mangle)]
pub extern "C" fn kjarni_version() -> *const std::ffi::c_char {
    static VERSION: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();
    VERSION.as_ptr() as *const std::ffi::c_char
}

/// Float array returned by FFI functions. Caller must free with kjarni_float_array_free.
#[repr(C)]
pub struct KjarniFloatArray {
    pub data: *mut f32,
    pub len: usize,
}

impl KjarniFloatArray {
    pub fn from_vec(v: Vec<f32>) -> Self {
        let len = v.len();
        let boxed = v.into_boxed_slice();
        let data = Box::into_raw(boxed) as *mut f32;
        Self { data, len }
    }

    pub fn empty() -> Self {
        Self {
            data: std::ptr::null_mut(),
            len: 0,
        }
    }
}

/// 2D float array for batch results. Caller must free with kjarni_float_2d_array_free.
#[repr(C)]
pub struct KjarniFloat2DArray {
    pub data: *mut f32,
    pub rows: usize,
    pub cols: usize,
}

impl KjarniFloat2DArray {
    pub fn from_vecs(vecs: Vec<Vec<f32>>) -> Self {
        if vecs.is_empty() {
            return Self {
                data: std::ptr::null_mut(),
                rows: 0,
                cols: 0,
            };
        }
        let rows = vecs.len();
        let cols = vecs[0].len();
        let flat: Vec<f32> = vecs.into_iter().flatten().collect();
        let boxed = flat.into_boxed_slice();
        let data = Box::into_raw(boxed) as *mut f32;
        Self { data, rows, cols }
    }

    pub fn from_flat(v: Vec<f32>, rows: usize, cols: usize) -> Self {
        // Ensure the vector length matches dimensions
        assert_eq!(v.len(), rows * cols, "Vector length must match rows*cols");

        // Convert Vec to Boxed Slice to prevent deallocation
        let mut boxed_slice = v.into_boxed_slice();

        // Get the raw pointer
        let data = boxed_slice.as_mut_ptr();

        // Forget the box so Rust doesn't free the memory at end of scope
        // Ownership is now passed to C (and eventually Python)
        std::mem::forget(boxed_slice);

        Self { data, rows, cols }
    }

    pub fn empty() -> Self {
        Self {
            data: std::ptr::null_mut(),
            rows: 0,
            cols: 0,
        }
    }
}

/// String array. Caller must free with kjarni_string_array_free.
#[repr(C)]
pub struct KjarniStringArray {
    pub strings: *mut *mut std::ffi::c_char,
    pub len: usize,
}

/// Free a float array allocated by Kjarni.
///
/// # Safety
///
/// - `arr` must be a value returned by a search/embedding call, passed here exactly once. Its
///   buffers are invalid to read afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_float_array_free(arr: KjarniFloatArray) {
    crate::panic::guard("kjarni_float_array_free", (), || unsafe {
        if !arr.data.is_null() && arr.len > 0 {
            let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(arr.data, arr.len));
        }
    })
}

/// Free a 2D float array allocated by Kjarni.
///
/// # Safety
///
/// - `arr` must be a value returned by `kjarni_embedder_encode_batch`, passed here exactly once.
///   Its buffers are invalid to read afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_float_2d_array_free(arr: KjarniFloat2DArray) {
    crate::panic::guard("kjarni_float_2d_array_free", (), || unsafe {
        if !arr.data.is_null() && arr.rows > 0 && arr.cols > 0 {
            let total = arr.rows * arr.cols;
            let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(arr.data, total));
        }
    })
}

/// Free a string allocated by Kjarni.
///
/// # Safety
///
/// - `s` must be a valid, writable pointer to a `std::ffi::c_char`. It is written on both
///   success and failure, and any buffers it receives are owned by the caller.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_string_free(s: *mut std::ffi::c_char) {
    crate::panic::guard("kjarni_string_free", (), || {
        if !s.is_null() {
            unsafe {
                let _ = std::ffi::CString::from_raw(s);
            }
        }
    })
}

/// Free a string array allocated by Kjarni.
///
/// # Safety
///
/// - `arr` must be a value returned by a call that returns one, passed here exactly once. Its
///   buffers are invalid to read afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_string_array_free(arr: KjarniStringArray) {
    crate::panic::guard("kjarni_string_array_free", (), || unsafe {
        if !arr.strings.is_null() && arr.len > 0 {
            let strings = std::slice::from_raw_parts_mut(arr.strings, arr.len);
            for s in strings.iter() {
                if !s.is_null() {
                    let _ = std::ffi::CString::from_raw(*s);
                }
            }
            let _ = Box::from_raw(strings.as_mut_ptr());
        }
    })
}

/// # Safety
///
/// - `a` must be null, or point to at least `len` readable `f32` values.
/// - `b` must be null, or point to at least `len` readable `f32` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_cosine_similarity(a: *const f32, b: *const f32, len: usize) -> f32 {
    crate::panic::guard("kjarni_cosine_similarity", 0.0, || -> f32 {
        unsafe {
            if a.is_null() || b.is_null() || len == 0 {
                return 0.0;
            }
            let a = std::slice::from_raw_parts(a, len);
            let b = std::slice::from_raw_parts(b, len);
            cosine_similarity(a, b)
        }
    })
}

#[cfg(test)]
mod ffi_bridge_tests {
    use super::*;
    use std::ffi::{CStr, CString};

    #[test]
    fn test_kjarni_init_returns_ok() {
        let result = kjarni_init();
        assert_eq!(result, KjarniErrorCode::Ok);
    }

    #[test]
    fn test_kjarni_init_idempotent() {
        // Should be safe to call multiple times
        let result1 = kjarni_init();
        let result2 = kjarni_init();
        let result3 = kjarni_init();

        assert_eq!(result1, KjarniErrorCode::Ok);
        assert_eq!(result2, KjarniErrorCode::Ok);
        assert_eq!(result3, KjarniErrorCode::Ok);
    }

    #[test]
    fn test_kjarni_version_not_null() {
        let version_ptr = kjarni_version();
        assert!(!version_ptr.is_null());
    }

    #[test]
    fn test_kjarni_version_valid_string() {
        let version_ptr = kjarni_version();
        let version_str = unsafe { CStr::from_ptr(version_ptr) };
        let version = version_str.to_str().expect("Version should be valid UTF-8");

        // Should match Cargo.toml version
        assert_eq!(version, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn test_kjarni_version_stable_pointer() {
        // Multiple calls should return the same pointer (static lifetime)
        let ptr1 = kjarni_version();
        let ptr2 = kjarni_version();
        let ptr3 = kjarni_version();

        assert_eq!(ptr1, ptr2);
        assert_eq!(ptr2, ptr3);
    }

    #[test]
    fn test_kjarni_shutdown_safe_to_call() {
        // Should not panic or crash
        kjarni_shutdown();
        kjarni_shutdown();
    }
    #[test]
    fn test_get_runtime_returns_valid_runtime() {
        let runtime = get_runtime();
        let result = runtime.block_on(async { 42 });
        assert_eq!(result, 42);
    }

    #[test]
    fn test_get_runtime_same_instance() {
        let runtime1 = get_runtime();
        let runtime2 = get_runtime();
        assert!(std::ptr::eq(runtime1, runtime2));
    }

    #[test]
    fn test_float_array_from_vec_empty() {
        let arr = KjarniFloatArray::from_vec(vec![]);

        assert_eq!(arr.len, 0);
        unsafe {
            kjarni_float_array_free(arr);
        }
    }

    #[test]
    fn test_float_array_from_vec_single_element() {
        let arr = KjarniFloatArray::from_vec(vec![2.5]);

        assert!(!arr.data.is_null());
        assert_eq!(arr.len, 1);

        let value = unsafe { *arr.data };
        assert!((value - 2.5).abs() < f32::EPSILON);

        unsafe {
            kjarni_float_array_free(arr);
        }
    }

    #[test]
    fn test_float_array_from_vec_multiple_elements() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let arr = KjarniFloatArray::from_vec(original.clone());

        assert!(!arr.data.is_null());
        assert_eq!(arr.len, 5);

        // Verify all data integrity
        let slice = unsafe { std::slice::from_raw_parts(arr.data, arr.len) };
        assert_eq!(slice, &original[..]);

        unsafe {
            kjarni_float_array_free(arr);
        }
    }

    #[test]
    fn test_float_array_from_vec_large() {
        // Test with realistic embedding size (e.g., 768 dimensions)
        let original: Vec<f32> = (0..768).map(|i| i as f32 * 0.001).collect();
        let arr = KjarniFloatArray::from_vec(original.clone());

        assert!(!arr.data.is_null());
        assert_eq!(arr.len, 768);

        let slice = unsafe { std::slice::from_raw_parts(arr.data, arr.len) };
        assert_eq!(slice, &original[..]);

        unsafe {
            kjarni_float_array_free(arr);
        }
    }

    #[test]
    fn test_float_array_from_vec_special_values() {
        let original = vec![
            0.0,
            -0.0,
            f32::MIN,
            f32::MAX,
            f32::MIN_POSITIVE,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        let arr = KjarniFloatArray::from_vec(original.clone());

        assert_eq!(arr.len, 8);

        let slice = unsafe { std::slice::from_raw_parts(arr.data, arr.len) };

        assert_eq!(slice[0], 0.0);
        assert_eq!(slice[2], f32::MIN);
        assert_eq!(slice[3], f32::MAX);
        assert_eq!(slice[4], f32::MIN_POSITIVE);
        assert_eq!(slice[5], f32::INFINITY);
        assert_eq!(slice[6], f32::NEG_INFINITY);
        assert!(slice[7].is_nan());

        unsafe {
            kjarni_float_array_free(arr);
        }
    }

    #[test]
    fn test_float_array_empty_constructor() {
        let arr = KjarniFloatArray::empty();

        assert!(arr.data.is_null());
        assert_eq!(arr.len, 0);

        // Should be safe to free an empty array
        unsafe {
            kjarni_float_array_free(arr);
        }
    }

    #[test]
    fn test_float_array_free_null_pointer() {
        let arr = KjarniFloatArray {
            data: std::ptr::null_mut(),
            len: 0,
        };
        unsafe {
            kjarni_float_array_free(arr);
        }
    }

    #[test]
    fn test_float_array_free_null_with_nonzero_len() {
        let arr = KjarniFloatArray {
            data: std::ptr::null_mut(),
            len: 100,
        };
        unsafe {
            kjarni_float_array_free(arr);
        }
    }

    #[test]
    fn test_float_2d_array_from_vecs_empty() {
        let arr = KjarniFloat2DArray::from_vecs(vec![]);

        assert!(arr.data.is_null());
        assert_eq!(arr.rows, 0);
        assert_eq!(arr.cols, 0);

        unsafe {
            kjarni_float_2d_array_free(arr);
        }
    }

    #[test]
    fn test_float_2d_array_from_vecs_single_row() {
        let original = vec![vec![1.0, 2.0, 3.0]];
        let arr = KjarniFloat2DArray::from_vecs(original.clone());

        assert!(!arr.data.is_null());
        assert_eq!(arr.rows, 1);
        assert_eq!(arr.cols, 3);

        let slice = unsafe { std::slice::from_raw_parts(arr.data, arr.rows * arr.cols) };
        assert_eq!(slice, &[1.0, 2.0, 3.0]);

        unsafe {
            kjarni_float_2d_array_free(arr);
        }
    }

    #[test]
    fn test_float_2d_array_from_vecs_multiple_rows() {
        let original = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let arr = KjarniFloat2DArray::from_vecs(original);

        assert!(!arr.data.is_null());
        assert_eq!(arr.rows, 3);
        assert_eq!(arr.cols, 3);

        // Verify row-major layout
        let slice = unsafe { std::slice::from_raw_parts(arr.data, 9) };
        assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        unsafe {
            kjarni_float_2d_array_free(arr);
        }
    }

    #[test]
    fn test_float_2d_array_from_vecs_batch_embeddings() {
        // Simulate batch of 4 embeddings with 384 dimensions
        let batch_size = 4;
        let embedding_dim = 384;
        let original: Vec<Vec<f32>> = (0..batch_size)
            .map(|i| {
                (0..embedding_dim)
                    .map(|j| (i * embedding_dim + j) as f32)
                    .collect()
            })
            .collect();

        let arr = KjarniFloat2DArray::from_vecs(original.clone());

        assert_eq!(arr.rows, batch_size);
        assert_eq!(arr.cols, embedding_dim);

        // Spot check some values
        let slice = unsafe { std::slice::from_raw_parts(arr.data, batch_size * embedding_dim) };
        assert_eq!(slice[0], 0.0); // First element of first row
        assert_eq!(slice[embedding_dim], embedding_dim as f32); // First element of second row
        assert_eq!(slice[2 * embedding_dim], (2 * embedding_dim) as f32); // First element of third row

        unsafe {
            kjarni_float_2d_array_free(arr);
        }
    }

    #[test]
    fn test_float_2d_array_from_flat() {
        let flat = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let arr = KjarniFloat2DArray::from_flat(flat.clone(), 2, 3);

        assert!(!arr.data.is_null());
        assert_eq!(arr.rows, 2);
        assert_eq!(arr.cols, 3);

        let slice = unsafe { std::slice::from_raw_parts(arr.data, 6) };
        assert_eq!(slice, &flat[..]);

        unsafe {
            kjarni_float_2d_array_free(arr);
        }
    }

    #[test]
    #[should_panic(expected = "Vector length must match rows*cols")]
    fn test_float_2d_array_from_flat_dimension_mismatch() {
        let flat = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // 5 elements
        let _ = KjarniFloat2DArray::from_flat(flat, 2, 3); // Expects 6
    }

    #[test]
    fn test_float_2d_array_from_flat_single_element() {
        let flat = vec![42.0];
        let arr = KjarniFloat2DArray::from_flat(flat, 1, 1);

        assert_eq!(arr.rows, 1);
        assert_eq!(arr.cols, 1);

        let value = unsafe { *arr.data };
        assert_eq!(value, 42.0);

        unsafe {
            kjarni_float_2d_array_free(arr);
        }
    }

    #[test]
    fn test_float_2d_array_empty_constructor() {
        let arr = KjarniFloat2DArray::empty();

        assert!(arr.data.is_null());
        assert_eq!(arr.rows, 0);
        assert_eq!(arr.cols, 0);

        unsafe {
            kjarni_float_2d_array_free(arr);
        }
    }

    #[test]
    fn test_float_2d_array_free_null_pointer() {
        let arr = KjarniFloat2DArray {
            data: std::ptr::null_mut(),
            rows: 0,
            cols: 0,
        };
        unsafe {
            kjarni_float_2d_array_free(arr);
        }
    }

    #[test]
    fn test_float_2d_array_free_null_with_nonzero_dimensions() {
        // Edge case: null pointer but dimensions > 0
        let arr = KjarniFloat2DArray {
            data: std::ptr::null_mut(),
            rows: 10,
            cols: 10,
        };
        unsafe {
            kjarni_float_2d_array_free(arr);
        }
    }

    #[test]
    fn test_float_2d_array_free_zero_rows() {
        // Edge case: valid pointer but rows = 0
        let arr = KjarniFloat2DArray::from_vecs(vec![]);
        unsafe {
            kjarni_float_2d_array_free(arr);
        }
    }

    #[test]
    fn test_string_array_free_null() {
        let arr = KjarniStringArray {
            strings: std::ptr::null_mut(),
            len: 0,
        };
        unsafe {
            kjarni_string_array_free(arr);
        }
    }

    #[test]
    fn test_string_array_free_null_with_nonzero_len() {
        let arr = KjarniStringArray {
            strings: std::ptr::null_mut(),
            len: 5,
        };
        unsafe {
            kjarni_string_array_free(arr);
        }
    }

    #[test]
    fn test_string_free_null() {
        unsafe {
            kjarni_string_free(std::ptr::null_mut());
        }
    }

    #[test]
    fn test_string_free_valid() {
        let s = CString::new("Hello, FFI!").unwrap();
        let ptr = s.into_raw();
        unsafe {
            kjarni_string_free(ptr);
        }
    }

    #[test]
    fn test_string_free_unicode() {
        let s = CString::new("こんにちは世界 🌍 émojis").unwrap();
        let ptr = s.into_raw();
        unsafe {
            kjarni_string_free(ptr);
        }
    }

    #[test]
    fn test_string_free_empty() {
        let s = CString::new("").unwrap();
        let ptr = s.into_raw();
        unsafe {
            kjarni_string_free(ptr);
        }
    }

    #[test]
    fn test_float_array_roundtrip_preserves_data() {
        let original: Vec<f32> = (0..1000).map(|i| i as f32 * 0.123).collect();
        let arr = KjarniFloatArray::from_vec(original.clone());

        // Simulate what a C caller would do: read the data
        let roundtrip: Vec<f32> = unsafe { std::slice::from_raw_parts(arr.data, arr.len).to_vec() };

        assert_eq!(original, roundtrip);

        unsafe {
            kjarni_float_array_free(arr);
        }
    }

    #[test]
    fn test_float_2d_array_roundtrip_row_access() {
        let original = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0],
        ];
        let arr = KjarniFloat2DArray::from_vecs(original.clone());

        // Simulate C caller accessing rows
        for (row_idx, expected) in original.iter().enumerate() {
            let row_start = unsafe { arr.data.add(row_idx * arr.cols) };
            let row = unsafe { std::slice::from_raw_parts(row_start, arr.cols) };
            assert_eq!(row, &expected[..]);
        }

        unsafe {
            kjarni_float_2d_array_free(arr);
        }
    }

    #[test]
    fn test_float_2d_array_roundtrip_element_access() {
        let arr = KjarniFloat2DArray::from_vecs(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

        let at = |row: usize, col: usize| unsafe { *arr.data.add(row * arr.cols + col) };
        assert_eq!(at(0, 0), 1.0);
        assert_eq!(at(0, 1), 2.0);
        assert_eq!(at(1, 0), 3.0);
        assert_eq!(at(1, 1), 4.0);

        unsafe {
            kjarni_float_2d_array_free(arr);
        }
    }

    #[test]
    fn test_float_array_alignment() {
        let arr = KjarniFloatArray::from_vec(vec![1.0; 256]);

        // f32 requires 4-byte alignment
        assert_eq!(arr.data as usize % std::mem::align_of::<f32>(), 0);

        unsafe {
            kjarni_float_array_free(arr);
        }
    }

    #[test]
    fn test_float_2d_array_alignment() {
        let arr = KjarniFloat2DArray::from_vecs(vec![vec![1.0; 128]; 8]);

        assert_eq!(arr.data as usize % std::mem::align_of::<f32>(), 0);

        unsafe {
            kjarni_float_2d_array_free(arr);
        }
    }

    #[test]
    fn test_float_array_repr_c_layout() {
        assert_eq!(
            std::mem::size_of::<KjarniFloatArray>(),
            std::mem::size_of::<*mut f32>() + std::mem::size_of::<usize>()
        );
    }

    #[test]
    fn test_float_2d_array_repr_c_layout() {
        assert_eq!(
            std::mem::size_of::<KjarniFloat2DArray>(),
            std::mem::size_of::<*mut f32>() + 2 * std::mem::size_of::<usize>()
        );
    }

    #[test]
    fn test_string_array_repr_c_layout() {
        assert_eq!(
            std::mem::size_of::<KjarniStringArray>(),
            std::mem::size_of::<*mut *mut std::ffi::c_char>() + std::mem::size_of::<usize>()
        );
    }

    #[test]
    fn test_runtime_thread_safe() {
        use std::thread;

        let handles: Vec<_> = (0..4)
            .map(|i| {
                thread::spawn(move || {
                    let rt = get_runtime();
                    rt.block_on(async move { i * 2 })
                })
            })
            .collect();

        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(results, vec![0, 2, 4, 6]);
    }

    #[test]
    fn test_init_thread_safe() {
        use std::thread;

        // The closure is not redundant: `kjarni_init` is `extern "C" fn`, which
        // does not satisfy `thread::spawn`'s `FnOnce()` bound directly. Clippy's
        // suggested rewrite does not compile.
        #[allow(clippy::redundant_closure)]
        let handles: Vec<_> = (0..8).map(|_| thread::spawn(|| kjarni_init())).collect();

        for handle in handles {
            let result = handle.join().unwrap();
            assert_eq!(result, KjarniErrorCode::Ok);
        }
    }

    #[test]
    fn test_float_array_concurrent_creation() {
        use std::thread;

        let handles: Vec<_> = (0..8)
            .map(|i| {
                thread::spawn(move || {
                    let data: Vec<f32> = (0..100).map(|j| (i * 100 + j) as f32).collect();
                    let arr = KjarniFloatArray::from_vec(data.clone());

                    let slice = unsafe { std::slice::from_raw_parts(arr.data, arr.len) };
                    assert_eq!(slice, &data[..]);

                    unsafe {
                        kjarni_float_array_free(arr);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }
    #[test]
    fn test_many_small_allocations() {
        // Simulate many small embedding returns
        for _ in 0..1000 {
            let arr = KjarniFloatArray::from_vec(vec![1.0, 2.0, 3.0]);
            unsafe {
                kjarni_float_array_free(arr);
            }
        }
    }

    #[test]
    fn test_large_allocation() {
        // Large batch: 1000 embeddings x 1024 dimensions
        let large: Vec<Vec<f32>> = (0..1000).map(|_| vec![0.5; 1024]).collect();

        let arr = KjarniFloat2DArray::from_vecs(large);
        assert_eq!(arr.rows, 1000);
        assert_eq!(arr.cols, 1024);

        unsafe {
            kjarni_float_2d_array_free(arr);
        }
    }

    #[test]
    fn test_alternating_alloc_free() {
        // Pattern that might expose memory issues
        let mut arrays = Vec::new();

        for i in 0..100 {
            arrays.push(KjarniFloatArray::from_vec(vec![i as f32; 64]));

            if i % 3 == 0 && !arrays.is_empty() {
                let arr = arrays.remove(0);
                unsafe {
                    kjarni_float_array_free(arr);
                }
            }
        }

        for arr in arrays {
            unsafe {
                kjarni_float_array_free(arr);
            }
        }
    }

    #[test]
    fn test_float_array_zero_len_nonzero_capacity() {
        let mut v = Vec::with_capacity(100);
        v.clear();
        let arr = KjarniFloatArray::from_vec(v);

        assert_eq!(arr.len, 0);

        unsafe {
            kjarni_float_array_free(arr);
        }
    }

    #[test]
    fn test_float_2d_array_single_empty_row() {
        // One row with zero columns
        let arr = KjarniFloat2DArray::from_vecs(vec![vec![]]);

        assert_eq!(arr.rows, 1);
        assert_eq!(arr.cols, 0);

        unsafe {
            kjarni_float_2d_array_free(arr);
        }
    }

    #[test]
    fn test_from_flat_memory_not_dropped() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let arr = KjarniFloat2DArray::from_flat(data, 2, 2);

        let slice = unsafe { std::slice::from_raw_parts(arr.data, 4) };
        assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0]);

        unsafe {
            kjarni_float_2d_array_free(arr);
        }
    }
}
