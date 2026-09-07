//! Chat FFI bindings for decoder model inference.

use crate::callback::{KjarniCancelToken, is_cancelled};
use crate::error::set_last_error;
use crate::{KjarniDevice, KjarniErrorCode, get_runtime};
use futures::StreamExt;
use kjarni::chat::Chat;
use kjarni::chat::types::{ChatError, ChatMode};
use kjarni::generation::GenerationOverrides;
use std::ffi::{CStr, CString, c_char, c_void};
use std::ptr;

/// Configuration for creating a Chat instance.
#[repr(C)]
pub struct KjarniChatConfig {
    /// Device to use (0 = CPU, 1 = GPU)
    pub device: KjarniDevice,
    /// Cache directory (NULL = default)
    pub cache_dir: *const c_char,
    /// Model name (required) - e.g. "llama3.2-1b-instruct"
    pub model_name: *const c_char,
    /// Model path (NULL = use registry)
    pub model_path: *const c_char,
    /// System prompt (NULL = use model default)
    pub system_prompt: *const c_char,
    /// Chat mode: 0 = default, 1 = creative, 2 = reasoning
    pub mode: i32,
    /// Suppress output
    pub quiet: i32,
}

/// Generation parameters. Use -1 sentinel values to keep model defaults.
#[repr(C)]
pub struct KjarniGenerationConfig {
    /// Sampling temperature. -1.0 = use default.
    pub temperature: f32,
    /// Top-k sampling. -1 = use default.
    pub top_k: i32,
    /// Top-p (nucleus) sampling. -1.0 = use default.
    pub top_p: f32,
    /// Min-p sampling threshold. -1.0 = use default.
    pub min_p: f32,
    /// Repetition penalty. -1.0 = use default.
    pub repetition_penalty: f32,
    /// Max new tokens to generate. -1 = use default.
    pub max_new_tokens: i32,
    /// Sampling mode. -1 = default, 0 = greedy, 1 = sample.
    pub do_sample: i32,
}

/// Token callback for streaming. Return false to stop generation.
pub type KjarniStreamCallbackFn =
    Option<extern "C" fn(text: *const c_char, user_data: *mut c_void) -> bool>;

/// Get default chat configuration.
#[unsafe(no_mangle)]
pub extern "C" fn kjarni_chat_config_default() -> KjarniChatConfig {
    KjarniChatConfig {
        device: KjarniDevice::Cpu,
        cache_dir: ptr::null(),
        model_name: ptr::null(),
        model_path: ptr::null(),
        system_prompt: ptr::null(),
        mode: 0,
        quiet: 0,
    }
}

/// Get default generation configuration (all sentinels = use model defaults).
#[unsafe(no_mangle)]
pub extern "C" fn kjarni_generation_config_default() -> KjarniGenerationConfig {
    KjarniGenerationConfig {
        temperature: -1.0,
        top_k: -1,
        top_p: -1.0,
        min_p: -1.0,
        repetition_penalty: -1.0,
        max_new_tokens: -1,
        do_sample: -1,
    }
}
/// Convert FFI generation config to Rust GenerationOverrides.
fn to_overrides(config: &KjarniGenerationConfig) -> GenerationOverrides {
    GenerationOverrides {
        temperature: if config.temperature >= 0.0 {
            Some(config.temperature)
        } else {
            None
        },
        top_k: if config.top_k >= 0 {
            Some(config.top_k as usize)
        } else {
            None
        },
        top_p: if config.top_p >= 0.0 {
            Some(config.top_p)
        } else {
            None
        },
        min_p: if config.min_p >= 0.0 {
            Some(config.min_p)
        } else {
            None
        },
        repetition_penalty: if config.repetition_penalty >= 0.0 {
            Some(config.repetition_penalty)
        } else {
            None
        },
        max_new_tokens: if config.max_new_tokens >= 0 {
            Some(config.max_new_tokens as usize)
        } else {
            None
        },
        do_sample: match config.do_sample {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        },
        ..Default::default()
    }
}

/// Convert FFI mode int to ChatMode.
fn to_chat_mode(mode: i32) -> ChatMode {
    match mode {
        1 => ChatMode::Creative,
        2 => ChatMode::Reasoning,
        _ => ChatMode::Default,
    }
}

/// Map ChatError to KjarniErrorCode.
fn chat_error_to_code(e: &ChatError) -> KjarniErrorCode {
    match e {
        ChatError::UnknownModel(_) => KjarniErrorCode::ModelNotFound,
        ChatError::ModelNotDownloaded(_) => KjarniErrorCode::ModelNotFound,
        ChatError::GpuUnavailable => KjarniErrorCode::GpuUnavailable,
        ChatError::InvalidConfig(_) => KjarniErrorCode::InvalidConfig,
        ChatError::NoChatTemplate(_) => KjarniErrorCode::InvalidConfig,
        ChatError::IncompatibleModel { .. } => KjarniErrorCode::InvalidConfig,
        ChatError::InvalidModel(_, _) => KjarniErrorCode::InvalidConfig,
        ChatError::DownloadFailed { .. } => KjarniErrorCode::LoadFailed,
        ChatError::LoadFailed { .. } => KjarniErrorCode::LoadFailed,
        ChatError::GenerationFailed(_) => KjarniErrorCode::InferenceFailed,
    }
}

// ---------------------------------------------------------------------------
// Opaque handles
// ---------------------------------------------------------------------------

/// Opaque handle to a Chat instance.
pub struct KjarniChat {
    inner: Chat,
}

/// Opaque handle to a stateful ChatConversation.
///
/// Owns the history and holds a raw pointer back to the parent Chat.
/// The parent Chat must outlive this handle.
pub struct KjarniChatConversation {
    chat: *const KjarniChat,
    history: kjarni::chat::types::History,
}

// ---------------------------------------------------------------------------
// Chat lifecycle
// ---------------------------------------------------------------------------

/// Create a new Chat instance.
///
/// # Safety
/// - `config` must be valid or NULL (uses defaults)
/// - `out` must be a valid pointer
/// - The returned handle must be freed with `kjarni_chat_free`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_chat_new(
    config: *const KjarniChatConfig,
    out: *mut *mut KjarniChat,
) -> KjarniErrorCode {
    unsafe { kjarni_chat_new_with_draft(config, std::ptr::null(), 0, out) }
}

/// Creates a chat handle with speculative decoding enabled.
///
/// A draft model proposes `draft_tokens` tokens per round and this model verifies
/// them in a single pass. Decode is bandwidth bound, so reading the target's
/// weights once for several tokens rather than once per token is the whole
/// saving: measured on Qwen2.5, 0.5B drafting for 1.5B runs 3.5x faster at eight
/// proposed tokens. The draft must share the target's vocabulary, which in
/// practice means a smaller model of the same family.
///
/// A separate entry point rather than two more fields on `KjarniChatConfig`:
/// that struct is returned by value from `kjarni_chat_config_default` and passed
/// back by pointer, so growing it would make every already-compiled caller hand
/// over a shorter allocation than this library reads. Adding a function cannot
/// break a binary that never calls it.
///
/// Pass NULL for `draft_model_name` to get exactly `kjarni_chat_new`.
///
/// # Safety
/// `config` must be null or a valid `KjarniChatConfig`. `draft_model_name` must
/// be null or a valid NUL-terminated string. `out` must be a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_chat_new_with_draft(
    config: *const KjarniChatConfig,
    draft_model_name: *const c_char,
    draft_tokens: usize,
    out: *mut *mut KjarniChat,
) -> KjarniErrorCode {
    crate::panic::guard(
        "kjarni_chat_new_with_draft",
        KjarniErrorCode::Panic,
        || -> KjarniErrorCode {
            unsafe {
                if out.is_null() {
                    return KjarniErrorCode::NullPointer;
                }

                let default_config = kjarni_chat_config_default();
                let config = if config.is_null() {
                    &default_config
                } else {
                    &*config
                };

                // Model name is required
                if config.model_name.is_null() {
                    set_last_error("model_name is required".to_string());
                    return KjarniErrorCode::InvalidConfig;
                }

                let model_name = match CStr::from_ptr(config.model_name).to_str() {
                    Ok(s) => s,
                    Err(_) => return KjarniErrorCode::InvalidUtf8,
                };

                let draft_name = if draft_model_name.is_null() {
                    None
                } else {
                    match CStr::from_ptr(draft_model_name).to_str() {
                        Ok(s) => Some(s),
                        Err(_) => return KjarniErrorCode::InvalidUtf8,
                    }
                };

                let result = get_runtime().block_on(async {
                    let mut builder = Chat::builder(model_name);

                    if let Some(d) = draft_name {
                        builder =
                            builder.draft(d, if draft_tokens == 0 { 4 } else { draft_tokens });
                    }

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

                    // System prompt
                    if !config.system_prompt.is_null() {
                        match CStr::from_ptr(config.system_prompt).to_str() {
                            Ok(s) => builder = builder.system(s),
                            Err(_) => return Err(KjarniErrorCode::InvalidUtf8),
                        }
                    }

                    // Mode
                    builder = builder.mode(to_chat_mode(config.mode));

                    // Quiet
                    if config.quiet != 0 {
                        builder = builder.quiet();
                    }

                    builder.build().await.map_err(|e| {
                        set_last_error(e.to_string());
                        chat_error_to_code(&e)
                    })
                });

                match result {
                    Ok(chat) => {
                        let handle = Box::new(KjarniChat { inner: chat });
                        *out = Box::into_raw(handle);
                        KjarniErrorCode::Ok
                    }
                    Err(e) => e,
                }
            }
        },
    )
}

/// Free a Chat instance.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_chat_free(chat: *mut KjarniChat) {
    crate::panic::guard("kjarni_chat_free", (), || unsafe {
        if !chat.is_null() {
            let _ = Box::from_raw(chat);
        }
    })
}

// ---------------------------------------------------------------------------
// Chat: send (blocking)
// ---------------------------------------------------------------------------

/// Send a message and get the full response.
///
/// # Safety
/// - `chat` must be a valid handle from `kjarni_chat_new`
/// - `message` must be a valid UTF-8 C string
/// - `gen_config` may be NULL (uses defaults)
/// - `out` must be a valid pointer; caller must free result with `kjarni_string_free`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_chat_send(
    chat: *mut KjarniChat,
    message: *const c_char,
    gen_config: *const KjarniGenerationConfig,
    out: *mut *mut c_char,
) -> KjarniErrorCode {
    crate::panic::guard(
        "kjarni_chat_send",
        KjarniErrorCode::Panic,
        || -> KjarniErrorCode {
            unsafe {
                if chat.is_null() || message.is_null() || out.is_null() {
                    return KjarniErrorCode::NullPointer;
                }

                let chat_ref = &(*chat).inner;

                let message = match CStr::from_ptr(message).to_str() {
                    Ok(s) => s,
                    Err(_) => return KjarniErrorCode::InvalidUtf8,
                };

                let result = get_runtime().block_on(async {
                    if gen_config.is_null() {
                        chat_ref.send(message).await
                    } else {
                        let overrides = to_overrides(&*gen_config);
                        chat_ref.send_with_config(message, &overrides).await
                    }
                });

                match result {
                    Ok(response) => match CString::new(response) {
                        Ok(cstr) => {
                            *out = cstr.into_raw();
                            KjarniErrorCode::Ok
                        }
                        Err(_) => {
                            set_last_error("Response contained null byte".to_string());
                            KjarniErrorCode::InferenceFailed
                        }
                    },
                    Err(e) => {
                        set_last_error(e.to_string());
                        *out = ptr::null_mut();
                        chat_error_to_code(&e)
                    }
                }
            }
        },
    )
}

// ---------------------------------------------------------------------------
// Chat: stream (callback per token)
// ---------------------------------------------------------------------------

/// Stream a response token by token via callback.
///
/// The callback receives each token's text as a C string. Return `false`
/// from the callback to stop generation early.
///
/// # Safety
/// - `chat` must be a valid handle
/// - `message` must be valid UTF-8
/// - `callback` must be a valid function pointer
/// - `cancel_token` may be NULL
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_chat_stream(
    chat: *mut KjarniChat,
    message: *const c_char,
    gen_config: *const KjarniGenerationConfig,
    callback: KjarniStreamCallbackFn,
    user_data: *mut c_void,
    cancel_token: *const KjarniCancelToken,
) -> KjarniErrorCode {
    crate::panic::guard(
        "kjarni_chat_stream",
        KjarniErrorCode::Panic,
        || -> KjarniErrorCode {
            unsafe {
                if chat.is_null() || message.is_null() {
                    return KjarniErrorCode::NullPointer;
                }

                let cb = match callback {
                    Some(cb) => cb,
                    None => return KjarniErrorCode::NullPointer,
                };

                let chat_ref = &(*chat).inner;

                let message = match CStr::from_ptr(message).to_str() {
                    Ok(s) => s,
                    Err(_) => return KjarniErrorCode::InvalidUtf8,
                };

                let result = get_runtime().block_on(async {
                    let stream_result = if gen_config.is_null() {
                        chat_ref.stream(message).await
                    } else {
                        let overrides = to_overrides(&*gen_config);
                        chat_ref.stream_with_config(message, overrides).await
                    };

                    let mut stream = stream_result?;

                    while let Some(token_result) = stream.next().await {
                        // Check cancellation
                        if is_cancelled(cancel_token) {
                            break;
                        }

                        match token_result {
                            Ok(text) => {
                                if let Ok(cstr) = CString::new(text) {
                                    let should_continue = cb(cstr.as_ptr(), user_data);
                                    if !should_continue {
                                        break;
                                    }
                                }
                                // Skip tokens with null bytes (shouldn't happen, but be safe)
                            }
                            Err(e) => {
                                return Err(e);
                            }
                        }
                    }

                    Ok(())
                });

                match result {
                    Ok(()) => KjarniErrorCode::Ok,
                    Err(e) => {
                        set_last_error(e.to_string());
                        chat_error_to_code(&e)
                    }
                }
            }
        },
    )
}
/// Send a message with explicit history. Does not modify the history
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_chat_send_with_history(
    chat: *mut KjarniChat,
    roles: *const i32,
    contents: *const *const c_char,
    history_len: usize,
    message: *const c_char,
    gen_config: *const KjarniGenerationConfig,
    out: *mut *mut c_char,
) -> KjarniErrorCode {
    crate::panic::guard(
        "kjarni_chat_send_with_history",
        KjarniErrorCode::Panic,
        || -> KjarniErrorCode {
            unsafe {
                if chat.is_null() || message.is_null() || out.is_null() {
                    return KjarniErrorCode::NullPointer;
                }

                if history_len > 0 && (roles.is_null() || contents.is_null()) {
                    return KjarniErrorCode::NullPointer;
                }

                let chat_ref = &(*chat).inner;

                let message = match CStr::from_ptr(message).to_str() {
                    Ok(s) => s,
                    Err(_) => return KjarniErrorCode::InvalidUtf8,
                };

                // Build History from arrays
                let mut history = kjarni::chat::types::History::new();
                for i in 0..history_len {
                    let role = *roles.add(i);
                    let content_ptr = *contents.add(i);
                    if content_ptr.is_null() {
                        return KjarniErrorCode::NullPointer;
                    }
                    let content = match CStr::from_ptr(content_ptr).to_str() {
                        Ok(s) => s,
                        Err(_) => return KjarniErrorCode::InvalidUtf8,
                    };

                    match role {
                        0 => {
                            // System - rebuild history with system prompt
                            history = kjarni::chat::types::History::with_system(content);
                        }
                        1 => history.push_user(content),
                        2 => history.push_assistant(content),
                        _ => {
                            set_last_error(format!("Invalid role: {}", role));
                            return KjarniErrorCode::InvalidConfig;
                        }
                    }
                }

                let result = get_runtime().block_on(async {
                    if gen_config.is_null() {
                        chat_ref.send_with_history(&history, message).await
                    } else {
                        let mut conversation = chat_ref.history_to_conversation(&history);
                        conversation.push_user(message);
                        let prompt = chat_ref.format_prompt(&conversation);
                        let overrides = to_overrides(&*gen_config);
                        chat_ref.generate(&prompt, &overrides).await
                    }
                });

                match result {
                    Ok(response) => match CString::new(response) {
                        Ok(cstr) => {
                            *out = cstr.into_raw();
                            KjarniErrorCode::Ok
                        }
                        Err(_) => {
                            set_last_error("Response contained null byte".to_string());
                            KjarniErrorCode::InferenceFailed
                        }
                    },
                    Err(e) => {
                        set_last_error(e.to_string());
                        *out = ptr::null_mut();
                        chat_error_to_code(&e)
                    }
                }
            }
        },
    )
}

/// Stream a response with explicit history. Does not modify the history.
///
/// The counterpart to `kjarni_chat_send_with_history`. `IChatClient` and other
/// stateless APIs hand back the whole transcript on every call, so streaming has
/// to accept a history the same way sending does; without this, a streaming
/// caller could only ever produce single-turn replies.
///
/// Roles follow the same encoding as `kjarni_chat_send_with_history`:
/// 0 = system, 1 = user, 2 = assistant.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_chat_stream_with_history(
    chat: *mut KjarniChat,
    roles: *const i32,
    contents: *const *const c_char,
    history_len: usize,
    message: *const c_char,
    gen_config: *const KjarniGenerationConfig,
    callback: KjarniStreamCallbackFn,
    user_data: *mut c_void,
    cancel_token: *const KjarniCancelToken,
) -> KjarniErrorCode {
    crate::panic::guard(
        "kjarni_chat_stream_with_history",
        KjarniErrorCode::Panic,
        || -> KjarniErrorCode {
            unsafe {
                if chat.is_null() || message.is_null() {
                    return KjarniErrorCode::NullPointer;
                }

                if history_len > 0 && (roles.is_null() || contents.is_null()) {
                    return KjarniErrorCode::NullPointer;
                }

                let cb = match callback {
                    Some(cb) => cb,
                    None => return KjarniErrorCode::NullPointer,
                };

                let chat_ref = &(*chat).inner;

                let message = match CStr::from_ptr(message).to_str() {
                    Ok(s) => s,
                    Err(_) => return KjarniErrorCode::InvalidUtf8,
                };

                let mut history = kjarni::chat::types::History::new();
                for i in 0..history_len {
                    let role = *roles.add(i);
                    let content_ptr = *contents.add(i);
                    if content_ptr.is_null() {
                        return KjarniErrorCode::NullPointer;
                    }
                    let content = match CStr::from_ptr(content_ptr).to_str() {
                        Ok(s) => s,
                        Err(_) => return KjarniErrorCode::InvalidUtf8,
                    };

                    match role {
                        0 => {
                            // System - rebuild history with system prompt
                            history = kjarni::chat::types::History::with_system(content);
                        }
                        1 => history.push_user(content),
                        2 => history.push_assistant(content),
                        _ => {
                            set_last_error(format!("Invalid role: {}", role));
                            return KjarniErrorCode::InvalidConfig;
                        }
                    }
                }

                let result = get_runtime().block_on(async {
                    let stream_result = if gen_config.is_null() {
                        chat_ref.stream_with_history(&history, message).await
                    } else {
                        let mut conversation = chat_ref.history_to_conversation(&history);
                        conversation.push_user(message);
                        let prompt = chat_ref.format_prompt(&conversation);
                        let overrides = to_overrides(&*gen_config);
                        chat_ref.generate_stream(prompt, overrides).await
                    };

                    let mut stream = stream_result?;

                    while let Some(token_result) = stream.next().await {
                        if is_cancelled(cancel_token) {
                            break;
                        }

                        match token_result {
                            Ok(text) => {
                                if let Ok(cstr) = CString::new(text) {
                                    let should_continue = cb(cstr.as_ptr(), user_data);
                                    if !should_continue {
                                        break;
                                    }
                                }
                            }
                            Err(e) => {
                                return Err(e);
                            }
                        }
                    }

                    Ok(())
                });

                match result {
                    Ok(()) => KjarniErrorCode::Ok,
                    Err(e) => {
                        set_last_error(e.to_string());
                        chat_error_to_code(&e)
                    }
                }
            }
        },
    )
}

/// Create a new stateful conversation from a Chat instance
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_chat_conversation_new(
    chat: *mut KjarniChat,
    out: *mut *mut KjarniChatConversation,
) -> KjarniErrorCode {
    crate::panic::guard(
        "kjarni_chat_conversation_new",
        KjarniErrorCode::Panic,
        || -> KjarniErrorCode {
            unsafe {
                if chat.is_null() || out.is_null() {
                    return KjarniErrorCode::NullPointer;
                }

                let chat_ref = &(*chat).inner;

                let history = if let Some(system) = chat_ref.system_prompt() {
                    kjarni::chat::types::History::with_system(system)
                } else {
                    kjarni::chat::types::History::new()
                };

                let handle = Box::new(KjarniChatConversation {
                    chat: chat as *const KjarniChat,
                    history,
                });

                *out = Box::into_raw(handle);
                KjarniErrorCode::Ok
            }
        },
    )
}

/// Free a ChatConversation.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_chat_conversation_free(convo: *mut KjarniChatConversation) {
    crate::panic::guard("kjarni_chat_conversation_free", (), || unsafe {
        if !convo.is_null() {
            let _ = Box::from_raw(convo);
        }
    })
}
/// Send a message in a conversation and get the response
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_chat_conversation_send(
    convo: *mut KjarniChatConversation,
    message: *const c_char,
    gen_config: *const KjarniGenerationConfig,
    out: *mut *mut c_char,
) -> KjarniErrorCode {
    crate::panic::guard(
        "kjarni_chat_conversation_send",
        KjarniErrorCode::Panic,
        || -> KjarniErrorCode {
            unsafe {
                if convo.is_null() || message.is_null() || out.is_null() {
                    return KjarniErrorCode::NullPointer;
                }

                let convo_ref = &mut *convo;

                // Get parent Chat reference
                if convo_ref.chat.is_null() {
                    set_last_error("Parent chat has been freed".to_string());
                    return KjarniErrorCode::NullPointer;
                }
                let chat_ref = &(*convo_ref.chat).inner;

                let message = match CStr::from_ptr(message).to_str() {
                    Ok(s) => s,
                    Err(_) => return KjarniErrorCode::InvalidUtf8,
                };

                // Add user message to history
                convo_ref.history.push_user(message);

                // Build prompt from full history
                let conversation = chat_ref.history_to_conversation(&convo_ref.history);
                let prompt = chat_ref.format_prompt(&conversation);

                let overrides = if gen_config.is_null() {
                    GenerationOverrides::default()
                } else {
                    to_overrides(&*gen_config)
                };

                let result =
                    get_runtime().block_on(async { chat_ref.generate(&prompt, &overrides).await });

                match result {
                    Ok(response) => {
                        // Add assistant response to history
                        convo_ref.history.push_assistant(&response);

                        match CString::new(response) {
                            Ok(cstr) => {
                                *out = cstr.into_raw();
                                KjarniErrorCode::Ok
                            }
                            Err(_) => {
                                set_last_error("Response contained null byte".to_string());
                                KjarniErrorCode::InferenceFailed
                            }
                        }
                    }
                    Err(e) => {
                        set_last_error(e.to_string());
                        *out = ptr::null_mut();
                        chat_error_to_code(&e)
                    }
                }
            }
        },
    )
}

// ---------------------------------------------------------------------------
// ChatConversation: stream (callback per token)
// ---------------------------------------------------------------------------

/// Stream a response in a conversation.
///
/// The user message is added to history before streaming.
/// After streaming completes, the full response is added to history.
///
/// # Safety
/// - Same requirements as `kjarni_chat_stream`
/// - Parent Chat must still be alive
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_chat_conversation_stream(
    convo: *mut KjarniChatConversation,
    message: *const c_char,
    gen_config: *const KjarniGenerationConfig,
    callback: KjarniStreamCallbackFn,
    user_data: *mut c_void,
    cancel_token: *const KjarniCancelToken,
) -> KjarniErrorCode {
    crate::panic::guard(
        "kjarni_chat_conversation_stream",
        KjarniErrorCode::Panic,
        || -> KjarniErrorCode {
            unsafe {
                if convo.is_null() || message.is_null() {
                    return KjarniErrorCode::NullPointer;
                }

                let cb = match callback {
                    Some(cb) => cb,
                    None => return KjarniErrorCode::NullPointer,
                };

                let convo_ref = &mut *convo;

                if convo_ref.chat.is_null() {
                    set_last_error("Parent chat has been freed".to_string());
                    return KjarniErrorCode::NullPointer;
                }
                let chat_ref = &(*convo_ref.chat).inner;

                let message = match CStr::from_ptr(message).to_str() {
                    Ok(s) => s,
                    Err(_) => return KjarniErrorCode::InvalidUtf8,
                };

                // Add user message
                convo_ref.history.push_user(message);

                // Build prompt
                let conversation = chat_ref.history_to_conversation(&convo_ref.history);
                let prompt = chat_ref.format_prompt(&conversation);

                let overrides = if gen_config.is_null() {
                    GenerationOverrides::default()
                } else {
                    to_overrides(&*gen_config)
                };

                let result = get_runtime().block_on(async {
                    let mut stream = chat_ref.generate_stream(prompt, overrides).await?;

                    let mut full_response = String::new();

                    while let Some(token_result) = stream.next().await {
                        if is_cancelled(cancel_token) {
                            break;
                        }

                        match token_result {
                            Ok(text) => {
                                full_response.push_str(&text);
                                if let Ok(cstr) = CString::new(text) {
                                    let should_continue = cb(cstr.as_ptr(), user_data);
                                    if !should_continue {
                                        break;
                                    }
                                }
                            }
                            Err(e) => {
                                // Still save partial response to history
                                if !full_response.is_empty() {
                                    convo_ref.history.push_assistant(&full_response);
                                }
                                return Err(e);
                            }
                        }
                    }

                    Ok(full_response)
                });

                match result {
                    Ok(response) => {
                        // Add full response to history
                        if !response.is_empty() {
                            convo_ref.history.push_assistant(&response);
                        }
                        KjarniErrorCode::Ok
                    }
                    Err(e) => {
                        set_last_error(e.to_string());
                        chat_error_to_code(&e)
                    }
                }
            }
        },
    )
}

// ---------------------------------------------------------------------------
// ChatConversation: history management
// ---------------------------------------------------------------------------

/// Get the number of messages in the conversation history.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_chat_conversation_len(
    convo: *const KjarniChatConversation,
) -> usize {
    crate::panic::guard("kjarni_chat_conversation_len", 0, || -> usize {
        unsafe {
            if convo.is_null() {
                return 0;
            }
            (*convo).history.len()
        }
    })
}

/// Clear the conversation history.
///
/// If `keep_system` is non-zero, the system prompt is preserved.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_chat_conversation_clear(
    convo: *mut KjarniChatConversation,
    keep_system: i32,
) {
    crate::panic::guard("kjarni_chat_conversation_clear", (), || unsafe {
        if !convo.is_null() {
            (*convo).history.clear(keep_system != 0);
        }
    })
}

// ---------------------------------------------------------------------------
// Chat: info
// ---------------------------------------------------------------------------

/// Get the model name. Returns number of bytes written (excluding null).
///
/// If `buf` is NULL or `buf_len` is 0, returns the required buffer size.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_chat_model_name(
    chat: *const KjarniChat,
    buf: *mut c_char,
    buf_len: usize,
) -> usize {
    crate::panic::guard("kjarni_chat_model_name", 0, || -> usize {
        unsafe {
            if chat.is_null() {
                return 0;
            }

            let name = (*chat).inner.model_name();
            let name_bytes = name.as_bytes();

            if buf.is_null() || buf_len == 0 {
                return name_bytes.len();
            }

            let copy_len = std::cmp::min(name_bytes.len(), buf_len.saturating_sub(1));
            std::ptr::copy_nonoverlapping(name_bytes.as_ptr(), buf as *mut u8, copy_len);
            *buf.add(copy_len) = 0; // null terminate

            copy_len
        }
    })
}

/// Get the context window size.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_chat_context_size(chat: *const KjarniChat) -> usize {
    crate::panic::guard("kjarni_chat_context_size", 0, || -> usize {
        unsafe {
            if chat.is_null() {
                return 0;
            }
            (*chat).inner.context_size()
        }
    })
}
