using System;
using System.IO;
using System.Reflection;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace Kjarni
{
    public enum KjarniErrorCode
    {
        Ok = 0,
        NullPointer = 1,
        InvalidUtf8 = 2,
        ModelNotFound = 3,
        LoadFailed = 4,
        InferenceFailed = 5,
        GpuUnavailable = 6,
        InvalidConfig = 7,
        Cancelled = 8,
        Timeout = 9,
        StreamEnded = 10,

        /// <summary>
        /// A panic in the native engine was caught at the ABI boundary. The operation
        /// failed and its outputs are undefined; the handle stays valid to dispose.
        /// </summary>
        Panic = 11,

        Unknown = 255,
    }

    public enum KjarniDevice
    {
        Cpu = 0,
        Gpu = 1,
    }

    public enum KjarniSearchMode
    {
        Keyword = 0,
        Semantic = 1,
        Hybrid = 2,
    }

    public enum KjarniProgressStage
    {
        Scanning = 0,
        Loading = 1,
        Embedding = 2,
        Writing = 3,
        Committing = 4,
        Searching = 5,
        Reranking = 6,
    }

    internal static class Native
    {
        private const string LibName = "kjarni_ffi";

        static Native()
        {
            NativeLibrary.SetDllImportResolver(typeof(Native).Assembly, ImportResolver);
        }

        private static IntPtr ImportResolver(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
        {
            if (libraryName != LibName)
            {
                return IntPtr.Zero;
            }

            string[] candidates;
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                candidates = new[] { "kjarni_ffi.dll", "libkjarni_ffi.dll" };
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                candidates = new[] { "libkjarni_ffi.dylib", "kjarni_ffi.dylib" };
            }
            else
            {
                candidates = new[] { "libkjarni_ffi.so", "kjarni_ffi.so" };
            }

            var assemblyDir = Path.GetDirectoryName(assembly.Location) ?? ".";
            var searchDirs = new[]
            {
                assemblyDir,
                Path.Combine(assemblyDir, "runtimes", GetRuntimeIdentifier(), "native"),
                Path.Combine(assemblyDir, "native"),
                Environment.CurrentDirectory,
            };

            foreach (var dir in searchDirs)
            {
                foreach (var candidate in candidates)
                {
                    var fullPath = Path.Combine(dir, candidate);
                    if (File.Exists(fullPath))
                    {
                        return NativeLibrary.Load(fullPath);
                    }
                }
            }

            foreach (var candidate in candidates)
            {
                if (NativeLibrary.TryLoad(candidate, out var handle))
                {
                    return handle;
                }
            }

            return IntPtr.Zero;
        }

        private static string GetRuntimeIdentifier()
        {
            var arch = RuntimeInformation.OSArchitecture switch
            {
                Architecture.X64 => "x64",
                Architecture.Arm64 => "arm64",
                Architecture.X86 => "x86",
                _ => "x64"
            };

            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return $"win-{arch}";
            if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
                return $"osx-{arch}";
            return $"linux-{arch}";
        }
        [StructLayout(LayoutKind.Sequential)]
        public struct KjarniFloatArray
        {
            public IntPtr Data;
            public UIntPtr Len;

            public float[] ToArray()
            {
                var len = (int)Len;
                if (len == 0 || Data == IntPtr.Zero) return Array.Empty<float>();
                var result = new float[len];
                Marshal.Copy(Data, result, 0, len);
                return result;
            }
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct KjarniFloat2DArray
        {
            public IntPtr Data;
            public UIntPtr Rows;
            public UIntPtr Cols;

            public float[][] ToArray()
            {
                var rows = (int)Rows;
                var cols = (int)Cols;
                if (rows == 0 || cols == 0 || Data == IntPtr.Zero)
                    return Array.Empty<float[]>();

                var flat = new float[rows * cols];
                Marshal.Copy(Data, flat, 0, rows * cols);

                var result = new float[rows][];
                for (int i = 0; i < rows; i++)
                {
                    result[i] = new float[cols];
                    Array.Copy(flat, i * cols, result[i], 0, cols);
                }
                return result;
            }
        }
        [StructLayout(LayoutKind.Sequential)]
        public struct KjarniEmbedderConfig
        {
            public KjarniDevice Device;
            public IntPtr CacheDir;
            public IntPtr ModelName;
            public IntPtr ModelPath;
            public int Normalize;
            public int Quiet;
        }
        [StructLayout(LayoutKind.Sequential)]
        public struct KjarniClassResult
        {
            public IntPtr Label;
            public float Score;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct KjarniClassResults
        {
            public IntPtr Results;
            public UIntPtr Len;

            public (string Label, float Score)[] ToArray()
            {
                var len = (int)Len;
                if (len == 0 || Results == IntPtr.Zero)
                    return Array.Empty<(string, float)>();

                var result = new (string, float)[len];
                var structSize = Marshal.SizeOf<KjarniClassResult>();

                for (int i = 0; i < len; i++)
                {
                    var ptr = IntPtr.Add(Results, i * structSize);
                    var item = Marshal.PtrToStructure<KjarniClassResult>(ptr);
                    var label = item.Label != IntPtr.Zero
                        ? Marshal.PtrToStringUTF8(item.Label) ?? ""
                        : "";
                    result[i] = (label, item.Score);
                }
                return result;
            }
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct KjarniClassifierConfig
        {
            public KjarniDevice Device;
            public IntPtr CacheDir;
            public IntPtr ModelName;
            public IntPtr ModelPath;
            public IntPtr Labels;
            public UIntPtr NumLabels;
            public int MultiLabel;
            public int Quiet;
        }
        [StructLayout(LayoutKind.Sequential)]
        public struct KjarniRerankResult
        {
            public UIntPtr Index;
            public float Score;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct KjarniRerankResults
        {
            public IntPtr Results;
            public UIntPtr Len;

            public (int Index, float Score)[] ToArray()
            {
                var len = (int)Len;
                if (len == 0 || Results == IntPtr.Zero)
                    return Array.Empty<(int, float)>();

                var result = new (int, float)[len];
                var structSize = Marshal.SizeOf<KjarniRerankResult>();

                for (int i = 0; i < len; i++)
                {
                    var ptr = IntPtr.Add(Results, i * structSize);
                    var item = Marshal.PtrToStructure<KjarniRerankResult>(ptr);
                    result[i] = ((int)item.Index, item.Score);
                }
                return result;
            }
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct KjarniRerankerConfig
        {
            public KjarniDevice Device;
            public IntPtr CacheDir;
            public IntPtr ModelName;
            public IntPtr ModelPath;
            public int Quiet;
        }
        [StructLayout(LayoutKind.Sequential)]
        public struct KjarniIndexStats
        {
            public UIntPtr DocumentsIndexed;
            public UIntPtr ChunksCreated;
            public UIntPtr Dimension;
            public ulong SizeBytes;
            public UIntPtr FilesProcessed;
            public UIntPtr FilesSkipped;
            public ulong ElapsedMs;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct KjarniIndexInfo
        {
            public IntPtr Path;
            public UIntPtr DocumentCount;
            public UIntPtr SegmentCount;
            public UIntPtr Dimension;
            public ulong SizeBytes;
            public IntPtr EmbeddingModel;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct KjarniIndexerConfig
        {
            public KjarniDevice Device;
            public IntPtr CacheDir;
            public IntPtr ModelName;
            public UIntPtr ChunkSize;
            public UIntPtr ChunkOverlap;
            public UIntPtr BatchSize;
            public IntPtr Extensions;
            public IntPtr ExcludePatterns;
            public int Recursive;
            public int IncludeHidden;
            public UIntPtr MaxFileSize;
            public int Quiet;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct KjarniProgress
        {
            public KjarniProgressStage Stage;
            public UIntPtr Current;
            public UIntPtr Total;
            public IntPtr Message;
        }

        // Progress callback delegate
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void KjarniProgressCallback(KjarniProgress progress, IntPtr userData);

        [StructLayout(LayoutKind.Sequential)]
        public struct KjarniSearchResult
        {
            public float Score;
            public UIntPtr DocumentId;
            public IntPtr Text;
            public IntPtr MetadataJson;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct KjarniSearchResults
        {
            public IntPtr Results;
            public UIntPtr Len;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct KjarniSearchOptions
        {
            public int Mode;
            public UIntPtr TopK;
            public int UseReranker;
            public float Threshold;
            public IntPtr SourcePattern;
            public IntPtr FilterKey;
            public IntPtr FilterValue;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct KjarniSearcherConfig
        {
            public KjarniDevice Device;
            public IntPtr CacheDir;
            public IntPtr ModelName;
            public IntPtr RerankModel;
            public KjarniSearchMode DefaultMode;
            public UIntPtr DefaultTopK;
            public int Quiet;
        }

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr kjarni_last_error_message();

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void kjarni_clear_error();

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr kjarni_error_name(KjarniErrorCode err);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniErrorCode kjarni_init();

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr kjarni_version();

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void kjarni_float_array_free(KjarniFloatArray arr);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void kjarni_float_2d_array_free(KjarniFloat2DArray arr);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void kjarni_string_free(IntPtr s);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void kjarni_class_results_free(KjarniClassResults results);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void kjarni_rerank_results_free(KjarniRerankResults results);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void kjarni_search_results_free(KjarniSearchResults results);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void kjarni_index_info_free(KjarniIndexInfo info);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr kjarni_cancel_token_new();

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void kjarni_cancel_token_cancel(IntPtr token);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool kjarni_cancel_token_is_cancelled(IntPtr token);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void kjarni_cancel_token_reset(IntPtr token);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void kjarni_cancel_token_free(IntPtr token);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniEmbedderConfig kjarni_embedder_config_default();

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniErrorCode kjarni_embedder_new(
            ref KjarniEmbedderConfig config,
            out IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void kjarni_embedder_free(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniErrorCode kjarni_embedder_encode(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string text,
            out KjarniFloatArray result);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniErrorCode kjarni_embedder_encode_batch(
            IntPtr handle,
            IntPtr[] texts,
            UIntPtr numTexts,
            out KjarniFloat2DArray result);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniErrorCode kjarni_embedder_similarity(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string text1,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string text2,
            out float result);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern UIntPtr kjarni_embedder_dim(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniClassifierConfig kjarni_classifier_config_default();

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniErrorCode kjarni_classifier_new(
            ref KjarniClassifierConfig config,
            out IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void kjarni_classifier_free(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniErrorCode kjarni_classifier_classify(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string text,
            out KjarniClassResults result);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern UIntPtr kjarni_classifier_num_labels(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniRerankerConfig kjarni_reranker_config_default();

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniErrorCode kjarni_reranker_new(
            ref KjarniRerankerConfig config,
            out IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void kjarni_reranker_free(IntPtr handle);

        /// Logit to 0..1. One implementation, in the engine, so the saturation
        /// branch cannot be omitted by a binding.
        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern float kjarni_sigmoid(float x);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniErrorCode kjarni_reranker_score(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string query,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string document,
            out float result);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniErrorCode kjarni_reranker_rerank(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string query,
            IntPtr[] documents,
            UIntPtr numDocs,
            out KjarniRerankResults result);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniErrorCode kjarni_reranker_rerank_top_k(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string query,
            IntPtr[] documents,
            UIntPtr numDocs,
            UIntPtr topK,
            out KjarniRerankResults result);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniIndexerConfig kjarni_indexer_config_default();

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniErrorCode kjarni_indexer_new(
            ref KjarniIndexerConfig config,
            out IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void kjarni_indexer_free(IntPtr handle);

[DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
public static extern KjarniErrorCode kjarni_indexer_create(
    IntPtr handle,
    [MarshalAs(UnmanagedType.LPUTF8Str)] string indexPath,
    IntPtr[] inputs,
    UIntPtr numInputs,
    int force,
    out KjarniIndexStats stats);

[DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
public static extern KjarniErrorCode kjarni_indexer_create_with_callback(
    IntPtr handle,
    [MarshalAs(UnmanagedType.LPUTF8Str)] string indexPath,
    IntPtr[] inputs,
    UIntPtr numInputs,
    int force,
    KjarniProgressCallback? progressCallback,
    IntPtr userData,
    IntPtr cancelToken,
    out KjarniIndexStats stats);

[DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
public static extern KjarniErrorCode kjarni_indexer_add(
    IntPtr handle,
    [MarshalAs(UnmanagedType.LPUTF8Str)] string indexPath,
    IntPtr[] inputs,
    UIntPtr numInputs,
    out UIntPtr documentsAdded);

[DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
public static extern KjarniErrorCode kjarni_indexer_add_with_callback(
    IntPtr handle,
    [MarshalAs(UnmanagedType.LPUTF8Str)] string indexPath,
    IntPtr[] inputs,
    UIntPtr numInputs,
    KjarniProgressCallback? progressCallback,
    IntPtr userData,
    IntPtr cancelToken,
    out UIntPtr documentsAdded);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniErrorCode kjarni_index_info(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string indexPath,
            out KjarniIndexInfo info);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniErrorCode kjarni_index_delete(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string indexPath);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern UIntPtr kjarni_indexer_model_name(IntPtr handle, IntPtr buf, UIntPtr buf_len);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern UIntPtr kjarni_indexer_dimension(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern UIntPtr kjarni_indexer_chunk_size(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniSearcherConfig kjarni_searcher_config_default();

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniSearchOptions kjarni_search_options_default();

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniErrorCode kjarni_searcher_new(
            ref KjarniSearcherConfig config,
            out IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void kjarni_searcher_free(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniErrorCode kjarni_searcher_search(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string indexPath,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string query,
            out KjarniSearchResults results);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniErrorCode kjarni_searcher_search_with_options(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string indexPath,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string query,
            ref KjarniSearchOptions options,
            out KjarniSearchResults results);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniErrorCode kjarni_search_keywords(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string indexPath,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string query,
            UIntPtr topK,
            out KjarniSearchResults results);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool kjarni_searcher_has_reranker(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniSearchMode kjarni_searcher_default_mode(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern UIntPtr kjarni_searcher_default_top_k(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern UIntPtr kjarni_searcher_model_name(IntPtr handle, IntPtr buf, UIntPtr buf_len);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern UIntPtr kjarni_searcher_reranker_model(IntPtr handle, IntPtr buf, UIntPtr buf_len);

        [DllImport(LibName)]
        public static extern float kjarni_cosine_similarity(float[] a, float[] b, nuint len);



[StructLayout(LayoutKind.Sequential)]
        public struct KjarniChatConfig
        {
            public KjarniDevice Device;
            public IntPtr CacheDir;
            public IntPtr ModelName;
            public IntPtr ModelPath;
            public IntPtr SystemPrompt;
            public int Mode;
            public int Quiet;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct KjarniGenerationConfig
        {
            public float Temperature;
            public int TopK;
            public float TopP;
            public float MinP;
            public float RepetitionPenalty;
            public int MaxNewTokens;
            public int DoSample;
        }

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public delegate bool KjarniStreamCallback(IntPtr text, IntPtr userData);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniChatConfig kjarni_chat_config_default();

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniGenerationConfig kjarni_generation_config_default();

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniErrorCode kjarni_chat_new(
            ref KjarniChatConfig config,
            out IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void kjarni_chat_free(IntPtr handle);

        // Send with nullable gen_config (pass IntPtr.Zero for defaults)
        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "kjarni_chat_send")]
        public static extern KjarniErrorCode kjarni_chat_send(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string message,
            IntPtr genConfig,
            out IntPtr result);

        // Send with explicit gen_config
        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "kjarni_chat_send")]
        public static extern KjarniErrorCode kjarni_chat_send_config(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string message,
            ref KjarniGenerationConfig genConfig,
            out IntPtr result);

        // Stream with nullable gen_config
        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "kjarni_chat_stream")]
        public static extern KjarniErrorCode kjarni_chat_stream(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string message,
            IntPtr genConfig,
            KjarniStreamCallback callback,
            IntPtr userData,
            IntPtr cancelToken);

        // Stream with explicit gen_config
        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "kjarni_chat_stream")]
        public static extern KjarniErrorCode kjarni_chat_stream_config(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string message,
            ref KjarniGenerationConfig genConfig,
            KjarniStreamCallback callback,
            IntPtr userData,
            IntPtr cancelToken);

        // Send with history
        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniErrorCode kjarni_chat_send_with_history(
            IntPtr handle,
            int[]? roles,
            IntPtr[]? contents,
            UIntPtr historyLen,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string message,
            IntPtr genConfig,
            out IntPtr result);

        // Send with history and an explicit generation config
        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "kjarni_chat_send_with_history")]
        public static extern KjarniErrorCode kjarni_chat_send_with_history_config(
            IntPtr handle,
            int[]? roles,
            IntPtr[]? contents,
            UIntPtr historyLen,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string message,
            ref KjarniGenerationConfig genConfig,
            out IntPtr result);

        // Stream with history
        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniErrorCode kjarni_chat_stream_with_history(
            IntPtr handle,
            int[]? roles,
            IntPtr[]? contents,
            UIntPtr historyLen,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string message,
            IntPtr genConfig,
            KjarniStreamCallback callback,
            IntPtr userData,
            IntPtr cancelToken);

        // Stream with history and an explicit generation config
        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "kjarni_chat_stream_with_history")]
        public static extern KjarniErrorCode kjarni_chat_stream_with_history_config(
            IntPtr handle,
            int[]? roles,
            IntPtr[]? contents,
            UIntPtr historyLen,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string message,
            ref KjarniGenerationConfig genConfig,
            KjarniStreamCallback callback,
            IntPtr userData,
            IntPtr cancelToken);

        // Conversation
        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniErrorCode kjarni_chat_conversation_new(
            IntPtr chat,
            out IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void kjarni_chat_conversation_free(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "kjarni_chat_conversation_send")]
        public static extern KjarniErrorCode kjarni_chat_conversation_send(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string message,
            IntPtr genConfig,
            out IntPtr result);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "kjarni_chat_conversation_send")]
        public static extern KjarniErrorCode kjarni_chat_conversation_send_config(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string message,
            ref KjarniGenerationConfig genConfig,
            out IntPtr result);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "kjarni_chat_conversation_stream")]
        public static extern KjarniErrorCode kjarni_chat_conversation_stream(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string message,
            IntPtr genConfig,
            KjarniStreamCallback callback,
            IntPtr userData,
            IntPtr cancelToken);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "kjarni_chat_conversation_stream")]
        public static extern KjarniErrorCode kjarni_chat_conversation_stream_config(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string message,
            ref KjarniGenerationConfig genConfig,
            KjarniStreamCallback callback,
            IntPtr userData,
            IntPtr cancelToken);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern UIntPtr kjarni_chat_conversation_len(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void kjarni_chat_conversation_clear(IntPtr handle, int keepSystem);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern UIntPtr kjarni_chat_model_name(IntPtr handle, IntPtr buf, UIntPtr bufLen);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern UIntPtr kjarni_chat_context_size(IntPtr handle);

        public static void CheckError(KjarniErrorCode err)
        {
            if (err != KjarniErrorCode.Ok)
            {
                var msgPtr = kjarni_last_error_message();
                var msg = msgPtr != IntPtr.Zero
                    ? Marshal.PtrToStringUTF8(msgPtr)
                    : $"Error: {err}";
                throw new KjarniException(msg ?? "Unknown error", err);
            }
        }

        public static string GetVersion()
        {
            var ptr = kjarni_version();
            return ptr != IntPtr.Zero ? Marshal.PtrToStringUTF8(ptr) ?? "unknown" : "unknown";
        }
    }
internal class Utf8StringArray : IDisposable
{
    private readonly IntPtr[] _ptrs;
    private bool _disposed;

    public Utf8StringArray(IEnumerable<string> strings)
    {
        var list = strings is IList<string> l ? l : new List<string>(strings);
        _ptrs = new IntPtr[list.Count];
        for (int i = 0; i < list.Count; i++)
        {
            var bytes = System.Text.Encoding.UTF8.GetBytes(list[i] + '\0');
            _ptrs[i] = Marshal.AllocHGlobal(bytes.Length);
            Marshal.Copy(bytes, 0, _ptrs[i], bytes.Length);
        }
    }

    public IntPtr[] Pointers => _ptrs;
    public int Length => _ptrs.Length;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var ptr in _ptrs)
        {
            if (ptr != IntPtr.Zero)
                Marshal.FreeHGlobal(ptr);
        }
    }
}

    public class Kjarni
    {
        public static void Initialize()
        {
            Native.CheckError(Native.kjarni_init());
        }

        public static string GetVersion()
        {
            return Native.GetVersion();
        }
    }

    public class KjarniException : Exception
    {
        public KjarniErrorCode ErrorCode { get; }

        public KjarniException(string message, KjarniErrorCode code)
            : base(message)
        {
            ErrorCode = code;
        }
    }
}