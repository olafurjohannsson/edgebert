using System;
using System.Runtime.InteropServices;

namespace Kjarni
{
    /// <summary>
    /// Single rerank result
    /// </summary>
    public readonly struct RerankResult
    {
        public int Index { get; }

        /// <summary>
        /// The cross-encoder's raw logit, matching what PyTorch returns. Roughly
        /// -11 to +11. See <see cref="Probability"/> for a 0..1 scale.
        /// </summary>
        public float Score { get; }

        public string Document { get; }

        /// <summary>
        /// <see cref="Score"/> on a 0..1 scale. Ordering is identical; the
        /// transform is monotonic.
        /// </summary>
        public float Probability => Sigmoid(Score);

        /// <summary>
        /// Logistic squash, for putting a logit on a 0..1 scale.
        ///
        /// Calls into the engine rather than reimplementing it. The formula is one
        /// line, but it needs a branch: the naive form overflows to NaN for large
        /// negative inputs, and five bindings each writing their own is five
        /// chances to omit it.
        /// </summary>
        public static float Sigmoid(float x) => Native.kjarni_sigmoid(x);

        internal RerankResult(int index, float score, string document)
        {
            Index = index;
            Score = score;
            Document = document;
        }
    }

    /// <summary>
    /// Text reranking model using cross-encoders.
    /// </summary>
    public class Reranker : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;

        /// <summary>
        /// Create a new Reranker with default settings.
        /// </summary>
        public Reranker(
            string? model = null,
            string device = "cpu",
            string? cacheDir = null,
            bool quiet = false)
        {
            var config = Native.kjarni_reranker_config_default();
            config.Device = device == "gpu" ? KjarniDevice.Gpu : KjarniDevice.Cpu;
            config.Quiet = quiet ? 1 : 0;

            IntPtr modelNamePtr = IntPtr.Zero;
            IntPtr cacheDirPtr = IntPtr.Zero;

            try
            {
                if (model != null)
                {
                    modelNamePtr = Marshal.StringToCoTaskMemUTF8(model);
                    config.ModelName = modelNamePtr;
                }

                if (cacheDir != null)
                {
                    cacheDirPtr = Marshal.StringToCoTaskMemUTF8(cacheDir);
                    config.CacheDir = cacheDirPtr;
                }

                var err = Native.kjarni_reranker_new(ref config, out _handle);
                Native.CheckError(err);
            }
            finally
            {
                if (modelNamePtr != IntPtr.Zero) Marshal.FreeCoTaskMem(modelNamePtr);
                if (cacheDirPtr != IntPtr.Zero) Marshal.FreeCoTaskMem(cacheDirPtr);
            }
        }

        /// <summary>
        /// Score a single query-document pair.
        /// </summary>
        public float Score(string query, string document)
        {
            ThrowIfDisposed();

            var err = Native.kjarni_reranker_score(_handle, query, document, out var result);
            Native.CheckError(err);

            return result;
        }

        /// <summary>
        /// Rerank documents by relevance to query.
        /// </summary>
        public RerankResult[] Rerank(string query, string[] documents)
        {
            ThrowIfDisposed();

            if (documents.Length == 0) return Array.Empty<RerankResult>();

            using var utf8Docs = new Utf8StringArray(documents);

            var err = Native.kjarni_reranker_rerank(
                _handle, query, utf8Docs.Pointers, (UIntPtr)utf8Docs.Length, out var results);
            Native.CheckError(err);

            try
            {
                var rawResults = results.ToArray();
                var output = new RerankResult[rawResults.Length];
                for (int i = 0; i < rawResults.Length; i++)
                {
                    output[i] = new RerankResult(
                        rawResults[i].Index,
                        rawResults[i].Score,
                        documents[rawResults[i].Index]
                    );
                }
                return output;
            }
            finally
            {
                Native.kjarni_rerank_results_free(results);
            }
        }

        /// <summary>
        /// Rerank and return top-k results.
        /// </summary>
        public RerankResult[] RerankTopK(string query, string[] documents, int k)
        {
            ThrowIfDisposed();

            if (documents.Length == 0) return Array.Empty<RerankResult>();

            using var utf8Docs = new Utf8StringArray(documents);

            var err = Native.kjarni_reranker_rerank_top_k(
                _handle, query, utf8Docs.Pointers, (UIntPtr)utf8Docs.Length, (UIntPtr)k, out var results);
            Native.CheckError(err);

            try
            {
                var rawResults = results.ToArray();
                var output = new RerankResult[rawResults.Length];
                for (int i = 0; i < rawResults.Length; i++)
                {
                    output[i] = new RerankResult(
                        rawResults[i].Index,
                        rawResults[i].Score,
                        documents[rawResults[i].Index]
                    );
                }
                return output;
            }
            finally
            {
                Native.kjarni_rerank_results_free(results);
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                if (_handle != IntPtr.Zero)
                {
                    Native.kjarni_reranker_free(_handle);
                    _handle = IntPtr.Zero;
                }
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }

        ~Reranker()
        {
            Dispose();
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(Reranker));
        }
    }
}