using System;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace Kjarni.Tests
{
    public class RerankerTests : IDisposable
    {
        private readonly Reranker _reranker;
        private readonly ITestOutputHelper _output;

        public RerankerTests(ITestOutputHelper output)
        {
            _output = output;
            _reranker = new Reranker(quiet: true);
        }

        public void Dispose() => _reranker.Dispose();

        [Theory]
        [InlineData(
            "What is machine learning?",
            "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            11.063f)]
        [InlineData(
            "What is machine learning?",
            "The weather today is sunny with a high of 72 degrees.",
            -11.091f)]
        [InlineData(
            "query",
            "document text",
            -10.537f)]
        [InlineData(
            "deep learning neural networks",
            "Neural networks learn hierarchical representations.",
            0.972f)]
        [InlineData(
            "deep learning neural networks",
            "Python is a programming language.",
            -11.327f)]
        [InlineData(
            "deep learning neural networks",
            "The cat sat on the mat.",
            -11.313f)]
        public void Score_ExactValues(string query, string document, float expected)
        {
            var score = _reranker.Score(query, document);
            _output.WriteLine($"\"{query}\" / \"{document}\": {score:F6}");

            // Raw cross-encoder logits, which is what torch returns: both
            // `transformers` and `sentence-transformers` give the unsquashed
            // value, and the checkpoint declares Identity as its activation.
            Assert.Equal(expected, score, 2);
        }

        [Fact]
        public void Score_RelevantHigherThanIrrelevant()
        {
            var relevant = _reranker.Score(
                "What is machine learning?",
                "Machine learning is a subset of artificial intelligence that enables systems to learn from data.");

            var irrelevant = _reranker.Score(
                "What is machine learning?",
                "The weather today is sunny with a high of 72 degrees.");

            _output.WriteLine($"Relevant: {relevant:F6}");
            _output.WriteLine($"Irrelevant: {irrelevant:F6}");

            Assert.Equal(11.063f, relevant, 2);
            Assert.Equal(-11.091f, irrelevant, 2);
            Assert.True(relevant > irrelevant);
        }

        [Fact]
        public void Score_IsDeterministic()
        {
            var a = _reranker.Score("query", "document text");
            var b = _reranker.Score("query", "document text");

            Assert.Equal(a, b);
        }

        [Fact]
        public void Rerank_ReturnsAllDocuments()
        {
            var docs = new[]
            {
                "Completely unrelated: baking cookies.",
                "Machine learning uses algorithms to learn from data.",
                "The stock market fluctuated wildly today.",
            };

            var results = _reranker.Rerank("What is machine learning?", docs);

            Assert.Equal(3, results.Length);
        }

        [Fact]
        public void Rerank_OrdersByRelevance()
        {
            var docs = new[]
            {
                "The weather is nice today.",
                "Machine learning is a branch of artificial intelligence.",
                "I need to buy groceries later.",
                "Deep learning uses neural networks with many layers.",
            };

            var results = _reranker.Rerank("What is machine learning?", docs);

            _output.WriteLine("Reranked order:");
            foreach (var r in results)
                _output.WriteLine($"  [{r.Index}] {r.Score:F6} {r.Document}");

            // ML-related docs should be top two
            Assert.Equal(1, results[0].Index); // "Machine learning is a branch..."
            Assert.Equal(3, results[1].Index); // "Deep learning uses neural networks..."
        }

        [Fact]
        public void Rerank_ScoresDescending()
        {
            var docs = new[]
            {
                "Python is a programming language.",
                "Neural networks learn hierarchical representations.",
                "The cat sat on the mat.",
            };

            var results = _reranker.Rerank("deep learning neural networks", docs);

            _output.WriteLine("Scores:");
            foreach (var r in results)
                _output.WriteLine($"  [{r.Index}] {r.Score:F6} {r.Document}");

            for (int i = 0; i < results.Length - 1; i++)
            {
                Assert.True(results[i].Score >= results[i + 1].Score,
                    $"Results not sorted: [{i}] {results[i].Score:F4} < [{i + 1}] {results[i + 1].Score:F4}");
            }

            // First result should be the neural networks doc
            Assert.Equal(1, results[0].Index);
        }

        [Fact]
        public void Rerank_DocumentFieldMatchesInput()
        {
            var docs = new[] { "First doc", "Second doc", "Third doc" };
            var results = _reranker.Rerank("query", docs);

            foreach (var r in results)
                Assert.Equal(docs[r.Index], r.Document);
        }

        [Fact]
        public void RerankTopK_ReturnsAtMostK()
        {
            var docs = new[]
            {
                "Doc one about ML.",
                "Doc two about weather.",
                "Doc three about cooking.",
                "Doc four about neural networks.",
                "Doc five about gardening.",
            };

            var results = _reranker.RerankTopK("machine learning", docs, k: 2);

            Assert.Equal(2, results.Length);
        }

        [Fact]
        public void RerankTopK_TopResultIsRelevant()
        {
            var docs = new[]
            {
                "Baking bread requires flour and water.",
                "Transformers revolutionized natural language processing.",
                "The sunset was beautiful yesterday evening.",
            };

            var results = _reranker.RerankTopK("NLP transformers attention", docs, k: 1);

            Assert.Single(results);
            Assert.Equal(1, results[0].Index);
            Assert.Equal(-5.181f, results[0].Score, 2);

            _output.WriteLine($"Top result: [{results[0].Index}] {results[0].Score:F6} {results[0].Document}");
        }

        [Fact]
        public void Rerank_EmptyArray_ReturnsEmpty()
        {
            var results = _reranker.Rerank("query", Array.Empty<string>());
            Assert.Empty(results);
        }

        [Fact]
        public void Rerank_SingleDocument()
        {
            var results = _reranker.Rerank("query", new[] { "Only document" });

            Assert.Single(results);
            Assert.Equal(0, results[0].Index);
            Assert.Equal("Only document", results[0].Document);
        }

        [Fact]
        public void Score_UnicodeText()
        {
            var score = _reranker.Score(
                "Que es el aprendizaje automatico?",
                "El aprendizaje automatico es una rama de la inteligencia artificial.");

            _output.WriteLine($"Spanish query-doc score: {score:F6}");
            Assert.False(float.IsNaN(score));
        }

        [Fact]
        public void Score_AfterDispose_Throws()
        {
            var reranker = new Reranker(quiet: true);
            reranker.Dispose();

            Assert.Throws<ObjectDisposedException>(() =>
                reranker.Score("query", "doc"));
        }

        [Fact]
        public void DoubleDispose_DoesNotThrow()
        {
            var reranker = new Reranker(quiet: true);
            reranker.Dispose();
            reranker.Dispose();
        }
        [Fact]
        public void Probability_IsOnAZeroToOneScale()
        {
            var results = _reranker.Rerank(
                "What is machine learning?",
                new[]
                {
                    "Machine learning is a subset of artificial intelligence.",
                    "The weather today is sunny.",
                });

            foreach (var r in results)
            {
                Assert.InRange(r.Probability, 0f, 1f);
            }

            // Monotonic, so ranking is unchanged by the transform.
            Assert.True(results[0].Score > results[1].Score);
            Assert.True(results[0].Probability > results[1].Probability);

            // sigmoid(0) is the midpoint, and the extremes saturate rather than
            // producing NaN.
            Assert.Equal(0.5f, RerankResult.Sigmoid(0f), 6);
            Assert.Equal(0f, RerankResult.Sigmoid(-1000f), 6);
            Assert.Equal(1f, RerankResult.Sigmoid(1000f), 6);
        }

    }
}