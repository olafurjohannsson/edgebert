//! Does prefix reuse actually reach the public API, and is it invisible?
//!
//! `PrefixCache` and `run_generation_loop_with_cache` were measured in
//! kjarni-models: a 2048 token prompt on Qwen2.5-0.5B costs 15.96s cold and 0.49s
//! with 1984 tokens already cached. But `Chat` and `Generator` both reach
//! generation through `DecoderGenerator::stream`, which passed `None` for the
//! cache regardless, so none of that saving was reachable without calling the
//! low-level loop by hand. These drive the public builder instead.
//!
//! Everything here is greedy. `temperature(0.0)` alone still samples: only
//! `greedy()` also sets `do_sample = false`, and under sampling two runs differ
//! for reasons that have nothing to do with the cache.

use kjarni::generator::Generator;
use std::time::Instant;

/// A long shared head is the point: a second prompt sharing it should prefill
/// only the differing tail. Short prompts hide the saving in noise.
fn shared_document() -> String {
    "The following is a reference document about geography. \
     Paris is the capital of France. Berlin is the capital of Germany. \
     Rome is the capital of Italy. Madrid is the capital of Spain. "
        .repeat(12)
}

async fn generator(prefix_cache: bool) -> Generator {
    let mut b = Generator::builder("qwen2.5-0.5b-instruct")
        .greedy()
        .max_tokens(8);
    if prefix_cache {
        b = b.prefix_cache(true);
    }
    b.build().await.expect("build generator")
}

/// The property that matters: a warm cache must be a pure speedup. Same prompt,
/// same greedy decode, same text, whether or not a previous call left a prefix
/// behind. A cache that changes the answer is worse than no cache.
#[tokio::test]
#[cfg_attr(
    debug_assertions,
    ignore = "decoder generation is orders of magnitude slower unoptimised; run with --release"
)]
async fn reuse_is_faster_and_does_not_change_the_answer() {
    let shared = shared_document();
    let first_prompt = format!("{shared}\nQ: What is the capital of France?\nA:");
    let second_prompt = format!("{shared}\nQ: What is the capital of Italy?\nA:");

    // Reference: the second prompt answered with no cache at all.
    let reference = generator(false)
        .await
        .generate(&second_prompt)
        .await
        .expect("reference generation");

    let cached = generator(true).await;

    let cold_start = Instant::now();
    cached
        .generate(&first_prompt)
        .await
        .expect("first generation");
    let cold = cold_start.elapsed();

    let warm_start = Instant::now();
    let warm_answer = cached
        .generate(&second_prompt)
        .await
        .expect("second generation");
    let warm = warm_start.elapsed();

    println!("cold {cold:?}, warm {warm:?}");
    println!("reference: {reference:?}\nwarm:      {warm_answer:?}");

    assert!(
        !reference.trim().is_empty(),
        "reference generation was empty"
    );
    assert_eq!(
        reference.trim(),
        warm_answer.trim(),
        "reusing a prefix changed the answer"
    );

    // Deliberately loose: this asserts the prefix was reused at all, not any
    // particular speedup, so a busy machine cannot make it flake.
    assert!(
        warm < cold,
        "warm call ({warm:?}) was not faster than cold ({cold:?}), prefix likely not reused"
    );
}

/// A cold cache must behave exactly like no cache, which is what makes the option
/// safe to turn on without re-validating output.
#[tokio::test]
#[cfg_attr(
    debug_assertions,
    ignore = "decoder generation is orders of magnitude slower unoptimised; run with --release"
)]
async fn a_cold_cache_matches_no_cache() {
    let prompt = format!(
        "{}\nQ: What is the capital of Spain?\nA:",
        shared_document()
    );

    let without = generator(false)
        .await
        .generate(&prompt)
        .await
        .expect("plain generation");
    let with = generator(true)
        .await
        .generate(&prompt)
        .await
        .expect("cached generation");

    assert_eq!(
        without.trim(),
        with.trim(),
        "prefix caching changed the answer on a cold cache"
    );
}

/// A prompt that shares nothing with what the cache holds must not inherit it.
#[tokio::test]
#[cfg_attr(
    debug_assertions,
    ignore = "decoder generation is orders of magnitude slower unoptimised; run with --release"
)]
async fn a_diverging_prompt_does_not_inherit_the_cache() {
    let warmed = format!(
        "{}\nQ: What is the capital of France?\nA:",
        shared_document()
    );
    let unrelated = "Q: Write one sentence about the ocean.\nA:";

    let reference = generator(false)
        .await
        .generate(unrelated)
        .await
        .expect("reference generation");

    let cached = generator(true).await;
    cached.generate(&warmed).await.expect("warm the cache");
    let after = cached
        .generate(unrelated)
        .await
        .expect("generation after warming");

    println!("reference: {reference:?}\nafter:     {after:?}");
    assert_eq!(
        reference.trim(),
        after.trim(),
        "an unrelated prompt inherited the warmed cache"
    );
}

/// The headline case: a conversation re-sends its whole transcript every turn, so
/// turn N normally pays to prefill everything said before it.
///
/// Measured here on Qwen2.5-0.5B with a long system prompt: without the cache all
/// three turns cost about 2.2s each, with it turn one costs the same and turns two
/// and three drop to roughly 210ms and 270ms. Answers are identical either way,
/// which is the part that has to stay true.
#[tokio::test]
#[cfg_attr(
    debug_assertions,
    ignore = "decoder generation is orders of magnitude slower unoptimised; run with --release"
)]
async fn a_conversation_only_prefills_the_new_turn() {
    use kjarni::chat::Chat;

    let system = "You are a helpful assistant. ".to_string()
        + &"Answer briefly and accurately using the reference below. \
            Paris is the capital of France. Berlin is the capital of Germany. \
            Rome is the capital of Italy. Madrid is the capital of Spain. "
            .repeat(10);

    let questions = [
        "What is the capital of France?",
        "And Germany?",
        "And Italy?",
    ];

    let mut answers: Vec<Vec<String>> = Vec::new();
    let mut turn_times: Vec<Vec<std::time::Duration>> = Vec::new();

    for cached in [false, true] {
        let mut builder = Chat::builder("qwen2.5-0.5b-instruct")
            .system(&system)
            .greedy()
            .max_tokens(12);
        if cached {
            builder = builder.prefix_cache(true);
        }
        let chat = builder.build().await.expect("build chat");
        let mut conversation = chat.conversation();

        let mut said = Vec::new();
        let mut took = Vec::new();
        for q in questions {
            let start = Instant::now();
            let answer = conversation.send(q).await.expect("send turn");
            took.push(start.elapsed());
            said.push(answer.trim().to_string());
        }
        println!("cached={cached}: {took:?} {said:?}");
        answers.push(said);
        turn_times.push(took);
    }

    // Reuse must be invisible in the output, whatever it does to the clock.
    assert_eq!(
        answers[0], answers[1],
        "prefix caching changed a conversation's answers"
    );

    // Turn one is cold either way; the saving is in the turns that follow. Loose
    // on purpose, so a busy machine cannot make this flake.
    let (plain, cached) = (&turn_times[0], &turn_times[1]);
    for turn in 1..questions.len() {
        assert!(
            cached[turn] < plain[turn],
            "turn {} was not faster with the cache ({:?} vs {:?})",
            turn + 1,
            cached[turn],
            plain[turn]
        );
    }
}
