mod commands;

use anyhow::Result;
use clap::Parser;

use kjarni_cli::{Cli, Commands, verbosity_to_log_level};

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // CLI verbosity flag takes precedence, otherwise use RUST_LOG env var
    if cli.verbose > 0 {
        let log_level = verbosity_to_log_level(cli.verbose);
        unsafe {
            std::env::set_var("RUST_LOG", log_level);
        }
    }

    env_logger::init();

    match cli.command {
        Commands::Model { action } => commands::model::run(action).await,

        Commands::Inspect { path } => commands::inspect::run(path.as_ref()).await,

        Commands::Generate {
            prompt,
            model,
            model_path,
            max_tokens,
            temperature,
            top_k,
            top_p,
            min_p,
            repetition_penalty,
            greedy,
            gpu,
            no_stream,
            quiet,
            draft,
            draft_tokens,
        } => {
            commands::generate::run(
                prompt.as_deref(),
                &model,
                model_path.as_deref(),
                max_tokens,
                temperature,
                top_k,
                top_p,
                min_p,
                repetition_penalty,
                greedy,
                gpu,
                no_stream,
                quiet,
                draft.as_deref(),
                draft_tokens,
            )
            .await
        }

        Commands::Embed {
            input,
            model,
            model_path,
            format,
            normalize,
            pooling,
            gpu,
            quiet,
        } => {
            commands::embed::run(
                input.as_deref(),
                &model,
                model_path.as_deref(),
                &format,
                normalize,
                &pooling,
                gpu,
                quiet,
            )
            .await
        }

        Commands::Summarize {
            input,
            model,
            model_path,
            min_length,
            max_length,
            num_beams,
            length_penalty,
            no_repeat_ngram,
            greedy,
            no_stream,
            gpu,
            quiet,
        } => {
            commands::summarize::run(
                input.as_deref(),
                &model,
                model_path.as_deref(),
                min_length,
                max_length,
                num_beams,
                length_penalty,
                no_repeat_ngram,
                greedy,
                no_stream,
                gpu,
                quiet,
            )
            .await
        }

        Commands::Translate {
            input,
            model,
            model_path,
            src,
            dst,
            max_length,
            num_beams,
            length_penalty,
            no_repeat_ngram,
            greedy,
            no_stream,
            gpu,
            quiet,
        } => {
            commands::translate::run(
                input.as_deref(),
                &model,
                model_path.as_deref(),
                src.as_deref(),
                dst.as_deref(),
                max_length,
                num_beams,
                length_penalty,
                no_repeat_ngram,
                greedy,
                no_stream,
                gpu,
                quiet,
            )
            .await
        }

        Commands::Transcribe {
            file,
            model,
            model_path,
            language,
            translate,
            timestamps,
            max_tokens,
            no_stream,
            gpu,
            quiet,
        } => {
            commands::transcribe::run(
                &file,
                &model,
                model_path.as_deref(),
                language.as_deref(),
                translate,
                timestamps,
                max_tokens,
                no_stream,
                gpu,
                quiet,
            )
            .await
        }

        Commands::Classify {
            input,
            model,
            model_path,
            labels,
            top_k,
            threshold,
            max_length,
            batch_size,
            multi_label,
            format,
            gpu,
            dtype,
            quiet,
        } => {
            commands::classify::run(
                &input,
                &model,
                model_path.as_deref(),
                labels.as_deref(),
                top_k,
                threshold,
                max_length,
                batch_size,
                multi_label,
                &format,
                gpu,
                dtype.as_deref(),
                quiet,
            )
            .await
        }

        Commands::Rerank {
            query,
            documents,
            model,
            model_path,
            top_k,
            format,
            gpu,
            quiet,
        } => {
            commands::rerank::run(
                &query,
                &documents,
                &model,
                model_path.as_deref(),
                top_k,
                &format,
                gpu,
                quiet,
            )
            .await
        }

        Commands::Chat {
            model,
            model_path,
            system,
            temperature,
            max_tokens,
            gpu,
            quiet,
            draft,
            draft_tokens,
        } => {
            commands::chat::run(
                &model,
                model_path.as_deref(),
                system.as_deref(),
                temperature,
                max_tokens,
                gpu,
                quiet,
                draft.as_deref(),
                draft_tokens,
            )
            .await
        }

        Commands::Index { action } => commands::index::run(action).await,

        Commands::Search {
            index_path,
            query,
            top_k,
            mode,
            model,
            rerank_model,
            format,
            gpu,
            quiet,
        } => {
            commands::search::run(
                &index_path,
                &query,
                top_k,
                &mode,
                &model,
                rerank_model.as_deref(),
                &format,
                gpu,
                quiet,
            )
            .await
        }

        Commands::Similarity {
            text1,
            text2,
            model,
            gpu,
            quiet,
        } => commands::similarity::run(&text1, &text2, &model, gpu, quiet).await,
    }
}
