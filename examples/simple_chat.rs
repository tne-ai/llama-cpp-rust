//! Simple Chat with Llama.cpp
//!
//! # Example
//!
//! ```bash
//! cargo run --example chat --features "huggingface,stream" -- \
//!   --hf-repo microsoft/Phi-3-mini-4k-instruct-gguf \
//!   --hf-model-path Phi-3-mini-4k-instruct-q4.gguf \
//!   --top-p 0.9 \
//!   --temperature 0.8
//! ```

use clap::Parser;
use llama_cpp::*;
use miette::{bail, IntoDiagnostic, Result};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::io::Write;
use std::{fs, io};

/// Simple chat with Llama.cpp API
#[derive(Parser, Default, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// If set load a model from the given path.
    #[arg(long)]
    model_path: Option<String>,

    /// A 🤗Hub repoistory id where to download a model from.
    #[arg(long)]
    hf_repo: Option<String>,

    /// A path to the GGUF model in the 🤗 Hub model repository.
    #[arg(long)]
    hf_model_path: Option<String>,

    /// A Git revision to download in the 🤗 Hub model repository.
    #[arg(long)]
    hf_model_revision: Option<String>,

    /// System prompt.
    #[arg(long, default_value = "You are a helpful AI assistant.")]
    system_prompt: String,

    /// Loads a system prompt from the given file path.
    #[arg(long)]
    system_prompt_file: Option<String>,

    /// Enable Top-K sampling.
    #[arg(long)]
    top_k: Option<i32>,

    /// Enable Nucleus sampling.
    #[arg(long)]
    top_p: Option<f32>,

    /// Enable Minimum P sampling.
    #[arg(long)]
    min_p: Option<f32>,

    /// The value used to module the next token probabilities.
    #[arg(long, default_value_t = 1.0)]
    temperature: f32,

    // If set to true, this parameter enables decoding strategies such as Top-K sampling and nucleus
    // sampling.
    #[arg(long, default_value_t = false)]
    do_sample: bool,

    /// The number of max tokens
    /// Defaults to 2048.
    #[arg(long, default_value_t = 2048)]
    max_new_tokens: usize,

    // Seed used by the sampler.
    #[arg(long, default_value_t = 42)]
    seed: u32,

    /// Output verbose outputs.
    #[arg(short, action, default_value_t = false)]
    verbose: bool,
}

struct PrintTextStream {
    output: String,
}

impl PrintTextStream {
    fn default() -> Self {
        PrintTextStream { output: String::new() }
    }

    fn output(&self) -> String {
        self.output.to_owned()
    }
}

impl TextStreamer for PrintTextStream {
    fn generated(&self, result: llama_cpp::Result<TextGenerationStream>) {
        match result {
            Ok(result) => {
                if result.details.finish_reason == FinishReason::None {
                    print!("{}", result.generated_text.as_str());
                    io::stdout().flush().unwrap();
                }
            }
            Err(err) => {
                eprintln!("{}", err);
            }
        }
    }
}

// https://github.com/ggerganov/llama.cpp/blob/master/examples/simple-chat/simple-chat.cpp
#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    if args.verbose {
        llama_set_log_level(LogLevel::Debug);
    } else {
        llama_set_log_level(LogLevel::Error);
    }

    let _handle = LlamaHandle::default();

    let model =
        if let Some(model_path) = args.model_path {
            println!("Loading the model {}", model_path);
            LlamaModel::from_file(model_path, None, None)?
        } else if let Some(repo_id) = args.hf_repo {
            if let Some(model_path) = args.hf_model_path {
                println!("Loading the Hugging Face model {}/{}", repo_id, model_path);
                LlamaModel::from_hf(repo_id, model_path, None, None, None).await?
            } else {
                bail!("--hf-model-path must be set")
            }
        } else {
            bail!("--model-path or --hf-repo must be set")
        };

    // Evaluate prompt and generate a response
    let mut messages: Vec<ChatMessage> = vec![];
    if let Some(system_prompt_file) = args.system_prompt_file {
        let system_prompt = fs::read_to_string(system_prompt_file).into_diagnostic()?;
        messages.push(ChatMessage::new("system", system_prompt.as_str()));
    } else {
        messages.push(ChatMessage::new("system", args.system_prompt.as_str()));
    }

    // Configure auto-regressive text generation parameters
    let mut params = GenerationParams::default();
    params.max_new_tokens = Some(args.max_new_tokens);
    params.temperature = Some(args.temperature);
    params.min_p = args.min_p;
    params.top_k = args.top_k;
    params.top_p = args.top_p;
    params.seed = Some(args.seed);

    let mut rl = DefaultEditor::new().into_diagnostic()?;

    loop {
        let readline = rl.readline(">> ");
        match readline {
            Ok(text) => {
                messages.push(ChatMessage::new("user", &text));

                let streamer = PrintTextStream::default();
                model.generate_stream(&messages, &params, &streamer)?;

                messages.push(ChatMessage::new("assistant", streamer.output().as_str()));
                println!();
            }
            Err(ReadlineError::Interrupted) => {
                println!("Use Ctrl+d to exit.");
                continue;
            }
            Err(ReadlineError::Eof) => {
                break;
            }
            Err(err) => {
                eprintln!("Error: {:?}", err);
            }
        }
    }

    Ok(())
}
