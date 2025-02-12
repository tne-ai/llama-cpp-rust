//! Simple inference server.
//!
//! # Example
//!
//! ```bash
//! cargo run --example server -- \
//!   --model-path ./Phi-3.5-mini-instruct-Q4_K_M.gguf
//! ```
use async_stream::{stream as async_stream, try_stream};
use axum::error_handling::HandleError;
use axum::extract::State;
use axum::http::{HeaderValue, Method, StatusCode};
use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Sse};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::builder::Str;
use clap::{arg, Parser};
use futures_util::{pin_mut, Stream, StreamExt};
use llama_cpp::{
    ChatMessage as LlamaChatMessage,
    FinishReason,
    GenerateStreamItem,
    GenerationParams,
    LlamaContext,
    LlamaContextParams,
    LlamaHandle,
    LlamaModel,
    LlamaTokenizer,
    LogLevel,
};
use miette::{IntoDiagnostic, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::SystemTime;
use tower_http::cors::{Any, CorsLayer};
use tracing::{error, info};
use uuid::uuid;

type ModelRegistry<Model> = HashMap<String, Arc<Model>>;

#[derive(Clone)]
struct AppState {
    model: Arc<LlamaModel>,
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    /// Server address
    #[arg(short, long, default_value = "127.0.0.1")]
    address: String,

    /// Server port
    #[arg(short, long, default_value_t = 8900)]
    port: u16,

    /// A path to GGUF model file.
    #[arg(long)]
    model_path: String,

    /// Enable verbose output to provide detailed logging and additional information during execution
    #[arg(long, default_value_t = false)]
    verbose: bool,
}

#[derive(Deserialize, Debug)]
struct GenerateParameters {
    do_sample: Option<bool>,
    frequency_penalty: Option<f32>,
    repetition_penalty: Option<f32>,
    max_new_tokens: Option<usize>,
    seed: Option<u64>,
    temperature: Option<f32>,
    top_k: Option<i32>,
    top_p: Option<f32>,
    top_n_tokens: Option<i32>,
    typical_p: Option<f32>,
}

#[derive(Serialize, Default, Debug)]
struct GenerateDetails {
    finish_reason: Option<String>,
    // generated_tokens: i32,
    // seed: i64,
}

#[derive(Deserialize, Debug)]
struct GenerateRequest {
    model: String,
    inputs: String,
    parameters: Option<GenerateParameters>,
    stream: Option<bool>,
}

#[derive(Serialize, Default)]
struct GenerateResponse {
    details: GenerateDetails,
    generated_text: String,
}

#[derive(Deserialize, Debug)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize, Debug)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: Option<bool>,
    max_tokens: Option<usize>,
    tok_k: Option<i32>,
    top_p: Option<f32>,
    temperature: Option<f32>,
    seed: Option<u64>,
    frequency_penalty: Option<f32>,
}

#[derive(Serialize, Debug)]
struct ChatCompletion {
    id: String,
    choices: Vec<Choice>,
    created: u64,
    model: String,
    system_finterprint: Option<String>,
    usage: Option<String>,
}

#[derive(Serialize, Debug)]
struct ChatCompletionMessage {
    content: Option<String>,
    role: Option<String>,
}


#[derive(Serialize, Debug)]
struct Choice {
    finish_reason: Option<String>,
    index: u64,
    message: ChatCompletionMessage,
}

#[derive(Serialize, Debug)]
struct ChatCompletionChunk {
    id: String,
    choices: Vec<ChoiceChunk>,
    created: u64,
    model: String,
    system_finterprint: Option<String>,
    usage: Option<String>,
}

#[derive(Serialize, Debug)]
struct ChoiceChunk {
    delta: ChoiceDelta,
    finish_reason: Option<String>,
    index: u64,
}

#[derive(Serialize, Debug)]
struct ChoiceDelta {
    content: Option<String>,
    role: Option<String>,
}


async fn generate(
    State(state): State<AppState>,
    Json(request): Json<GenerateRequest>,
) -> impl IntoResponse {
    let tokenizer = LlamaTokenizer::new();
    let messages = [
        LlamaChatMessage::new("user", request.inputs.as_str()),
    ];
    let prompt = tokenizer.apply_chat_template(state.model.as_ref(), &messages, true).expect("failed to apply chat template");
    let mut gen_params = GenerationParams::default();
    if let Some(params) = request.parameters.as_ref() {
        gen_params.top_p = params.top_p;
        gen_params.top_k = params.top_k;
        gen_params.do_sample = params.do_sample.unwrap_or(false);
        gen_params.seed = params.seed;
        gen_params.typical_p = params.typical_p;
        gen_params.max_new_tokens = params.max_new_tokens;
        gen_params.frequency_penalty = params.frequency_penalty;
        gen_params.repetition_penalty = params.repetition_penalty;
        gen_params.temperature = params.temperature;
    }

    if request.stream.unwrap_or(false) {
        let streamer = Box::pin(async_stream! {
            let stream = state.model.generate_stream(&messages, &gen_params);
            pin_mut!(stream);

            while let Some(Ok(chunk)) = stream.next().await {
                let data = GenerateResponse {
                    details: GenerateDetails {
                        finish_reason: chunk.details.finish_reason.map_or(None, |r| Some(r.as_str().to_owned())),
                    },
                    generated_text: chunk.generated_text.to_owned(),
                };
                let event: Result<Event, fmt::Error> = Ok(Event::default().json_data(data).unwrap());
                yield event;
            }
        });

        Sse::new(streamer)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        let output = state.model.generate(&messages, gen_params).expect("failed to generate");
        let response = GenerateResponse {
            details: GenerateDetails {
                finish_reason: Some("stop".to_owned()),
            },
            generated_text: output.generated_text,
        };

        Json(response).into_response()
    }
}

async fn chat_completion(
    State(state): State<AppState>,
    Json(payload): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let mut messages = Vec::<LlamaChatMessage>::with_capacity(payload.messages.len());
    for m in &payload.messages {
        messages.push(LlamaChatMessage::new(m.role.as_str(), m.content.as_str()));
    }

    let tokenizer = LlamaTokenizer::new();
    let prompt = tokenizer.apply_chat_template(state.model.as_ref(), &messages, true).expect("failed to apply chat template");

    let mut params = GenerationParams::default();
    params.top_k = payload.tok_k;
    params.top_p = payload.top_p;
    params.temperature = payload.temperature;
    params.seed = payload.seed;
    params.max_new_tokens = payload.max_tokens;
    params.frequency_penalty = payload.frequency_penalty;

    let id = uuid::Uuid::new_v4().to_string();
    let created = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap();
    let mut index = 0;

    if payload.stream.unwrap_or(false) {
        let streamer = Box::pin(async_stream! {
            let stream = state.model.generate_stream(&messages, &params);
            pin_mut!(stream);

            while let Some(Ok(chunk)) = stream.next().await {
                let data = ChatCompletionChunk {
                    id: id.to_owned(),
                    created: created.as_secs(),
                    model: payload.model.to_owned(),
                    choices: vec![
                        ChoiceChunk{
                            delta: ChoiceDelta{
                                content: Some(chunk.generated_text),
                                role: Some("assistant".to_owned()),
                            },
                            finish_reason: chunk.details.finish_reason.map(|reason| {
                                let finish_reason = match reason {
                                    FinishReason::Stop => "stop",
                                    FinishReason::MaxTokens => "length",
                                    _ => "",
                                };
                                finish_reason.to_owned()
                            }),
                            index,
                        }
                    ],
                    system_finterprint: None,
                    usage: None,
                };

                let event: Result<Event, fmt::Error> = Ok(Event::default().json_data(data).unwrap());
                yield event;

                index += 1;
            }
        });

        Sse::new(streamer)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        let output = state.model.generate(&messages, params).expect("failed to generate");
        let response = ChatCompletion {
            id,
            created: created.as_secs(),
            model: payload.model.to_owned(),
            choices: vec![
                Choice {
                    message: ChatCompletionMessage {
                        role: Some("assistant".to_owned()),
                        content: Some(output.generated_text),
                    },
                    finish_reason: output.details.finish_reason.map_or(None, |reason| {
                        match reason {
                            FinishReason::MaxTokens => Some("length".to_owned()),
                            FinishReason::Stop => Some("stop".to_owned()),
                            _ => None,
                        }
                    }),
                    index: 0,
                },
            ],
            system_finterprint: None,
            usage: None,
        };

        Json(response).into_response()
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let handle = LlamaHandle::default();

    let args = Args::parse();
    if args.verbose {
        handle.set_log_level(LogLevel::Info)
    } else {
        handle.set_log_level(LogLevel::Error)
    }

    let mut ctx_params = LlamaContextParams::default();
    ctx_params.set_n_ctx(4096);
    ctx_params.set_n_batch(4096);

    let model = LlamaModel::from_file(args.model_path, None, Some(ctx_params))
        .expect("failed to load model");

    let mut state = AppState {
        model: Arc::new(model),
    };

    let router = Router::new()
        .route("/generate", post(generate))
        .route("/v1/chat/completions", post(chat_completion))
        .with_state(state)
        .layer(
            CorsLayer::new()
                .allow_origin("*".parse::<HeaderValue>().unwrap())
                .allow_headers(Any)
                .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        );

    let addr = format!("{}:{}", args.address, args.port);
    let listener = tokio::net::TcpListener::bind(addr).await.expect("failed to bind");
    info!("listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, router).await.unwrap();
}
