use async_stream::stream as async_stream;
use axum::error_handling::HandleError;
use axum::extract::State;
use axum::http::{HeaderValue, StatusCode};
use axum::response::sse::KeepAlive;
use axum::response::{IntoResponse, Sse};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::builder::Str;
use clap::Parser;
use futures_util::Stream;
use llama_cpp::{llama_set_log_level, ChatMessage, GenerateStreamItem, GenerationParams, LlamaHandle, LlamaModel, LlamaTokenizer, LogLevel, TextStreamer};
use miette::{IntoDiagnostic, Result};
use reqwest::Method;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tracing::info;

type ModelRegistry<Model> = HashMap<String, Arc<Model>>;

#[derive(Clone, Default)]
struct AppState {}

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    /// Server address
    #[arg(short, long, default_value = "127.0.0.1")]
    address: String,

    /// Server port
    #[arg(short, long, default_value_t = 8900)]
    port: u16,
}

// -------------------------------------------------------------------
// Types definition
// -------------------------------------------------------------------

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
    finish_reason: String,
    generated_tokens: i32,
    seed: i64,
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
    generated_text: String,
}

async fn list_models(
    State(state): State<AppState>,
) {
    //
}


async fn generate(
    State(state): State<AppState>,
    Json(request): Json<GenerateRequest>,
) -> impl IntoResponse {
    let model_path = "./Phi-3-mini-4k-instruct-q4.gguf";
    let model = LlamaModel::from_file(model_path, None, None).expect("failed to load model");
    let tokenizer = LlamaTokenizer::new();
    let messages = [
        ChatMessage::new("user", request.inputs.as_str()),
    ];
    let prompt = tokenizer.apply_chat_template(&model, &messages, true).expect("failed to apply chat template");
    let mut gen_params = GenerationParams::default();
    if let Some(params) = request.parameters.as_ref() {
        gen_params.top_p = params.top_p;
        gen_params.top_k = params.top_k;
        gen_params.do_sample = params.do_sample.unwrap_or(false);
        gen_params.seed = params.seed;
        gen_params.typical_p = params.typical_p;
        gen_params.max_new_tokens = params.max_new_tokens;
    }

    if request.stream.unwrap_or(false) {
        // let streamer = Box::pin(async_stream! {
        //     yield "OK";
        // });
        //
        // // model.generate_stream(&messages, &gen_params).expect("failed to generate stream");
        //
        // Sse::new(streamer)
        //     .keep_alive(KeepAlive::default())
        //     .into_response()
        "".into_response()
    } else {
        let output = model.generate(&messages, &gen_params).expect("failed to generate");
        let response = GenerateResponse {
            generated_text: output.generated_text,
        };

        Json(response).into_response()
    }
}

async fn generate_stream(
    State(state): State<AppState>,
) {
    //
}

async fn chat_completion(
    State(state): State<AppState>,
) {}

async fn list_lora_adapters(
    State(state): State<AppState>,
) {
    //
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    llama_set_log_level(LogLevel::Error);

    let args = Args::parse();

    let _handle = LlamaHandle::default();

    let mut state = AppState::default();
    let router = Router::new()
        // .route("/models", get(list_models))
        .route("/generate", post(generate))
        // .route("/generate_stream", post(generate_stream))
        // .route("/v1/chat/completions", post(chat_completion))
        // .route("/lora/adapters", get(list_lora_adapters))
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
