use axum::extract::State;
use axum::http::HeaderValue;
use axum::routing::{get, post};
use axum::Router;
use clap::Parser;
use reqwest::Method;
use tower_http::cors::{Any, CorsLayer};
use tracing::info;

#[derive(Clone, Debug, Default)]
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

async fn list_models(
    State(state): State<AppState>,
) {
    //
}

async fn generate(
    State(state): State<AppState>,
) {
    //
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

    let mut state = AppState::default();
    let router = Router::new()
        .route("/models", get(list_models))
        .route("/generate", post(generate))
        .route("/generate_stream", post(generate_stream))
        .route("/v1/chat/completions", post(chat_completion))
        .route("/lora/adapters", get(list_lora_adapters))
        .with_state(state)
        .layer(
            CorsLayer::new()
                .allow_origin("*".parse::<HeaderValue>().unwrap())
                .allow_headers(Any)
                .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000").await.unwrap();
    info!("listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, router).await.unwrap();
}
