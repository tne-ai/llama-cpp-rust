//! Rust bindings of llama.cpp

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

#[cfg(feature = "stream")]
use async_stream::try_stream;
#[cfg(feature = "stream")]
use futures_util::stream::Stream;

use hf_hub::{Repo, RepoType};
use log::{debug, error, info, warn};
use miette::{bail, miette, Diagnostic, IntoDiagnostic, Severity};
use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::fmt::{Debug, Formatter};
use std::path::Path;
use std::ptr::{null, null_mut};
use std::rc::Rc;
use std::{env, fmt};
use thiserror::Error;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

/// The type of result being returned by llama-cpp-rust.
pub type Result<T> = miette::Result<T>;

#[derive(Error, Diagnostic, Debug)]
pub enum Error {
    #[error("{0}")]
    #[diagnostic(severity(Warning))]
    InvalidInput(String),
    #[error("{0}")]
    #[diagnostic(severity(Error))]
    OutOfMemory(String),
    #[error("unknown error")]
    Other(String),
}

/// The type of token id.
pub type Token = llama_token;

/// The RAII type that initialize and deinitialize resources for Llama.cpp.
pub struct LlamaHandle {}

impl LlamaHandle {
    /// Returns the default handle.
    pub fn default() -> Self {
        unsafe {
            llama_backend_init();
        }
        LlamaHandle {}
    }
}

impl Drop for LlamaHandle {
    fn drop(&mut self) {
        unsafe {
            llama_backend_free();
        }
    }
}

pub enum RopeScalingType {
    None,
    Linear,
    Yarn,
}

impl Into<llama_rope_scaling_type> for RopeScalingType {
    fn into(self) -> llama_rope_scaling_type {
        match self {
            RopeScalingType::None => llama_rope_scaling_type_LLAMA_ROPE_SCALING_TYPE_NONE,
            RopeScalingType::Linear => llama_rope_scaling_type_LLAMA_ROPE_SCALING_TYPE_LINEAR,
            RopeScalingType::Yarn => llama_rope_scaling_type_LLAMA_ROPE_SCALING_TYPE_YARN,
        }
    }
}

pub enum PoolingType {
    None,
    Mean,
    CLS,
    Last,
    Rank,
}

impl Into<llama_pooling_type> for PoolingType {
    fn into(self) -> llama_pooling_type {
        match self {
            PoolingType::None => llama_pooling_type_LLAMA_POOLING_TYPE_NONE,
            PoolingType::Mean => llama_pooling_type_LLAMA_POOLING_TYPE_MEAN,
            PoolingType::CLS => llama_pooling_type_LLAMA_POOLING_TYPE_CLS,
            PoolingType::Last => llama_pooling_type_LLAMA_POOLING_TYPE_LAST,
            PoolingType::Rank => llama_pooling_type_LLAMA_POOLING_TYPE_RANK,
        }
    }
}

/// The type to specify how to split the model into multiple GPUs.
pub enum SplitMode {
    None,
    Layer,
    Row,
}

impl Into<llama_split_mode> for SplitMode {
    fn into(self) -> llama_split_mode {
        match self {
            SplitMode::None => llama_split_mode_LLAMA_SPLIT_MODE_NONE,
            SplitMode::Layer => llama_split_mode_LLAMA_SPLIT_MODE_LAYER,
            SplitMode::Row => llama_split_mode_LLAMA_SPLIT_MODE_ROW,
        }
    }
}

pub enum AttentionType {
    Causal,
    NonCausal,
}

impl Into<llama_attention_type> for AttentionType {
    fn into(self) -> llama_attention_type {
        match self {
            AttentionType::Causal => llama_attention_type_LLAMA_ATTENTION_TYPE_CAUSAL,
            AttentionType::NonCausal => llama_attention_type_LLAMA_ATTENTION_TYPE_NON_CAUSAL,
        }
    }
}

pub enum LogLevel {
    None,
    Debug,
    Info,
    Warn,
    Error,
}

impl Into<ggml_log_level> for LogLevel {
    fn into(self) -> ggml_log_level {
        match self {
            LogLevel::None => ggml_log_level_GGML_LOG_LEVEL_NONE,
            LogLevel::Debug => ggml_log_level_GGML_LOG_LEVEL_DEBUG,
            LogLevel::Info => ggml_log_level_GGML_LOG_LEVEL_INFO,
            LogLevel::Warn => ggml_log_level_GGML_LOG_LEVEL_WARN,
            LogLevel::Error => ggml_log_level_GGML_LOG_LEVEL_ERROR,
        }
    }
}


pub fn llama_set_log_level(log_level: LogLevel) {
    unsafe extern "C" fn print_error(level: ggml_log_level, text: *const c_char, _: *mut c_void) {
        if level >= ggml_log_level_GGML_LOG_LEVEL_ERROR {
            let log = unsafe { CStr::from_ptr(text).to_str().unwrap() };
            error!("{}", log);
        }
    }
    unsafe extern "C" fn print_warn(level: ggml_log_level, text: *const c_char, _: *mut c_void) {
        if level >= ggml_log_level_GGML_LOG_LEVEL_WARN {
            let log = unsafe { CStr::from_ptr(text).to_str().unwrap() };
            warn!("{}", log);
        }
    }
    unsafe extern "C" fn print_info(level: ggml_log_level, text: *const c_char, _: *mut c_void) {
        if level >= ggml_log_level_GGML_LOG_LEVEL_WARN {
            let log = unsafe { CStr::from_ptr(text).to_str().unwrap() };
            info!("{}", log);
        }
    }
    unsafe extern "C" fn print_debug(level: ggml_log_level, text: *const c_char, _: *mut c_void) {
        if level >= ggml_log_level_GGML_LOG_LEVEL_WARN {
            let log = unsafe { CStr::from_ptr(text).to_str().unwrap() };
            debug!("{}", log);
        }
    }

    unsafe {
        match log_level {
            LogLevel::Error => llama_log_set(Some(print_error), null_mut()),
            LogLevel::Warn => llama_log_set(Some(print_warn), null_mut()),
            LogLevel::Info => llama_log_set(Some(print_info), null_mut()),
            LogLevel::Debug => llama_log_set(Some(print_debug), null_mut()),
            _ => (),
        }
    }
}

#[derive(Debug, Default)]
pub struct GenerationParams {
    /// The number of highest probability vocabulary tokens to keep for top-k filtering.
    pub top_k: Option<i32>,
    /// Top-p parameter for Nucleus sampling. Only the probable tokens with probabilities that add up
    /// to `top_p` or higher are kept for generation.
    pub top_p: Option<f32>,
    /// The Minimum P sampling parameter.
    pub min_p: Option<f32>,
    /// The value used to module the next token probabilities.
    pub temperature: Option<f32>,
    /// The maximum numbers of tokens to generate, ignore the current number of tokens.
    /// Default to 512.
    pub max_new_tokens: Option<usize>,
    /// Locally Typical Sampling parameter.
    pub typical_p: Option<f32>,
    /// Whether or not to use sampling. Use greedy decoding otherwise.
    pub do_sample: bool,
    /// The parameter for freqeuncy penalty that helps us avoid using the same words too often.
    pub frequency_penalty: Option<f32>,
    /// The parameter for repetition penalty. 1.0 means no penalty.
    pub repetition_penalty: Option<f32>,
    /// The parameter for presence penalty that encourages using different words.
    pub presence_penalty: Option<f32>,
    ///
    pub penalize_newline: Option<bool>,
    /// Seed used by the sampler.
    pub seed: Option<u64>,
}

/// The enumerable type to identify the reason why the model stopped generating tokens.
#[derive(Eq, PartialEq)]
pub enum FinishReason {
    None,
    /// Token generation reached a natural stopping point or a configured stop sequence.
    Stop,
    /// Token generation reached the configured maximum output tokens.
    MaxTokens,
    /// Token generation stopped because the content potentially contains safety violations.
    Safety,
    /// All other reasons that stopped the token generation.
    Other,
}

impl FinishReason {
    /// Returns the string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            FinishReason::None => "",
            FinishReason::Stop => "stop",
            FinishReason::MaxTokens => "max_tokens",
            FinishReason::Safety => "safety",
            FinishReason::Other => "other",
        }
    }
}

impl Debug for FinishReason {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug)]
pub struct GenerateDetails {
    pub finish_reason: FinishReason,
}

#[derive(Debug)]
pub struct GenerateOutput {
    pub details: GenerateDetails,
    pub generated_text: String,
}

#[derive(Debug)]
pub struct GenerateStreamItem {
    ///
    pub details: GenerateDetails,
    /// An index of current generated item.
    pub index: usize,
    /// A chunk of generated text at the time.
    pub generated_text: String,
}

struct LoRA {
    name: String,
    adapter: *mut llama_lora_adapter,
    scale: f32,
}

unsafe impl Send for LoRA {}

///
pub struct LlamaModel {
    ctx: LlamaContext,
    pimpl: *mut llama_model,
    loras: HashMap<String, LoRA>,
}

impl LlamaModel {
    /// Load the model from Hugging Face Hub.
    ///
    /// If `HF_TOKEN` environment variable is set, HF token will be set to API requests.
    ///
    /// # Parameters
    ///
    /// * `repo`: The Hugging Face model repository ID to download from.
    /// * `model_path`: The GGUF model file to download.
    /// * `revision`: The Git revision to download from.
    /// * `model_params`: The parameters for loading a GGUF model.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp::{LlamaModel, LlamaModelParams};
    ///
    /// let repo_id = "microsoft/Phi-3-mini-4k-instruct-gguf";
    /// let model_path = "Phi-3-mini-4k-instruct-q4.gguf";
    /// let revision = None;
    /// let model_params = LlamaModelParams::default();
    ///
    /// let model = LlamaModel::from_hf(repo_id, model_path, revision, Some(model_params), None)?;
    /// ```
    #[cfg(feature = "huggingface")]
    pub async fn from_hf<S: AsRef<str>>(
        repo_id: S,
        model_path: S,
        revision: Option<S>,
        model_params: Option<LlamaModelParams>,
        ctx: Option<LlamaContext>,
    ) -> Result<Self> {
        let api = hf_hub::api::tokio::ApiBuilder::new()
            .with_token(env::var("HF_TOKEN").ok())
            .build()
            .into_diagnostic()?;
        let repo = if let Some(revision) = revision {
            Repo::with_revision(repo_id.as_ref().to_string(), RepoType::Model, revision.as_ref().to_string())
        } else {
            Repo::new(repo_id.as_ref().to_string(), RepoType::Model)
        };
        let model_path = api.repo(repo).get(model_path.as_ref()).await.into_diagnostic()?;

        LlamaModel::from_file(model_path.as_path(), model_params, ctx)
    }

    /// Load a model from the file.
    ///
    /// # Parameters
    ///
    /// * `model_path`: The path to GGUF model file.
    /// * `model_params`: The parameters for the model.
    /// * `ctx`: The context of Llama.cpp.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp::{LlamaModel, LlamaModelParams};
    ///
    /// // Load the Phi-3-4k model with default configurations.
    /// let model = LlamaModel::from_file("./Phi-3-mini-4k-instruct-q4.gguf", None, None);
    /// ```
    ///
    pub fn from_file<P: AsRef<Path>>(
        model_path: P,
        model_params: Option<LlamaModelParams>,
        ctx: Option<LlamaContext>,
    ) -> Result<Self> {
        let model_params = model_params.unwrap_or(LlamaModelParams::default());
        let c_path_model = CString::new(model_path.as_ref().to_str().unwrap()).unwrap();
        let model = unsafe {
            llama_load_model_from_file(c_path_model.as_ptr(), model_params.pimpl)
        };
        if model.is_null() {
            let msg = format!("unable to load the model `{}`", model_path.as_ref().to_str().unwrap());
            return Err(Error::InvalidInput(msg).into());
        }

        // Prepare the context from the given context or create from default params.
        let ctx = ctx.unwrap_or_else(|| {
            let ctx_params = unsafe { llama_context_default_params() };
            let ctx = unsafe { llama_new_context_with_model(model, ctx_params) };
            LlamaContext { pimpl: ctx }
        });

        let model = LlamaModel { ctx, pimpl: model, loras: HashMap::new() };

        Ok(model)
    }

    // Returns the description of the model type
    pub fn description(&self) -> String {
        let mut desc = [0 as c_char; 256];
        unsafe {
            llama_model_desc(self.pimpl, desc.as_mut_ptr(), desc.len());
            CStr::from_ptr(desc.as_ptr()).to_str().unwrap().to_string()
        }
    }

    /// Returns the total size of all tensors in the model in in bytes.
    pub fn model_size(&self) -> usize {
        unsafe { llama_model_size(self.pimpl) as usize }
    }

    /// Returns the total number of parameters in the model.
    pub fn num_params(&self) -> usize {
        unsafe { llama_model_n_params(self.pimpl) as usize }
    }

    /// Returns true if the model contains an encoder.
    pub fn has_encoder(&self) -> bool {
        unsafe { llama_model_has_encoder(self.pimpl) }
    }

    /// Returns true if the model contaisn a decoder.
    pub fn has_decoder(&self) -> bool {
        unsafe { llama_model_has_decoder(self.pimpl) }
    }

    /// Returns the beginning-of-sequence.
    pub fn bos_token(&self) -> Token {
        unsafe { llama_token_eos(self.pimpl) }
    }

    /// Returns the end-of-sequence.
    pub fn eos_token(&self) -> Token {
        unsafe { llama_token_eos(self.pimpl) }
    }

    /// Returns end-of-turn token.
    pub fn eot_token(&self) -> Token {
        unsafe { llama_token_eot(self.pimpl) }
    }

    /// Returns classification token.
    pub fn cls_token(&self) -> Token {
        unsafe { llama_token_cls(self.pimpl) }
    }

    /// Returns sequence separator token.
    pub fn sep_token(&self) -> Token {
        unsafe { llama_token_sep(self.pimpl) }
    }

    /// Returns next-line token.
    pub fn nl_tokne(&self) -> Token {
        unsafe { llama_token_nl(self.pimpl) }
    }

    /// Returns padding token.
    pub fn pad_token(&self) -> Token {
        unsafe { llama_token_pad(self.pimpl) }
    }

    pub fn add_bos_token(&self) -> bool {
        unsafe { llama_add_bos_token(self.pimpl) }
    }

    pub fn add_eos_token(&self) -> bool {
        unsafe { llama_add_eos_token(self.pimpl) }
    }

    pub fn n_vocab(&self) -> i32 {
        unsafe { llama_n_vocab(self.pimpl) }
    }

    /// Auto-regressive text generation.
    ///
    /// # Parameters
    ///
    /// * `messages`: The list of input messages.
    /// * `params`: The parameters of text generation.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use llama_cpp::{ChatMessage, GenerationParams, LlamaModel};
    ///
    /// // Load a Phi-3.5-mini-instruct model in the current directory.
    /// let model = LlamaModel::from_file("Phi-3.5-mini-instruct-Q4_K_M.gguf", None, None)?;
    /// let messages = &[
    ///     ChatMessage::new("user", "How to cook Pasta alla Trapanese?"),
    /// ];
    ///
    /// let generated = model.generate(&messages, GenerationParams::default())?;
    /// println!("{:?}", generated);
    /// ```
    ///
    pub fn generate(&self, messages: &[ChatMessage], params: GenerationParams) -> Result<GenerateOutput> {
        // Prepare the sampler from the given params
        let sampler = Sampler::new(self, &params);

        // Tokenize the prompt
        let tokenizer = LlamaTokenizer::new();
        let prompt = tokenizer.apply_chat_template(self, messages, true)?;
        let mut input_ids = tokenizer.encode(self, &prompt, true, true)?;

        // Prepare the single batch
        let mut batch = unsafe {
            llama_batch_get_one(input_ids.as_mut_ptr(), input_ids.len() as i32)
        };

        let lctx = self.ctx.pimpl;
        let mut output = String::new();
        let mut new_token_id = vec![0 as Token; 1];
        loop {
            // Check if we have enough space in the context
            let n_ctx = unsafe { llama_n_ctx(lctx) };
            let n_ctx_used = unsafe { llama_get_kv_cache_used_cells(lctx) };
            if (n_ctx_used + batch.n_tokens) as u32 > n_ctx {
                let result = GenerateOutput {
                    details: GenerateDetails {
                        finish_reason: FinishReason::MaxTokens,
                    },
                    generated_text: output,
                };

                return Ok(result);
            }

            // Decode a batch of tokens
            let ret = unsafe { llama_decode(lctx, batch) };
            if ret != 0 {
                bail!(
                    severity = Severity::Error,
                    "failed to decode input tokens",
                );
            }

            let new_token_id_ = unsafe { llama_sampler_sample(sampler.pimpl, lctx, -1) };
            let is_eog = unsafe { llama_token_is_eog(self.pimpl, new_token_id_) };
            if is_eog {
                break;
            }
            new_token_id[0] = new_token_id_;

            // Accepting the token pdates the internal state of certain samplers.
            unsafe { llama_sampler_accept(sampler.pimpl, new_token_id_) };

            // Convert the token to a string.
            let mut buf = vec![0 as c_char; 256];
            let n = unsafe {
                llama_token_to_piece(
                    self.pimpl,
                    new_token_id_,
                    buf.as_mut_ptr(),
                    buf.len() as i32,
                    0,
                    true,
                )
            };
            if n < 0 {
                bail!(
                    severity = Severity::Error,
                    "failed to convert token to piece",
                );
            }
            buf.truncate(n as usize);

            let piece = unsafe { CStr::from_ptr(buf.as_ptr()) }.to_str().into_diagnostic()?;
            output.push_str(piece);

            // Prepare the next batch with the sampled token
            batch = unsafe { llama_batch_get_one(new_token_id.as_mut_ptr(), 1) };
        }

        Ok(GenerateOutput {
            details: GenerateDetails {
                finish_reason: FinishReason::Stop,
            },
            generated_text: output,
        })
    }

    /// Generates a stream of tokens.
    ///
    /// # Parameters
    ///
    /// * `messages`:
    /// * `params`:
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// ```
    #[cfg(feature = "stream")]
    pub fn generate_stream<'a>(
        &'a self,
        messages: &'a [ChatMessage],
        params: &'a GenerationParams,
    ) -> impl Stream<Item=Result<GenerateStreamItem>> + 'a {
        try_stream! {
            // Prepare the sampler from the given parameters.
            let sampler = Sampler::new(self, params);

            // Tokenize the prompt.
            let tokenizer = LlamaTokenizer::new();
            let prompt = tokenizer.apply_chat_template(self, messages, true)?;
            let mut input_ids = tokenizer.encode(self, &prompt, true, true).unwrap();

            // Configure the context parameters
            let mut ctx_params = LlamaContextParams::default();
            if let Some(n_ctx) = params.max_new_tokens {
                let max_new_tokens = (n_ctx + input_ids.len()) as u32;
                ctx_params.set_n_ctx(max_new_tokens);
                ctx_params.set_n_batch(max_new_tokens);
            }
            let ctx = LlamaContext::new(self, ctx_params)?;

            // Configure the single batch from the given prompt.
            let mut batch = unsafe {
                llama_batch_get_one(input_ids.as_mut_ptr(), input_ids.len() as i32)
            };

            let mut new_token_ids = vec![0 as Token; 1];
            let mut index = 0usize;

            loop {
                // Check if we have enough space in the context
                let n_ctx = unsafe { llama_n_ctx(ctx.pimpl) };
                let n_ctx_used = unsafe { llama_get_kv_cache_used_cells(ctx.pimpl) };
                if (n_ctx_used + batch.n_tokens) as u32 > n_ctx {
                    yield GenerateStreamItem {
                        details: GenerateDetails {
                            finish_reason: FinishReason::MaxTokens,
                        },
                        index,
                        generated_text: "".to_owned(),
                    };
                    break;
                }

                // Decode a batch of tokens
                self.decode_batch(&ctx, batch)?;

                let new_token_id = unsafe { llama_sampler_sample(sampler.pimpl, ctx.pimpl, -1) };

                // Accepting the token pdates the internal state of certain samplers.
                unsafe { llama_sampler_accept(sampler.pimpl, new_token_id) };

                let is_eog = unsafe { llama_token_is_eog(self.pimpl, new_token_id) };
                if is_eog {
                    yield GenerateStreamItem {
                        details: GenerateDetails {
                            finish_reason: FinishReason::Stop,
                        },
                        index,
                        generated_text: "".to_owned(),
                    };
                    break;
                }

                // Convert the token to a string.
                let piece = tokenizer.convert_token_to_piece(self, new_token_id, true)?;
                yield GenerateStreamItem {
                    details: GenerateDetails {
                        finish_reason: FinishReason::None,
                    },
                    index,
                    generated_text: piece.to_owned(),
                };

                // Prepare the next batch with the sampled token
                new_token_ids[0] = new_token_id;
                batch = unsafe { llama_batch_get_one(new_token_ids.as_mut_ptr(), 1) };

                index += 1;
            }
        }
    }

    /// Loads a LoRA adapter from the file.
    ///
    /// This is not thread safe.
    ///
    /// # Parameters
    ///
    /// * `lora_path`: The path to LoRA adapter.
    /// * `adapter_name`: The name of LoRA adapter to reference.
    /// * `scale`:
    ///
    /// # Examples
    ///
    pub fn load_lora_adapter<P: AsRef<Path>, S: AsRef<str>>(
        &mut self,
        lora_path: P,
        adapter_name: S,
        scale: f32,
    ) -> Result<()> {
        let c_lora_path = CString::new(lora_path.as_ref().to_str().unwrap()).into_diagnostic()?;
        let adapter = unsafe { llama_lora_adapter_init(self.pimpl, c_lora_path.as_ptr()) };
        unsafe { llama_lora_adapter_set(self.ctx.pimpl, adapter, scale); };

        let lora = LoRA {
            name: adapter_name.as_ref().to_owned(),
            adapter,
            scale,
        };
        self.loras.insert(adapter_name.as_ref().to_string(), lora);

        Ok(())
    }

    /// Removes a LoRA adapter from the model.
    ///
    /// This is not thread safe.
    ///
    /// # Parameters
    ///
    /// * `adapter_name`: The name of the LoRA adapter to remove.
    ///
    /// # Examples
    ///
    pub fn remove_lora_adapter<S: AsRef<str>>(&mut self, adapter_name: S) -> Result<()> {
        match self.loras.get(adapter_name.as_ref()) {
            Some(lora) => {
                unsafe { llama_lora_adapter_remove(self.ctx.pimpl, lora.adapter); };
                self.loras.remove(adapter_name.as_ref());

                Ok(())
            }
            None => bail!(
                severity = Severity::Warning,
                "adapter `{}` is not present.", adapter_name.as_ref(),
            )
        }
    }

    /// Clear all LoRA adapters.
    pub fn remove_lora_adapters(&mut self) -> Result<()> {
        unsafe { llama_lora_adapter_clear(self.ctx.pimpl) };
        self.loras.clear();

        Ok(())
    }

    fn decode_batch(&self, ctx: &LlamaContext, batch: llama_batch) -> Result<()> {
        match unsafe { llama_decode(ctx.pimpl, batch) } {
            0 => Ok(()),
            1 => Err(miette!("Try reducing the size of the batch or increase the context size.")),
            _ => Err(miette!("failed to decode batch"))
        }
    }
}

impl fmt::Display for LlamaModel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.description())
    }
}

unsafe impl Send for LlamaModel {}
unsafe impl Sync for LlamaModel {}

impl Drop for LlamaModel {
    fn drop(&mut self) {
        unsafe {
            llama_free_model(self.pimpl);
        }
    }
}

pub struct LlamaContext {
    pimpl: *mut llama_context,
}


impl LlamaContext {
    pub fn new(model: &LlamaModel, params: LlamaContextParams) -> Result<Self> {
        let pimpl = unsafe {
            llama_new_context_with_model(model.pimpl, params.pimpl)
        };
        if pimpl.is_null() {
            bail!(severity = Severity::Error, "failed to allocate llama_context")
        } else {
            Ok(LlamaContext { pimpl })
        }
    }

    /// Returns the number of tokens in the KV cache.
    pub fn get_kv_cache_token_count(&self) -> i32 {
        unsafe {
            llama_get_kv_cache_token_count(self.pimpl)
        }
    }

    /// Returns the number of used KV cells.
    pub fn get_kv_cache_used_cells(&self) -> i32 {
        unsafe {
            llama_get_kv_cache_used_cells(self.pimpl)
        }
    }

    /// Clear the kV cache.
    pub fn kv_cache_clear(&self) {
        unsafe {
            llama_kv_cache_clear(self.pimpl)
        }
    }
}

unsafe impl Send for LlamaContext {}

impl Drop for LlamaContext {
    fn drop(&mut self) {
        unsafe { llama_free(self.pimpl) }
    }
}

/// The wrapper type of llama_model_params.
pub struct LlamaModelParams {
    pimpl: llama_model_params,
}

impl LlamaModelParams {
    /// Returns the default model params.
    pub fn default() -> Self {
        let mut pimpl = unsafe { llama_model_default_params() };
        pimpl.n_gpu_layers = 99;
        LlamaModelParams { pimpl }
    }

    /// Sets the number of layers to offload to the GPU.
    pub fn set_n_gpu_layers(&mut self, value: i32) {
        self.pimpl.n_gpu_layers = value;
    }
    /// Sets the how to split the model across multiple GPUs.
    pub fn set_split_mode(&mut self, mode: SplitMode) {
        self.pimpl.split_mode = mode.into();
    }
    /// Sets the GPU that is used for the entire model when split_mode is LLAMA_SPLIT_MODE_NONE.
    pub fn set_main_gpu(&mut self, value: i32) {
        self.pimpl.main_gpu = value;
    }
    // pub fn set_rpc_servers<S: AsRef<str>>(&mut self, servers: Vec<S>) {
    //     let servers = servers.iter().map(|s| s.as_ref().to_string()).collect::<Vec<String>>().join(",");
    //     let c_servers = CString::new(servers).unwrap();
    //     todo!()
    // }

    pub fn set_progress_callback(&mut self) {
        todo!()
    }

    pub fn set_kv_overrides(&mut self) {
        todo!()
    }

    pub fn set_vocab_only(&mut self, value: bool) {
        self.pimpl.vocab_only = value;
    }

    pub fn set_use_mmap(&mut self, value: bool) {
        self.pimpl.use_mmap = value;
    }

    pub fn set_check_tensors(&mut self, value: bool) {
        self.pimpl.check_tensors = value;
    }
}

/// The wrapper type of `llama_context_params`.
pub struct LlamaContextParams {
    pimpl: llama_context_params,
}

impl LlamaContextParams {
    /// Returns context params initialized with default settings.
    pub fn default() -> Self {
        let mut pimpl = unsafe { llama_context_default_params() };
        if let Some(n_cpus) = std::thread::available_parallelism().map(|n| n.get() as i32).ok() {
            pimpl.n_threads = n_cpus;
            pimpl.n_threads_batch = n_cpus;
        }

        LlamaContextParams { pimpl }
    }

    /// Sets the context size.
    pub fn set_n_ctx(&mut self, value: u32) {
        self.pimpl.n_ctx = value;
    }
    pub fn set_n_batch(&mut self, value: u32) {
        self.pimpl.n_batch = value;
    }
    pub fn set_n_ubatch(&mut self, value: u32) {
        self.pimpl.n_ubatch = value;
    }
    pub fn set_n_seq_max(&mut self, value: u32) {
        self.pimpl.n_seq_max = value;
    }
    pub fn set_n_threads(&mut self, value: i32) {
        self.pimpl.n_threads = value;
    }
    pub fn set_n_threads_batch(&mut self, value: i32) {
        self.pimpl.n_threads_batch = value;
    }
    pub fn set_rope_scaling_type(&mut self, value: RopeScalingType) {
        self.pimpl.rope_scaling_type = value.into();
    }
    pub fn set_pooling_type(&mut self, value: PoolingType) {
        self.pimpl.pooling_type = value.into();
    }
    pub fn set_attention_type(&mut self, value: AttentionType) {
        self.pimpl.attention_type = value.into();
    }
    pub fn set_rope_freq_base(&mut self, value: f32) {
        self.pimpl.rope_freq_base = value;
    }
    pub fn set_rope_freq_scale(&mut self, value: f32) {
        self.pimpl.rope_freq_scale = value;
    }
    pub fn set_yarn_ext_factor(&mut self, value: f32) {
        self.pimpl.yarn_ext_factor = value;
    }
    pub fn set_yarn_attn_factor(&mut self, value: f32) {
        self.pimpl.yarn_attn_factor = value;
    }
    pub fn set_yarn_beta_fast(&mut self, value: f32) {
        self.pimpl.yarn_beta_fast = value;
    }
    pub fn set_yarn_beta_slow(&mut self, value: f32) {
        self.pimpl.yarn_beta_slow = value;
    }
    pub fn set_yarn_orig_ctx(&mut self, value: u32) {
        self.pimpl.yarn_orig_ctx = value;
    }
    pub fn set_defrag_thold(&mut self, value: f32) {
        self.pimpl.defrag_thold = value;
    }
    pub fn set_logits_all(&mut self, value: bool) {
        self.pimpl.logits_all = value;
    }

    /// If true, extracts embeddings.
    pub fn set_embeddings(&mut self, value: bool) {
        self.pimpl.embeddings = value;
    }
    pub fn set_offload_kqv(&mut self, value: bool) {
        self.pimpl.offload_kqv = value
    }

    /// Enable or disable using Flash Attention.
    pub fn set_flash_attn(&mut self, enable: bool) {
        self.pimpl.flash_attn = enable;
    }

    /// Sets whether to measure performance.
    pub fn set_no_perf(&mut self, value: bool) {
        self.pimpl.no_perf = value;
    }
}

unsafe impl Send for LlamaContextParams {}

impl Into<llama_context_params> for LlamaContextParams {
    fn into(self) -> llama_context_params {
        self.pimpl
    }
}

pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    /// Returns a chat message initialized with `role` and `conteent`.
    pub fn new<S: AsRef<str>>(role: S, content: S) -> Self {
        ChatMessage {
            role: role.as_ref().to_owned(),
            content: content.as_ref().to_owned(),
        }
    }
}

pub struct LlamaTokenizer {}

impl LlamaTokenizer {
    /// Returns the tokenizer.
    pub fn new() -> Self {
        LlamaTokenizer {}
    }

    /// Converts a list of messages with "role" and "content" keys to a chat message.
    ///
    /// # Parameters
    ///
    /// * `model`: The reference of the `LlamaModel`.
    /// * `messages`: The list of messages.
    /// * `add_generation_prompt`: Whether to end the prompt with tokens that indicate the start of an assistant message.
    ///
    /// # Example
    ///
    pub fn apply_chat_template(&self, model: &LlamaModel, messages: &[ChatMessage], add_generation_prompt: bool) -> Result<String> {
        let mut alloc_size = 0usize;
        let mut roles = Vec::with_capacity(messages.len());
        let mut contents = Vec::with_capacity(messages.len());
        let mut chat = Vec::with_capacity(messages.len());
        for m in messages {
            let c_role = CString::new(m.role.as_str()).unwrap();
            let c_content = CString::new(m.content.as_str()).unwrap();
            roles.push(c_role);
            contents.push(c_content);

            let chat_msg = llama_chat_message {
                role: roles.last().unwrap().as_ptr(),
                content: contents.last().unwrap().as_ptr(),
            };
            chat.push(chat_msg);

            // Recommended to alloc size is 2 * (total number of characters)
            alloc_size += ((m.role.len() + m.content.len()) * 2) as usize;
        }

        let mut c_formatted: Vec<c_char> = vec![0; alloc_size];
        let n_chars = unsafe {
            llama_chat_apply_template(
                model.pimpl,
                std::ptr::null(),
                chat.as_mut_ptr(),
                chat.len(),
                add_generation_prompt,
                c_formatted.as_mut_ptr(),
                alloc_size as i32,
            )
        };
        if n_chars < 0 {
            bail!("unable to apply chat template");
        }

        // If buffer is too small, we need to resize the buffer and apply again.
        c_formatted.resize(n_chars as usize, 0);
        if n_chars > alloc_size as i32 {
            unsafe {
                llama_chat_apply_template(
                    model.pimpl,
                    null(),
                    chat.as_mut_ptr(),
                    chat.len(),
                    add_generation_prompt,
                    c_formatted.as_mut_ptr(),
                    c_formatted.len() as i32,
                );
            }
        }

        let c_formatted = unsafe { CStr::from_ptr(c_formatted.as_ptr()) };
        Ok(c_formatted.to_str().unwrap().to_string())
    }

    /// Toeknizes a text into a vector of tokens.
    ///
    /// # Parameters
    ///
    /// * `text`: A text to tokenize.
    /// * `add_special`: Whether to add a special token.
    /// * `parse_special`: Whether to parse a special token.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use llama_cpp::{LlamaModel, LlamaTokenizer};
    ///
    /// let model = LlamaModel::from_file("./Phi-3-mini-4k-instruct-q4.gguf", None, None)?;
    /// let tokenizer = LlamaTokenizer::new();
    ///
    /// let prompt = "What can you help?";
    /// let token_ids = tokenizer.encode(&model, prompt, true, true);
    /// println!("{:?}", token_ids);
    /// ```
    ///
    pub fn encode<S: AsRef<str>>(&self, model: &LlamaModel, text: S, add_special: bool, parse_special: bool) -> Result<Vec<Token>> {
        // Find the number of tokens
        let c_text = CString::new(text.as_ref()).unwrap();
        let n_tokens = unsafe {
            -llama_tokenize(
                model.pimpl,
                c_text.as_ptr(),
                text.as_ref().len() as i32,
                null_mut(),
                0,
                add_special,
                parse_special,
            )
        };

        let mut tokens = vec![0 as Token; n_tokens as usize];
        let rc = unsafe {
            llama_tokenize(
                model.pimpl,
                c_text.as_ptr(),
                text.as_ref().len() as i32,
                tokens.as_mut_ptr(),
                n_tokens,
                add_special,
                parse_special,
            )
        };
        if rc < 0 {
            bail!("failed to tokenize prompt")
        } else {
            Ok(tokens)
        }
    }

    /// Detokenizes a vector of tokens into a string.
    ///
    /// # Parameters
    ///
    pub fn decode(&self, model: &LlamaModel, tokens: Vec<Token>, remove_special: bool, unparse_special: bool) -> Result<String> {
        let mut text: Vec<c_char> = vec![0; tokens.len()];
        let n_chars = unsafe {
            llama_detokenize(model.pimpl, tokens.as_ptr(), tokens.len() as i32, text.as_mut_ptr(), text.len() as i32, remove_special, unparse_special)
        };
        if n_chars < 0 {
            text.resize(-n_chars as usize + 1, 0);
            unsafe {
                llama_detokenize(model.pimpl, tokens.as_ptr(), tokens.len() as i32, text.as_mut_ptr(), -n_chars, remove_special, unparse_special);
            }
        }

        let text = unsafe { CStr::from_ptr(text.as_ptr()).to_str().unwrap().to_owned() };
        Ok(text)
    }

    /// Convert token id to text.
    pub fn convert_token_to_piece(&self, model: &LlamaModel, token: Token, skip_special: bool) -> Result<String> {
        let mut buf = vec![0; 256];
        let n_chars = unsafe {
            llama_token_to_piece(model.pimpl, token, buf.as_mut_ptr(), 256, 0, skip_special)
        };
        if n_chars < 0 {
            bail!("failed to convert token to piece")
        } else {
            buf.truncate(n_chars as usize);
            let piece = unsafe { CStr::from_ptr(buf.as_ptr()) }.to_str().expect("unable to decode");
            Ok(piece.to_owned())
        }
    }
}

/// The wrapper type of `llama_sampler`.
pub struct Sampler {
    pimpl: *mut llama_sampler,
}

unsafe impl Send for Sampler {}

impl Sampler {
    /// Returns a barebones sampler.
    pub fn default() -> Self {
        let pimpl = unsafe {
            let sparams = llama_sampler_chain_default_params();
            llama_sampler_chain_init(sparams)
        };
        Sampler { pimpl }
    }

    /// Returns a sampler initialized with given prameters.
    ///
    /// # Parameters
    ///
    /// `model`:
    /// `params`:
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use llama_cpp::{GenerationParams, LlamaModel, Sampler};
    ///
    /// let model = LlamaModel::from_file("./Phi-3-mini-4k-instruct-q4.gguf", None, None)?;
    ///
    /// let mut gen_params = GenerationParams::default();
    /// gen_params.top_p = Some(0.9);
    /// gen_params.temperature = Some(0.6);
    ///
    /// let sampler = Sampler::new(&model, &gen_params);
    /// ```
    pub fn new(model: &LlamaModel, params: &GenerationParams) -> Self {
        let mut sampler = Sampler::default();
        if let Some(top_k) = params.top_k {
            sampler.set_top_k(top_k);
        }
        if let Some(top_p) = params.top_p {
            sampler.set_top_p(top_p, 1);
        }
        if let Some(min_p) = params.min_p {
            sampler.add_min_p(min_p, 1);
        }
        if let Some(typical) = params.typical_p {
            sampler.set_typical_p(typical, 1);
        }
        if let Some(temp) = params.temperature {
            sampler.set_temperature(temp);
        }
        if params.repetition_penalty.is_some() ||
            params.frequency_penalty.is_some() ||
            params.presence_penalty.is_some() {
            sampler.add_penalties(
                model,
                -1,
                params.repetition_penalty.unwrap_or(1.0),
                params.frequency_penalty.unwrap_or(0.0),
                params.presence_penalty.unwrap_or(0.0),
                false,
                true,
            );
        }
        if params.do_sample {
            sampler.add_dist(params.seed.unwrap_or(LLAMA_DEFAULT_SEED as u64) as u32);
        } else {
            sampler.add_greedy();
        }

        sampler
    }

    /// Set nucleus sampling parameters.
    pub fn add_min_p(&mut self, top_p: f32, min_keep: usize) {
        unsafe {
            let sampler = llama_sampler_init_top_p(top_p, min_keep);
            llama_sampler_chain_add(self.pimpl, sampler);
        }
    }

    /// Add Top-K sampler into the sampler chain.
    pub fn set_top_k(&mut self, top_k: i32) {
        unsafe {
            let sampler = llama_sampler_init_top_k(top_k);
            llama_sampler_chain_add(self.pimpl, sampler);
        }
    }
    // Add Nucleus sampling into the sampler chain.
    pub fn set_top_p(&mut self, top_p: f32, min_keep: usize) {
        unsafe {
            let sampelr = llama_sampler_init_top_p(top_p, min_keep);
            llama_sampler_chain_add(self.pimpl, sampelr);
        }
    }

    /// Add locally typical sampling into the sampler chain.
    pub fn set_typical_p(&mut self, p: f32, min_keep: usize) {
        unsafe {
            let sampler = llama_sampler_init_typical(p, min_keep);
            llama_sampler_chain_add(self.pimpl, sampler);
        }
    }

    /// Updates the temperature.
    pub fn set_temperature(&mut self, temperature: f32) {
        unsafe {
            let sampler = llama_sampler_init_temp(temperature);
            llama_sampler_chain_add(self.pimpl, sampler);
        }
    }

    pub fn add_penalties(
        &mut self,
        model: &LlamaModel,
        penalty_last_n: i32,
        penalty_repeat: f32,
        penalty_freq: f32,
        penalty_present: f32,
        penalize_newline: bool,
        ignore_eos: bool,
    ) {
        unsafe {
            llama_sampler_chain_add(
                self.pimpl,
                llama_sampler_init_penalties(
                    llama_n_vocab(model.pimpl),
                    llama_token_eos(model.pimpl),
                    llama_token_nl(model.pimpl),
                    penalty_last_n,
                    penalty_repeat,
                    penalty_freq,
                    penalty_present,
                    penalize_newline,
                    ignore_eos,
                ),
            );
        }
    }

    pub fn add_temperature_ext(&mut self, temperature: f32, delta: f32, exponent: f32) {
        unsafe {
            let sampler = llama_sampler_init_temp_ext(temperature, delta, exponent);
            llama_sampler_chain_add(self.pimpl, sampler);
        }
    }

    pub fn add_greedy(&mut self) {
        unsafe {
            let sampler = llama_sampler_init_greedy();
            llama_sampler_chain_add(self.pimpl, sampler);
        }
    }

    pub fn add_dist(&mut self, seed: u32) {
        unsafe {
            let sampler = llama_sampler_init_dist(seed);
            llama_sampler_chain_add(self.pimpl, sampler);
        }
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe { llama_sampler_free(self.pimpl) }
    }
}

pub struct SamplerChain {
    pimpl: *mut llama_sampler,
}

impl SamplerChain {
    /// Create a sampler chain from given params.
    pub fn new(params: SamplerChainParams) -> Result<Self> {
        let pimpl = unsafe { llama_sampler_chain_init(params.internal) };
        if pimpl.is_null() {
            bail!("unable to create sampler chain")
        } else {
            Ok(SamplerChain { pimpl })
        }
    }

    pub fn add(&mut self, sampler: Rc<SamplerChain>) {
        unsafe { llama_sampler_chain_add(self.pimpl, Rc::clone(&sampler).pimpl) }
    }

    pub fn len(&self) -> usize {
        unsafe { llama_sampler_chain_n(self.pimpl) as usize }
    }
}

impl Drop for SamplerChain {
    fn drop(&mut self) {
        unsafe { llama_sampler_free(self.pimpl) }
    }
}

pub struct SamplerChainParams {
    internal: llama_sampler_chain_params,
}

impl SamplerChainParams {
    /// Returns default sampler chain parameters.
    pub fn default() -> Self {
        SamplerChainParams {
            internal: unsafe { llama_sampler_chain_default_params() },
        }
    }

    pub fn set_no_perf(&mut self, value: bool) {
        self.internal.no_perf = value;
    }
}

/// Input data that can contain inputs about one or many sequences.
pub struct LlamaBatch {
    pimpl: llama_batch,
    need_free: bool,
}

impl LlamaBatch {
    /// Returns a number of tokens in the batch.
    pub fn n_tokens(&self) -> i32 {
        self.pimpl.n_tokens
    }

    pub fn token_ids(&self) -> &[Token] {
        unsafe {
            std::slice::from_raw_parts(self.pimpl.token, self.pimpl.n_tokens as usize)
        }
    }

    pub fn embeds(&self) -> &[f32] {
        unsafe {
            std::slice::from_raw_parts(self.pimpl.embd, self.pimpl.n_tokens as usize)
        }
    }
}

impl Drop for LlamaBatch {
    fn drop(&mut self) {
        if self.need_free {
            unsafe { llama_batch_free(self.pimpl) }
        }
    }
}

pub enum ModelKVOverrideType {
    Int,
    Float,
    Bool,
    Str,
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_MODEL_PATH: &'static str = "./models/Phi-3.5-mini-instruct-Q4_K_M.gguf";
    const TEST_LORA_PATH: &'static str = "./models/Phi3.5-mini-F16-LoRA.gguf";
    #[test]
    fn test_set_log_level() {
        llama_set_log_level(LogLevel::Error);
    }

    #[cfg(feature = "huggingface")]
    #[tokio::test]
    async fn test_model_from_hf() -> Result<()> {
        let repo_id = "microsoft/Phi-3-mini-4k-instruct-gguf";
        let model_path = "Phi-3-mini-4k-instruct-q4.gguf";
        let revision = None;
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::from_hf(
            repo_id,
            model_path,
            revision,
            Some(model_params),
            None,
        ).await?;
        assert!(model.model_size() > 0);

        Ok(())
    }

    #[test]
    fn test_model_from_file() -> Result<()> {
        LlamaModel::from_file(TEST_MODEL_PATH, None, None)?;

        let err_model = LlamaModel::from_file("/path/to/invalid.gguf", None, None);
        assert!(err_model.is_err());

        Ok(())
    }

    #[test]
    fn test_model_generate() -> Result<()> {
        let model = LlamaModel::from_file(TEST_MODEL_PATH, None, None)?;
        let messages = vec![
            ChatMessage::new("system", "You are a helpful assistant."),
            ChatMessage::new("user", "What can you help me?"),
        ];
        let mut params = GenerationParams::default();
        params.max_new_tokens = Some(2048);
        let output = model.generate(&messages, params)?;
        // println!("{:?}", output);
        assert_eq!(output.details.finish_reason, FinishReason::Stop);
        assert!(output.generated_text.len() > 0);

        Ok(())
    }

    #[test]
    fn test_model_lora_adapter() -> Result<()> {
        let _handle = LlamaHandle::default();

        let mut model = LlamaModel::from_file(TEST_MODEL_PATH, None, None)?;
        let adapter_name = "test";
        model.load_lora_adapter(TEST_LORA_PATH, adapter_name, 0.5)?;
        model.remove_lora_adapter(adapter_name)?;

        Ok(())
    }

    #[test]
    fn test_model_size() -> Result<()> {
        todo!()
    }

    #[test]
    fn test_sampler_chain() -> Result<()> {
        let mut sampler = Sampler::default();
        sampler.set_top_k(10);
        sampler.add_greedy();

        Ok(())
    }

    #[test]
    fn test_tokenizer_encode() -> Result<()> {
        let model = LlamaModel::from_file(TEST_MODEL_PATH, None, None)?;
        let tokenizer = LlamaTokenizer::new();
        let prompt = "<|system|>\nYou are a helpful AI assistant.<|end|>\n<|user|>\nWhat can you help me?<|end|>\n<|assistant|>\n";
        let token_ids = tokenizer.encode(&model, prompt, true, true)?;
        assert!(token_ids.len() > 0);

        Ok(())
    }

    #[test]
    fn test_tokenizer_convert_token_to_piece() -> Result<()> {
        let _handle = LlamaHandle::default();
        let model = LlamaModel::from_file(TEST_MODEL_PATH, None, None)?;
        let tokenizer = LlamaTokenizer::new();
        let tokens = tokenizer.encode(&model, "Hello Llama", true, true)?;
        for token in tokens {
            let piece = tokenizer.convert_token_to_piece(&model, token, false)?;
            println!("{:?}", piece);
        }

        Ok(())
    }

    #[test]
    fn test_tokenizer_apply_chat_template() -> Result<()> {
        let model = LlamaModel::from_file(TEST_MODEL_PATH, None, None)?;
        let tokenizer = LlamaTokenizer::new();
        let messages = vec![
            ChatMessage::new("system", "You are helpful AI assistant."),
            ChatMessage::new("user", "What can you help?"),
        ];
        let formatted = tokenizer.apply_chat_template(&model, &messages, true)?;
        assert_eq!(formatted, "<|system|>\nYou are helpful AI assistant.<|end|>\n<|user|>\nWhat can you help?<|end|>\n<|assistant|>\n");
        println!("{:?}", formatted);

        Ok(())
    }
}

