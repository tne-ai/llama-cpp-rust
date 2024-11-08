# Rust bindings for Llama.cpp

## Examples

The following [simple-chat.rs](./examples/simple_chat)
downloads [Phi-3-mini-4k-instruct-gguf](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/blob/main/Phi-3-mini-4k-instruct-q4.gguf)
from [microsoft/Phi-3-mini-4k-instruct-gguf](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/)
repository.

```bash
cargo run --example simple_chat --features "huggingface,stream" -- \
  --hf-repo microsoft/Phi-3-mini-4k-instruct-gguf \
  --hf-model-path Phi-3-mini-4k-instruct-q4.gguf \
  --top-p 0.9 \
  --temperature 0.8
```
