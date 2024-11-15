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

## Tips

### Download a model from 🤗 Hub and convert to GGUF format

Use `huggingface-cli` to download a model from 🤗 Hub.

```shell
MODEL_ID=microsoft/Phi-3.5-mini-instruct
LOCAL_DIR=`basename ${MODEL_ID}`

huggingface-cli download ${MODEL_ID} --local-dir ${LOCAL_DIR}
```

To convert 🤗 Model format to GGUF, you can use `convert_hf_to_gguf.py` in `llama.cpp`.
Before use the script, you have to install dependencies from `requirements.txt`.

```shell
MODEL_PATH=./Phi-3.5-mini-instruct
python convert_hf_to_gguf.py --use-temp-file ${MODEL_PATH}
```

### Quanitze GGUF model

After you build a `llama.cpp`, `llama-quantize` is available for quantization of GGUF model.

```shell
MODEL_PATH=./Phi-3.5-mini-instruct-F16.gguf
QUANT_TYPE=Q4_K_M

./llama-quantize ${MODEL_PATH} Phi-3.5-mini-instruct-${QUANT_TYPE}.gguf ${QUANT_TYPE}
```
