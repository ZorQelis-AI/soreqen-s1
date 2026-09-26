# Running the GGUF builds

Every model ships in three quantisations: `Q4_K_M` (the default), `Q8_0` and `F16`.

## llama.cpp

```bash
llama-cli -hf zorqelis-ai/soreqen-s1-GGUF:Q4_K_M --jinja
llama-server -hf zorqelis-ai/soreqen-s1-mega-GGUF:Q8_0 --jinja --port 8080
```

`--jinja` makes llama.cpp use the chat template packaged with the model. Without it,
thinking and tool-call markers are not rendered correctly.

## Ollama

```bash
ollama run hf.co/zorqelis-ai/soreqen-s1-mini-GGUF:Q4_K_M
```

## Vision

The GGUF builds are text only. The vision tower is not included, so image input
needs the safetensors weights (see `transformers_quickstart.py`) or the API.
