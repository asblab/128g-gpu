#!/bin/bash
# gpt-oss-120b native MXFP4 on spark (GB10, llama.cpp CUDA sbsa build).
# Same model file as evo-x2 — the cross-hardware anchor row.
cd ~/llama/llama.cpp/build/bin
exec ./llama-server \
  -m $HOME/models/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf \
  --alias gptoss-120b \
  --n-gpu-layers 999 \
  --ctx-size 131072 --threads 20 -ub 2048 -b 2048 \
  -fa on --parallel 1 --jinja \
  --host 127.0.0.1 --port 8080
