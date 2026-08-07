#!/bin/bash
# qwen-coder-q8 on spark (llama.cpp CUDA) — the quant-isolation experiment:
# SAME Q8_0 GGUF + flags as evo-x2's run-qwencoder.sh (which scored 21/30),
# only the box/backend differs. Answers: was 21-vs-17 the quant or the box?
cd ~/llama/llama.cpp/build/bin
exec ./llama-server \
  -m $HOME/models/qwen3-coder-next-gguf/Qwen3-Coder-Next-Q8_0/Qwen3-Coder-Next-Q8_0-00001-of-00004.gguf \
  --alias qwen-coder-q8 \
  --n-gpu-layers 999 \
  --ctx-size 65536 --threads 20 -ub 1024 -b 2048 \
  --no-mmap --parallel 1 --jinja \
  --host 127.0.0.1 --port 8080
