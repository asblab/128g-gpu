#!/bin/bash
# gpt-oss-120b, native MXFP4 (63.4GB) — zero quant loss, 128K ctx fits easily.
# -fa on is fine here (standard attention). --jinja for the harmony chat template.
cd ~/llama/llama-b10092
LD_LIBRARY_PATH=$PWD exec ./llama-server \
  -m $HOME/models/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf \
  --alias gptoss-120b \
  --n-gpu-layers 999 \
  --ctx-size 131072 --threads 16 -ub 2048 -b 2048 \
  -fa on --no-mmap --parallel 1 --jinja \
  --host 127.0.0.1 --port 8080
