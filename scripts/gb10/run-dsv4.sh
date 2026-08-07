#!/bin/bash
# DeepSeek V4 Flash UD-IQ3_XXS on spark (GB10, llama.cpp CUDA sbsa build).
# CUDA has the fused V4 attention kernels (PR 25545) Vulkan lacks — the pp test.
cd ~/llama/llama.cpp/build/bin
exec ./llama-server \
  -m $HOME/models/dsv4-flash/DeepSeek-V4-Flash-UD-IQ3_XXS-00001-of-00004.gguf \
  --alias dsv4-flash \
  --n-gpu-layers 999 \
  --ctx-size 32768 --threads 20 -ub 2048 -b 2048 \
  --parallel 1 \
  --temp 1.0 --top-p 1.0 --min-p 0.0 \
  --host 127.0.0.1 --port 8080
