#!/bin/bash
# DeepSeek V4 Flash UD-IQ3_XXS, fully GPU-resident (Vulkan, GTT).
# -fa auto (NOT on): measured 2026-07-23, fa on costs ~11% pp on this model
# (55.9 vs 62.9 tok/s @1277-tok); auto picks the faster path. --no-mmap per
# Strix Halo guidance (kyuz0). MTP unavailable: this GGUF has no MTP layers.
cd ~/llama/llama-b10092
LD_LIBRARY_PATH=$PWD exec ./llama-server \
  -m $HOME/models/dsv4-flash/DeepSeek-V4-Flash-UD-IQ3_XXS-00001-of-00004.gguf \
  --alias dsv4-flash \
  --n-gpu-layers 999 \
  --ctx-size 16384 --threads 16 -ub 2048 -b 2048 \
  --no-mmap --parallel 1 \
  --temp 1.0 --top-p 1.0 --min-p 0.0 \
  --host 127.0.0.1 --port 8080
