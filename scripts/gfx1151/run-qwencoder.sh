#!/bin/bash
# qwen-coder-q8 on evo-x2 (llama.cpp Vulkan, wave-2). -fa auto per measured lesson.
cd ~/llama/llama-b10092
LD_LIBRARY_PATH=$PWD exec ./llama-server \
  -m $HOME/models/qwen3-coder-next-gguf/Qwen3-Coder-Next-Q8_0/Qwen3-Coder-Next-Q8_0-00001-of-00004.gguf \
  --alias qwen-coder-q8 \
  --n-gpu-layers 999 \
  --ctx-size 65536 --threads 16 -ub 1024 -b 2048 \
  --no-mmap --parallel 1 --jinja \
  --host 127.0.0.1 --port 8080
