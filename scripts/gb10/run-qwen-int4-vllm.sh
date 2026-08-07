#!/bin/bash
# Qwen3-Coder-Next-80B INT4 AutoRound on a DGX Spark via vLLM (eugr/spark-vllm container).
# Needs: docker + NVIDIA container toolkit (stock DGX OS), ~/spark-vllm-docker clone,
# model at ~/models/qwen3-coder-next-int4 (hf download Intel/Qwen3-Coder-Next-int4-AutoRound,
# 41 GB). Measured 73.6 gen tok/s single-stream vs 51.6 for the same model in FP8.
#
# Two flags are load-bearing on SM121a and the server will not serve this checkpoint
# correctly without them:
#   --apply-mod mods/fix-qwen3-next-autoround   (the container image's own patch)
#   VLLM_MARLIN_USE_ATOMIC_ADD=1                (Marlin INT4 GEMM path)
#
# The image is pinned by digest. ':latest' is rebuilt daily; three separate dailies were
# A/B'd against this one and all were within noise on throughput, so tracking the tag buys
# nothing and silently changes the stack under measurements.
IMAGE=eugr/spark-vllm@sha256:ab975f8b2fff7bd96271d84ca15862cbbb4efe2b4577553156354ed4113548a7
set -euo pipefail
cd ~/spark-vllm-docker
VLLM_SPARK_EXTRA_DOCKER_ARGS="-v $HOME/models:/models" ./launch-cluster.sh \
  --apply-mod mods/fix-qwen3-next-autoround -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  --solo -t "$IMAGE" -p 8000:8000 -d \
  exec vllm serve /models/qwen3-coder-next-int4 \
  --served-model-name qwen3-coder-next \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --gpu-memory-utilization 0.8 \
  --host 0.0.0.0 --port 8000 \
  --load-format fastsafetensors \
  --enable-prefix-caching \
  --max-model-len 262144
