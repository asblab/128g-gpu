#!/bin/bash
# Qwen3-Coder-Next-80B FP8 on spark via vLLM (eugr/spark-vllm container).
# Needs: docker + NVIDIA container toolkit (stock DGX OS), ~/spark-vllm-docker clone,
# model at ~/models/qwen3-coder-next-fp8 (hf download Qwen/Qwen3-Coder-Next-FP8).
# Flags per NVIDIA forum HOW-TO 359571: flashinfer >> default backend; prefix caching
# is CRITICAL for agentic workloads (0% cache hits without it).
cd ~/spark-vllm-docker
VLLM_SPARK_EXTRA_DOCKER_ARGS="-v $HOME/models:/models" ./launch-cluster.sh \
  --solo -t eugr/spark-vllm:latest -p 8000:8000 -d \
  exec vllm serve /models/qwen3-coder-next-fp8 \
  --served-model-name qwen3-coder-next \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --gpu-memory-utilization 0.8 \
  --host 0.0.0.0 --port 8000 \
  --load-format fastsafetensors \
  --attention-backend flashinfer \
  --enable-prefix-caching
