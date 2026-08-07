# DGX Spark (GB10) — serving notes and measurements

Verified-by-effect setup and benchmarks for serving large MoE models on the **NVIDIA
DGX Spark** (GB10 Grace-Blackwell superchip: 20-core Grace aarch64 CPU + Blackwell GPU,
128 GB unified LPDDR5X at ~273 GB/s, OS sees 121 GiB).

Test machine: DGX Spark running stock DGX OS (Ubuntu 24.04 base), kernel
`6.17.0-1026-nvidia`. Everything below was measured on this box between 2026-07-24 and
2026-08-07 — no numbers are quoted from elsewhere except where cited. The driver moved
580.173.02 → 595.84 during that window (610.43.02 was tested and rolled back); each
measurement states the driver it was taken on. Sibling repo for the same models on AMD
Strix Halo: [Strix Halo page](strix-gfx1151.html).

Two results from the later window are worth reading before the tables:

- **INT4 AutoRound serves the 80B coder at 73.6 gen tok/s**, 43% above the same model in
  FP8 (51.6) and 49% above Q8_0 in llama.cpp (49.3). See
  [the quantization matrix](#quantization-x-backend-matrix-measured).
- **Greedy decoding on this stack is not reproducible.** Six identical `temperature: 0`
  requests return six different logprob vectors. This is not a sampling setting and no
  serving flag tested changes it; it bounds what any eval on this machine can resolve.
  See [Reproducibility](#reproducibility-greedy-decoding-does-not-repeat).

## llama.cpp CUDA build (aarch64/sbsa)

llama.cpp master `0a50d99`, built from source in ~4 minutes on the 20 Grace cores:

```
export PATH=/usr/local/cuda/bin:$PATH CUDACXX=/usr/local/cuda/bin/nvcc
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 20 --target llama-server llama-bench llama-cli
```

Gotcha: `nvcc` is not on the non-login PATH on DGX OS — without `CUDACXX` cmake fails
with "No CMAKE_CUDA_COMPILER could be found" despite the toolkit being installed.

## Measured — llama.cpp CUDA, single stream

`llama-bench -p 1024 -n 128` (gpt-oss with `-fa 1`, V4 Flash at default flash-attention;
flags chosen to match the Strix Halo runs exactly):

| model | quant / size | pp1024 tok/s | tg128 tok/s |
|---|---|---|---|
| gpt-oss-120b (116.8B, 5.1B active) | MXFP4, 59.0 GiB | 1616.2 ± 88.5 | 49.2 ± 0.5 |
| DeepSeek V4 Flash (284B, 13B active) | UD-IQ3_XXS, 95.9 GiB | 340.6 ± 4.1 | 16.4 ± 0.1 |

Server-API measurement (llama-server, ~1.3k-token chat prompt, salted against prompt
cache): gpt-oss-120b **pp 2109.9 / gen 48.1 tok/s** at 131,072 context, fully resident.

## Cross-hardware: GB10 vs Strix Halo, same GGUF files, same methodology

Both boxes are 128 GB unified-memory machines at ~270 GB/s. Identical model files and
`llama-bench` flags; Strix numbers from [Strix Halo page](strix-gfx1151.html)
(best backend per cell shown — Vulkan for gpt-oss, ROCm for V4 generation).

| model | metric | Strix Halo (best) | DGX Spark (CUDA) | ratio |
|---|---|---|---|---|
| gpt-oss-120b | pp1024 | 561.7 (Vulkan) | 1616.2 | 2.9x |
| gpt-oss-120b | tg128 | 53.6 (Vulkan) | 49.2 | 0.92x |
| DeepSeek V4 Flash | pp1024 | 113.9 (Vulkan) | 340.6 | 3.0x |
| DeepSeek V4 Flash | tg128 | 13.9 (ROCm) | 16.4 | 1.18x |

Observations (all measured, same day-class conditions):

- **Prompt processing is ~3x on the GB10** for both models. On V4 Flash the gap has a
  specific cause: the CUDA backend runs the model's fused attention ops (Lightning
  Indexer etc., llama.cpp PR #25545) on GPU, while Vulkan still falls back to CPU for
  them ([issue #25579](https://github.com/ggml-org/llama.cpp/issues/25579)).
- **Generation is bandwidth-bound on both** (~273 vs ~256 GB/s): Strix Halo edges the
  GB10 on gpt-oss tg (53.6 vs 49.2), the GB10 leads on V4 Flash tg (16.4 vs 13.9).
- 16.4 tg on V4 Flash IQ3_XXS is the best single-stream generation number we have
  measured for this model on a 128 GB-class unified-memory box.

## vLLM lane — Qwen3-Coder-Next-80B FP8 (measured)

The [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) prebuilt image
(`eugr/spark-vllm:latest`, 20.3 GB, GB10-targeted) serves
`Qwen/Qwen3-Coder-Next-FP8` (80B total, 3B active) on a single Spark — launch command in
[`scripts/gb10/run-qwen-vllm.sh`](scripts/gb10/run-qwen-vllm.sh), flags per the
[NVIDIA forum HOW-TO](https://forums.developer.nvidia.com/t/how-to-run-qwen3-coder-next-on-spark/359571).

Measured on this box (2026-07-24):

- Engine ready in ~4 minutes from container start (torch.compile ~28 s of that).
- `max_model_len` **262,144** at `--gpu-memory-utilization 0.8`, flashinfer backend.
- Structured tool calling verified: with `--tool-call-parser qwen3_coder`, an OpenAI
  `tools` request returns a well-formed `tool_calls` array and
  `finish_reason: "tool_calls"`.
- Two-point timing (salted ~1.3k-token prompt; wall-clock incl. HTTP overhead):
  prompt 1349 tokens in 0.74 s (~1800 tok/s), generation ~**51.6 tok/s** single-stream —
  above the ~43 tok/s community report for this configuration.

## vLLM lane — Qwen3-Coder-Next-80B INT4 AutoRound (measured, driver 595.84)

`Intel/Qwen3-Coder-Next-int4-AutoRound` (41 GB on disk) on the same container image,
launch command in
[`scripts/gb10/run-qwen-int4-vllm.sh`](scripts/gb10/run-qwen-int4-vllm.sh):

- **73.6 gen tok/s** single-stream, **4054-4190 pp tok/s**, `max_model_len` 262,144 with
  prefix caching enabled. That is +43% generation over the same model in FP8 (51.6) and
  +49% over Q8_0 in llama.cpp (49.3); prompt processing is ~2.3x the FP8 figure.
- Reproducibility of the throughput number: 15 control readings across the measurement
  window spanned **72.55-74.07 tok/s** (sd ~0.4), so differences under ~1% are noise.
- Two flags are load-bearing on SM121a — the container's own
  `mods/fix-qwen3-next-autoround` patch, and `VLLM_MARLIN_USE_ATOMIC_ADD=1` for the
  Marlin INT4 GEMM path. Without them this checkpoint does not serve correctly.
- Weights load in ~180 s with `--load-format fastsafetensors`.
- **First request after boot pays ~19 s TTFT** (prefill-shape JIT compile at realistic
  prompt lengths, not the 16-token warmup shape), then ~1.4 s thereafter. Issuing one
  realistic-shape request at service start absorbs it.

Generation rate falls with context depth, measured on the INT4 configuration:

| context | gen tok/s | pp tok/s |
|---|---|---|
| ~6k | 73.3 | 4107 |
| ~33k | 63.3 | — |
| ~63k | 58.6 | 3516 |

The 73.6 headline is a shallow-context figure; agentic workloads that carry 30-60k of
context run at ~59-63 tok/s.

### Speculative decoding (DFlash drafter)

`z-lab/Qwen3-Coder-Next-DFlash` (0.5B BF16 drafter) via
`--speculative-config '{"method":"dflash",...}'`. At `num_speculative_tokens: 15` the
scheduler OOMs unless `--max-num-batched-tokens 16384` is also set.

| configuration | gen tok/s (salted bench) | mean acceptance |
|---|---|---|
| FP8 + DFlash, flashinfer | 89.3 | 8.39 |
| INT4 + DFlash, flashinfer | 254.7 | up to 12.6 |
| INT4 + DFlash, flash_attn | 212.7 | 10.6 |
| INT4 + DFlash, flash_attn + prefix caching | 181.6 | — |

**These bench figures overstate what agentic traffic sees.** The benchmark prompt is a
long run of near-identical generated functions, which is close to the best case for a
block-diffusion drafter — hence acceptance lengths of 10-12. On the 30-task coding suite
the same configurations landed at 40-57 s median wall/task versus 51 s for INT4 without
speculation, i.e. no reliable end-to-end win. Read 254.7 as a ceiling for highly
predictable continuations, not a serving rate.

One published constraint did not hold here: prefix caching **does** engage under DFlash
on this stack (the recipes that specify it must be off were written against a different
build). `--disable-hybrid-kv-cache-manager` fails outright on KV spec unification.

### Quantization x backend matrix (measured)

Every serving option tried for this model on this machine, single-stream generation:

| stack | quantization | gen tok/s |
|---|---|---|
| vLLM | INT4 AutoRound | **73.6** |
| llama.cpp CUDA | Q4_K_M | 69.0 |
| vLLM | FP8 | 51.6 |
| llama.cpp CUDA | Q8_0 | 49.3 |
| vLLM | AWQ 4-bit | 34.4 |
| vLLM | NVFP4 | ~14 |

- **AWQ is 2.1x slower than AutoRound INT4** despite both being 4-bit, via
  `CompressedTensorsWNA16MarlinMoEMethod` on SM121a — slower even than Q8_0 in llama.cpp.
- **NVFP4 is ~3x slower than FP8** on this GPU; there is no fast NVFP4 path for this
  architecture in the versions tested.
- llama.cpp Q4_K_M is the closest non-vLLM option (69.0 gen) but its prompt processing is
  ~2.7x worse (1486-1560 vs ~4100 pp tok/s).

Engine knobs that produced no measurable change on single-stream INT4 (all within the
~1% noise band): `--async-scheduling` (+0.7%), `--max-num-seqs 4` (+0.9%),
`CUDAGraphMode.FULL` vs the default `FULL_AND_PIECEWISE` (-1.9%), and locking GPU clocks
with `-lgc 3003,3003` (0% — the GPU holds ~2535 MHz under this memory-bound workload
regardless). Three separate daily rebuilds of the container image were also A/B'd against
the pinned digest and all were within noise, so the image is pinned rather than tracked.

## Driver and CUDA toolkit comparison (measured)

Three configurations, same llama.cpp commit (`0a50d99`), same flags, same model files
(`llama-bench -p 1024 -n 128`; gpt-oss with `-fa 1`):

| config | gpt-oss pp1024 | gpt-oss tg128 | V4 Flash pp1024 | V4 Flash tg128 |
|---|---|---|---|---|
| driver 580.173.02, CUDA 13.0 native build | 1616.2 ± 88.5 | 49.2 ± 0.5 | 340.6 ± 4.1 | 16.4 ± 0.1 |
| driver 580.173.02, CUDA 13.3.0 container build | 1560.6 ± 205.8 | 49.3 ± 0.6 | 320.5 ± 5.5 | 16.7 ± 0.1 |
| driver 595.84, CUDA 13.0 native build (same binaries) | 1457.3 ± 9.7 | 47.7 ± 0.3 | 349.5 ± 2.9 | 16.2 ± 0.1 |

- Newer CUDA userspace alone (13.3 vs 13.0 on the same driver) is parity within noise.
- Driver 595.84 costs gpt-oss ~5-10% pp and ~3% tg in llama.cpp; V4 Flash unchanged.
- **What 595.84 fixes: vLLM CUDA graph capture.** On 580, graph capture was broken on
  this GPU (SM121a); on 595.84 vLLM captures full decode graphs cleanly
  (51 piecewise + 35 full-decode graphs, 10 s, 2.9 GiB) with single-stream serving
  parity. Graph benefits are expected under concurrency rather than single-stream.

Driver upgrade path that worked under Secure Boot (the `nvidia-driver-595-open`
metapackage pulls a DKMS package whose configure blocks forever on an interactive MOK
prompt in headless sessions): install `linux-modules-nvidia-595-open-<kernel>` (signed,
prebuilt) + `nvidia-firmware-595` + the 595 userspace libs, purge the DKMS and driver
metapackages, remove the 580 module packages so exactly one driver remains, reboot.

### Driver 610.43.02 — tested and rolled back

610.43.02 was installed the same way (Canonical-signed exact-kernel package set, Secure
Boot intact; the `nvidia-driver-610-open` metapackage pulls `nvidia-dkms-610-open` and
hits the same MOK trap, so install the component packages without it).

Throughput was a wash — 73.83 gen tok/s on the INT4 configuration versus 72.55-73.59 on
595.84, inside the noise band. It was rolled back anyway:

**On 610.43.02 every vLLM container stop leaked ~80 GB of unified memory** — 4 stops out
of 4, including graceful `docker stop -t 120`, with no compute apps remaining and shm and
caches clean. The only recovery found was a reboot. The identical unit and container on
595.84 stop cleanly (2 GB used / 118 GB available afterwards). Acceptance test for anyone
retrying a 610-series driver here: start a serving container, issue one completion, stop
the container, and check that `free -g` shows under 10 GB used.

## Reproducibility: greedy decoding does not repeat

Measured on the INT4 configuration, driver 595.84, and reproduced on two other serving
configurations on this machine.

**Six identical requests at `temperature: 0` return six different logprob vectors.** Same
prompt, `max_tokens: 1`, sequential (not batched). The top-1 logprob moves by ~0.55
logprob units without speculative decoding and ~0.67 with it, and the second- and
third-ranked candidates change places between runs:

```
run 0  '``' -2.002      run 3  '``' -1.731      run 5  '``' -2.284
```

Greedy `argmax` therefore becomes a coin flip at any position whose top-2 margin is
smaller than that drift. Over 14,520 token positions sampled from real task prompts:

| top-2 margin | share of positions |
|---|---|
| exactly 0 | 0.55% |
| ≤ 0.25 | 2.63% |
| ≤ 0.55 (within the observed drift) | **4.35%** |
| ≤ 1.0 | 7.70% |
| median margin | 10.875 |

Most positions are decided by a wide margin; it is the low-margin tail that moves. At
4.4% of positions at risk, the probability of at least one flip in a 512-token generation
is ~100%, and one flip changes every token after it. Measured directly: two runs of the
**same** configuration over 35 prompts × 512 tokens diverged on **34 of 35** prompts.

What it is not:

| candidate cause | test | result |
|---|---|---|
| sampling not actually greedy | `temperature: 0`, then also `top_p: 1, top_k: -1` | still diverges; greedy is applied |
| `VLLM_MARLIN_USE_ATOMIC_ADD=1` (atomic float accumulation is order-dependent) | removed, rebuilt, retested | no change |
| prefix caching | `--no-enable-prefix-caching`, rebuilt, retested | no change |
| speculative decoding | control vs DFlash arm, spec flag the only difference | no change (0.553 vs 0.665 drift) |

A drift of ~0.5 logprob units is far larger than floating-point re-association would
explain. Untested candidates: non-deterministic reductions in the hybrid gated-delta-net
kernels, chunked-prefill boundary effects, and batch-shape-dependent kernel selection.

**Consequence for benchmarking this machine.** Repeated runs of one configuration on the
30-task suite scored 17 and 16, with 5 of the 30 individual tasks flipping outcome; a
second configuration scored 19, 17 and 20 across three runs, with 11 of 30 tasks flipping
between two of them. Differences of one or two tasks between configurations on a 30-task
suite carry no information on this hardware.

Two things that cost time when reproducing any of this:

- `--enable-prefix-caching` is vLLM V1's **default**. Removing the flag disables nothing;
  use `--no-enable-prefix-caching`. Check the live process arguments, not the script.
- After stopping a container, the outgoing one keeps answering `/v1/models` while
  `docker stop` drains. An endpoint check is not evidence the new configuration is up —
  read the serving process arguments instead.

Separately, this checkpoint ships a `generation_config.json` with `do_sample: true`,
`top_k: 40`, `top_p: 0.95`, and vLLM adopts those as server-wide sampling defaults (it
logs a warning). Pass `--generation-config vllm` if you want the server's defaults to be
yours rather than the checkpoint's. This is not the cause of the above.

## Coding-agent eval rows (30-task polyglot suite)

Same agent harness and suite across models (python/javascript/go, hidden tests, 600 s
per-task cap, one agent session per task). Pass = hidden test suite green.

**What this suite can resolve.** At n=30 a pass count carries a binomial standard error of
~2.7 tasks, so the 95% interval spans roughly ±10 tasks and the smallest difference
detectable at 80% power is ~11 tasks. On top of that sits the non-reproducibility described
above. Two consequences, both of which the rows below illustrate: a one- or two-task
difference between configurations means nothing, and **median wall/task is confounded with
accuracy** — a failed task burns up to the 600 s cap, so configurations that fail more look
slower for reasons unrelated to token rate. Both figures are given, the second computed
over passing tasks only.

| model | serving | scored pass | wall/task, all | wall/task, passing |
|---|---|---|---|---|
| Qwen3-Coder-Next-80B INT4 AutoRound | vLLM | 17/30 (56%) | 51 s | 37 s |
| Qwen3-Coder-Next-80B INT4 AutoRound (repeat, identical config) | vLLM | 16/30 (53%) | 89 s | 43 s |
| Qwen3-Coder-Next-80B FP8 | vLLM | 17/30 (56%) | 77 s | 66 s |
| gpt-oss-120b MXFP4 | llama.cpp CUDA | 8/30 (26%) | 63 s | — |
| DeepSeek V4 Flash IQ3_XXS | llama.cpp CUDA | 3/30 (10%) | 600 s | — |

The two INT4 rows are the same configuration run twice: 17 vs 16 passes and 51 vs 89 s
median wall/task, with 5 of the 30 individual tasks changing outcome. That spread is the
measurement's noise floor, not a difference between configurations.

Speculative-decoding variants of the INT4 configuration were also scored and landed at
12-16/30 with 40-57 s median wall/task. Across every DFlash variant tried the pass count
came in at or below the non-speculative rows while generating 2.5-3.5x faster on the
throughput bench — consistent with a small accuracy cost, though no individual comparison
separates from noise at this sample size. The reproducibility result above means output
correctness cannot be demonstrated either way on this machine.

## Scripts

- [`scripts/gb10/run-gptoss.sh`](scripts/gb10/run-gptoss.sh) — gpt-oss-120b llama-server launch
  (131k ctx, `-fa on`, `--jinja` for the harmony template).
- [`scripts/gb10/run-dsv4.sh`](scripts/gb10/run-dsv4.sh) — DeepSeek V4 Flash llama-server launch
  (32k ctx, fully GPU-resident).
- [`scripts/gb10/run-qwen-int4-vllm.sh`](scripts/gb10/run-qwen-int4-vllm.sh) —
  Qwen3-Coder-Next-80B INT4 AutoRound via vLLM (73.6 gen tok/s, 262k ctx, prefix caching;
  image pinned by digest, and the two SM121a-specific flags the checkpoint requires).
