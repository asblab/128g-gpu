# 128g-gpu — LLM serving on 128 GB unified-memory machines

Verified-by-effect setup notes, benchmarks, and coding-agent eval results from two
128 GB unified-memory GPU machines, measured 2026-07-23 to 2026-08-07 with identical
model files and methodology:

- **[NVIDIA DGX Spark (GB10)](spark-gb10.html)** — Grace-Blackwell superchip, 20-core
  aarch64 CPU + Blackwell GPU, 128 GB LPDDR5X ~273 GB/s, DGX OS, CUDA.
- **[AMD Strix Halo (gfx1151)](strix-gfx1151.html)** — GMKtec EVO-X2, Ryzen AI MAX+ 395
  + Radeon 8060S iGPU, 128 GB LPDDR5-8000 ~256 GB/s, Debian 13, Vulkan + ROCm.

All numbers were measured on these two boxes; nothing is quoted from elsewhere except
where cited. Launch scripts for every configuration are under [`scripts/`](https://github.com/asblab/128g-gpu/tree/main/scripts).

## Cross-hardware: same GGUF, same llama.cpp flags

`llama-bench -p 1024 -n 128`; best backend per cell (details on the node pages):

| model | metric | Strix Halo (best) | DGX Spark (CUDA) | ratio |
|---|---|---|---|---|
| gpt-oss-120b MXFP4 | pp1024 | 561.7 (Vulkan) | 1616.2 | 2.9x |
| gpt-oss-120b MXFP4 | tg128 | 53.6 (Vulkan) | 49.2 | 0.92x |
| DeepSeek V4 Flash IQ3_XXS | pp1024 | 113.9 (Vulkan) | 340.6 | 3.0x |
| DeepSeek V4 Flash IQ3_XXS | tg128 | 13.9 (ROCm) | 16.4 | 1.18x |

Prompt processing is ~3x on the GB10 (compute + CUDA kernel coverage); generation is
bandwidth-bound on both, so they trade blows model by model.

## Coding-agent eval leaderboard (30-task polyglot suite)

Same agent harness and suite everywhere: python/javascript/go, hidden tests, 600 s
per-task cap, one agent session per task. Pass = hidden test suite green.

| model | machine / serving | scored pass | median wall/task |
|---|---|---|---|
| **Qwen3-Coder-Next-80B Q8_0** | **DGX Spark / llama.cpp CUDA** | **19, 17, 20 /30 (3 runs, mean 62%)** | 73-98 s |
| **Qwen3-Coder-Next-80B Q8_0** | **Strix Halo / llama.cpp Vulkan** | **21, 16, 16 /30 (3 runs, mean 59%)** | 94-233 s |
| **Qwen3-Coder-Next-80B INT4 AutoRound** | **DGX Spark / vLLM** | **17, 16 /30 (2 runs, mean 55%)** | 51-89 s |
| Qwen3-Coder-Next-80B FP8 | DGX Spark / vLLM | 17/30 (56%) | 77 s |
| Qwen3.6-35B-A3B NVFP4 (thinking) | DGX Spark / vLLM | 16/30 (53%) | 127 s |
| gpt-oss-120b MXFP4 | DGX Spark / llama.cpp CUDA | 8/30 (26%) | 63 s |
| gpt-oss-120b MXFP4 | Strix Halo / llama.cpp Vulkan | 7/30 (23%) | 69 s |
| Step-3.7-Flash UD-Q4_K_S | Strix Halo / llama.cpp Vulkan | 6/30 (20%) | 600 s |
| DeepSeek V4 Flash IQ3_XXS | DGX Spark / llama.cpp CUDA | 3/30 (10%) | 600 s |
| DeepSeek V4 Flash IQ3_XXS | Strix Halo / llama.cpp Vulkan | 3/30 (10%) | 600 s |
| DeepSeek V4 Flash IQ3_XXS (1800 s lane, not comparable) | DGX Spark / llama.cpp CUDA | 7/30 (23%) | 569 s |
| GLM-4.5-Air UD-Q6_K_XL | Strix Halo / llama.cpp Vulkan | 3/30 (10%) | 297 s |

Observations the table supports:

- Serving throughput does not predict agentic-coding accuracy: the fastest
  prompt-processing model here (gpt-oss-120b) scores 26%, half the coding-tuned MoE.
- The same GGUF scores the same across machines and backends (26% vs 23%; 10% vs 10%)
  — the suite measures the model, not the serving stack.
- Generation speed gates slow reasoners: DeepSeek V4 Flash passes the small smoke tasks
  (5/5) but times out on substantial ones at ~12-16 tok/s under a 600 s cap.
- Raw tok/s does not predict wall-clock either: the thinking-mode 35B generates at
  ~70 tok/s (vs ~50 for the 80B coder) yet is 65% slower per task end-to-end, because
  it emits a median 8.4K tokens of reasoning per task vs the coder's 3K of work. The
  non-thinking coding specialist wins on accuracy and latency simultaneously.
- **Run-to-run variance is large at n=30, and it has a measured cause**: six independent
  runs of the same Q8_0 config spanned 16-21 passes (53-70%). Single rows carry roughly
  ±10 percentage points; treat any two rows within ~4 tasks of each other as tied. After
  3 runs per box, the two machines are statistically indistinguishable on accuracy (means
  62% and 59%), and the Q8-vs-FP8-vs-INT4 spread (means 61%, a single 56% row, and 55%)
  is not established. Part of this is simply binomial — the smallest difference a 30-task
  suite can detect at 80% power is ~11 tasks — but on the DGX Spark the serving stack also
  [does not reproduce its own output](spark-gb10.html#reproducibility-greedy-decoding-does-not-repeat):
  six identical `temperature: 0` requests return six different logprob vectors, and two
  runs of one configuration flip 5 of 30 tasks. What variance cannot explain: the
  Qwen3-Coder-Next family at ~60% versus every other tested model at ≤26%.
- **Quantization moved throughput far more than accuracy** on the GB10: INT4 AutoRound
  serves the 80B coder at 73.6 gen tok/s versus 51.6 for FP8 and 49.3 for Q8_0, while the
  accuracy rows stay inside each other's noise. Full matrix, including two 4-bit formats
  that are *slower* than 8-bit on this GPU, on the [DGX Spark page](spark-gb10.html).
- Extra time is not a cure: DeepSeek V4 Flash given a 1800 s cap improved 10% → 23%
  but plateaued at gpt-oss level; its remaining failures are quality, not clock.
- Class is everything: two more 100B-class generalists (GLM-4.5-Air Q6, Step-3.7-Flash
  Q4) landed at 10-20% — the coding-specialized MoE beats bigger generalists by 3-7x
  on this suite.
