# AMD Strix Halo (gfx1151) — serving notes and measurements

Verified-by-effect setup and benchmarks for serving large MoE models on **AMD Strix Halo**
(Ryzen AI MAX+ 395 / Radeon 8060S, ROCm target `gfx1151`) with 128 GB unified LPDDR5-8000.

Test machine: GMKtec NucBox EVO-X2 (128 GB, 8x16 soldered), Debian 13 (trixie),
kernel 6.12, Mesa RADV 25.0.7. Everything below was measured on this box on 2026-07-23 —
no numbers are quoted from elsewhere except where cited.

## The memory recipe (the part most guides get wrong)

A big fixed BIOS VRAM carve is the wrong configuration for LLM serving on Strix Halo.
Set the carve to the minimum and let the GPU map memory dynamically through GTT:

1. **BIOS** → Advanced → GFX Configuration:
   - `iGPU Configuration` = `UMA_SPECIFIED`
   - `UMA Frame buffer Size` = `512M`
2. **BIOS** → Main: `Power Mode Select` = `Performance Mode` (EVO-X2 ships in Balance).
3. **Kernel** (GRUB `GRUB_CMDLINE_LINUX_DEFAULT`):
   ```
   amdgpu.gttsize=131072 ttm.pages_limit=31457280
   ```
4. Your user needs the `render` group to reach `/dev/dri/renderD128` (RADV silently
   falls back to llvmpipe without it — check `vulkaninfo --summary`).

Result on 128 GB: `mem_info_gtt_total` = 128 GiB, OS sees 124 GiB, and llama.cpp Vulkan
can hold a ~103 GB model fully GPU-resident.

With the same model split 64 GB carve + CPU experts (`--n-cpu-moe`), generation ran ~35%
slower and prompt processing ~30% slower than fully GTT-resident. Details in the table.

Tip: no keyboard needed for BIOS on headless boxes — `systemctl reboot --firmware-setup`.

## DeepSeek V4 Flash 284B (13B active) — measured

Model: `unsloth/DeepSeek-V4-Flash-GGUF` `UD-IQ3_XXS` (103 GB).
Runtime: llama.cpp **b10092** prebuilt `ubuntu-vulkan-x64` release binary (no custom build).
Launch flags: see [`scripts/gfx1151/run-dsv4.sh`](scripts/gfx1151/run-dsv4.sh); bench method:
[`scripts/gfx1151/bench.sh`](scripts/gfx1151/bench.sh) (~1270-token prompt, 128 generated).

| config | prompt tok/s | gen tok/s |
|---|---|---|
| 64G carve, `--n-cpu-moe 28` (17 similar) | 44.8 | 7.8 |
| **512M carve + GTT 128G, fully resident** | **62.9** | **11.9** |

Sanity anchor: the community reference for this box class is 13.27 tg on the 12% smaller
`UD-IQ2_XXS` ([strix-halo-guide](https://github.com/hogeheer499-commits/strix-halo-guide));
scaled for quant size that predicts ~11.7 on IQ3_XXS — matching what we measure. This is
quant-parity, i.e. the Vulkan stack is running as fast as currently known for this model size.

## gpt-oss-120b (116.8B, 5.1B active) — measured

Model: `ggml-org/gpt-oss-120b-GGUF` `MXFP4` (63.4 GB). This is the format the model was
trained in — there is no additional quantization step, so this row has no quant-fidelity
caveat. Runtime: same llama.cpp b10092 Vulkan binary. Serves fully GPU-resident at
**131,072 context** with room to spare. Launch flags: [`scripts/gfx1151/run-gptoss.sh`](scripts/gfx1151/run-gptoss.sh).

| method | prompt tok/s | gen tok/s |
|---|---|---|
| server API, 1282-token prompt | 710.4 | 51.2 |
| server API, 85-token prompt | 97.2 | 52.8 |
| `llama-bench -p 1024 -n 128 -fa 1` | 561.7 ± 4.2 | 53.6 ± 0.4 |

Unlike DeepSeek V4, gpt-oss uses attention ops that all have Vulkan kernels, so nothing
falls back to CPU — prompt processing is an order of magnitude faster than the V4 Flash
number on the same box (710 vs 63 tok/s at ~1.3k-token prompts).

## Backend A/B: native Vulkan vs ROCm 7.2.4 (measured)

The [kyuz0 toolbox images](https://github.com/kyuz0/amd-strix-halo-toolboxes) package
llama.cpp built against ROCm for gfx1151 — no local ROCm install needed. Run via rootless
podman with `--device /dev/dri --device /dev/kfd --group-add keep-groups`:

```
podman run --rm --device /dev/dri --device /dev/kfd --group-add keep-groups \
  --security-opt seccomp=unconfined -v $HOME/models:/models:ro \
  docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-7.2.4 \
  /usr/local/bin/llama-bench -m /models/... -p 1024 -n 128 -fa 1
```

Same machine, same model file, same flags (`llama-bench -p 1024 -n 128`):

| model | backend | prompt tok/s | gen tok/s |
|---|---|---|---|
| gpt-oss-120b MXFP4 (`-fa 1`) | Vulkan (native b10092) | 561.7 ± 4.2 | 53.6 ± 0.4 |
| gpt-oss-120b MXFP4 (`-fa 1`) | ROCm 7.2.4 (container b10107) | 498.6 ± 6.1 | 51.6 ± 0.1 |
| DeepSeek V4 Flash IQ3_XXS | Vulkan (native b10092) | 113.9 ± 0.4 | 11.9 ± 0.0 |
| DeepSeek V4 Flash IQ3_XXS | ROCm 7.2.4 (container b10107) | 113.0 ± 0.6 | **13.9 ± 0.1** |

The winner is model-dependent. On a standard-attention MoE (gpt-oss), Vulkan RADV leads
ROCm by ~13% prompt / ~4% generation. On DeepSeek V4 Flash the picture inverts: prompt
processing is at parity (the b10107 HIP build evidently still doesn't run the V4 fused
ops on GPU), but ROCm generates **~17% faster** (13.9 vs 11.9 tok/s) — the best V4 Flash
generation number measured on this box, beating the IQ2_XXS community reference (13.27)
on a bigger quant. The ROCm container ran without issue on the stock Debian 6.12 kernel
in these runs (the toolbox docs recommend ≥6.18.4 for gfx1151 stability; a long-run soak
has not been done here yet).

Note the llama-bench pp1024 numbers for V4 Flash (~114 tok/s both backends) are roughly
2x the server-API pp measurement (62.9 at ~1.3k-token chat prompts) — raw batch
throughput vs end-to-end server timings; compare within a method, not across.

## Known ceiling (as of 2026-07-23)

DeepSeek V4's new attention ops (Lightning Indexer, HC pre/comb/post) have **no Vulkan
kernels** — llama.cpp assigns them to CPU (`resolve_fused_ops` warnings at load), which
throttles prompt processing especially. CUDA kernels landed in
[llama.cpp PR #25545](https://github.com/ggml-org/llama.cpp/pull/25545); Vulkan still falls
back ([issue #25579](https://github.com/ggml-org/llama.cpp/issues/25579)).

Paths up from here, in rough order of expected value:

- **ROCm/HIP build for gfx1151** — HIP inherits the CUDA kernels, so a
  `GGML_HIP=ON -DAMDGPU_TARGETS=gfx1151` build should run the fused ops on GPU
  (plus rocWMMA flash-attention, which wins on long-context generation per
  [llm-tracker](https://llm-tracker.info/_TOORG/Strix-Halo)). Needs the ROCm 7.2 runtime.
- **Watch upstream** for a Vulkan Lightning-Indexer kernel — binary-swap upgrade when it lands.
- **MTP speculative decoding** — tested 2026-07-23: llama.cpp b10092 already exposes it
  (`--spec-type draft-mtp`), but the unsloth `UD-IQ3_XXS` GGUF **does not contain the MTP
  layers** — the server refuses to start (`model doesn't contain MTP layers`). Needs a
  GGUF conversion that keeps the MTP head; potential 1.5–2× generation at no quality cost.
- `UD-IQ2_XXS` (90.9 GB) trades a little quality for ~13+ tg.

Two more measured notes on V4 Flash flags:

- `-fa on` (explicit flash attention) **costs** ~11% prompt throughput on this model
  (55.9–56.1 vs 62.9 tok/s at ~1.3k-token prompts, generation unchanged) — the default
  `-fa auto` picks the faster path. The blanket "always `-fa 1` on Strix Halo" guidance
  does not hold for this architecture.
- Repeat benches against a running server hit the prompt cache (`prompt_n` collapses to
  the uncached tail), silently turning a pp measurement into noise — salt the prompt.

## Coding-agent eval rows (30-task polyglot suite)

Same agent harness and suite as the [DGX Spark page](spark-gb10.html)
(python/javascript/go, hidden tests, 600 s per-task cap, one agent session per task):

| model | serving | scored pass | median wall/task | notes |
|---|---|---|---|---|
| gpt-oss-120b MXFP4 | llama.cpp Vulkan | 7/30 (23%) | 69 s | matches the DGX Spark CUDA row (8/30) for the same GGUF — model-limited, not stack-limited |
| DeepSeek V4 Flash IQ3_XXS | llama.cpp Vulkan | 3/30 (10%) | 600 s | timeout-gated at ~12 tok/s generation; identical score to the 38%-faster Spark run |

## Scripts

- [`scripts/gfx1151/run-dsv4.sh`](scripts/gfx1151/run-dsv4.sh) — DeepSeek V4 Flash llama-server launch
  (fully GPU-resident, 16k ctx, `-ub 2048`, `-fa auto`, Unsloth-recommended sampling).
- [`scripts/gfx1151/run-gptoss.sh`](scripts/gfx1151/run-gptoss.sh) — gpt-oss-120b llama-server launch
  (131k ctx, `-fa on`, `--jinja` for the harmony template).
- [`scripts/gfx1151/bench.sh`](scripts/gfx1151/bench.sh) — repeatable pp/tg measurement against the
  running server via `/v1/chat/completions` timings (salt the prompt on repeats — see
  the flags notes above).
