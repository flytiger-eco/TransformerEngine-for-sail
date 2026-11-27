# RTC Grouped GEMM: HGTX Tag Extraction and Replay Guide

Every TE grouped GEMM launch carries an HGTX tag (PPU's NVTX equivalent) and,
optionally, a TFLOPS / tensor-core utilization annotation. This document
explains how to collect and extract these tags, and how to turn a single
collected tag line back into a runnable replay.

Components:

| Component | Location | Purpose |
|---|---|---|
| C++ instrumentation | `transformer_engine/common/gemm/te_profiling_interface.h` | tag assembly, HGTX ranges, TFLOPS timing |
| Launch sites | `cutlass_grouped_gemm_rtc.cu` / `cutlass_grouped_gemm_ppu.cuh` / `cublaslt_gemm.cu` | wrap the three backends' launches |
| Replay tool | `benchmarks/gemm/replay_tag.py` | tag line -> runnable harness |
| Sample | `benchmarks/gemm/uneven_m_demo.py` | generates an uneven-m tag |

---

## 1. Prerequisites

- Build gate: when CMake finds `<PPU_SDK>/targets/x86_64-linux/include/hgtx3/hgToolsExt.h`
  it defines `NVTE_ENABLE_HGTX` for the 3 instrumented TUs; without the SDK
  every entry point compiles to a no-op.
- After rebuilding, deploy (the Python runtime loads the repo-root .so, not
  the build/cmake one):

```bash
cmake --build build/cmake -j16 --target transformer_engine
cp build/cmake/libtransformer_engine.so .
```

- PPU SM89 selects the backend automatically. To reproduce a particular arm,
  replay pins CUTLASS or cuBLAS for the recorded layout and checks the emitted tag.

## 2. Environment Switches (export before the process starts)

| Switch | Effect |
|---|---|
| `PPU_LIB_PERF_INSTRUMENT=1` | push/pop an HGTX range around every launch (domain `tengine`) |
| `PPU_LIB_SHOW_PARAMS=1` | print one `[TE] --format=...` full tag line to stdout per launch |
| `PPU_LIB_TFLOPS=1` | CUDA-event timing; right after the case range pops, a short `[TE-TFLOPS]` range plus a stdout line |

Notes:

- Switches are read once at first use -- they must be set **before the
  process starts**; changing them mid-run has no effect.
- `SHOW_PARAMS` alone works too (pure stdout, no HGTX events emitted).
- `TFLOPS=1` calls `cudaEventSynchronize` on the pop side, so it is a
  **profiling-only** switch; never use it for throughput measurement. Do NOT
  enable it during acu capture (kernel replay would double-count the replayed
  passes into the event elapsed).

## 3. The Two Tag Carriers and Their Semantics

### 3.1 Full stdout line (`SHOW_PARAMS=1`, most complete)

```
[TE] --format=GroupedGemmRtc,data_type:bf16,groups:8,layout:NT,m:[256,512,1024,2048,512,1024,256,4096],n:1536,k:2096,tile:256x256x64s3,accumulate:0,gpu:0
```

| Field | Meaning |
|---|---|
| op | `GroupedGemmRtc` (NVRTC CUTLASS) / `GroupedGemmPpu` (statically compiled) / `GroupedGemmCublas` (per-group fallback) |
| data_type | A/B data type (accumulator type shows up in the hggc kernel name, e.g. `acc_fp32`) |
| groups | number of groups G |
| layout | **RTC arm**: fwd=`NT`, dgrad=`NN`, wgrad=`TN`; **cuBLAS arm**: the caller's transa/transb |
| m | uniform across groups -> scalar (`m:2048` means every group); uneven -> `m:[...]`, one value per group, **i.e. DeepGemm's distribution field** |
| n / k | group-shared output / contraction dims |
| tile | the RTC tile `{M}x{N}x{K}s{stages}` actually picked (reproducible via `NVTE_GG_FORCE_TILE=M,N,K,stages`) |
| accumulate | whether C accumulates (beta != 0) |
| gpu | device id |

TFLOPS line (with `TFLOPS=1`):

```
[TE-TFLOPS] op=GroupedGemmRtc flops=2.110e+11 time_us=736.3 tflops=286.55 peak=415.30 util=69.0%
```

Peak table: SM80(810E)=139.2, SM89(890M)=415.3 TFLOPS (selected automatically
from the compute capability); other devices report `peak=n/a`.

### 3.2 HGTX range text (goes into traces / asysrep / acu)

```
TE_GroupedGemmRtc_bf16_g16_NT_m2048_n1536_k2096_gpu0
```

Character set is restricted to `[A-Za-z0-9_]` so acu filters need no escaping.
**Limitation: when the per-group m values differ, the range name collapses to
the first group** (`m256` above is only m_splits[0]); the full distribution
exists only in the stdout line. When replaying uneven-m cases from production,
always run with `SHOW_PARAMS=1` and `tee` the stdout to a file.

## 4. Collection and Extraction

### 4.1 asys timeline (tags + kernels, full picture)

```bash
NVTE_USE_CUTLASS_GROUPED_GEMM=1 PPU_LIB_PERF_INSTRUMENT=1 PPU_LIB_TFLOPS=1 \
  PPU_LIB_SHOW_PARAMS=1 \
  /usr/local/PPU_SDK/asight/bin/asys profile -t hgtx,hggc -o out python3 app.py
/usr/local/PPU_SDK/asight/bin/asys export -f true out.asysrep   # -> out.sqlite
```

Three common extractions (read the `.asysrep` directly; `-o <file>` writes
csv/txt/xlsx):

```bash
# (1) every range summarized (including each [TE-TFLOPS] annotation; the
#     text column may be truncated)
asys stats -r hgtx_sum out.asysrep

# (2) kernels grouped by tag (tag <-> kernel <-> timing; the core report;
#     regex keeps only the given tag)
asys stats -r "hgtx_kern_sum:range-include=TE_GroupedGemmRtc" out.asysrep

# (3) the tag's projection on the GPU timeline (GPU active time vs the host
#     range duration)
asys stats -r "hgtx_ppu_proj_sum:rows=8" out.asysrep
```

For full message texts or custom statistics, query the sqlite export:

```sql
SELECT start, end, text FROM HGTX_EVENTS;                     -- ranges + TFLOPS, full text
SELECT s.value, k.start, k.end                                -- kernel real name + duration(ns)
  FROM HGPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.shortName;
```

Full rule list: `asys stats --help-reports none <file>`. asys has **no kernel
filter**; `--filter-hgtx 'tag@tengine[/idx]'` narrows any report to the time
window of one range instance.

### 4.2 acu (per-kernel metrics, tag/kernel filterable)

```bash
# filter by kernel name + full metric set (-s skips the first N matching
# launches, -c collects that many)
NVTE_USE_CUTLASS_GROUPED_GEMM=1 \
  acu --profile=on -k regex:nvte_grouped_gemm3 --set full -s 3 -c 1 -o rep python3 app.py
acu --import rep.acurep --page raw --csv-file rep_raw.csv
# TFU headline = cu__we_pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed
```

## 5. End-to-End Cases: deep-M Collect -> Extract -> Replay

### Case 1: collect + TFLOPS annotation + extract (g16, MPE=2048, K=2096, N=1536, bf16 fwd)

Harness (`GroupedLinear(g, k, n)` + `m_splits`; copy as-is):

```python
import torch, transformer_engine.pytorch as te
g, k, n, mpe = 16, 2096, 1536, 2048
net = te.GroupedLinear(g, k, n, bias=False, params_dtype=torch.bfloat16).cuda()
x = torch.randn(mpe * g, k, dtype=torch.bfloat16, device="cuda")
splits = torch.tensor([mpe] * g, dtype=torch.int32, device="cuda")
for _ in range(5):
    y = net(x, splits); torch.cuda.synchronize()
```

Measured output (two lines per launch):

```
[TE] --format=GroupedGemmRtc,data_type:bf16,groups:16,layout:NT,m:2048,n:1536,k:2096,tile:256x256x64s3,accumulate:0,gpu:0
[TE-TFLOPS] op=GroupedGemmRtc flops=2.110e+11 time_us=736.3 tflops=286.55 peak=415.30 util=69.0%
```

Measured extraction (`hgtx_kern_sum`):

```
HGTX Range: tengine:TE_GroupedGemmRtc_bf16_g16_NT_m2048_n1536_k2096_gpu0
Kernel:     nvte_grouped_gemm3_bf16_bf16_bf16_acc_fp32_LA_RM_LB_CM_LC_RM_sm89_tb256x256x64_s3_Cbf16
5 instances  avg 730.2us  med 730.1us  StdDev 0.7us
```

acu `--set full` cross-check: TFU=**70.0%**, occupancy 25% (Reg 248/thread,
waves=1), DRAM 13.8% -- consistent with the TFLOPS annotation (69.0%, event
timing). The kernel name is self-describing: `LA_RM_LB_CM_LC_RM` = fwd operand
layout (dgrad=`LA_RM_LB_RM_LC_RM`, wgrad=`LA_CM_LB_RM_LC_RM`),
`tb256x256x64_s3` = tile/stages, `Cbf16` = output type.

### Case 2: generating an uneven-m tag

```bash
NVTE_USE_CUTLASS_GROUPED_GEMM=1 PPU_LIB_SHOW_PARAMS=1 \
  python3 benchmarks/gemm/uneven_m_demo.py
# [TE] --format=GroupedGemmRtc,data_type:bf16,groups:8,layout:NT,m:[256,512,1024,2048,512,1024,256,4096],n:1536,k:2096,tile:256x256x64s3,accumulate:0,gpu:0
```

### Case 3: one-line tag -> replay

```bash
python3 benchmarks/gemm/replay_tag.py --tag "[TE] --format=GroupedGemmRtc,data_type:bf16,groups:8,layout:NT,m:[256,512,1024,2048,512,1024,256,4096],n:1536,k:2096,tile:256x256x64s3,accumulate:0,gpu:0"
```

Tool behavior:

- parses the tag (scalar `m` is expanded to all G groups; `m:[...]` is used
  as `m_splits` verbatim) and assembles the `GroupedLinear(g, k, n)` harness;
- sets `NVTE_USE_CUTLASS_GROUPED_GEMM=1` automatically when
  `op=GroupedGemmRtc`;
- converts the tag's tile into `NVTE_GG_FORCE_TILE=M,N,K,stages` to pin it
  (`--no-force-tile` disables; adaptive selection is deterministic for the
  same shape, so an unpinned replay picks the same tile anyway);
- **echoes the live tag at the end** -- diff it field-by-field against the
  input; a mismatch means the shape is wrong (e.g. a missing RTC switch shows
  up as `GroupedGemmCublas,layout:TN`, immediately visible).

Measured: both the uneven-m and the uniform-m (`m:2048` scalar) tags echo
back **character-identical**.

Layout vs replay form:

| tag layout | business form | replay | echoed lines per iteration |
|---|---|---|---|
| `NT` | fwd | forward | 1 |
| `NN` | dgrad | `backward()` | 2: dgrad(NN) + wgrad(TN) |
| `TN` | wgrad | same (compare the 2nd line) | 2 |

### Case 4: production capture -> batch replay loop

```bash
# 1) capture: tags to a file (joinable with the asysrep by tag)
PPU_LIB_SHOW_PARAMS=1 NVTE_USE_CUTLASS_GROUPED_GEMM=1 \
  python3 train.py 2>&1 | tee run_tags.txt
grep --line-buffered '^\[TE\]' run_tags.txt > tags.txt

# 2) batch replay (one process per line so FORCE_TILE-style state cannot leak)
while IFS= read -r line; do
  python3 benchmarks/gemm/replay_tag.py --tag "$line" --warmup 2 --iters 3
done < tags.txt

# 3) verify: the replay process's echoed tag must match the input; wrap the
#    replay in acu (same -k regex) and compare Kernel Name character-identical
#    plus Grid/Block -- that proves launch equivalence.
```

## 6. Known Limitations

- **The asight timeline does not render instantaneous markers**: the TFLOPS
  annotation is therefore implemented as a short PushPop range immediately
  after the case range (visible in the GUI, addressable by
  `acu --hgtx-include`).
- **Uneven m exists only on stdout**: the trace file alone is not enough to
  replay an uneven-m case (see section 3.2).
- `column=` args with bracketed names (e.g. `Avg (ns)`) fail to parse in
  `hgtx_*` reports; slice columns via sqlite instead.
- Replay does not support `fp8` data_type or `accumulate=1` (GroupedLinear
  does not expose them; the tool warns).
- `TFLOPS=1` and acu capture are mutually exclusive (section 2).

## 7. Local Reference Artifacts (not committed)

`profile/hgtx_tflops_asys/`: `mpe2048_asys.asysrep` / `mpe2048_acu.acurep`
(the report pair of case 1 in section 5), `uneven_m_asys.asysrep` (the trace
of case 2), `cmds.txt` (every command), `smoke_grouped_linear.py` /
`grouped_gemm_mpe2048_k2096_n1536.py` (harnesses).

## 8. BF16 MegaMoE and DeepEP forward replay

The two-rank benchmark maps the eight-expert NT tag with `m=[4490,4490,4490,4490,4490,4490,4489,4489]`, `n=1024`, and `k=2048` to the first SwiGLU projection. Each SM89 PPU owns eight experts and supplies 17,959 tokens with top-k two, H=2,048 and I=512. Half the routes cross the peer link.

After workspace setup, the public BF16 forward issues three kernels per rank:

1. **Pre-dispatch** builds per-expert lists of source slots and routing weights, records inverse slot indices and stages the inputs needed by the peer.
2. **L1** uses persistent copy CTAs to fetch 256-row tiles from both sources. Each tile signals readiness independently. Compute CTAs project gate and up together, apply weighted SwiGLU and write BF16 activations. Copy CTAs join the compute queue after their last tile.
3. **L2** projects FC2 directly into the expert owner's symmetric buffer. For this shape, once 1,024 output columns are ready on both owners, 15 CTAs fetch and sum their two routes in FP32 while the other 24 project the remaining columns.

After queuing both layers, the host checks the collective pre-dispatch status to surface invalid IDs. The route lists, readiness signals and FC2 results stay on device. First use and workspace growth run a collective allocation and a second pre-dispatch. Other shapes, precisions and topologies can use the existing bulk, generic peer or all-to-all implementations. The backward pass recomputes activations.

```bash
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
    benchmarks/gemm/benchmark_mega_moe_deepep.py --expect-three-kernels --deep-ep-sms 39 \
    --route-variants 3 --warmup 8 --rounds 10 --iters 12 --output /tmp/mega-moe-deepep.json
```

The benchmark verifies the three physical GPU launches and compares BF16 outputs with DeepEP and an independent reference. It alternates method order and route maps, recording event and synchronized wall times for the slower rank. DeepEP V2 uses all 39 CUs for the sequential baseline. Its SM89 JIT needs PCCL's CUDA-compatible device headers in `$EP_PCCL_ROOT_DIR/include_cuda`. Use `--automatic-rtc` to exercise the installation's dispatch settings or `--deepep-implementation legacy` with Megatron-LM on `PYTHONPATH` to measure the legacy dispatcher.

On two pairs of 39-CU SM89 PPUs (PyTorch 2.10.0 and DeepEP 2.2.0), three seeds gave 1.521–1.531 ms for the three-kernel path and 1.831–1.834 ms for DeepEP. This reduced median forward latency by 16.4–17.0%. BF16 outputs matched elementwise for all nine routing tables on both ranks.

## 9. MegaMoE backward on captured grouped GEMM shapes

`benchmark_mega_moe_backward_tags.py` runs the owner-local MegaMoE gradient stages with the full per-expert row distribution from each tag. For RTC, NN records dgrad and TN records wgrad. The script also replays accumulating cuBLAS NT wgrad and derives the two adjoints of each RTC NT forward capture. The CSV keeps the source line, original backend and layout, expert splits, operation, and whether the case was captured directly or derived.

```bash
tag=gg_tags_20261118-174428.txt
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" python benchmarks/gemm/benchmark_mega_moe_backward_tags.py \
    "$tag" --backend megamoe --output /tmp/mega-grads.csv
```

Use `--backend tag` to replay the original backend and recorded tile of each directly captured backward GEMM. Use `--backend legacy` to compile the original MegaMoE dgrad and wgrad kernels from `--legacy-revision` and reproduce their launch geometry. Each stage uses identical seeded BF16 operands across backends. The comparison tests every expert against sampled full-length FP32 dot products and measures repeated CUDA-event intervals after warmup. MegaMoE traces also check the optimized gradient kernel identity.

The 179-line capture contains 66 direct dgrads, 24 direct wgrads and 89 forward tags. Including both adjoints of each forward tag gives 268 cases. The capture records only operand precision for its 14 accumulating wgrads. Replaying those with both BF16 and FP32 destinations gives 282 measurements. For accumulating wgrad, the legacy comparison adds the old BF16 result into the destination, and the original cuBLAS comparison uses beta=1 directly. Use `--line N` for a single tag or `--shards N --shard I` to distribute the cases over independent processes.

To time the public two-rank backward, select which projection each forward tag represents. The benchmark generates a seeded top-2 route map that reproduces the exact row counts at each expert owner. It verifies the full input, probability and parameter gradients against independent per-expert FP32 math with the operator's BF16 rounding. The CUDA event interval covers dispatch, FC1 recomputation, SwiGLU backward, both dgrads, both wgrads and returning source gradients. The JSON also records kernel names, start times, durations and stream IDs for each rank.

```bash
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
    benchmarks/gemm/benchmark_mega_moe_backward.py --tag-file gg_tags_20261118-174428.txt \
    --cases 26:fc1 29:fc1 151:fc2 1:fc1 --output /tmp/megamoe-ep2-backward.json
```

On the 39-CU SM89 PPU, the median single-stage dgrad time over 155 cases went from 32.393 ms to 0.542 ms compared with MegaMoE at `91a4ee48`. Across 127 wgrad cases it went from 28.064 ms to 0.510 ms. Against each captured backend, 65/66 dgrads beat the recorded RTC tile (median 1.34×), and 5/10 wgrads beat RTC (median 1.00×). All 14 accumulating wgrads beat cuBLAS with both BF16 and FP32 destinations (median 1.51× and 1.52×). The largest regression against the captured backend was line 12 dgrad, 0.334 ms versus 0.278 ms. ACU measured 70.56% / 72.17% TFU for FC1 dgrad / wgrad and 72.42% / 61.89% for FC2 on line 26.

The full two-rank backward medians for lines 26, 29, 151 and 1 were 3.231, 2.640, 7.568 and 1.621 ms respectively, down from 116.762, 78.796, 309.846 and 50.981 ms. With identical owner kernels, the complete P2P backward including overlap was 1.90–2.28× faster than the collective variant. With the same two-CU reservation for P2P, overlapping the return with wgrad reduced total backward time by 0.8–6.2% compared with a serialized return. Kineto measured 215–818 µs of return/wgrad overlap across the eight rank/case pairs.

## 10. BF16 EP8 / top-k eight forward replay

`benchmark_mega_moe_ep8.py` compares full dispatch, expert computation and combine on eight peer-accessible SM89 PPUs. The default case has 64 experts (eight per rank), 2,048 received rows per expert, H=3,584 and I=3,072. FC1 is 3,584 -> 6,144, weighted SwiGLU reduces to 3,072, and FC2 is 3,072 -> 3,584. Each rank supplies 2,048 tokens and each token selects eight distinct experts. Random route maps preserve exactly 256 rows from every source to every expert, giving 87.5% remote routes.

The EP8 fused path extends the three-kernel protocol in section 8 to eight source ranks and up to eight routes per token. L1 packs each input tile once, uses 256x256x64 GEMMs with three pipeline stages, and applies weighted SwiGLU in its epilogue. L2 overlaps the second projection partition with first-partition return and reduction. Expert tails are zero-padded to 256 rows. The final reduction preserves route order, accumulates in FP32, and stores BF16 once.

On eight 39-CU ZW-M890P PPUs with ICN64 connectivity, PyTorch 2.11 and PCCL 2.2.0, the measured complete-forward latencies were:

| Seed | DeepEP V2 + RTC grouped GEMM | Fused MegaMoE, tuned |
|---|---:|---:|
| 73 | 4.228080 ms | 4.680850 ms |
| 991 | 4.229800 ms | 4.681490 ms |
| 5741 | 4.229700 ms | 4.685250 ms |
| Pooled median | 4.229145 ms | 4.683245 ms |

Each seed restarts all eight ranks and uses three route maps, eight warmups and ten alternating AB/BA rounds of twelve forwards per implementation. The table pools 360 samples per implementation, taking the slowest rank for every sample. Timing excludes initialization, RTC compilation, workspace growth and reference calculations. Profiler collection is disabled for these measurements. Pooled p10/p90 were 4.219/4.252 ms for DeepEP and 4.668/4.711 ms for MegaMoE.

The previous generic EP8 peer path took 147.909 ms on the same case, so the tuned fused path is 31.58x faster. It remains 10.74% slower than DeepEP. The earlier DeepEP baseline was 4.229195 ms, consistent with the new measurement. The generic path uses 16x64x32 tiles with repeated weight requests and synchronous K-loop loads; FC1 requests each source input again for each of 48 output tiles. Those source-level request counts do not measure actual ICN or HBM traffic because caches can serve repeated requests.

The selected partition uses eight copy CTAs, four combine CTAs and batches four route loads before accumulating in the original order. Defaults remain eight copy CTAs, fifteen combine CTAs and one route per batch. A short sweep gave 4.970 ms with defaults and 4.690 ms with the selected settings; sixteen copy CTAs, only two combine CTAs, or serial combine did not improve this case. These are case-specific tuning results, not universal settings.

With a built Transformer Engine and a compatible DeepEP V2 installation, run from the repository root:

```bash
for seed in 73 991 5741; do
  PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=1 \
    NVTE_MOE_EP8_FUSED=1 NVTE_MOE_FUSED_COPY_SMS=8 \
    NVTE_MOE_FUSED_COMBINE_SMS=4 NVTE_MOE_FUSED_ROUTE_BATCH=4 \
    python -m torch.distributed.run --standalone --nproc-per-node=8 \
    benchmarks/gemm/benchmark_mega_moe_ep8.py --seed "$seed" \
    --route-variants 3 --warmup 8 --rounds 10 --iters 12 \
    --output "/tmp/mega-moe-ep8-${seed}.json"
done
```

The benchmark enables the RTC grouped GEMM backend and uses 39 CUs for DeepEP by default. DeepEP's SM89 JIT also needs the PCCL CUDA-compatible headers described in section 8. Add `--profile` to collect kernel identities outside the timed loop. Set `NVTE_MOE_EP8_FUSED=0` to replay the generic EP8 peer path.

All 72 rank/map/seed checks passed the independent FP32 reference with BF16 storage rounding. Maximum relative L2 was 0.001403459 for MegaMoE versus reference, 0.002536992 for DeepEP versus reference, and 0.002278959 between implementations; maximum absolute error was 0.015625. The outputs need not match elementwise: this DeepEP configuration pre-reduces routes belonging to the same owner into BF16 before return, whereas MegaMoE returns each route and performs one ordered FP32 top-k sum.

Seven EP8 checks also passed under both default and tuned settings: unbalanced source/expert counts, empty sources, skipped/duplicate/zero-weight routes, owners with no expert rows, capacity growth and cross-stream reuse. Their maximum relative L2 was 0.000388306 and maximum absolute error was 0.0078125. The existing EP2 fused-forward and bulk-transport tests both passed, including backward, fallback, and workspace transitions.

The fused public path admits EP2/EP8, BF16, top-k one through eight, H and I divisible by 256, at most sixteen local experts, and aligned contiguous expert weights. At least one rank must supply 2,048 tokens; other sources may be empty. All peers must pass the IPC, cooperative-launch and device capability checks. Other inputs retain their existing fallback behavior. The selected case reserves about 1.6724 GiB of workspace per rank for the worst route imbalance; that reservation is larger than the balanced case's actual owner row count.

## 11. EP8 training with saved SwiGLU activations

The balanced BF16 case in section 10 has 2,048 source tokens per rank and 2,048 rows per expert. The training replay uses 64 experts, top-k eight, FC1 [6144,3584] and FC2 [3584,3072]. It checks the output, input gradient, routing-weight gradient and both lists of expert weight gradients before timing. The DeepEP training baseline uses non-expanded dispatch, local grouping and the cached handle for backward dispatch and gradient return. Both saved-activation implementations use the same activation derivative. For each expert projection the DeepEP baseline uses TE RTC GroupGEMM with the GroupedLinear layouts TN for forward, NN for dgrad and NT for wgrad.

The following historical comparison used MegaMoE gradient GEMMs in the DeepEP backward pass. On eight ICN64-connected SM89 PPUs, three warmups and eight timed iterations gave these median latencies in ms. Each sample takes the maximum across ranks for its interval. The total measures forward plus backward in one iteration.

| Training path | Forward | Backward | Total |
| --- | ---: | ---: | ---: |
| MegaMoE, saved activations | 8.615 | 11.905 | 20.522 |
| MegaMoE, FC1 recompute | 4.834 | 14.183 | 19.005 |
| DeepEP, RTC forward and MegaMoE dgrad/wgrad | 5.036 | 8.241 | 13.267 |

Saving the activations reduced backward by 2.278 ms. Forward took 3.781 ms longer, increasing the total by 1.518 ms. The saved path retained 592 MiB per owner for packed input, gate/up, weighted BF16 activation and unweighted FP32 activation, excluding route metadata, workspace and temporaries. The backward profile contained one pull dgrad, one activation derivative, one scatter dgrad and two local weight-gradient GEMMs.

With the dependencies from section 8, run from the repository root.

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=1 \
  NVTE_MOE_FUSED_COPY_SMS=8 NVTE_MOE_FUSED_COMBINE_SMS=4 NVTE_MOE_FUSED_ROUTE_BATCH=4 \
  python -m torch.distributed.run --standalone --nproc-per-node=8 \
  benchmarks/gemm/benchmark_mega_moe_ep8_training.py --include-recompute --profile
```

## Capturing and replaying a distributed MegaMoE call

Enable `NVTE_MOE_CAPTURE_TAG=routes` for eager capture. Each source rank emits
one self-contained line per call, both to stdout and as its forward NVTX range.
The readable prefix includes EP rank, group size, local token count, top-k,
local expert count, dtype and FC1/FC2 input/output widths. For example,
`[TE] --format=MegaMoE,v:2,call:demo/0/0/0.1/3,rank:1,ep:2,dtype:torch.bfloat16,t:2048,topk:2,groups:8,fc1:3584x6144,fc2:3072x3584,...`
illustrates the prefix; replay needs the complete lines.

The readable `m_src` vector counts valid slots from this source to each global expert.
Summing these vectors gives the logical M of each expert GEMM before tile
padding; replay includes this sum as `rows_per_global_expert`.

The payload encodes every global expert ID in source-token and top-k slot order,
including duplicate experts, -1 slots and empty source ranks. IDs use 1, 2 or 4
bytes depending on expert count, then zlib and URL-safe base64. In `routes`
mode, a bit mask preserves zero routing weights; replay sets each other weight
to 1/top-k. Choose `NVTE_MOE_CAPTURE_TAG=exact` to include lossless FP32
routing weights. Both modes preserve tensor shapes, strides, alignment and
gradient requirements, and regenerate hidden states, parameters and upstream
gradients from the replay seed.

Use one run label across workers, unique for each capture session. The call key
combines this label (default: torchrun rendezvous ID), restart count, process
group name, ordered global ranks and a group-local invocation counter. Enable
capture at the same call boundary on participating ranks and retain their call
order. Capture copies local routing data to CPU and adds no distributed
collectives. For timing runs, unset the capture variable. Compiled and fake
tensor execution bypass collection.

Full per-token strings can exceed a profiler UI's display limit or a shared
pipe's atomic write size. Use separate worker stdout logs, e.g.:

```bash
export NVTE_MOE_CAPTURE_TAG=routes NVTE_MOE_CAPTURE_RUN=demo
torchrun --standalone --nproc-per-node=8 --log-dir ./run-logs --redirects 3 train.py
find ./run-logs -name stdout.log -exec grep -hF '[TE] --format=MegaMoE,' {} + > run.tags
cut -d, -f3 run.tags | sort -u
unset NVTE_MOE_CAPTURE_TAG
# Substitute a call value listed above. Omit --call for a single invocation.
torchrun --standalone --nproc-per-node=8 benchmarks/gemm/replay_mega_moe.py \
  run.tags --call 'demo/0/0/0.1.2.3.4.5.6.7/0' --output step.json
```

Collect the stdout files from every host for a multi-node group. Replay maps
the EP ranks onto the new workers, checks completeness, and compares outputs
and requested gradients against TE's all-to-all transport. Timing JSON reports
rank-max forward, backward and total CUDA-event samples and medians. The
reader also accepts previously captured v1 manifests and their original shards.

For the Qwen EP16 training capture, start one torchrun agent on each of two
eight-PPU hosts in the same ICN fabric. Use the same PyTorch build on both
hosts. Set `NODE_RANK` to 0 and 1 respectively and `MASTER_ADDR` to the
first host's reachable address. The opt-in uses cross-host CUDA symmetric
memory for expert inputs and output slots.

```bash
export NVTE_MOE_EP16_FUSED=1
torchrun --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  benchmarks/gemm/replay_mega_moe.py \
  qa/L0_pytorch_mega_moe_rtc/tags/qwen35-ep16-r5-call0.tags \
  --profile-ep16-path --output qwen-ep16.json
# Run on rank zero's host after torchrun exits.
python qa/L0_pytorch_mega_moe_rtc/verify_ep16.py qwen-ep16.json
```

The two-node Qwen CI job measures merge-base MegaMoE, PR MegaMoE and DeepEP with TE RTC
GroupGEMM on the same 16 PPUs and route capture. The second sweep reverses their order.
The baseline uses the path selected by its own wheel. The report records how many
baseline ranks executed all-to-all in each direction.
Each backend runs five warmups and twenty timed replays per sweep. The PR report
includes single-layer forward, backward and total latency, plus a chart for the
six kernel stages using the same profiling convention as the single-node gate.
The `qwen-ep16-rank-paths` artifact contains the timing samples, rank-path
verification and plotted data.

The single-node 890P gate replays Megatron EP8, DeepSeek-V4.1 calls 0 and 1,
and Qwen35B-A3B call 3. It runs each BF16 distributed tag through merge-base MegaMoE, PR MegaMoE, and DeepEP intranode dispatch/combine with
the PR wheel's RTC grouped GEMMs. The second sweep reverses the three-way order.
The four middle arms of each single-node case share one PR-wheel process group.
Each arm restores the capture and seed, checks the reference, and performs its own warmup and timing.
Use `--backend deepep` on the same replay command to measure DeepEP separately.
For a source with zero tokens, its DeepEP adapter pads one all-dropped row and
trims the result. This padding is timed and sends no expert rows.
The Actions job summary and the artifact's `comparison.md` list forward,
backward and total latency in milliseconds. Each `stages-<case>.png` plots all three implementations across six kernel stages.
The report records the merge-base SHA and shows both baseline / PR and DeepEP / PR latency ratios.
Each profiling step repeats the same captured invocation four times. Stage values sum
rank-local kernel durations, take the maximum across ranks, then average three steps.
The raw reports include matched device kernel names for each stage. DeepEP / PR above 1 means PR
MegaMoE is faster. The 5% regression check compares MegaMoE head with base. The checked-in Megatron example has 64 experts, H=3584,
I=3072 and eight source tags. To add a model case, place its `.tags` under
`qa/L0_pytorch_mega_moe_rtc/tags/` or `MEGA_MOE_CI_CAPTURES` (default
`/mnt/wl_nas/te-ci/mega-moe/captures`) and add a line with EP size, filename,
call key and a unique model/call label to `qa/L0_pytorch_mega_moe_rtc/cases.txt`. Numerical failures
block the job. Manual and perf/refactor runs also enforce a 5% total latency
regression threshold.
