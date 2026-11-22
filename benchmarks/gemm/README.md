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
