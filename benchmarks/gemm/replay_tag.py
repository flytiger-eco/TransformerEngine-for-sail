#!/usr/bin/env python3
"""Replay a TE grouped GEMM from a collected '[TE] --format=...' tag line.

The tag is the exact stdout line printed under PPU_LIB_SHOW_PARAMS=1 (one per
launch, e.g. from `PPU_LIB_SHOW_PARAMS=1 python app 2>&1 | tee run_tags.txt`).
This tool turns it back into a runnable harness and re-prints the live tag so
you can diff it against the input:

  python3 replay_tag.py --tag "[TE] --format=GroupedGemmRtc,data_type:bf16,groups:8,layout:NT,m:[256,512,1024,2048,512,1024,256,4096],n:1536,k:2096,tile:256x256x64s3,accumulate:0,gpu:0"

Behavior:
  * layout NT  -> forward replay via GroupedLinear (one tag line per launch).
  * layout NN/TN -> backward replay: one GroupedLinear.backward launches
    dgrad (NN) then wgrad (TN), i.e. two tag lines per iteration; match each
    against the corresponding production line.
  * the tag's m/n/k are the GEMM axes of the captured call, so they map back
    to a layer as NT: (tokens, out, in), NN: (tokens, in, out), and
    TN: (out, in, per-group tokens) -- wgrad keeps its token split in k.
  * op GroupedGemmRtc sets NVTE_USE_CUTLASS_GROUPED_GEMM=1 automatically;
    the echoed live tag must show the same op as the input, otherwise the
    arm fell back.
  * the tag's tile (RTC arm) is pinned via NVTE_GG_FORCE_TILE by default;
    --no-force-tile lets the adaptive path re-select (same shape selects the
    same tile deterministically).
  * accumulate=1 is not reproducible through GroupedLinear (warned, replayed
    as accumulate=0).
"""
import argparse
import os
import re
import sys


def parse_tag(tag: str):
    body = tag.split("--format=", 1)[1] if "--format=" in tag else tag
    op, rest = body.split(",", 1)
    # m:[256,512,...] embeds commas: only split on commas outside brackets.
    fields = dict(re.findall(r"(\w+):(\[[^\]]*\]|[^,]+)", rest))
    return op, fields


def extent(v: str):
    v = v.strip()
    if v.startswith("["):
        return [int(x) for x in v[1:-1].split(",") if x]
    return [int(v)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True,
                    help="one '[TE] --format=...' line, quoted")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--no-force-tile", action="store_true",
                    help="do not pin NVTE_GG_FORCE_TILE from the tag")
    args = ap.parse_args()

    op, f = parse_tag(args.tag)
    if not op.startswith("GroupedGemm"):
        sys.exit(f"unsupported op: {op}")
    g = int(f["groups"])
    layout, dtype_s = f["layout"], f["data_type"]
    if layout not in ("NT", "NN", "TN"):
        sys.exit(f"unsupported layout: {layout}")
    n = int(f["n"])
    m_raw = extent(f["m"])
    # Uniform extents serialize as a scalar (m:2048 means all groups share
    # it); bracketed lists carry one entry per group.
    m = m_raw * g if len(m_raw) == 1 else m_raw
    if len(m) != g:
        sys.exit(f"m list has {len(m)} entries but groups={g}")
    acc, tile = f.get("accumulate", "0"), f.get("tile")
    gpu = int(f.get("gpu", 0) or 0)

    # The tag records the (m, n, k) of the GEMM the backend actually ran, not
    # the layer's (in_features, out_features) pair, and the axes permute with
    # the operand role of the captured launch:
    #   NT (forward): (m, n, k) = (tokens, out_features, in_features)
    #   NN (dgrad):   (m, n, k) = (tokens, in_features, out_features)
    #   TN (wgrad):   (m, n, k) = (out_features, in_features, tokens), i.e. the
    #                 per-group token split sits in k and may be bracketed.
    # Reading them as layer dimensions swaps in/out for the backward forms.
    if layout == "TN":
        if len(set(m)) != 1:
            sys.exit("TN (wgrad) tag should carry one N_out per group; "
                     f"got m={m}")
        k_raw = extent(f["k"])
        split_list = k_raw * g if len(k_raw) == 1 else k_raw
        if len(split_list) != g:
            sys.exit(f"k list has {len(split_list)} entries but groups={g}")
        in_features, out_features = n, m[0]
    elif layout == "NN":
        split_list = m
        in_features, out_features = n, int(f["k"])
    else:
        split_list = m
        in_features, out_features = int(f["k"]), n

    # Env vars must exist before the C++ side reads them (first launch config).
    os.environ["PPU_LIB_SHOW_PARAMS"] = "1"   # echo the live tag for diffing
    # Set both routing switches explicitly per operation. Relying on the RTC
    # tag alone lets an inherited environment replay a cuBLAS capture through
    # CUTLASS (or a PPU/static capture through RTC), so the echoed tag would
    # not describe the requested backend.
    if op == "GroupedGemmRtc":
        # RTC is opt-in on both levels: without these the launch silently falls
        # back to the static CUTLASS arm or cuBLAS and the echoed tag shows a
        # different op.
        os.environ["NVTE_USE_CUTLASS_GROUPED_GEMM"] = "1"
        os.environ["NVTE_USE_CUTLASS_GROUPED_GEMM_RTC"] = "1"
    elif op == "GroupedGemmPpu":
        # Statically compiled PPU CUTLASS arm: CUTLASS on, RTC off.
        os.environ["NVTE_USE_CUTLASS_GROUPED_GEMM"] = "1"
        os.environ["NVTE_USE_CUTLASS_GROUPED_GEMM_RTC"] = "0"
    elif op == "GroupedGemmCublas":
        os.environ["NVTE_USE_CUTLASS_GROUPED_GEMM"] = "0"
    if tile and not args.no_force_tile:
        tm, tn, tks = tile.split("x")
        tk, st = tks.split("s")
        os.environ["NVTE_GG_FORCE_TILE"] = f"{tm},{tn},{tk},{st}"

    import torch
    import transformer_engine.pytorch as te

    # Honour the captured device: replaying on the default GPU can select a
    # different architecture (and therefore a different backend) than the one
    # the tag was collected on.
    torch.cuda.set_device(gpu)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(dtype_s)
    if dtype is None:
        sys.exit(f"unsupported data_type for this tool: {dtype_s} (fp8 needs "
                 f"a quantizer harness, not covered here)")
    if acc == "1":
        print("WARNING: accumulate=1 is not reproducible via GroupedLinear; "
              "replaying with accumulate=0.")

    total_m = sum(split_list)
    dev = f"cuda:{gpu}"
    torch.manual_seed(0)
    net = te.GroupedLinear(g, in_features, out_features, bias=False,
                           params_dtype=dtype).to(dev)
    x = torch.randn(total_m, in_features, dtype=dtype, device=dev)
    splits = torch.tensor(split_list, dtype=torch.int32, device=dev)

    if layout == "NT":
        for _ in range(args.warmup + args.iters):
            y = net(x, splits)
            torch.cuda.synchronize()
    else:
        print(f"NOTE: layout {layout} is a backward form; running "
              f"GroupedLinear.backward -> launches dgrad (NN) then wgrad (TN).")
        gy = torch.randn(total_m, out_features, dtype=dtype, device=dev)
        for _ in range(args.warmup + args.iters):
            xg = x.clone().requires_grad_(True)
            y = net(xg, splits)
            y.backward(gy)
            torch.cuda.synchronize()

    pinned = tile if (tile and not args.no_force_tile) else "adaptive"
    print(f"\n[replay] op={op} g={g} layout={layout} tokens={total_m} "
          f"in={in_features} out={out_features} dtype={dtype_s} tile={pinned}")
    print("[replay] diff the [TE] lines above against the input tag: "
          "op/data_type/groups/layout/m/n/k must match; tile matches when "
          "pinned.")


if __name__ == "__main__":
    main()
