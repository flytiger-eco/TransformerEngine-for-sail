#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.
set -euo pipefail
wheel_dir=$1
qa="$SOURCE_DIR/qa/L0_pytorch_mega_moe_rtc"
out="$wheel_dir/qwen-ep16"
name=Qwen35-EP16-r5
call=qwen35-ep16-r5/0/135/0.1.2.3.4.5.6.7.8.9.10.11.12.13.14.15/0
[[ $NNODES == 2 && $NPROC_PER_NODE == 8 ]]
scratch=$(mktemp -d)
cd "$scratch"
unset PYTHONPATH
python3 -m venv --system-site-packages venv
py="$scratch/venv/bin/python"
"$py" -m pip install --only-binary=:all: --no-deps packaging==24.2 nvidia-cudnn-frontend==1.25.0
"$py" -m pip install --only-binary=:all: pydantic importlib-metadata einops onnx onnxscript nvdlfw-inspect
"$py" -m pip install --ignore-installed --no-deps "$wheel_dir"/transformer_engine*.whl
export EP_PCCL_ROOT_DIR=/usr/local/pccl DEEPEP_NUM_MAX_ICN_PEERS=16
"$py" -m pip install --no-deps --no-build-isolation \
    --index-url https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple \
    'deep-ep==2.2.1+v0.1.0.ppu2.2.0'
export OMP_NUM_THREADS=1 NVTE_MOE_EP16_FUSED=1
mkdir -p "$out/head" "$out/deepep"
for sweep in 0 1; do
    arms=(head deepep)
    if (( sweep == 1 )); then arms=(deepep head); fi
    for arm in "${arms[@]}"; do
        backend=megamoe
        flags=()
        if [[ $arm == deepep ]]; then backend=deepep; fi
        if (( sweep == 0 )); then
            flags+=(--profile-stages)
            if [[ $arm == head ]]; then flags+=(--profile-ep16-path); fi
        fi
        report="$out/$arm/$name-$sweep.json"
        "$py" -m torch.distributed.run --nnodes="$NNODES" --nproc-per-node="$NPROC_PER_NODE" \
            --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
            "$SOURCE_DIR/benchmarks/gemm/replay_mega_moe.py" "$qa/tags/qwen35-ep16-r5-call0.tags" \
            --call "$call" --backend "$backend" "${flags[@]}" --warmup 5 --iters 20 --output "$report"
        if [[ $NODE_RANK == 0 && $arm == head && $sweep == 0 ]]; then
            "$py" "$qa/verify_ep16.py" "$report"
            cp "$report" "$out/qwen-ep16-paths.json"
            cp "$report" "$wheel_dir/qwen-ep16-paths.json"
        fi
    done
done
if [[ $NODE_RANK == 0 ]]; then
    "$py" -m pip install --only-binary=:all: 'matplotlib>=3.7,<4'
    printf '%s 16 %s\n' "$name" "$call" > "$out/cases.txt"
    "$py" "$qa/compare.py" "$out" --head-vs-deepep
fi
