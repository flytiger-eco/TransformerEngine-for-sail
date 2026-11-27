#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.
set -euo pipefail
wheel_dir=$1
qa="$SOURCE_DIR/qa/L0_pytorch_mega_moe_rtc"
report="$wheel_dir/qwen-ep16-paths.json"
[[ $NNODES == 2 && $NPROC_PER_NODE == 8 ]]
scratch=$(mktemp -d)
cd "$scratch"
unset PYTHONPATH
python3 -m venv --system-site-packages venv
py="$scratch/venv/bin/python"
"$py" -m pip install --only-binary=:all: --no-deps nvidia-cudnn-frontend==1.25.0
"$py" -m pip install --only-binary=:all: pydantic importlib-metadata einops onnx onnxscript nvdlfw-inspect
"$py" -m pip install --ignore-installed --no-deps "$wheel_dir"/transformer_engine*.whl
export OMP_NUM_THREADS=1 NVTE_MOE_EP16_FUSED=1
"$py" -m torch.distributed.run --nnodes="$NNODES" --nproc-per-node="$NPROC_PER_NODE" \
    --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
    "$SOURCE_DIR/benchmarks/gemm/replay_mega_moe.py" "$qa/tags/qwen35-ep16-r5-call0.tags" \
    --call qwen35-ep16-r5/0/135/0.1.2.3.4.5.6.7.8.9.10.11.12.13.14.15/0 \
    --backend megamoe --warmup 1 --iters 1 --profile-ep16-path --output "$report"
if [[ $NODE_RANK == 0 ]]; then
    "$py" "$qa/verify_ep16.py" "$report"
fi
