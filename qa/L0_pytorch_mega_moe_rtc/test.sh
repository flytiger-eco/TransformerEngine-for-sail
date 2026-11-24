#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.
set -euo pipefail
source_dir=$1
scratch=$2
out=$3
replay_case() {
    local world=$1 tag=$2 route_dir=$3 name=$4
    shift 4
    echo "$name $world $tag" >> "$out/cases.txt"
    for sweep in 0 1; do
        arms=(base head)
        if (( sweep == 1 )); then arms=(head base); fi
        for arm in "${arms[@]}"; do
            "$scratch/$arm-venv/bin/python" -m torch.distributed.run --standalone \
                --nproc-per-node="$world" "$source_dir/benchmarks/gemm/replay_mega_moe.py" \
                "$tag" --directory "$route_dir" "$@" --output "$out/$arm/$name-$sweep.json"
        done
    done
}

for world in 2 8; do
    route_dir="$out/routes-ep$world"
    mkdir "$route_dir"
    NVTE_MOE_CAPTURE_DIR="$route_dir" "$scratch/head-venv/bin/python" -m torch.distributed.run \
        --standalone --nproc-per-node="$world" "$source_dir/qa/L0_pytorch_mega_moe_rtc/capture_case.py"
    manifests=("$route_dir"/*/manifest.json)
    tag=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["tag"])' "${manifests[0]}")
    backend=()
    if (( world == 8 )); then backend=(--expect-ep8-training); fi
    replay_case "$world" "$tag" "$route_dir" "ep$world" "${backend[@]}"
done
while read -r world tag extra; do
    [[ -z "$world" || "$world" == \#* ]] && continue
    [[ "$world" =~ ^[1-8]$ && "$tag" =~ ^te_mega_moe:v1:[a-f0-9]{32}$ && -z "$extra" ]] || exit 2
    replay_case "$world" "$tag" "${MEGA_MOE_CI_CAPTURES:-/mnt/wl_nas/te-ci/mega-moe/captures}" "${tag##*:}"
done < "$source_dir/qa/L0_pytorch_mega_moe_rtc/cases.txt"
python3 "$source_dir/qa/L0_pytorch_mega_moe_rtc/compare.py" "$out"
