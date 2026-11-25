#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.
set -euo pipefail
source_dir=$1
scratch=$2
out=$3
replay_case() {
    local world=$1 tag=$2 log=$3 name=$4
    shift 4
    echo "$name $world $tag" >> "$out/cases.txt"
    for sweep in 0 1; do
        arms=(base head)
        if (( sweep == 1 )); then arms=(head base); fi
        for arm in "${arms[@]}"; do
            "$scratch/$arm-venv/bin/python" -m torch.distributed.run --standalone \
                --nproc-per-node="$world" "$source_dir/benchmarks/gemm/replay_mega_moe.py" \
                "$log" --call "$tag" "$@" --output "$out/$arm/$name-$sweep.json"
        done
    done
}

for world in 2 8; do
    logs="$out/logs-ep$world"
    mode=routes
    if (( world == 2 )); then mode=exact; fi
    NVTE_MOE_CAPTURE_TAG=$mode NVTE_MOE_CAPTURE_RUN=ci "$scratch/head-venv/bin/python" -m torch.distributed.run \
        --standalone --nproc-per-node="$world" --log-dir "$logs" --redirects 3 "$source_dir/qa/L0_pytorch_mega_moe_rtc/capture_case.py"
    log="$out/ep$world.tags"
    find "$logs" -name stdout.log -exec grep -hF '[TE] --format=MegaMoE,' {} + > "$log"
    tag=$(head -n 1 "$log" | cut -d, -f3 | cut -d: -f2-)
    backend=()
    if (( world == 8 )); then backend=(--expect-ep8-training); fi
    replay_case "$world" "$tag" "$log" "ep$world" "${backend[@]}"
done
while read -r world file call extra; do
    [[ -z "$world" || "$world" == \#* ]] && continue
    [[ "$world" =~ ^[1-8]$ && "$file" =~ ^[a-zA-Z0-9_-]+\.tags$ && -n "$call" && -z "$extra" ]] || exit 2
    log="$source_dir/qa/L0_pytorch_mega_moe_rtc/tags/$file"
    if [[ ! -f "$log" ]]; then log="${MEGA_MOE_CI_CAPTURES:-/mnt/wl_nas/te-ci/mega-moe/captures}/$file"; fi
    replay_case "$world" "$call" "$log" "model-$(printf '%s' "$file/$call" | sha256sum | cut -c1-16)"
done < "$source_dir/qa/L0_pytorch_mega_moe_rtc/cases.txt"
python3 "$source_dir/qa/L0_pytorch_mega_moe_rtc/compare.py" "$out"
