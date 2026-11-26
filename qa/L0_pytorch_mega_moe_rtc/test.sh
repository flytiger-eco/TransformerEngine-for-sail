#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.
set -euo pipefail
source_dir=$1
scratch=$2
out=$3
mkdir -p "$out/deepep"
"$scratch/head-venv/bin/python" -m pip install --no-deps --no-build-isolation \
    --index-url https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple \
    'deep-ep==2.2.1+v0.1.0.ppu2.2.0'
"$scratch/head-venv/bin/python" -m pip install --only-binary=:all: 'matplotlib>=3.7,<4'
replay_case() {
    local world=$1 tag=$2 log=$3 name=$4
    shift 4
    echo "$name $world $tag" >> "$out/cases.txt"
    for sweep in 0 1; do
        arms=(base head deepep)
        if (( sweep == 1 )); then arms=(deepep head base); fi
        for arm in "${arms[@]}"; do
            local venv=$arm
            local flags=("$@")
            if [[ $arm == deepep ]]; then
                venv=head
                flags=(--backend deepep)
            fi
            if [[ $arm != base && $sweep == 0 ]]; then flags+=(--profile-stages); fi
            "$scratch/$venv-venv/bin/python" -m torch.distributed.run --standalone \
                --nproc-per-node="$world" "$source_dir/benchmarks/gemm/replay_mega_moe.py" \
                "$log" --call "$tag" "${flags[@]}" --output "$out/$arm/$name-$sweep.json"
        done
    done
}

for world in 2 8; do
    logs="$scratch/logs-ep$world"
    mode=routes
    if (( world == 2 )); then mode=exact; fi
    if ! NVTE_MOE_CAPTURE_TAG=$mode NVTE_MOE_CAPTURE_RUN=ci "$scratch/head-venv/bin/python" -m torch.distributed.run \
        --standalone --nproc-per-node="$world" --log-dir "$logs" --redirects 3 "$source_dir/qa/L0_pytorch_mega_moe_rtc/capture_case.py"; then
        find "$logs" -name stderr.log -exec tail -n 40 {} \;
        exit 1
    fi
    log="$out/ep$world.tags"
    find "$logs" -name stdout.log -exec grep -hF '[TE] --format=MegaMoE,' {} + > "$log"
    tag=$(head -n 1 "$log" | cut -d, -f3 | cut -d: -f2-)
    backend=()
    if (( world == 8 )); then backend=(--expect-ep8-training); fi
    replay_case "$world" "$tag" "$log" "Synthetic-EP$world" "${backend[@]}"
done
while read -r world file call name extra; do
    [[ -z "$world" || "$world" == \#* ]] && continue
    [[ "$world" =~ ^[1-8]$ && "$file" =~ ^[a-zA-Z0-9_-]+\.tags$ && -n "$call" && "$name" =~ ^[a-zA-Z0-9_.-]+$ && -z "$extra" ]] || exit 2
    log="$source_dir/qa/L0_pytorch_mega_moe_rtc/tags/$file"
    if [[ ! -f "$log" ]]; then log="${MEGA_MOE_CI_CAPTURES:-/mnt/wl_nas/te-ci/mega-moe/captures}/$file"; fi
    replay_case "$world" "$call" "$log" "$name"
done < "$source_dir/qa/L0_pytorch_mega_moe_rtc/cases.txt"
"$scratch/head-venv/bin/python" "$source_dir/qa/L0_pytorch_mega_moe_rtc/compare.py" "$out"
