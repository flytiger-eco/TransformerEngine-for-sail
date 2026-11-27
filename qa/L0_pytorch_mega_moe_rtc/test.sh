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
while read -r world file call name extra; do
    [[ -z "$world" || "$world" == \#* ]] && continue
    [[ "$world" =~ ^[1-8]$ && "$file" =~ ^[a-zA-Z0-9_-]+\.tags$ && -n "$call" && "$name" =~ ^[a-zA-Z0-9_.-]+$ && -z "$extra" ]] || exit 2
    log="$source_dir/qa/L0_pytorch_mega_moe_rtc/tags/$file"
    if [[ ! -f "$log" ]]; then log="${MEGA_MOE_CI_CAPTURES:-/mnt/wl_nas/te-ci/mega-moe/captures}/$file"; fi
    echo "$name $world $call" >> "$out/cases.txt"
    launch=(-m torch.distributed.run --standalone --nproc-per-node="$world")
    for sweep in 0 1; do
        "$scratch/base-venv/bin/python" "${launch[@]}" "$source_dir/benchmarks/gemm/replay_mega_moe.py" \
            "$log" --call "$call" --output "$out/base/$name-$sweep.json"
        if (( sweep == 0 )); then
            "$scratch/head-venv/bin/python" "${launch[@]}" \
                "$source_dir/qa/L0_pytorch_mega_moe_rtc/replay_head_sweeps.py" \
                "$out" "$name" "$log" --call "$call"
        fi
    done
done < "$source_dir/qa/L0_pytorch_mega_moe_rtc/cases.txt"
"$scratch/head-venv/bin/python" "$source_dir/qa/L0_pytorch_mega_moe_rtc/compare.py" "$out"
