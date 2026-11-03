#!/usr/bin/env bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.

# 用法：bash test.sh <head 源码目录> <merge-base SHA> <结果目录>
# 构建两个独立 wheel/venv，在同一个 PPU pod 中以 AB/BA 顺序测量全部 case。
set -euo pipefail
SOURCE=$(realpath "$1")
BASE_SHA="$2"
OUT=$(realpath -m "$3")
SCRATCH=$(mktemp -d /tmp/te-rtc-ci.XXXXXX)
mkdir -p "$OUT/base" "$OUT/head"
GATE="$SOURCE/.github/scripts/rtc_grouped_gemm_gate.py"
REPLAY="$SOURCE/benchmarks/gemm/replay_tag.py"
CASES="$SOURCE/benchmarks/gemm/grouped_gemm_rtc_ci_cases.txt"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NVTE_FRAMEWORK=pytorch NVTE_WITH_NCCL_EP=0 NVTE_CUDA_ARCHS="80;89"
export MAX_JOBS="${MAX_JOBS:-16}" NVTE_BUILD_THREADS_PER_JOB="${NVTE_BUILD_THREADS_PER_JOB:-4}"
# 容器里的 PYTHONPATH/headers 指针不能指向另一版本的 TE。
unset PYTHONPATH NVTE_CUTLASS3_INCLUDE_DIR

git -C "$SOURCE" worktree add --detach "$SCRATCH/base" "$BASE_SHA"

build_arm() {
    local arm="$1" tree="$2"
    git -C "$tree" submodule update --init --recursive
    python3 -m venv --system-site-packages "$SCRATCH/$arm-venv"
    local py="$SCRATCH/$arm-venv/bin/python"
    # 先完成纯 wheel 的工具安装；后续依赖生成 metadata 时必须已能使用新 packaging。
    "$py" -m pip install --upgrade --only-binary=:all: \
        packaging==24.2 setuptools==75.8.0 wheel==0.45.1 pybind11==3.1.0
    "$py" - <<'PY'
import packaging
import setuptools
from packaging.utils import canonicalize_version
from setuptools import _core_metadata

assert canonicalize_version("1.0.0", strip_trailing_zero=False) == "1.0.0"
assert _core_metadata._distribution_fullname("rtc-ci", "1.0.0") == "rtc_ci-1.0.0"
print(f"[RTC CI] packaging={packaging.__version__} ({packaging.__file__})")
print(f"[RTC CI] setuptools={setuptools.__version__} ({setuptools.__file__})")
PY
    # TE 只从包 metadata 定位 C++ 头文件，不使用 frontend Python/DSL 运行时。
    # 固定到项目要求的版本，避免安装额外 CUDA bindings 或替换 SDK 的系统依赖。
    "$py" -m pip install --upgrade --no-deps --only-binary=:all: nvidia-cudnn-frontend==1.25.0
    "$py" - <<'PY'
from importlib.metadata import distribution
from pathlib import Path

frontend = distribution("nvidia-cudnn-frontend")
header = Path(frontend.locate_file("include/cudnn_frontend.h"))
assert header.is_file(), f"cuDNN frontend 缺少构建头文件：{header}"
print(f"[RTC CI] cuDNN frontend={frontend.version}, header={header}")
PY
    mkdir -p "$SCRATCH/$arm-wheel"
    (
        cd "$tree"
        "$py" -m pip wheel -v --no-build-isolation --no-deps . -w "$SCRATCH/$arm-wheel"
    )
    # 不卸载容器预装的 TE/torch；venv 中的当前 wheel 必须优先于系统包。
    "$py" -m pip install --ignore-installed --no-deps "$SCRATCH/$arm-wheel"/transformer_engine*.whl
}

build_arm base "$SCRATCH/base" 2>&1 | tee "$OUT/base-build.log"
build_arm head "$SOURCE" 2>&1 | tee "$OUT/head-build.log"

# 固定从 head 取同一份 harness/case；不能用 base 自己的旧测试替代。
cd "$SCRATCH"
for sweep in 0 1 2 3; do
    if (( sweep % 2 == 0 )); then
        arms=(base head)
    else
        arms=(head base)
    fi
    for arm in "${arms[@]}"; do
        export RTC_CI_EXPECTED_PREFIX="$SCRATCH/$arm-venv"
        python3 "$GATE" measure --python "$SCRATCH/$arm-venv/bin/python" \
            --replay "$REPLAY" --cases "$CASES" --output "$OUT/$arm/sweep-$sweep" \
            --warmup 5 --iters 20 --samples 5 \
            2>&1 | tee "$OUT/$arm/sweep-$sweep.log" || true
    done
done
# 测量失败也继续收集其他 case；最终判定在 runner 上始终执行。
