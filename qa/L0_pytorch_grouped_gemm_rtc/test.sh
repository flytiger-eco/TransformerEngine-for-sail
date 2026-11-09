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
    # --no-deps 安装 TE 前补齐非 GPU 运行依赖；约束间接依赖不得替换 SDK torch。
    "$py" - <<'PY'
import json
import sys
from importlib.metadata import version
from pathlib import Path
import torch

prefix = Path(sys.prefix)
sdk = {"version": version("torch"), "path": str(Path(torch.__file__).resolve())}
(prefix / "rtc-ci-sdk.json").write_text(json.dumps(sdk), encoding="utf-8")
(prefix / "rtc-ci-sdk-constraints.txt").write_text(
    f"torch=={sdk['version']}\npackaging==24.2\nnvidia-cudnn-frontend==1.25.0\n",
    encoding="utf-8",
)
print(f"[RTC CI] 保留 SDK torch：{sdk}")
PY
    "$py" -m pip install --only-binary=:all: --constraint "$SCRATCH/$arm-venv/rtc-ci-sdk-constraints.txt" \
        pydantic 'importlib-metadata>=1.0' einops onnx onnxscript nvdlfw-inspect
    mkdir -p "$SCRATCH/$arm-wheel"
    (
        cd "$tree"
        "$py" -m pip wheel -v --no-build-isolation --no-deps . -w "$SCRATCH/$arm-wheel"
    )
    # 不卸载容器预装的 TE/torch；venv 中的当前 wheel 必须优先于系统包。
    "$py" -m pip install --ignore-installed --no-deps "$SCRATCH/$arm-wheel"/transformer_engine*.whl
    # 在源码树外完整导入；错误立即终止，不能重复运行全部必失败的 worker。
    (
        cd "$SCRATCH"
        "$py" - <<'PY'
import json
import sys
from importlib.metadata import version
from pathlib import Path
import torch
import transformer_engine
import transformer_engine.pytorch
from transformer_engine.common import _get_shared_object_file
from transformer_engine.pytorch.cpp_extensions.gemm import general_grouped_gemm

prefix = Path(sys.prefix).resolve()
sdk = json.loads((prefix / "rtc-ci-sdk.json").read_text(encoding="utf-8"))
assert version("torch") == sdk["version"], "SDK torch 版本被替换"
assert str(Path(torch.__file__).resolve()) == sdk["path"], "SDK torch 路径被替换"
for path in (transformer_engine.__file__, transformer_engine.pytorch.__file__,
             _get_shared_object_file("core"), _get_shared_object_file("torch")):
    assert Path(path).resolve().is_relative_to(prefix), f"TE 必须来自当前 venv：{path}"
    print(f"[RTC CI] 导入预检：{path}")
site = Path(transformer_engine.__file__).resolve().parent.parent
headers = site / "include/cutlass3/cutlass/cutlass.h"
assert headers.is_file(), f"wheel 缺少 RTC headers：{headers}"
assert callable(general_grouped_gemm), "GroupedGemm 入口不可调用"
assert torch.cuda.is_available(), "必须有可用 PPU，不能跳过测试"
assert torch.cuda.get_device_capability(0) in ((8, 0), (8, 9)), "设备不支持 RTC"
print(f"[RTC CI] 运行环境预检通过，RTC headers={headers}")
PY
    )
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
