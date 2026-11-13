#!/usr/bin/env bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.

# 用法：bash test.sh <head 源码目录> <merge-base SHA> <结果目录> [wheel 复用目录]
# 构建两个独立 wheel/venv，在同一个 PPU pod 中以 AB/BA 顺序测量全部 case。
# 第四个参数（可选）指向 NAS 上的 wheel 复用目录，每份 wheel 旁带 commit 文件记录来源 SHA：
#   - head arm：优先安装 ppu-890-build 已产出的 head wheel，跳过本地编译；
#   - base arm：按 merge-base SHA 缓存，命中则复用，未命中则构建并写入。
# 任何 wheel 复用前都以 commit 文件校验 SHA，校验不过一律回退本地构建。
# 复用与本地构建的 wheel 构建命令/环境完全一致（同一镜像、NVTE_CUDA_ARCHS="80;89"、
# pip wheel --no-build-isolation --no-deps），配对测量的可比性不受影响。
set -euo pipefail
SOURCE=$(realpath "$1")
BASE_SHA="$2"
OUT=$(realpath -m "$3")
WHEEL_REUSE="${4:-}"
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

# 校验一个候选 wheel 目录是否属于指定 SHA；输出目录路径（stdout）供调用方使用。
# 目录必须含 transformer_engine*.whl 和 commit 文件，且 commit 与期望 SHA 一致。
verify_wheel_dir() {
    local dir="$1" sha="$2"
    [[ -n "$dir" && -d "$dir" ]] || return 1
    local marker
    marker=$(cat "$dir/commit" 2>/dev/null || true)
    [[ "$marker" == "$sha" ]] || return 1
    ls "$dir"/transformer_engine*.whl >/dev/null 2>&1 || return 1
    printf '%s\n' "$dir"
}

# 在 NAS 复用目录下写入一份带 commit 标记的 wheel（base 缓存用）。
# 为避免读到写一半的目录，先写入临时目录再原子改名。
publish_wheel_cache() {
    local dest="$1" src_dir="$2" sha="$3"
    mkdir -p "$(dirname "$dest")"
    local tmp
    tmp=$(mktemp -d "$(dirname "$dest")/.tmp-XXXXXX")
    cp "$src_dir"/transformer_engine*.whl "$tmp/"
    printf '%s\n' "$sha" > "$tmp/commit"
    rm -rf "$dest" 2>/dev/null || true
    mv "$tmp" "$dest"
}

build_arm() {
    # sha：当前 arm 的 commit（wheel 复用校验用）；reuse_dir：候选复用 wheel 目录。均可省略。
    # 旧式两参调用（CPU 单测）下两者为空，直接走本地构建。
    local arm="$1" tree="$2"
    local sha="" reuse_dir=""
    if (( $# >= 3 )); then sha="$3"; fi
    if (( $# >= 4 )); then reuse_dir="$4"; fi
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
    local reuse_from=""
    # 未绑定（旧调用方只传 arm/tree，如 CPU 单测）时安全回退本地构建。
    if [[ -n "${WHEEL_REUSE:-}" && -n "$reuse_dir" ]]; then
        reuse_from=$(verify_wheel_dir "$reuse_dir" "$sha" || true)
    fi
    if [[ -n "$reuse_from" ]]; then
        echo "[RTC CI] $arm arm 复用 wheel：$reuse_from（commit=$sha，跳过本地编译）"
        cp "$reuse_from"/transformer_engine*.whl "$SCRATCH/$arm-wheel/"
    else
        echo "[RTC CI] $arm arm 本地构建 wheel（commit=${sha}）"
        (
            cd "$tree"
            "$py" -m pip wheel -v --no-build-isolation --no-deps . -w "$SCRATCH/$arm-wheel"
        )
        # base arm 构建成功后写回缓存（head arm 的复用源由 ppu-890-build 负责，不在此缓存）。
        if [[ "$arm" == "base" && -n "${WHEEL_REUSE:-}" ]]; then
            publish_wheel_cache "$WHEEL_REUSE/cache-$sha" "$SCRATCH/$arm-wheel" "$sha" \
                && echo "[RTC CI] base wheel 已写入缓存：$WHEEL_REUSE/cache-$sha" \
                || echo "[RTC CI] warning: base wheel 缓存写入失败（不影响本次测量）"
        fi
    fi
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

build_arm base "$SCRATCH/base" "$BASE_SHA" "${WHEEL_REUSE:+$WHEEL_REUSE/cache-$BASE_SHA}" 2>&1 | tee "$OUT/base-build.log"
build_arm head "$SOURCE" "$(git -C "$SOURCE" rev-parse HEAD)" "${WHEEL_REUSE:+$WHEEL_REUSE/build-$(git -C "$SOURCE" rev-parse HEAD)}" 2>&1 | tee "$OUT/head-build.log"

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
