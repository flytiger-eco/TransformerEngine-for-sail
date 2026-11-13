# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.
"""门禁控制面的 CPU 测试；无需安装 torch/TE 或连接 PPU。"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GATE = load_module("rtc_gate", ROOT / ".github/scripts/rtc_grouped_gemm_gate.py")
REPLAY = load_module("replay_tag", ROOT / "benchmarks/gemm/replay_tag.py")
CASES = GATE.load_cases(GATE.DEFAULT_CASES)


def test_case_inventory():
    assert len(CASES) == 27
    fields = [REPLAY.parse_tag(tag)[1] for tag in CASES]
    assert {f["layout"] for f in fields} == {"NT", "NN", "TN"}
    assert {int(f["groups"]) for f in fields} == {1, 8, 16, 32}
    assert {f["tile"] for f in fields} == {"256x256x64s3", "256x256x64s4"}
    assert fields[0]["k"] == "2096"
    assert REPLAY.extent(fields[0]["m"]) == [256, 512, 1024, 2048, 512, 1024, 256, 4096]
    for f in fields[-3:]:
        splits = REPLAY.extent(f["m"])
        assert len(splits) == int(f["groups"])
        assert sum(splits) == 35918
    for f in fields:
        assert f["data_type"] == "bf16" and f["accumulate"] == "0" and f["gpu"] == "0"


@pytest.mark.parametrize("tag", CASES)
def test_operand_mapping(tag):
    _, f = REPLAY.parse_tag(tag)
    layout, a, b, outputs, splits, single = REPLAY.rtc_operand_specs(f)
    groups = int(f["groups"])
    assert len(a) == len(b) == len(outputs) == len(splits) == groups
    for index in range(groups):
        # 还原 lhs @ rhs 的二维形状，检查 GEMM 轴没有错误置换。
        lhs = b[index][::-1] if layout == "NT" else b[index]
        rhs = a[index][::-1] if layout == "TN" else a[index]
        assert lhs[1] == rhs[0]
        assert outputs[index] == (lhs[0], rhs[1])
    assert single == (f["layout"] != "TN")
    assert layout == {"NT": "TN", "TN": "NT", "NN": "NN"}[f["layout"]]
    assert splits == (
        [int(f["k"])] * groups
        if f["layout"] == "TN"
        else REPLAY.extent(f["m"]) * (groups if not f["m"].startswith("[") else 1)
    )


def test_scalar_and_invalid_split():
    _, f = REPLAY.parse_tag(CASES[1])
    assert REPLAY.rtc_operand_specs(f)[-2] == [1536] * 16
    f["m"] = "[2048]"
    with pytest.raises(ValueError, match="16"):
        REPLAY.rtc_operand_specs(f)


@pytest.mark.parametrize(
    "subject,labels,manual,expected",
    [
        ("perf(gemm): optimize", [], False, "perf"),
        ("[perf:890p] optimize", [], False, "perf"),
        ("refactor: cleanup", [], False, "refactor"),
        ("[refactor] cleanup", [], False, "refactor"),
        ("fix: correct output", [], False, "advisory"),
        ("fix: correct output", ["perf"], False, "perf"),
        ("docs: description", ["refactor"], False, "refactor"),
        ("docs: description", [], True, "enforced"),
    ],
)
def test_policy(subject, labels, manual, expected):
    assert GATE.policy_for(subject, labels, manual) == expected


@pytest.mark.parametrize(
    "path,expected",
    [
        ("transformer_engine/common/gemm/rtc/cutlass_grouped_gemm.cu", True),
        ("transformer_engine/common/gemm/actlize_overlay/include/cutlass/a.hpp", True),
        ("transformer_engine/common/CMakeLists.txt", True),
        ("transformer_engine/pytorch/csrc/extensions/gemm.cpp", True),
        ("build_tools/build_ext.py", True),
        ("3rdparty/actlize", True),
        ("benchmarks/gemm/replay_tag.py", True),
        ("transformer_engine/common/util/rtc.h", True),
        ("transformer_engine/common/gemm/cutlass_grouped_gemm_rtc.cu", True),
        ("transformer_engine/common/gemm/ppu_adaptive_tile.hpp", True),
        ("transformer_engine/pytorch/cpp_extensions/gemm.py", True),
        ("3rdparty/cudafy-for-sail", True),
        ("qa/L0_pytorch_grouped_gemm_rtc/test.sh", True),
        (".github/workflows/ppu-890-rtc-grouped-gemm-gate.yml", True),
        ("docs/installation.rst", False),
        ("docs/transformer_engine/common/util/rtc.h", False),
        ("transformer_engine/common/gemm/cublaslt_gemm.cu", False),
        ("transformer_engine/common/normalization/rtc_dispatch.cpp", False),
        ("transformer_engine/common/util/logging.h", False),
        ("transformer_engine/pytorch/csrc/extensions/attention.cpp", False),
        ("transformer_engine/pytorch/cpp_extensions/normalization.py", False),
        ("build_tools/jax.py", False),
        ("3rdparty/nccl", False),
        ("benchmarks/gemm/benchmark_gemm.py", False),
        ("transformer_engine/pytorch/cpp_extensions/gemm.py.bak", False),
        ("transformer_engine/pytorch/attention/dot_product_attention.py", False),
    ],
)
def test_scope(path, expected):
    assert bool(GATE.SCOPE.search(path)) == expected


def workflow_config():
    import yaml

    return yaml.load(
        (ROOT / ".github/workflows/ppu-890-rtc-grouped-gemm-gate.yml").read_text(),
        Loader=yaml.BaseLoader,
    )


def test_workflow_paths_match_scope():
    workflow = workflow_config()
    trigger = workflow["on"]["pull_request"]
    assert trigger["branches"] == ["v2.18-SAIL"]
    assert set(trigger["paths"]) == set(GATE.SCOPE_PATHS)
    assert len(trigger["paths"]) == len(GATE.SCOPE_PATHS)
    assert "workflow_dispatch" in workflow["on"]
    assert "push" not in workflow["on"] and "schedule" not in workflow["on"]
    assert {"edited", "labeled", "unlabeled", "synchronize"} <= set(trigger["types"])
    for pattern in trigger["paths"]:
        assert GATE.SCOPE.fullmatch(pattern.replace("**", "a/b").replace("*", "a"))
    scope = next(step for step in workflow["jobs"]["gate"]["steps"] if step.get("id") == "scope")
    assert "run = manual or any(gate.SCOPE.search(path) for path in changed)" in scope["run"]


def payload(times):
    return {
        "case_digest": hashlib.sha256("\n".join(CASES).encode()).hexdigest(),
        "settings": {"iters": 20, "samples": 5, "warmup": 5},
        "errors": [],
        "results": [
            {
                "id": i + 1,
                "tag": tag,
                "median_us": t,
                "device": "ZW-M890P",
                "capability": [8, 9],
                "torch": "2.11",
            }
            for i, (tag, t) in enumerate(zip(CASES, times))
        ],
    }


def test_no_regression_hidden_by_geomean():
    base, head = payload([100.0] * 27), payload([50.0] * 26 + [106.0])
    rows = GATE.compare(CASES, [(base, head)] * 4, 0.05)
    assert sum(row["regression"] for row in rows) == 1
    assert rows[-1]["regression"]


def test_paired_sweep_median_and_boundary():
    base = payload([100.0] * 27)
    heads = [payload([t] * 27) for t in (104.0, 105.0, 105.0, 300.0)]
    rows = GATE.compare(CASES, [(base, h) for h in heads], 0.05)
    assert rows[0]["latency_ratio"] == pytest.approx(1.05)
    assert not any(row["regression"] for row in rows)


@pytest.mark.parametrize(
    "corruption", ["missing", "errors", "tag", "nan", "zero", "device", "digest", "settings"]
)
def test_fail_closed(corruption):
    base, head = payload([100.0] * 27), payload([100.0] * 27)
    if corruption == "missing":
        head["results"].pop()
    elif corruption == "errors":
        head["errors"] = ["RTC fallback"]
    elif corruption == "tag":
        head["results"][0]["tag"] = "wrong"
    elif corruption in ("nan", "zero"):
        head["results"][0]["median_us"] = float("nan") if corruption == "nan" else 0
    elif corruption == "device":
        head["results"][0]["device"] = "other"
    elif corruption == "digest":
        head["case_digest"] = "old case list"
    else:
        head["settings"]["iters"] = 1
    with pytest.raises(ValueError):
        GATE.compare(CASES, [(base, head)], 0.05)


@pytest.mark.parametrize(
    "policy,expected", [("advisory", 0), ("perf", 2), ("refactor", 2), ("enforced", 2)]
)
def test_report_exit_codes(tmp_path, policy, expected):
    for arm, times in (("base", [100.0] * 27), ("head", [106.0] * 27)):
        directory = tmp_path / arm / "sweep-0"
        directory.mkdir(parents=True)
        (directory / "results.json").write_text(json.dumps(payload(times)))
    args = argparse.Namespace(
        output=str(tmp_path),
        cases=GATE.DEFAULT_CASES,
        sweeps=1,
        tolerance=0.05,
        policy=policy,
        base_sha="base",
        head_sha="head",
    )
    assert GATE.report(args) == expected
    assert json.loads((tmp_path / "gate.json").read_text())["verdict"] == "REGRESSION"
    assert "27/27" in (tmp_path / "gate.md").read_text()
    # 即使非性能变更，缺失或错误测量也不能显示成成功。
    args.sweeps = 2
    assert GATE.report(args) == 3


@pytest.mark.parametrize(
    "head_time,missing,expected,verdict",
    [(100.0, False, 0, "PASS"), (106.0, False, 2, "REGRESSION"), (100.0, True, 3, "ERROR")],
)
def test_report_readonly_input(tmp_path, monkeypatch, head_time, missing, expected, verdict):
    measurements = tmp_path / "nas"
    output = tmp_path / "runner" / "report"
    for arm, time in (("base", 100.0), ("head", head_time)):
        directory = measurements / arm / "sweep-0"
        directory.mkdir(parents=True)
        if not (missing and arm == "head"):
            (directory / "results.json").write_text(json.dumps(payload([time] * 27)))
    before = {p: p.read_bytes() for p in measurements.rglob("*.json")}
    original_open, original_mkdir = Path.open, Path.mkdir

    # root 也必须受到写入限制，避免 chmod 在本地测试中失效。
    def readonly_open(path, mode="r", *args, **kwargs):
        if path.is_relative_to(measurements) and any(flag in mode for flag in "wax+"):
            raise PermissionError(f"只读 NAS：{path}")
        return original_open(path, mode, *args, **kwargs)

    def readonly_mkdir(path, *args, **kwargs):
        if path.is_relative_to(measurements):
            raise PermissionError(f"只读 NAS：{path}")
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", readonly_open)
    monkeypatch.setattr(Path, "mkdir", readonly_mkdir)
    args = argparse.Namespace(
        input=str(measurements),
        output=str(output),
        cases=GATE.DEFAULT_CASES,
        sweeps=1,
        tolerance=0.05,
        policy="enforced",
        base_sha="base",
        head_sha="head",
    )
    assert GATE.report(args) == expected
    assert json.loads((output / "gate.json").read_text())["verdict"] == verdict
    assert verdict in (output / "gate.md").read_text()
    assert {p: p.read_bytes() for p in measurements.rglob("*.json")} == before
    assert not (measurements / "gate.json").exists()


@pytest.mark.parametrize("separate_input", [False, True])
def test_report_cli_input_compatibility(tmp_path, separate_input):
    output = tmp_path / "report"
    measurements = tmp_path / "nas" if separate_input else output
    for arm in ("base", "head"):
        directory = measurements / arm / "sweep-0"
        directory.mkdir(parents=True)
        (directory / "results.json").write_text(json.dumps(payload([100.0] * 27)))
    command = [
        sys.executable,
        str(ROOT / ".github/scripts/rtc_grouped_gemm_gate.py"),
        "report",
        "--output",
        str(output),
        "--base-sha",
        "base",
        "--head-sha",
        "head",
        "--sweeps",
        "1",
    ]
    if separate_input:
        command += ["--input", str(measurements)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert json.loads((output / "gate.json").read_text())["verdict"] == "PASS"


@pytest.mark.parametrize(
    "failure", ["", "packaging==24.2", "nvidia-cudnn-frontend==1.25.0", "pydantic", "preflight"]
)
def test_build_bootstrap_before_frontend(tmp_path, failure):
    script = (ROOT / "qa/L0_pytorch_grouped_gemm_rtc/test.sh").read_text()
    function = re.search(r"(?ms)^build_arm\(\) \{.*?^\}", script).group()
    calls = tmp_path / "calls.jsonl"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "code = sys.stdin.read() if args == ['-'] else ''\n"
        "with open(os.environ['CALLS'], 'a') as stream:\n"
        "    stream.write(json.dumps({'args': args, 'code': code, 'cwd': os.getcwd()}) + '\\n')\n"
        "if os.environ['FAILURE'] == 'preflight' and 'general_grouped_gemm' in code:\n"
        "    sys.exit(17)\n"
        "if os.environ['FAILURE'] and os.environ['FAILURE'] in args:\n"
        "    sys.exit(17)\n"
        "if args[:3] == ['-m', 'pip', 'wheel']:\n"
        "    (Path(args[args.index('-w') + 1]) / 'transformer_engine-test.whl').touch()\n"
    )
    fake_python.chmod(0o755)
    tree = tmp_path / "tree"
    tree.mkdir()
    shell = (
        "set -euo pipefail\n"
        "git() { :; }\n"
        "python3() {\n"
        '  [[ "$*" == "-m venv --system-site-packages $SCRATCH/base-venv" ]]\n'
        '  mkdir -p "$SCRATCH/base-venv/bin"\n'
        '  ln -s "$FAKE_PYTHON" "$SCRATCH/base-venv/bin/python"\n'
        "}\n" + function + '\nbuild_arm base "$TREE"\n'
    )
    result = subprocess.run(
        ["bash"],
        input=shell,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "SCRATCH": str(tmp_path),
            "TREE": str(tree),
            "FAKE_PYTHON": str(fake_python),
            "CALLS": str(calls),
            "FAILURE": failure,
        },
    )
    recorded = [json.loads(line) for line in calls.read_text().splitlines()]
    assert result.returncode == (17 if failure else 0), result.stderr
    bootstrap = recorded[0]["args"]
    assert bootstrap[:3] == ["-m", "pip", "install"]
    assert "--upgrade" in bootstrap and "--only-binary=:all:" in bootstrap
    assert "packaging==24.2" in bootstrap and "setuptools==75.8.0" in bootstrap
    assert not any("cudnn" in arg for arg in bootstrap)
    if failure == "packaging==24.2":
        assert len(recorded) == 1
        return
    assert recorded[1]["args"] == ["-"]
    assert "strip_trailing_zero=False" in recorded[1]["code"]
    compile(recorded[1]["code"], str(ROOT / "qa/L0_pytorch_grouped_gemm_rtc/test.sh"), "exec")
    frontend = recorded[2]["args"]
    assert frontend[:3] == ["-m", "pip", "install"]
    assert "--no-deps" in frontend and "--only-binary=:all:" in frontend
    assert "nvidia-cudnn-frontend==1.25.0" in frontend
    if failure == "nvidia-cudnn-frontend==1.25.0":
        assert len(recorded) == 3
        return
    assert "cudnn_frontend.h" in recorded[3]["code"]
    assert "torch==" in recorded[4]["code"]
    runtime = recorded[5]["args"]
    assert runtime[:3] == ["-m", "pip", "install"]
    assert "--constraint" in runtime and "--only-binary=:all:" in runtime
    assert "--no-deps" not in runtime
    assert {
        "pydantic",
        "importlib-metadata>=1.0",
        "einops",
        "onnx",
        "onnxscript",
        "nvdlfw-inspect",
    } <= set(runtime)
    assert not any(arg.startswith(("torch", "nvidia-")) for arg in runtime)
    if failure == "pydantic":
        assert len(recorded) == 6
        return
    assert recorded[6]["args"][:3] == ["-m", "pip", "wheel"]
    assert "--no-build-isolation" in recorded[6]["args"]
    assert "--no-deps" in recorded[7]["args"]
    assert "--ignore-installed" in recorded[7]["args"]
    assert "general_grouped_gemm" in recorded[8]["code"]
    assert recorded[8]["cwd"] == str(tmp_path), "导入预检不能在源码目录内执行"
    assert len(recorded) == 9
    for call in recorded:
        if call["code"]:
            compile(call["code"], str(ROOT / "qa/L0_pytorch_grouped_gemm_rtc/test.sh"), "exec")


def test_runtime_dependencies_match_project():
    import ast

    def declared(path, variable):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == variable:
                return set(ast.literal_eval(node.value))
            if isinstance(node, ast.Return) and isinstance(node.value, ast.List):
                return set(ast.literal_eval(node.value))
        raise AssertionError(f"未找到依赖声明：{path}")

    requirements = declared(ROOT / "setup.py", "install_reqs")
    requirements |= declared(ROOT / "build_tools/pytorch.py", None)
    required = requirements - {"torch>=2.1", "packaging", "nvidia-cudnn-frontend>=1.25.0"}
    script = (ROOT / "qa/L0_pytorch_grouped_gemm_rtc/test.sh").read_text()
    runtime = shlex.split(re.search(r'--constraint "[^"\n]+" \\\n(.*?)\n', script).group(1))
    assert set(runtime) == required


@pytest.mark.parametrize(
    "failure",
    [
        "",
        "pydantic",
        "system-te",
        "system-core",
        "torch-version",
        "torch-path",
        "headers",
        "device",
    ],
)
def test_import_preflight(tmp_path, monkeypatch, failure):
    import builtins
    import importlib.metadata

    script = (ROOT / "qa/L0_pytorch_grouped_gemm_rtc/test.sh").read_text()
    code = re.findall(r'"\$py" - <<\'PY\'\n(.*?)\nPY', script, re.S)[-1]
    site = tmp_path / "lib/site-packages"
    package = site / "transformer_engine"
    header = site / "include/cutlass3/cutlass/cutlass.h"
    header.parent.mkdir(parents=True)
    if failure != "headers":
        header.touch()
    torch = ModuleType("torch")
    torch.__file__ = str(tmp_path / "sdk/torch/__init__.py")
    torch.cuda = SimpleNamespace(
        is_available=lambda: failure != "device", get_device_capability=lambda _: (8, 9)
    )
    sdk = {"version": "2.11.0", "path": torch.__file__}
    (tmp_path / "rtc-ci-sdk.json").write_text(json.dumps(sdk))
    if failure == "torch-path":
        torch.__file__ = str(tmp_path / "other/torch/__init__.py")
    te = ModuleType("transformer_engine")
    te.__file__ = str(package / "__init__.py")
    if failure == "system-te":
        te.__file__ = "/system/transformer_engine/__init__.py"
    te.pytorch = ModuleType("transformer_engine.pytorch")
    te.pytorch.__file__ = str(package / "pytorch/__init__.py")
    common = ModuleType("transformer_engine.common")
    common._get_shared_object_file = lambda name: (
        Path("/system/libtransformer_engine.so")
        if failure == "system-core" and name == "core"
        else site / f"{name}.so"
    )
    gemm = ModuleType("transformer_engine.pytorch.cpp_extensions.gemm")
    gemm.general_grouped_gemm = lambda: None
    for name, module in {
        "torch": torch,
        "transformer_engine": te,
        "transformer_engine.pytorch": te.pytorch,
        "transformer_engine.common": common,
        "transformer_engine.pytorch.cpp_extensions.gemm": gemm,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda _: "2.10.0" if failure == "torch-version" else "2.11.0",
    )
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if failure == "pydantic" and name == "transformer_engine.pytorch":
            raise ModuleNotFoundError("No module named 'pydantic'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    if failure:
        with pytest.raises(ModuleNotFoundError if failure == "pydantic" else AssertionError):
            exec(compile(code, "rtc-preflight", "exec"), {})
    else:
        exec(compile(code, "rtc-preflight", "exec"), {})


def test_workflow_report_paths():
    yaml = pytest.importorskip("yaml")
    workflow = yaml.load(
        (ROOT / ".github/workflows/ppu-890-rtc-grouped-gemm-gate.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    steps = workflow["jobs"]["gate"]["steps"]
    report = next(step for step in steps if "--tolerance" in step.get("run", ""))
    assert report["env"]["REPORT_OUT"] == "${{ runner.temp }}/rtc-grouped-gemm-report"
    assert "$GITHUB_ENV" not in report["run"]
    assert '--input "$NAS_OUT" --output "$REPORT_OUT"' in report["run"]
    assert 'mkdir -p "$NAS_OUT"' not in report["run"]
    assert 'cat "$REPORT_OUT/gate.md"' in report["run"]
    assert 'exit "$rc"' in report["run"]
    comment = workflow["jobs"]["publish"]["steps"][-1]
    assert "process.env.REPORT_OUT" in comment["with"]["script"]
    assert "dir + '/gate.md'" in comment["with"]["script"]
    assert comment["env"]["REPORT_OUT"] == report["env"]["REPORT_OUT"]
    deps = next(step for step in steps if "GATE_DEPS" in step.get("env", {}))
    assert deps["env"]["GATE_DEPS"] == report["env"]["PYTHONPATH"]
    assert "--plot" in report["run"]
    artifact = next(
        step for step in steps if "base/**/*.json" in step.get("with", {}).get("path", "")
    )
    paths = artifact["with"]["path"].splitlines()
    assert "${{ runner.temp }}/rtc-grouped-gemm-report/gate.*" in paths
    assert not any("env.REPORT_OUT" in path for path in paths)
    assert "${{ env.NAS_OUT }}/*.log" in paths
    assert "${{ env.NAS_OUT }}/base/**/*.json" in paths
    assert "${{ env.NAS_OUT }}/head/**/*.json" in paths


@pytest.mark.parametrize("missing", [False, True])
def test_report_paths_after_failed_step(tmp_path, missing):
    yaml = pytest.importorskip("yaml")
    workflow = yaml.load(
        (ROOT / ".github/workflows/ppu-890-rtc-grouped-gemm-gate.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    steps = workflow["jobs"]["gate"]["steps"]
    report = next(step for step in steps if "--tolerance" in step.get("run", ""))
    source = tmp_path / "nas"
    if not missing:
        for arm in ("base", "head"):
            for sweep in range(4):
                directory = source / arm / f"sweep-{sweep}"
                directory.mkdir(parents=True)
                (directory / "results.json").write_text(json.dumps(payload([100.0] * 27)))
    directory = report["env"]["REPORT_OUT"].replace("${{ runner.temp }}", str(tmp_path))
    summary = tmp_path / "summary.md"
    env = {
        **os.environ,
        "NAS_OUT": str(source),
        "REPORT_OUT": directory,
        "BASE_SHA": "base",
        "HEAD_SHA": "head",
        "POLICY": "advisory",
        "GITHUB_STEP_SUMMARY": str(summary),
    }
    env.pop("GITHUB_ENV", None)
    result = subprocess.run(
        ["bash"], input=report["run"], cwd=ROOT, env=env, text=True, capture_output=True
    )
    assert result.returncode == (3 if missing else 0), result.stderr
    assert ("ERROR" if missing else "PASS") in summary.read_text()
    # 发布 job 通过 artifact 接收报告，不依赖前一 job 的环境。
    comment = workflow["jobs"]["publish"]["steps"][-1]
    comment_directory = comment["env"]["REPORT_OUT"].replace("${{ runner.temp }}", str(tmp_path))
    assert (Path(comment_directory) / "gate.md").is_file()
    artifact = next(step for step in steps if step.get("uses") == "actions/upload-artifact@v4")
    pattern = artifact["with"]["path"].splitlines()[0].replace("${{ runner.temp }}", str(tmp_path))
    assert {p.name for p in Path(pattern).parent.glob(Path(pattern).name)} == {
        "gate.md",
        "gate.json",
        "gate.png",
    }


def test_workflow_publisher_isolated():
    workflow = workflow_config()
    gate, publisher = workflow["jobs"]["gate"], workflow["jobs"]["publish"]
    assert gate["permissions"] == {"contents": "read"}
    assert publisher["permissions"] == {"contents": "write", "pull-requests": "write"}
    assert publisher["needs"] == "gate"
    assert "always()" in publisher["if"] and "!cancelled()" in publisher["if"]
    assert "head.repo.full_name == github.repository" in publisher["if"]
    assert "github.event_name == 'pull_request'" in publisher["if"]
    assert publisher["steps"][0]["uses"] == "actions/download-artifact@v4"
    publish = publisher["steps"][-1]
    assert publish["env"]["PR_NUMBER"] == "${{ github.event.pull_request.number }}"
    assert "context.issue" not in publish["with"]["script"]
    assert publisher["steps"][0]["with"]["artifact-ids"] == "${{ needs.gate.outputs.report_id }}"
    assert gate["outputs"]["report_id"] == "${{ steps.report_artifact.outputs.artifact-id }}"
    assert not any(
        step.get("run") or step.get("uses", "").startswith("actions/checkout")
        for step in publisher["steps"]
    )
    artifact = next(step for step in gate["steps"] if step.get("id") == "report_artifact")
    assert "always()" in artifact["if"]
    assert artifact["with"]["path"] == "${{ runner.temp }}/rtc-grouped-gemm-report/gate.*"


@pytest.mark.parametrize("missing,regression", [(False, False), (False, True), (True, False)])
def test_report_plot_cli(tmp_path, missing, regression):
    if not missing:
        for arm, latency in (("base", 100.0), ("head", 106.0 if regression else 98.0)):
            directory = tmp_path / arm / "sweep-0"
            directory.mkdir(parents=True)
            (directory / "results.json").write_text(json.dumps(payload([latency] * 27)))
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / ".github/scripts/rtc_grouped_gemm_gate.py"),
            "report",
            "--output",
            str(tmp_path),
            "--base-sha",
            "base",
            "--head-sha",
            "head",
            "--sweeps",
            "1",
            "--plot",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == (3 if missing else 2 if regression else 0), result.stderr
    assert (tmp_path / "gate.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    from PIL import Image

    with Image.open(tmp_path / "gate.png") as image:
        assert image.size == (2240, 1260)


@pytest.mark.parametrize("missing,regression", [(False, False), (False, True), (True, False)])
def test_plot_failure_preserves_verdict(tmp_path, monkeypatch, capsys, missing, regression):
    if not missing:
        for arm, latency in (("base", 100.0), ("head", 106.0 if regression else 98.0)):
            directory = tmp_path / arm / "sweep-0"
            directory.mkdir(parents=True)
            (directory / "results.json").write_text(json.dumps(payload([latency] * 27)))

    def unavailable(*_):
        raise ImportError("模拟绘图依赖不可用")

    monkeypatch.setattr(GATE, "plot_report", unavailable)
    args = argparse.Namespace(
        output=str(tmp_path),
        cases=GATE.DEFAULT_CASES,
        sweeps=1,
        tolerance=0.05,
        policy="enforced",
        base_sha="base",
        head_sha="head",
        plot=True,
    )
    assert GATE.report(args) == (3 if missing else 2 if regression else 0)
    verdict = "ERROR" if missing else "REGRESSION" if regression else "PASS"
    assert json.loads((tmp_path / "gate.json").read_text())["verdict"] == verdict
    assert verdict in (tmp_path / "gate.md").read_text()
    assert "图表生成失败" in capsys.readouterr().err
    assert not (tmp_path / "gate.png").exists()


@pytest.mark.parametrize("error", [False, True])
def test_plot_paired_data_and_error(tmp_path, monkeypatch, error):
    import matplotlib.figure
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    rows = GATE.compare(CASES, [(payload([100.0] * 27), payload([110.0] * 27))], 0.05)
    # 强制独立中位数与配对比率不同，确认下图采用门禁真正使用的比率。
    rows[0].update(latency_ratio=0.98, regression=False)
    gate = {
        "rows": rows,
        "error": "缺测" if error else None,
        "verdict": "ERROR" if error else "REGRESSION",
        "base_sha": "base",
        "head_sha": "head",
        "sweeps": 4,
        "tolerance": 0.05,
    }
    figures = []
    original = matplotlib.figure.Figure.savefig

    def save(figure, *args, **kwargs):
        figures.append(figure)
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", save)
    GATE.plot_report(gate, tmp_path / "gate.png")
    axes = figures[0].axes
    if error:
        assert not any(axis.patches for axis in axes)
        assert "ERROR" in axes[0].texts[0].get_text()
    else:
        assert len(axes[0].patches) == 54 and len(axes[1].patches) == 27
        assert axes[0].patches[0].get_height() == 100
        assert axes[0].patches[27].get_height() == 110
        assert axes[1].patches[0].get_height() == pytest.approx(-2)
        assert axes[1].patches[0].get_facecolor() == to_rgba("#2ca02c")
        assert axes[1].patches[1].get_facecolor() == to_rgba("#d62728")
        assert axes[1].lines[1].get_ydata() == [5.0, 5.0]
    assert not plt.fignum_exists(figures[0].number)


@pytest.mark.parametrize(
    "scenario",
    [
        "new",
        "existing",
        "conflict",
        "create-race",
        "forbidden",
        "missing-png",
        "bad-png",
        "oversized",
        "exhausted",
        "error-report",
        "missing-commit",
        "missing-pr",
        "invalid-pr",
        "zero-pr",
        "fractional-pr",
        "unsafe-pr",
    ],
)
@pytest.mark.parametrize("context_number", [38, None, 999])
def test_publisher_api(scenario, context_number):
    import quickjs

    script = workflow_config()["jobs"]["publish"]["steps"][-1]["with"]["script"]
    vm = quickjs.Context()
    vm.eval("const scenario = " + json.dumps(scenario) + ";")
    vm.eval("const contextNumber = " + json.dumps(context_number) + ";")
    vm.eval(
        r"""
        const calls = [], warnings = [];
        let done = false, failure = null, branches = 0, commits = 0, updates = 0, reads = 0;
        const context = {serverUrl: 'https://github.com', repo: {owner: 'owner', repo: 'repo'},
                         runId: 123};
        // 事件文件缺失时，禁止通过 context.issue 获取 PR 编号。
        Object.defineProperty(context, 'issue', {get: () => {
            if (contextNumber === null) throw new Error('事件文件不存在');
            return {number: contextNumber};
        }});
        const invalidPr = {'missing-pr': undefined, 'invalid-pr': 'abc', 'zero-pr': '0',
                           'fractional-pr': '38.5', 'unsafe-pr': '9007199254740992'};
        const prNumber = scenario in invalidPr ? invalidPr[scenario] : '38';
        const process = {env: {REPORT_OUT: '/report', GITHUB_RUN_ATTEMPT: '2', PR_NUMBER: prNumber}};
        const core = {warning: message => warnings.push(message)};
        function fail(status) { throw Object.assign(new Error('模拟 API 失败'), {status}); }
        const fs = {readFileSync: path => {
            reads++;
            if (path.endsWith('.md')) return scenario === 'error-report' ? 'ERROR: 缺测' : 'PASS';
            if (scenario === 'missing-png') throw new Error('PNG 不存在');
            return {length: scenario === 'oversized' ? 6000000 : 8,
                    subarray: () => ({toString: () => scenario === 'bad-png' ? 'bad' : '89504e470d0a1a0a'}),
                    toString: () => 'iVBORw0KGgo='};
        }};
        const require = name => { if (name !== 'fs') throw new Error(name); return fs; };
        const git = {};
        for (const name of ['createBlob', 'getRef', 'getCommit', 'createTree',
                            'createCommit', 'createRef', 'updateRef']) {
            git[name] = async args => {
                calls.push({name, args});
                if (scenario === 'forbidden') fail(403);
                if (name === 'getRef') {
                    branches++;
                    if (['new', 'create-race', 'error-report'].includes(scenario) && branches === 1) fail(404);
                    return {data: {object: {sha: 'parent-' + branches}}};
                }
                if (name === 'getCommit') {
                    if (scenario === 'missing-commit') fail(404);
                    return {data: {tree: {sha: 'old-tree'}}};
                }
                if (name === 'createCommit') return {data: {sha: 'commit-' + (++commits)}};
                if (name === 'createRef' && scenario === 'create-race') fail(422);
                if (name === 'updateRef') {
                    updates++;
                    if (scenario === 'exhausted' || (scenario === 'conflict' && updates === 1)) fail(422);
                }
                return {data: {sha: name === 'createBlob' ? 'blob' : 'tree'}};
            };
        }
        const issues = {listComments: 'list',
                        createComment: async args => calls.push({name: 'createComment', args}),
                        updateComment: async args => calls.push({name: 'updateComment', args})};
        const github = {rest: {git, issues}, paginate: async (method, args) => {
            calls.push({name: 'listComments', args});
            return scenario === 'existing' ? [
                {id: 7, user: {login: 'someone'}, body: '<!-- te-rtc-grouped-gemm-gate:ZW-M890P -->'},
                {id: 9, user: {login: 'github-actions[bot]'}, body: '<!-- te-rtc-grouped-gemm-gate:ZW-M890P -->'}
            ] : [];
        }};
    """
    )
    vm.eval(
        "(async () => {\n"
        + script
        + "\n})().then(() => done = true, error => {failure = String(error); done = true;});"
    )
    for _ in range(200):
        if not vm.execute_pending_job():
            break
    assert vm.eval("done")
    calls = json.loads(vm.eval("JSON.stringify(calls)"))
    if scenario in ("missing-pr", "invalid-pr", "zero-pr", "fractional-pr", "unsafe-pr"):
        assert "PR_NUMBER" in vm.eval("failure")
        assert not calls and vm.eval("reads") == 0
        return
    assert vm.eval("failure") is None
    comments = [call for call in calls if call["name"] in ("createComment", "updateComment")]
    assert len(comments) == 1
    body = comments[0]["args"]["body"]
    assert body.startswith("<!-- te-rtc-grouped-gemm-gate:ZW-M890P -->")
    assert ("ERROR: 缺测" if scenario == "error-report" else "PASS") in body
    assert "actions/runs/123" in body
    if scenario == "existing":
        assert comments[0]["name"] == "updateComment" and comments[0]["args"]["comment_id"] == 9
    fallback = scenario in (
        "forbidden",
        "missing-png",
        "bad-png",
        "oversized",
        "exhausted",
        "missing-commit",
    )
    if fallback:
        assert "图表未能内嵌" in body and "![" not in body
        assert json.loads(vm.eval("JSON.stringify(warnings)"))
    else:
        assert "/raw/commit-" in body and "/rtc-grouped-gemm/pr-38/run-123-2/gate.png" in body
        assert "![RTC GroupGemm" in body
    for call in calls:
        args = call["args"]
        if "ref" in args:
            assert args["ref"] in ("heads/ci-reports", "refs/heads/ci-reports")
        if call["name"] in ("listComments", "createComment"):
            assert args["issue_number"] == 38
        if call["name"] == "createCommit":
            assert "PR #38" in args["message"]
        if call["name"] == "updateRef":
            assert args["force"] is False
        if call["name"] == "createTree":
            assert args["tree"] == [
                {
                    "path": "rtc-grouped-gemm/pr-38/run-123-2/gate.png",
                    "mode": "100644",
                    "type": "blob",
                    "sha": "blob",
                }
            ]
    commits = [call for call in calls if call["name"] == "createCommit"]
    if scenario in ("new", "create-race", "error-report"):
        assert commits[0]["args"]["parents"] == []
    if scenario in ("conflict", "create-race"):
        assert len(commits) == 2
        assert commits[1]["args"]["parents"] == ["parent-2"]
    if scenario == "exhausted":
        assert len(commits) == 5
    if scenario == "missing-commit":
        assert not any(
            call["name"] in ("createTree", "createCommit", "updateRef", "createRef")
            for call in calls
        )
    trees = [call for call in calls if call["name"] == "createTree"]
    for tree, commit in zip(trees, commits):
        assert tree["args"].get("base_tree") == ("old-tree" if commit["args"]["parents"] else None)


def test_worker_failure_is_recorded(tmp_path):
    args = argparse.Namespace(
        output=str(tmp_path),
        cases=GATE.DEFAULT_CASES,
        replay=str(tmp_path / "nonexistent.py"),
        python=sys.executable,
        warmup=1,
        iters=1,
        samples=1,
        timeout=5,
        only_case=1,
    )
    assert GATE.measure(args) == 1
    result = json.loads((tmp_path / "results.json").read_text())
    assert len(result["errors"]) == 1
    assert not result["results"]


def test_workflow_bootstraps_cpu_python():
    yaml = pytest.importorskip("yaml")
    path = ROOT / ".github/workflows/ppu-890-rtc-grouped-gemm-gate.yml"
    workflow = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
    steps = workflow["jobs"]["gate"]["steps"]
    setups = [i for i, step in enumerate(steps) if step.get("uses") == "actions/setup-python@v5"]
    assert len(setups) == 1, "精简 runner 必须显式准备包含 pip 的 Python"
    setup_index = setups[0]
    setup = steps[setup_index]
    assert setup["with"]["python-version"] == "3.12"
    assert setup["with"]["update-environment"] == "true"
    assert setup["env"]["AGENT_TOOLSDIRECTORY"] == "${{ runner.temp }}/rtc-python"
    assert not setup.get("if")
    assert setup.get("continue-on-error", "false") == "false"
    # 仅检查 runner 的 run 步骤；PPU pod 的 command 使用 SDK 镜像自己的 Python。
    consumers = [(i, step) for i, step in enumerate(steps) if "python3" in step.get("run", "")]
    assert len(consumers) >= 3
    for index, step in consumers:
        assert index > setup_index, "安装依赖、范围判定和报告都必须在 Python 准备之后"
        assert step["shell"] == "bash"
    install = next(step["run"] for _, step in consumers if "python3 -m pip install" in step["run"])
    assert "python3 -m pip --version" in install
    assert install.index("python3 -m pip --version") < install.index("python3 -m pip install")
    assert '--target "$GATE_DEPS"' in install
    assert "pytest==8.2.1 pyyaml==6.0.2 matplotlib==3.10.0 quickjs==1.19.4" in install
    assert "--only-binary=:all:" in install


def test_workflow_and_shell_syntax():
    yaml = pytest.importorskip("yaml")
    path = ROOT / ".github/workflows/ppu-890-rtc-grouped-gemm-gate.yml"
    workflow = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
    assert "pull_request" in workflow["on"] and "pull_request_target" not in workflow["on"]
    job = workflow["jobs"]["gate"]
    assert "head.repo.full_name == github.repository" in job["if"]
    steps = job["steps"]
    assert steps[0]["with"]["persist-credentials"] == "false"
    for step in steps:
        command = step.get("run") or step.get("with", {}).get("command")
        if not command:
            continue
        subprocess.run(["bash", "-n"], input=command, text=True, check=True, capture_output=True)
        python = re.search(r"python3 - <<'PY'\n(.*?)\nPY", command, re.S)
        if python:
            compile(python[1], str(path), "exec")
    subprocess.run(
        ["bash", "-n", str(ROOT / "qa/L0_pytorch_grouped_gemm_rtc/test.sh")],
        check=True,
        capture_output=True,
    )
    report_step = next(s for s in steps if "--tolerance" in s.get("run", ""))
    assert "always()" in report_step["if"]
    assert "--tolerance 0.05" in report_step["run"]


def study_cpu_functions():
    """只加载采集器的纯函数，不执行CLI、导入torch或创建GPU context。"""
    import ast
    import statistics

    namespace = {"statistics": statistics, "hashlib": hashlib, "Path": Path, "json": json, "re": re}
    for filename, names in (("atomic_study.py", {"nexpand_case_manifest", "nexpand_statistics",
                                               "nexpand_validate_codegen", "nexpand_validate_capture",
                                               "nexpand_validate_current_anchor", "nexpand_validate_runtime", "nexpand_csv_row"}),
                             ("repro.py", {"packed_study_config", "packed_study_environment"})):
        path = ROOT / "profile/gg_fp32_accum" / filename
        tree = ast.parse(path.read_text())
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        assert {node.name for node in nodes} == names
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


def test_nexpand_24_case_contract():
    functions = study_cpu_functions()
    cases = functions.nexpand_case_manifest()
    assert len(cases) == 26 and sum(not c["control"] for c in cases) == 24
    assert len({c["id"] for c in cases}) == 26
    for case in cases:
        k, n = case["model_KN"]
        expected = {"fwd": [2048, n, k], "dgrad": [2048, k, n], "wgrad": [n, k, 2048]}
        assert case["MNK"] == expected[case["direction"]]
        assert case["groups"] == 16 and case["splits"] == [2048] * 16
        assert case["beta"] == int(case["direction"] == "wgrad")
        assert case["dtype"] == ("fp32" if case["beta"] else "bf16")
    assert all(c["MNK"] == [2048, 4096, 2048] and not c["beta"] for c in cases[-2:])


@pytest.mark.parametrize("ne,kmax,expected", [(4, 512, 2), (4, 1024, 6), (4, 2048, 8),
                                              (8, 512, 2), (8, 1024, 4), (8, 2048, 6)])
def test_nexpand_admission_inventory(ne, kmax, expected):
    functions = study_cpu_functions()
    if ne == 8:
        for case in functions.nexpand_case_manifest():
            if not case["beta"]:
                with pytest.raises(AssertionError, match="NE8 is not a supported contiguous runtime arm"):
                    functions.packed_study_config("contiguous", case["layout"], 0, ne, kmax,
                                                 *case["MNK"], case["groups"])
        return
    configs = [functions.packed_study_config("contiguous", c["layout"], c["beta"],
               1 if c["beta"] else ne, kmax, *c["MNK"], c["groups"])
               for c in functions.nexpand_case_manifest()[:24]]
    assert sum(c["effective_ne"] > 1 for c in configs) == expected
    assert sum(c["bypass_reason"] == "withC_NE1" for c in configs) == 8


@pytest.mark.parametrize("ne,kmax", [(4, 512), (4, 1024), (4, 2048), (8, 512), (8, 1024), (8, 2048)])
def test_nexpand_k2048_controls(ne, kmax):
    functions = study_cpu_functions()
    for case in functions.nexpand_case_manifest()[-2:]:
        if ne == 8:
            with pytest.raises(AssertionError, match="NE8 is not a supported contiguous runtime arm"):
                functions.packed_study_config("contiguous", case["layout"], 0, ne, kmax,
                                             *case["MNK"], case["groups"])
            continue
        config = functions.packed_study_config("contiguous", case["layout"], 0, ne, kmax,
                                               *case["MNK"], case["groups"])
        assert config["effective_ne"] == (ne if kmax == 2048 else 1)
        assert config["expected_work"] == 16 * 8 * 16 // config["effective_ne"]


def test_nexpand_environment_isolation():
    functions = study_cpu_functions()
    environment = {"PATH": "/sdk/bin", "NVTE_GG_FORCE_TILE": "bad", "NVTE_GG_FORCE_WITH_C": "0",
                   "NVTE_GG_EXTRA_DEFINES": "-DBAD=1", "NVTE_GG_RTC_LOAD_CUBIN_DIR": "/old",
                   "NVTE_GG_ALLOW_FP32_ACCUM": "1", "NVTE_GG_RTC_LINEINFO": "1",
                   "NVTE_CUTLASS3_INCLUDE_DIR": "/frozen", "NVTE_GG_N_EXPAND": "8"}
    config = functions.packed_study_config("production", "TN", 0, 1, 512, 2048, 4096, 512, 16)
    env = functions.packed_study_environment(environment, config, "/new")
    assert env["NVTE_GG_CONTIGUOUS"] == "0" and env["NVTE_GG_RTC_CACHE_DIR"] == "/new"
    assert env["NVTE_GG_ALLOW_FP32_ACCUM"] == env["NVTE_GG_RTC_LINEINFO"] == "1"
    assert env["NVTE_CUTLASS3_INCLUDE_DIR"] == "/frozen"
    assert not any(key in env for key in ("NVTE_GG_FORCE_TILE", "NVTE_GG_EXTRA_DEFINES",
                                        "NVTE_GG_FORCE_WITH_C", "NVTE_GG_N_EXPAND", "NVTE_GG_RTC_LOAD_CUBIN_DIR"))
    config = functions.packed_study_config("contiguous", "TN", 0, 4, 512, 2048, 4096, 512, 16)
    env = functions.packed_study_environment(environment, config, "/new")
    assert env["NVTE_GG_CONTIGUOUS_BN"] == "256" and env["NVTE_GG_CONTIGUOUS_N_EXPAND"] == "4"
    assert "NVTE_GG_CONTIGUOUS_N_EXPAND_K_MAX" not in env and "NVTE_GG_FORCE_TILE" not in env


def test_contiguous_aligned_environment_isolation():
    functions = study_cpu_functions()
    inherited = {"NVTE_GG_CONTIGUOUS_ALIGNED": "1", "PATH": "/sdk/bin"}
    for arm in ("production", "contiguous"):
        config = functions.packed_study_config(arm, "TN", 0, 4, 512, 2048, 2048, 512, 16)
        assert "NVTE_GG_CONTIGUOUS_ALIGNED" not in functions.packed_study_environment(inherited, config, "/new")
        assert config["full_tile"] == (arm == "contiguous")


def test_contiguous_aligned_host_device_wiring():
    directory = ROOT / "transformer_engine/common/gemm"
    host = (directory / "cutlass_grouped_gemm_rtc.cu").read_text()
    kernel = (directory / "actlize_overlay/include/cutlass/gemm/kernel/nvte_contiguous_grouped.hpp").read_text()
    epi = (directory / "actlize_overlay/include/cutlass/epilogue/collective/ppu_epilogue_vectorized_evt.hpp").read_text()
    rtc = (directory / "rtc/cutlass_grouped_gemm.cu").read_text()
    assert "contiguous_aligned" not in host and "bool Aligned" not in kernel
    assert 'cfg.contiguous_mode ? "_fulltile" : ""' in host
    assert "NVTE_GG_CONTIGUOUS_ALIGNED" not in host
    assert "NVTE_GG_CONTIGUOUS_ALIGNED" not in rtc
    assert "ptr != 0 && ptr % 16 == 0" in host
    assert re.search(r"compute_contiguous<WithC>\([\s\S]*?acc, mma, tid, storage\.c_pool\);", kernel)
    assert "template <bool WithC, class TileShapeType, class Coord, class Acc, class Mma>" in epi
    assert "Acc const& acc, Mma mma, int thread_idx, char* c_pool)" in epi
    contiguous_epi = epi.split("CUTLASS_DEVICE static void compute_contiguous(", 1)[1]
    assert "FullTile" not in contiguous_epi and "residue_mnk" not in contiguous_epi
    assert "static_assert(fragment_origins_aligned" in contiguous_epi
    assert "Copy::get_layoutD_TV()" in contiguous_epi
    assert "max_common_vector(registers, global)" in contiguous_epi
    assert re.search(r"\(void\) vector_bytes\(result, d\);\s*copy\(copy_c, result, d\);", contiguous_epi)
    assert re.search(r"\(void\) vector_bytes\(source, d\);\s*copy\(copy_c, d, source\);", contiguous_epi)
    # Ordinary RTC still predicates both C reads and D writes on partial fragments.
    ordinary_epi = epi.split("CUTLASS_DEVICE static void compute_contiguous(", 1)[0]
    assert "if (any_valid)" in ordinary_epi and "if (all_valid)" in ordinary_epi
    assert "clear(tCrC_frag)" in ordinary_epi and "tCrC_frag(i) = tCgC_frag(i)" in ordinary_epi
    assert "get<0>(coord) < get<0>(residue_mnk)" in ordinary_epi
    assert "get<1>(coord) < get<1>(residue_mnk)" in ordinary_epi
    assert "tCaD_frag(i) = tDrD_frag(i)" in ordinary_epi


def test_contiguous_full_tile_mainloop_and_params():
    directory = ROOT / "transformer_engine/common/gemm"
    kernel = (directory / "actlize_overlay/include/cutlass/gemm/kernel/nvte_contiguous_grouped.hpp").read_text()
    loop = (directory / "actlize_overlay/include/cutlass/gemm/collective/ppu_mma_aiu_multistage_batch_array.hpp").read_text()
    params = (directory / "actlize_overlay/include/cutlass/gemm/kernel/nvte_contiguous_grouped_params.hpp").read_text()
    assert "sizeof(Params) == 80" in params
    assert "__builtin_offsetof(Params, work_count) == 40" in params
    assert "__builtin_offsetof(Params, beta) == 76" in params
    assert 'static_assert(NExpand == 1 || BN == 256' in kernel
    assert "if constexpr (WithC) { if (w.k == 0) continue; }" in kernel
    assert "mainloop.compute_contiguous(" in kernel
    assert re.search(r"mainloop\.compute_nexpand\([\s\S]*?w.ni, cute::true_type\{\}\)", kernel)
    assert "class FullTile = cute::false_type" in loop
    assert "static_assert(256 / size<2>(TileShape{}) >= DispatchPolicy::Stages)" in loop
    assert "if constexpr (FullTile::value) issue_initial_stage();" in loop
    assert "else if (k_tile_count > 0) issue_initial_stage();" in loop
    assert "for (int pipe = 0; pipe < DispatchPolicy::Stages; ++pipe) issue_stage(pipe, full_tile)" in loop
    assert "if constexpr (decltype(initial_full_tile)::value) issue_load();" in loop
    assert "else if (remaining > 0) issue_load();" in loop
    # Drain/refill guards remain generic, not the initial full-stage guard.
    assert "issue_stage(read_pipe, cute::false_type{})" in loop
    assert re.search(r"\(\*this\)\(accum, gA, gB, src_accum, k_tile_iter, k_tile_count,\s*"
                     r"cute::true_type\{\}, thread_idx, smem_buf, cute::true_type\{\}\);", loop)
    assert "AfterPrefetch" not in loop


def test_contiguous_python_native_contract_wiring():
    python = (ROOT / "transformer_engine/pytorch/cpp_extensions/gemm.py").read_text()
    native = (ROOT / "transformer_engine/pytorch/csrc/extensions/gemm.cpp").read_text()
    host = (ROOT / "transformer_engine/common/gemm/cutlass_grouped_gemm_rtc.cu").read_text()
    assert "if m and n:" in python and "extent % 256" in python
    assert "split_axis = 2 if layout == \"NT\" else 0" in python
    assert "if actual != expected:" in python
    assert "t.untyped_storage()._cdata != storage._cdata" in python
    assert "t.storage_offset() != offset" in python
    assert "t.numel() and t.data_ptr() % 16" in python
    assert "if (m > 0 && n > 0)" in native and "m % 256 == 0 && n % 256 == 0 && k % 256 == 0" in native
    assert "m_splits[i] == (nt ? k : m)" in native
    assert "storage.unsafeGetStorageImpl() == packed.impl && offset == packed.next_offset" in native
    assert "count <= capacity - offset" in native
    assert "ptr != 0 && ptr % 16 == 0" in native and "ptr != 0 && ptr % 16 == 0" in host
    assert "if (m > 0 && n > 0)" in host and "m % 256 == 0 && n % 256 == 0 && k % 256 == 0" in host
    # Native grouped-storage adapter uses live CSR, validates both cached mirrors,
    # passes actual D dtype, and preserves caller output identity.
    for field in ("first_dims", "last_dims", "tensor_offsets", "operand.offsets", "operand.tensor_shapes"):
        assert field in python
    assert "data.dtype != operand.dtype" in python
    assert "d[0].dtype, layout=layout" in python and "return out" in python


@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
@pytest.mark.parametrize("delta", [-256, -1, 0, 1, 256])
def test_contiguous_study_per_group_threshold_contract(layout, delta):
    config = study_cpu_functions().packed_study_config
    args = ("contiguous", layout, 0, 4, 1024, 256, 1024, 1024 + delta, 2)
    if abs(delta) == 1:
        with pytest.raises(AssertionError, match="per-group MNK256"):
            config(*args)
    else:
        result = config(*args)
        assert result["effective_ne"] == (4 if delta <= 0 else 1)
    # An aligned total still contains invalid individual ragged M or K.
    with pytest.raises(AssertionError, match="per-group MNK256"):
        config("contiguous", layout, 0, 4, 1024, 256, 1024, 256, 2, splits=[128, 384])


def test_grouped_gemm_retired_experiment_identifiers_absent():
    directory = ROOT / "transformer_engine/common/gemm"
    sources = [directory / "cutlass_grouped_gemm_rtc.cu",
               directory / "rtc/cutlass_grouped_gemm.cu"]
    sources += [directory / "actlize_overlay/include/cutlass" / relative for relative in (
        "gemm/kernel/nvte_contiguous_grouped.hpp", "gemm/kernel/ppu_aiu_gemm_array_group.hpp",
        "gemm/collective/ppu_mma_aiu_multistage_batch_array.hpp",
        "epilogue/collective/ppu_epilogue_vectorized_evt.hpp")]
    for path in sources:
        text = path.read_text()
        for retired in ("NVTE_GG_ATOMIC", "NVTE_GG_PROLOGUE_C", "NVTE_GG_C_LATE", "NVTE_GG_C_SMEM",
                        "NVTE_GG_EXTRA_DEFINES", "NVTE_GG_WGRAD_C_FUSION", "NVTE_GG_C_EVT_CLASSIC",
                        "NVTE_GG_FORCE_WITH_C", "NVTE_GG_NO_EXTRA_NVRTC_FLAGS",
                        "NVTE_GG_CONTIGUOUS_MMA_INIT", "mma_init", "MmaInit", "AfterPrefetch"):
            assert retired not in text, (path, retired)
    repro = (ROOT / "profile/gg_fp32_accum/repro.py").read_text()
    assert "mma_init" not in repro and "mma-init" not in repro


def test_contiguous_accumulators_unconditionally_initialized():
    directory = ROOT / "transformer_engine/common/gemm/actlize_overlay/include/cutlass"
    kernel = (directory / "gemm/kernel/nvte_contiguous_grouped.hpp").read_text()
    loop = (directory / "gemm/collective/ppu_mma_aiu_multistage_batch_array.hpp").read_text()
    ordinary = (directory / "gemm/kernel/ppu_aiu_gemm_array_group.hpp").read_text()
    assert re.search(r"Tensor acc = partition_fragment_C\([^;]+;\s*clear\(acc\);", kernel)
    assert re.search(r"store_column\(column\);\s*clear\(acc\);", kernel)
    assert "for (int column = 0; column < NExpand; ++column) {\n        clear(acc)" in kernel
    assert "clear(accumulators);" in ordinary
    assert "cute::gemm(tiled_mma, accum, tCrA(_,_,idx), tCrB(_,_,idx), accum)" in loop
    assert "repeat_like(shape(accum), _0{})" not in loop


def test_nexpand_round_statistics_and_incomplete_rejection():
    functions = study_cpu_functions()
    values = [[40, 42, 60], [45, 44, 46], [43, 44, 45]]
    result = functions.nexpand_statistics(values)
    assert result["round_medians"] == [42, 45, 44] and result["median"] == 44
    assert (result["min"], result["max"]) == (40, 60)
    for invalid in (values[:2], [[1, 2, 3], [1, 2], [1, 2, 3]], [[1, 2, float("nan")]] * 3):
        with pytest.raises(AssertionError):
            functions.nexpand_statistics(invalid)


def test_nexpand_profile_restores_c_and_retains_failure_receipts():
    import ast

    repro = ast.parse((ROOT / "profile/gg_fp32_accum/repro.py").read_text())
    launcher = next(node for node in ast.walk(repro) if isinstance(node, ast.FunctionDef) and node.name == "launch_packed")
    text = ast.unparse(launcher)
    assert text.index("d.copy_(c)") < text.index("general_grouped_gemm(")
    assert "accumulate=bool(a.accumulate)" in text
    study = ast.parse((ROOT / "profile/gg_fp32_accum/atomic_study.py").read_text())
    branch = next(node for node in study.body if isinstance(node, ast.If) and ast.unparse(node.test) == "a.phase == 'nexpand-matrix'")
    profiler = next(node for node in ast.walk(branch) if isinstance(node, ast.FunctionDef) and node.name == "profiled")
    final = next(node for node in ast.walk(profiler) if isinstance(node, ast.Try) and node.finalbody)
    assert "foreign_contenders(log)" in ast.unparse(ast.Module(body=final.finalbody, type_ignores=[]))
    assert "assert target and (not foreign)" in ast.unparse(profiler)


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("kmax", [512, 1024, 2048])
def test_nexpand_compile_kmax_options(mode, kmax):
    """离线入口不把无效Kmax配置添加到production/NE1/default512。"""
    import ast

    path = ROOT / "profile/gg_fp32_accum/repro.py"
    tree = ast.parse(path.read_text())
    branch = next(node for node in tree.body if isinstance(node, ast.If) and
                  ast.unparse(node.test) == "a.compile_only")
    nodes = [node for node in branch.body if isinstance(node, ast.If) and
             "a.compile_kmax" in ast.unparse(node.test)]
    assert len(nodes) == 1
    namespace = {"a": SimpleNamespace(compile_contiguous=mode, compile_kmax=kmax), "defines": {}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    expected = {"NVTE_GG_CONTIGUOUS_N_EXPAND_K_MAX": kmax} if mode > 1 and kmax != 512 else {}
    assert namespace["defines"] == expected


def test_nexpand_kmax_host_device_wiring():
    """门禁前检查三档Traits、模板ABI及RTC映射均接线；实际机器码另验收。"""
    directory = ROOT / "transformer_engine/common/gemm"
    host = (directory / "cutlass_grouped_gemm_rtc.cu").read_text()
    kernel = (directory / "actlize_overlay/include/cutlass/gemm/kernel/nvte_contiguous_grouped.hpp").read_text()
    rtc = (directory / "rtc/cutlass_grouped_gemm.cu").read_text()
    assert "ContiguousTraits<TA, TB, FP32, false, 4, BN," in host
    for kmax in (512, 1024, 2048):
        assert f"compile_expand(cute::Int<{kmax}>{{}})" in host
    assert "cfg.contiguous_kmax = n_expand > 1 ? requested_kmax : 512" in host
    assert "k <= requested_kmax" in host and "w.k <= KMax) {" in kernel
    assert "int NExpand = 1, int KMax = 512" in kernel
    assert "NVTE_GG_CONTIGUOUS_MODE == 2 ? NVTE_GG_CONTIGUOUS_N_EXPAND_K_MAX : 512" in rtc
    assert "std::stoi(kmax_option" not in host
    for field in ("requested_ne=", "effective_ne=", "requested_kmax=", "effective_kmax=",
                  "eligible_groups=", "bypass_reason="):
        assert field in host


@pytest.fixture
def nexpand_codegen_receipt():
    cases = []
    configs = [(mode, bn, c, layout, dtype, 512)
               for mode, bn, c in ((0, 256, 0), (0, 256, 1), (1, 128, 0), (1, 128, 1),
                                  (1, 256, 0), (2, 128, 0), (2, 256, 0))
               for layout in ("TN", "NN", "NT") for dtype in ("bf16", "fp32")]
    configs += [(2, bn, 0, layout, dtype, kmax) for kmax in (1024, 2048)
                for bn in (128, 256) for layout in ("TN", "NN", "NT") for dtype in ("bf16", "fp32")]
    for mode, bn, c, layout, dtype, kmax in configs:
        ta, tb = {"TN": (0, 1), "NN": (0, 0), "NT": (1, 0)}[layout]
        macros = {"NVTE_GG_CONTIGUOUS_MODE": mode, "NVTE_GG_WITH_C": c,
                  "NVTE_GG_TRANS_A": ta, "NVTE_GG_TRANS_B": tb,
                  "NVTE_GG_OUT_FP32": int(dtype == "fp32"), "NVTE_GG_DTYPE": 0,
                  "NVTE_GG_ARCH": 89, "NVTE_GG_TILE_M": 256, "NVTE_GG_TILE_N": bn,
                  "NVTE_GG_TILE_K": 64, "NVTE_GG_STAGES": 3}
        if kmax != 512:
            macros["NVTE_GG_CONTIGUOUS_N_EXPAND_K_MAX"] = kmax
        cases.append({"mode": mode, "bn": bn, "with_C": c, "layout": layout,
                      "dtype": dtype, "kmax": kmax, "compile_result": {
                          "compile_only": "PASS", "gpu_execution": False, "stack_size": 0,
                          "atomic_instruction_count": 0, "options": [f"-D{k}={v}" for k, v in macros.items()]},
                      "default_comparison": {"machine_identity_identical": True,
                                             "resources_identical": True, "raw_cubin_identical": False}
                                             if kmax == 512 else None})
    return {"status": "PASS", "gpu_execution": False, "cases": cases}


def test_nexpand_codegen_complete_inventory(nexpand_codegen_receipt):
    study_cpu_functions().nexpand_validate_codegen(nexpand_codegen_receipt)


@pytest.mark.parametrize("damage", ["partial", "duplicate", "wrong_mode", "stack", "atomic", "gpu",
                                     "machine", "resource", "missing_macro", "duplicate_macro"])
def test_nexpand_codegen_rejects_invalid_receipt(nexpand_codegen_receipt, damage):
    gate = nexpand_codegen_receipt
    if damage == "partial":
        gate["cases"].pop()
    elif damage == "duplicate":
        gate["cases"][-1] = gate["cases"][-2]
    elif damage == "wrong_mode":
        gate["cases"][-1]["mode"] = 1
    elif damage in ("stack", "atomic", "gpu"):
        key = {"stack": "stack_size", "atomic": "atomic_instruction_count", "gpu": "gpu_execution"}[damage]
        gate["cases"][0]["compile_result"][key] = 1
    elif damage in ("machine", "resource"):
        key = "machine_identity_identical" if damage == "machine" else "resources_identical"
        gate["cases"][0]["default_comparison"][key] = False
    elif damage == "missing_macro":
        gate["cases"][-1]["compile_result"]["options"].pop()
    else:
        gate["cases"][-1]["compile_result"]["options"].append("-DNVTE_GG_CONTIGUOUS_N_EXPAND_K_MAX=2048")
    with pytest.raises(AssertionError):
        study_cpu_functions().nexpand_validate_codegen(gate)


@pytest.fixture
def nexpand_capture_receipt():
    functions = study_cpu_functions()
    manifest = functions.nexpand_case_manifest()
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    record = {"status": "PASS", "performance_valid": True, "contention": False, "manifest": manifest,
              "case_digest": digest, "core_24_complete": True, "samples_per_round": 3, "rounds": 3,
              "stage": "baseline", "kmax": 512, "device": {"uuid": "old_gpu"},
              "baseline_audit": {"anchors_sha256": "frozen_anchor"}, "cases": {}}
    for case in manifest:
        summary = {arm: functions.nexpand_statistics([[40, 42, 44]] * 3) for arm in ("production", "contiguous")}
        record["cases"][case["id"]] = {"status": "PASS", "case": case, "summary": summary,
            "order": [["production", "contiguous"] if (case["index"] + r) % 2 == 0
                      else ["contiguous", "production"] for r in range(3)],
            "reports": [{"arm": arm} for _ in range(3) for arm in ("production", "contiguous")],
            "production_anchor_identity": "PASS", "current_anchor_identity": "PASS"}
    return record


@pytest.mark.parametrize("damage", [None, "partial", "failed", "contention", "samples", "wrong_order", "wrong_stage", "wrong_kmax"])
def test_nexpand_complete_capture_gate(nexpand_capture_receipt, damage):
    gate = nexpand_capture_receipt
    case = gate["cases"][gate["manifest"][0]["id"]]
    if damage == "partial":
        gate["cases"].pop(gate["manifest"][-1]["id"])
    elif damage == "failed":
        case["status"] = "FAIL"
    elif damage == "contention":
        gate["contention"] = True
    elif damage == "samples":
        case["summary"]["production"]["samples"] = [[40, 42, 44]] * 2
    elif damage == "wrong_order":
        case["order"][0].reverse()
    elif damage == "wrong_stage":
        gate["stage"] = "kmax"
    elif damage == "wrong_kmax":
        gate["kmax"] = 1024
    if damage is None:
        study_cpu_functions().nexpand_validate_capture(gate, gate["case_digest"], "baseline", 512)
    else:
        with pytest.raises(AssertionError):
            study_cpu_functions().nexpand_validate_capture(gate, gate["case_digest"], "baseline", 512)


@pytest.mark.parametrize("stage", ["baseline", "kmax"])
def test_nexpand_first_cross_device_baseline_not_circular(nexpand_capture_receipt, stage):
    gate = nexpand_capture_receipt
    validate = study_cpu_functions().nexpand_validate_current_anchor
    arguments = (stage, "new_gpu", gate, None, gate["case_digest"], gate["baseline_audit"])
    if stage == "baseline":
        validate(*arguments)
    else:
        with pytest.raises(AssertionError, match="同卡"):
            validate(*arguments)


@pytest.mark.parametrize("damage", [None, "wrong_gpu", "partial", "anchor_machine", "audit_binding"])
def test_nexpand_cross_device_current_anchor(nexpand_capture_receipt, damage):
    import copy

    baseline = nexpand_capture_receipt
    anchor = copy.deepcopy(baseline)
    anchor["device"]["uuid"] = "new_gpu"
    if damage == "wrong_gpu":
        anchor["device"]["uuid"] = "third_gpu"
    elif damage == "partial":
        anchor["cases"].pop(anchor["manifest"][-1]["id"])
    elif damage == "anchor_machine":
        anchor["cases"][anchor["manifest"][0]["id"]]["current_anchor_identity"] = "FAIL"
    elif damage == "audit_binding":
        anchor["baseline_audit"] = {}
    arguments = ("kmax", "new_gpu", baseline, anchor, baseline["case_digest"], baseline["baseline_audit"])
    if damage is None:
        study_cpu_functions().nexpand_validate_current_anchor(*arguments)
    else:
        with pytest.raises(AssertionError):
            study_cpu_functions().nexpand_validate_current_anchor(*arguments)


@pytest.mark.parametrize("damage", [None, "missing", "requested_only", "threshold", "eligible", "reason"])
def test_nexpand_actual_host_launch_receipt(damage):
    functions = study_cpu_functions()
    config = functions.packed_study_config("contiguous", "TN", 0, 4, 1024, 2048, 4096, 1024, 16)
    fields = {"symbol": "actual_symbol", "work": config["expected_work"], "requested_ne": 4,
              "effective_ne": 4, "requested_kmax": 1024, "effective_kmax": 1024,
              "eligible_groups": 16, "bypass_reason": "eligible"}
    if damage == "threshold":
        fields["effective_kmax"] = 512
    elif damage == "eligible":
        fields["eligible_groups"] = 0
    elif damage == "reason":
        fields["bypass_reason"] = "no_eligible_K"
    elif damage == "requested_only":
        fields.pop("effective_ne")
    log = "NVTE_GG_CONTIGUOUS LAUNCHED " + " ".join(f"{k}={v}" for k, v in fields.items()) + " grid=16"
    if damage == "missing":
        log = ""
    if damage is None:
        functions.nexpand_validate_runtime(log, "actual_symbol", config)
    else:
        with pytest.raises(AssertionError):
            functions.nexpand_validate_runtime(log, "actual_symbol", config)


def test_nexpand_metadata_production_and_bypass_match_host():
    functions = study_cpu_functions()
    production = functions.packed_study_config("production", "TN", 0, 1, 2048, 2048, 4096, 1024, 16)
    assert production["mode"] == 0 and production["expected_work"] is None
    for n, k, reason in ((768, 2304, "N_not_divisible"), (4096, 2304, "no_eligible_K")):
        config = functions.packed_study_config("contiguous", "TN", 0, 4, 2048, 2048, n, k, 16)
        assert config["bypass_reason"] == reason and config["effective_kmax"] == 512


def test_nexpand_prerequisite_failure_receipt_without_gpu(tmp_path):
    out = tmp_path / "missing_baseline"
    result = subprocess.run([sys.executable, str(ROOT / "profile/gg_fp32_accum/atomic_study.py"),
        "nexpand-matrix", "--out", str(out), "--nexpand-stage", "kmax", "--nexpand-kmax", "2048",
        "--full", "--samples", "3", "--repeats", "3", "--require-verify", str(tmp_path / "missing_verify")],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), text=True, capture_output=True)
    assert result.returncode != 0
    receipt = json.loads((out / "result.json").read_text())
    assert receipt["status"] == "FAIL" and receipt["failure_phase"] == "prerequisites"
    assert receipt["gpu_execution"] is False and not receipt["performance_valid"]
    assert "阶段0" in receipt["error"]


@pytest.mark.parametrize("complete", [False, True])
def test_nexpand_full_ledger_preserves_negative_and_pending(nexpand_capture_receipt, complete):
    functions = study_cpu_functions()
    record = nexpand_capture_receipt
    rows = []
    for case in record["manifest"]:
        p = functions.nexpand_statistics([[44, 45, 46]] * 3)
        x = functions.nexpand_statistics([[40, 42, 44]] * 3)
        cfg = functions.packed_study_config("contiguous", case["layout"], case["beta"],
              1 if case["beta"] else 4, 1024, *case["MNK"], case["groups"])
        summary = {"production": p, "contiguous": x, "config": cfg,
                   "delta_pp": x["median"] - p["median"], "relative_pct": 100 * (x["median"] / p["median"] - 1)}
        row = functions.nexpand_csv_row(case, summary if complete else None, "gpu", "PASS" if complete else "PENDING", [])
        if complete:
            assert row["delta_pp"] < 0 and row["relative_pct"] < 0
            assert json.loads(row["paired_round_delta_pp"]) == [-3, -3, -3]
            assert len(json.loads(row["current_samples"])) == 3
            assert row["kmax"] == 1024 and row["effective_kmax"] in (512, 1024)
        else:
            assert row["production_tfu"] is row["current_tfu"] is None and row["status"] == "PENDING"
        rows.append(row)
    assert len(rows[:24]) == 24 and len(rows[24:]) == 2
    assert sum(row["beta"] for row in rows[:24]) == 8


@pytest.mark.parametrize("missing", [True, False])
def test_nexpand_independent_audit_fails_closed_without_gpu(tmp_path, nexpand_capture_receipt, missing):
    capture, out = tmp_path / "capture", tmp_path / "audit"
    capture.mkdir()
    if not missing:
        receipt = nexpand_capture_receipt
        receipt["cases"].pop(receipt["manifest"][-1]["id"])
        (capture / "result.json").write_text(json.dumps(receipt))
    result = subprocess.run([sys.executable, str(ROOT / "profile/gg_fp32_accum/atomic_study.py"),
        "audit-nexpand", "--out", str(out), "--nexpand-capture", str(capture)],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), text=True, capture_output=True)
    assert result.returncode != 0
    audit = json.loads((out / "result.json").read_text())
    assert audit["status"] == "FAIL" and audit["gpu_execution"] is False and not audit["performance_valid"]
    assert audit["full_reports"] == audit["full_actions"] == 0


def test_nexpand_independent_audit_keeps_report_parents_alive():
    import ast

    tree = ast.parse((ROOT / "profile/gg_fp32_accum/atomic_study.py").read_text())
    branch = next(node for node in tree.body if isinstance(node, ast.If) and
                  ast.unparse(node.test) == "a.phase == 'audit-nexpand'")
    actions = next(node for node in ast.walk(branch) if isinstance(node, ast.FunctionDef) and node.name == "report_actions")
    returned = next(node for node in actions.body if isinstance(node, ast.Return)).value
    assert isinstance(returned, ast.Tuple)
    assert [ast.unparse(node) for node in returned.elts[:2]] == ["context", "ranges"]
    assert "capture / 'source_snapshot'" in ast.unparse(branch)
    assert "capture / 'libraries'" in ast.unparse(branch)


@pytest.mark.parametrize("failure", ["unbound", "missing", "partial", "overwrite"])
def test_nexpand_codegen_audit_fails_closed_without_gpu(tmp_path, nexpand_codegen_receipt, failure):
    """独立编译审计拒绝缺门禁、残缺清单和覆盖历史回执，不执行GPU。"""
    out, gate = tmp_path / "audit", tmp_path / "codegen.json"
    command = [sys.executable, str(ROOT / "profile/gg_fp32_accum/atomic_study.py"),
               "audit-nexpand-codegen", "--out", str(out)]
    if failure != "unbound":
        command += ["--nexpand-codegen", str(gate)]
    if failure == "partial":
        nexpand_codegen_receipt["cases"].pop()
        gate.write_text(json.dumps(nexpand_codegen_receipt))
    if failure == "overwrite":
        out.mkdir()
        original = json.dumps({"status": "PASS", "historical": True}) + "\n"
        (out / "result.json").write_text(original)
    result = subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                            text=True, capture_output=True)
    assert result.returncode != 0
    if failure == "overwrite":
        assert (out / "result.json").read_text() == original
    else:
        receipt = json.loads((out / "result.json").read_text())
        assert receipt["status"] == "FAIL" and receipt["gpu_execution"] is False
        assert "artifact_sha256" not in receipt
        if failure == "unbound":
            assert "显式绑定66组" in receipt["error"]
