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
        ("docs/installation.rst", False),
        ("transformer_engine/pytorch/attention/dot_product_attention.py", False),
    ],
)
def test_scope(path, expected):
    assert bool(GATE.SCOPE.search(path)) == expected


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
    comment = next(step for step in steps if step.get("uses") == "actions/github-script@v7")
    assert "process.env.REPORT_OUT + '/gate.md'" in comment["with"]["script"]
    assert comment["env"]["REPORT_OUT"] == report["env"]["REPORT_OUT"]
    artifact = next(step for step in steps if step.get("uses") == "actions/upload-artifact@v4")
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
    # 后续步骤只使用自己的 env/表达式，不继承报告步骤的动态环境。
    comment = next(step for step in steps if step.get("uses") == "actions/github-script@v7")
    comment_directory = comment["env"]["REPORT_OUT"].replace("${{ runner.temp }}", str(tmp_path))
    assert (Path(comment_directory) / "gate.md").is_file()
    artifact = next(step for step in steps if step.get("uses") == "actions/upload-artifact@v4")
    pattern = artifact["with"]["path"].splitlines()[0].replace("${{ runner.temp }}", str(tmp_path))
    assert {p.name for p in Path(pattern).parent.glob(Path(pattern).name)} == {
        "gate.md",
        "gate.json",
    }


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
    assert '--target "$RUNNER_TEMP/rtc-gate-deps"' in install
    assert "pytest==8.2.1 pyyaml==6.0.2" in install


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
