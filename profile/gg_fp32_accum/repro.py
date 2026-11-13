# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.

"""PPU RTC grouped GEMM：in-place 累加、尾 tile 与兼容性定位。"""
import argparse
import json
import os
from pathlib import Path

def packed_study_config(arm, layout, accumulate, ne, kmax, m, n, k, groups, splits=None):
    """Independent admission/work ledger, including actual ragged M (TN/NN) or K (NT)."""
    assert arm in ("production", "contiguous")
    assert layout in ("TN", "NN", "NT") and ne in (1, 4, 8)
    assert kmax in (512, 1024, 2048) and min(m, n, groups) > 0 and k >= 0
    assert not accumulate or (layout == "NT" and ne == 1)
    splits = list(splits) if splits is not None else [k if layout == "NT" else m] * groups
    assert len(splits) == groups and all(type(s) is int and s >= 0 for s in splits)
    shapes = [(m, n, s) if layout == "NT" else (s, n, k) for s in splits]
    if arm == "contiguous":
        assert ne in (1, 4), "NE8 is not a supported contiguous runtime arm"
        assert all(not (gm and gn) or all(d % 256 == 0 for d in (gm, gn, gk))
                   for gm, gn, gk in shapes), "NVTE_GG_CONTIGUOUS requires per-group MNK256"
    bn = 128 if accumulate else 256
    eligible_groups = sum(gm > 0 and 0 < gk <= kmax and gn % (bn * ne) == 0
                          for gm, gn, gk in shapes) if not accumulate and ne > 1 else 0
    effective = ne if eligible_groups and arm == "contiguous" else 1
    work = sum((gm // 256) * (gn // bn // effective) for gm, gn, _ in shapes)
    return {"arm": arm, "requested_ne": ne, "effective_ne": effective,
            "kmax": kmax, "effective_kmax": kmax if effective > 1 else 512,
            "tile_n": bn if arm == "contiguous" else None,
            "full_tile": arm == "contiguous",
            "mode": 0 if arm == "production" else 2 if effective > 1 else 1,
            "eligible_groups": eligible_groups if arm == "contiguous" else 0,
            "expected_work": None if arm == "production" else work,
            "splits": splits, "per_group_MNK": shapes,
            "bypass_reason": ("production" if arm == "production" else
                              "withC_NE1" if accumulate else "requested_NE1" if ne == 1 else
                              "no_work" if work == 0 else "eligible" if eligible_groups else
                              "N_not_divisible" if n % (bn * ne) else "no_eligible_K")}


def packed_study_environment(environment, config, cache):
    """只保留RTC采集必需开关，不继承历史tile/epilogue实验。"""
    keep = {"NVTE_GG_ALLOW_FP32_ACCUM", "NVTE_GG_RTC_LINEINFO"}
    env = {key: value for key, value in environment.items()
           if not key.startswith("NVTE_GG_") or key in keep}
    env["NVTE_GG_RTC_CACHE_DIR"] = str(cache)
    env["NVTE_GG_CONTIGUOUS"] = "1" if config["arm"] == "contiguous" else "0"
    if config["arm"] == "contiguous":
        env["NVTE_GG_CONTIGUOUS_BN"] = str(config["tile_n"])
        env["NVTE_GG_CONTIGUOUS_N_EXPAND"] = str(config["requested_ne"])
        # 默认512无需新host开关，阶段0仍使用未修改的生产库。
        if config["kmax"] != 512:
            env["NVTE_GG_CONTIGUOUS_N_EXPAND_K_MAX"] = str(config["kmax"])
    return env


def contiguous_contract_cases(many_groups):
    """Public-runtime probes; each case has its own process, cache, log and trace."""
    cases = []
    for api in ("gemm", "native"):
        for layout in ("TN", "NN", "NT"):
            for dtype in ("bf16", "fp32"):
                for beta in (0, 1):
                    cases.append(dict(id=f"{api}-{layout}-{dtype}-b{beta}-ragged",
                                      suite=api, api=api, layout=layout, dtype=dtype,
                                      beta=beta, splits=[256, 768, 0, 512], repeats=1))
        for size, splits in (("single", [256]), ("multiple", [256, 512]),
                             ("many", [256] * many_groups)):
            cases.append(dict(id=f"{api}-TN-fp32-b1-{size}-persistent", suite=api,
                              api=api, layout="TN", dtype="fp32", beta=1,
                              splits=splits, n=1024, repeats=3))
        for layout in ("TN", "NN", "NT"):
            for dtype in ("bf16", "fp32"):
                for beta in (0, 1):
                    cases.append(dict(id=f"{api}-{layout}-{dtype}-b{beta}-zeroK", suite=api,
                                      api=api, layout=layout, dtype=dtype, beta=beta, k=0,
                                      splits=[0, 0, 0, 0] if layout == "NT" else [256, 768, 0, 512],
                                      zero_k=True, repeats=1))
            for empty_axis in ("M", "N"):
                m, n = (0, 128) if empty_axis == "M" else (128, 0)
                cases.append(dict(id=f"{api}-{layout}-empty-{empty_axis}", suite=api,
                                  api=api, layout=layout, dtype="fp32", beta=1,
                                  splits=[8, 8] if layout == "NT" else [m, m],
                                  m=m, n=n, k=8, no_work=True, repeats=1))
        for dtype in ("bf16", "fp32"):
            for layout, splits, n in (("TN", [256, 768, 0, 512], 1024),
                                      ("NN", [256, 768, 0, 512], 1024),
                                      ("NT", [256, 768, 0, 512], 1024),
                                      ("TN", [256, 768, 0, 512], 768)):
                cases.append(dict(id=f"{api}-{layout}-{dtype}-b0-N{n}", suite=api,
                                  api=api, layout=layout, dtype=dtype, beta=0,
                                  splits=splits, n=n, repeats=3))
    invalid = ("M", "N", "K", "split-count", "split-mismatch", "split-negative",
               "split-fractional", "split-order", "split-sum", "ptr16-A", "ptr16-D",
               "stride-A", "stride-D", "separate-A", "separate-D")
    for bad in invalid:
        cases.append(dict(id=f"invalid-{bad}", suite="invalid", api="gemm", bad=bad,
                          layout="NT" if bad == "K" else "TN", dtype="bf16", beta=1,
                          splits=[256, 768, 0, 512], repeats=1))
    for bad in ("offset-live", "offset-cache", "offset-order", "offset-end",
                "shape-cache", "first-live-cache", "first-negative", "first-count",
                "first-dtype", "ptr16-D"):
        cases.append(dict(id=f"native-invalid-{bad}", suite="invalid", api="native",
                          bad=bad, layout="TN", dtype="fp32", beta=1,
                          splits=[256, 768, 0, 512], repeats=1))
    for bad in (None, "M", "N", "K", "layout", "split-count", "split-negative",
                "split-fractional", "split-order", "stride-A", "stride-D", "ptr16-D",
                "separate-A", "workspace-alignment", "workspace-capacity", "grouped-entry",
                "discrete-in-entry", "discrete-out-entry"):
        cases.append(dict(id=f"direct-{bad or 'valid'}", suite="direct", api="direct",
                          bad=bad, layout="NT" if bad == "K" else "TN", dtype="bf16",
                          beta=0, splits=[256, 768, 0, 512], repeats=1))
    for layout in ("TN", "NN", "NT"):
        for empty_axis in ("M", "N"):
            m, n = (0, 128) if empty_axis == "M" else (128, 0)
            cases.append(dict(id=f"direct-{layout}-empty-{empty_axis}", suite="direct", api="direct",
                              layout=layout, dtype="fp32", beta=1, splits=[8, 8] if layout == "NT" else [m, m],
                              m=m, n=n, k=8, no_work=True, repeats=1))
        for beta in (0, 1):
            cases.append(dict(id=f"direct-{layout}-fp32-b{beta}-zeroK", suite="direct", api="direct",
                              layout=layout, dtype="fp32", beta=beta, k=0, zero_k=True,
                              splits=[0, 0] if layout == "NT" else [256, 0], repeats=1))
    for api in ("module", "ops"):
        for fin, fout in ((256, 1024), (1024, 256)):
            cases.append(dict(id=f"{api}-{fin}x{fout}", suite="linear", api=api,
                              fin=fin, fout=fout, splits=[256, 768, 0, 512]))
        for variant, flags in (
                ("alternate-parameters", {"grouped": api == "ops"}),
                ("main-grad", {"main_grad": True, "repeats": 2}),
                ("main-grad-overwrite", {"main_grad": True, "overwrite": True, "repeats": 2}),
                ("main-grad-dummy", {"main_grad": True, "dummy": True}),
                ("delayed", {"delay": True}),
                ("delayed-main-grad", {"main_grad": True, "delay": True})):
            cases.append(dict(id=f"{api}-{variant}", suite="linear", api=api,
                              fin=256, fout=1024, splits=[256, 768, 0, 512], **flags))
        for bad in ("split-count", "split-sum", "split-negative", "split-fractional",
                    "M", "N", "K", "separate-A", "stride-A", "parameter-dtype"):
            cases.append(dict(id=f"{api}-invalid-{bad}", suite="linear", api=api,
                              fin=128 if bad == "K" else 256,
                              fout=128 if bad == "N" else 1024,
                              bad=bad, splits=[256, 768, 0, 512]))
    for api in ("gemm", "native", "direct"):
        for operand in ("A", "B", "D"):
            cases.append(dict(id=f"{api}-invalid-lazy-negative-{operand}",
                              suite="invalid" if api != "direct" else "direct", api=api,
                              bad=f"lazy-negative-{operand}", layout="TN", dtype="bf16", beta=0,
                              splits=[256, 768, 0, 512], repeats=1))
        for layout in ("TN", "NN"):
            cases.append(dict(id=f"{api}-{layout}-all-empty-experts", suite=api,
                              api=api, layout=layout, dtype="fp32", beta=1,
                              splits=[0, 0, 0, 0], n=128, k=8, no_work=True, repeats=1))
        for layout in ("TN", "NN", "NT"):
            cases.append(dict(id=f"{api}-{layout}-first-last-empty-experts", suite=api,
                              api=api, layout=layout, dtype="bf16", beta=0,
                              splits=[0, 256, 512, 0], n=1024, repeats=3))
    for no_work in (False, True):
        cases.append(dict(id=f"direct-invalid-workspace-backing-{'empty' if no_work else 'work'}",
                          suite="direct", api="direct", bad="workspace-backing", layout="TN",
                          dtype="bf16", beta=0, splits=[0, 0] if no_work else [256, 256],
                          n=128 if no_work else 256, k=8 if no_work else 256,
                          no_work=no_work, repeats=1))
    for api in ("module", "ops"):
        for grouped in (False, True):
            cases.append(dict(id=f"{api}-delayed-{'grouped' if grouped else 'discrete'}-repeat",
                              suite="linear", api=api, fin=256, fout=1024,
                              splits=[256, 768, 0, 512], delay=True, grouped=grouped, repeats=2))
    for api in ("module", "ops"):
        for grouped in (False, True):
            for pattern, splits, fused in (("all-empty", [0, 0, 0, 0], False),
                                          ("all-empty-main-grad", [0, 0, 0, 0], True),
                                          ("first-last-empty", [0, 256, 512, 0], False)):
                cases.append(dict(id=f"{api}-{pattern}-{'grouped' if grouped else 'discrete'}",
                                  suite="linear", api=api, fin=256, fout=1024, grouped=grouped,
                                  splits=splits, main_grad=fused, linear_empty=not any(splits)))
    for grouped in (False, True):
        cases.append(dict(id=f"ops-invalid-main-grad-stride-{'grouped' if grouped else 'discrete'}",
                          suite="linear", api="ops", fin=256, fout=1024,
                          splits=[256, 768, 0, 512], grouped=grouped,
                          main_grad=True, bad="main-grad-stride"))
    for api in ("module", "ops"):
        for grouped in (False, True):
            cases.append(dict(id=f"{api}-forward-owner-{'grouped' if grouped else 'discrete'}",
                              suite="linear", api=api, fin=256, fout=1024,
                              splits=[256, 768, 0, 512], grouped=grouped,
                              main_grad=True, dummy=True, replace_before_backward=True))
            for fused in (False, True):
                cases.append(dict(id=f"{api}-delayed-owner-{'grouped' if grouped else 'discrete'}-{'fused' if fused else 'result'}",
                                  suite="linear", api=api, fin=256, fout=1024,
                                  splits=[256, 768, 0, 512], grouped=grouped, delay=True,
                                  main_grad=fused, dummy=fused, replace_before_wgrad=True))
    for api in ("gemm", "native"):
        for dtype in ("bf16", "fp32"):
            for kmax in (512, 1024, 2048):
                cases.append(dict(id=f"{api}-NT-{dtype}-kmax{kmax}-mixed", suite=api, api=api,
                                  layout="NT", dtype=dtype, beta=0, n=1024, kmax=kmax,
                                  splits=[0, kmax - 256, kmax, kmax + 256], repeats=3))
                for delta in (-256, 0, 256):
                    cases.append(dict(id=f"{api}-TN-{dtype}-kmax{kmax}-delta{delta}",
                                      suite=api, api=api, layout="TN", dtype=dtype, beta=0,
                                      n=1024, k=kmax + delta, kmax=kmax,
                                      splits=[256, 0, 512], repeats=1))
        for kmax in (512, 1024, 2048):
            for delta in (-1, 1):
                cases.append(dict(id=f"{api}-invalid-kmax{kmax}-delta{delta}",
                                  suite="invalid", api=api, layout="TN", dtype="bf16", beta=0,
                                  bad="K-boundary", n=1024, k=kmax + delta, kmax=kmax,
                                  splits=[256, 0, 512], repeats=1))
    for field in ("first_dims", "last_dims", "tensor_offsets"):
        for no_work in (False, True):
            cases.append(dict(id=f"native-invalid-metadata-backing-{field}-{'empty' if no_work else 'work'}",
                              suite="invalid", api="native", bad="metadata-backing", field=field,
                              layout="TN", dtype="fp32", beta=1,
                              splits=[0, 0, 0, 0] if no_work else [256, 768, 0, 512],
                              no_work=no_work, repeats=1))
        cases.append(dict(id=f"native-invalid-metadata-lazy-{field}", suite="invalid", api="native",
                          bad="metadata-lazy", field=field, layout="TN", dtype="fp32", beta=1,
                          splits=[256, 768, 0, 512], repeats=1))
    for api in ("gemm", "module", "ops"):
        cases.append(dict(id=f"{api}-invalid-split-backing", suite="invalid" if api == "gemm" else "linear",
                          api=api, bad="split-backing", layout="TN", dtype="bf16", beta=1,
                          fin=256, fout=1024, splits=[256, 768, 0, 512], repeats=1))
    for no_work in (False, True):
        cases.append(dict(id=f"native-invalid-rowwise-backing-{'empty' if no_work else 'work'}",
                          suite="invalid", api="native", bad="rowwise-backing", layout="TN",
                          dtype="fp32", beta=1, splits=[0, 0, 0, 0] if no_work else [256, 768, 0, 512],
                          no_work=no_work, repeats=1))
    for api in ("module", "ops"):
        cases.append(dict(id=f"{api}-invalid-dgrad-input-alias", suite="linear", api=api,
                          bad="dgrad-input-alias", fin=256, fout=1024,
                          splits=[256, 768, 0, 512]))
        cases.append(dict(id=f"{api}-input-dgrad-disjoint-storage", suite="linear", api=api,
                          input_dgrad_shared=True, fin=256, fout=1024,
                          splits=[256, 768, 0, 512]))
        for grouped in (False, True):
            cases.append(dict(id=f"{api}-delayed-{'grouped' if grouped else 'discrete'}-accumulate",
                              suite="linear", api=api, fin=256, fout=1024,
                              splits=[256, 768, 0, 512], grouped=grouped, delay=True,
                              accumulate_leaf=True, repeats=2))
    cases.append(dict(id="module-invalid-weight-shape-live", suite="linear", api="module",
                      bad="weight-shape-live", fin=256, fout=1024,
                      splits=[256, 768, 0, 512], grouped=True))
    for no_work in (False, True):
        for option in ("NVTE_GG_RTC_LINEINFO", "NVTE_GG_DISABLE_RTC_CACHE",
                       "NVTE_CUTLASS_GROUPED_GEMM_RTC_VERBOSE"):
            cases.append(dict(id=f"direct-invalid-rtc-option-{option}-{'empty' if no_work else 'work'}",
                              suite="direct", api="direct", bad="rtc-option", option=option,
                              layout="TN", dtype="bf16", beta=0,
                              splits=[0, 0] if no_work else [256, 256],
                              n=128 if no_work else 256, k=8 if no_work else 256,
                              no_work=no_work, repeats=1))
    for layout in ("TN", "NN", "NT"):
        cases.append(dict(id=f"ordinary-{layout}-irregular", suite="ordinary", api="gemm",
                          ordinary=True, layout=layout, dtype="fp32", beta=1,
                          splits=[24, 40, 0, 56], repeats=1))
    cases.append(dict(id="gemm-TN-bf16-b0-large-output-zeroK", suite="gemm", api="gemm",
                      layout="TN", dtype="bf16", beta=0, splits=[33024], n=65536, k=0,
                      zero_k=True, large_output=True, repeats=2))
    for representation, layout in (("discrete-in", "TN"), ("discrete-out", "NT")):
        for dtype in ("bf16", "fp32"):
            cases.append(dict(id=f"native-{representation}-{dtype}", suite="native", api="native",
                              representation=representation, layout=layout, dtype=dtype, beta=0, n=1024,
                              splits=[256, 768, 0, 512], repeats=2))
    for option in ("bias", "bias_scale", "alpha", "beta", "grad"):
        cases.append(dict(id=f"native-invalid-api-{option}", suite="invalid", api="native",
                          bad="api-option", option=option, layout="TN", dtype="bf16", beta=0,
                          splits=[256, 768, 0, 512], repeats=1))
    for api in ("gemm", "native", "direct"):
        for no_work in (False, True):
            cases.append(dict(id=f"{api}-invalid-capture-{'empty' if no_work else 'work'}",
                              suite="invalid", api=api, bad="capture", layout="TN", dtype="bf16",
                              beta=0, splits=[0, 0] if no_work else [256, 256],
                              n=128 if no_work else 256, k=8 if no_work else 256,
                              no_work=no_work, repeats=1))
    for invalid in ("workspace-short", "workspace-dtype", "workspace-rank", "workspace-cpu",
                    "workspace-strided", "workspace-missing", "lists"):
        cases.append(dict(id=f"direct-invalid-{invalid}", suite="direct", api="direct", bad=invalid,
                          layout="TN", dtype="bf16", beta=0, splits=[256, 768, 0, 512], repeats=1))
    default_cases = []
    for state in ("unset", "0"):
        default_cases.append(dict(id=f"default-{state}-native-explicit-true", suite="native",
            api="native", default_backend=True, enable_state=state, intent=True,
            layout="TN", dtype="fp32", beta=0, splits=[0, 256, 512, 0], repeats=2))
        default_cases.append(dict(id=f"default-{state}-native-explicit-false-guard", suite="invalid",
            api="native", default_backend=True, enable_state=state, intent=False, bad="false-guard",
            rejection_stage="legacy-grouped-tensor-guard",
            layout="TN", dtype="bf16", beta=0, splits=[256, 256], repeats=1))
        for bad in ("split-bool", "dtype-A", "dtype-D", "output-alias", "workspace-alias"):
            default_cases.append(dict(id=f"default-{state}-direct-invalid-{bad}", suite="direct",
                api="direct", default_backend=True, enable_state=state, bad=bad,
                layout="TN", dtype="bf16", beta=0, splits=[256, 256], repeats=1))
        for case in cases:
            if case["api"] not in ("native", "direct", "module", "ops") and not case.get("ordinary"):
                continue
            if case["api"] == "direct" and case.get("bad") in (
                    "workspace-capacity", "grouped-entry", "discrete-in-entry", "discrete-out-entry"):
                continue
            default_case = dict(case, id=f"default-{state}-{case['id']}",
                                default_backend=True, enable_state=state)
            if case["api"] == "ops" and case.get("bad") == "parameter-dtype":
                default_case["rejection_stage"] = "ops-pre-fuser-weight-consistency"
            default_cases.append(default_case)
        for api in ("module", "ops"):
            default_cases.append(dict(id=f"default-{state}-{api}-invalid-capture-cpu-splits", suite="linear",
                api=api, default_backend=True, enable_state=state, bad="capture", cpu_splits=True,
                grouped=True, fin=256, fout=1024, splits=[256, 0, 512, 0]))
            default_cases.append(dict(id=f"default-{state}-{api}-main-grad-bf16", suite="linear",
                api=api, default_backend=True, enable_state=state, main_grad=True, main_grad_dtype="bf16",
                grouped=True, fin=256, fout=1024, splits=[256, 0, 512, 0], repeats=2))
            default_cases.append(dict(id=f"default-{state}-{api}-legacy-bias", suite="linear",
                api=api, default_backend=True, enable_state=state, ordinary=True,
                legacy_feature="bias", fin=256, fout=256, splits=[7, 9, 0]))
            for delayed in (False, True):
                default_cases.append(dict(id=f"default-{state}-{api}-saved-intent-{'delayed' if delayed else 'backward'}",
                    suite="linear", api=api, default_backend=True, enable_state=state,
                    grouped=True, fin=256, fout=1024, splits=[0, 256, 512, 0],
                    main_grad=True, overwrite=True, repeats=2, delay=delayed,
                    change_enable_after_forward=True))
    return cases + default_cases


def run_contiguous_contract(args):
    """Capture native stdout/stderr outside Python, including RTC host launch receipts."""
    import fnmatch
    import re
    import subprocess
    import sys

    cases = [c for c in contiguous_contract_cases(args.contract_many_groups)
             if c["suite"] in args.contract_scenarios
             and bool(c.get("default_backend")) == args.contract_default_backend]
    if args.contract_case:
        unmatched = [s for s in args.contract_case
                     if not any(fnmatch.fnmatchcase(c["id"], s) for c in cases)]
        assert not unmatched, f"No contract cases match selectors: {unmatched}"
        cases = [c for c in cases if any(fnmatch.fnmatchcase(c["id"], s)
                                        for s in args.contract_case)]
    assert cases, "No contract cases selected"
    result = {"status": "RUNNING", "gpu_execution": True, "cases": [],
              "command": sys.argv, "selected_cases": [c["id"] for c in cases]}
    receipt = args.dump / "contiguous_contract.json"
    assert not receipt.exists(), "Use a fresh --dump directory; existing contract evidence is immutable"
    for case in cases:
        dest = args.dump / case["id"]
        dest.mkdir(exist_ok=False)  # Never overwrite R0 or partial/failed evidence.
        (dest / "case.json").write_text(json.dumps(case, indent=2) + "\n")
        command = [sys.executable, str(Path(__file__).resolve()), "--contiguous-contract",
                   "--contract-worker", case["id"], "--contract-ne", str(args.contract_ne),
                   "--contract-many-groups", str(args.contract_many_groups), "--dump", str(dest)]
        with_c = bool(case.get("beta") or case.get("main_grad"))
        config = {"arm": "production" if case.get("ordinary") else "contiguous",
                  "tile_n": 128 if with_c else 256,
                  "requested_ne": 1 if with_c else args.contract_ne, "kmax": case.get("kmax", 512)}
        env = packed_study_environment(os.environ, config, dest / "cache")
        if case.get("default_backend"):
            for name in ("NVTE_GG_CONTIGUOUS", "NVTE_GG_CONTIGUOUS_BN",
                         "NVTE_GG_CONTIGUOUS_N_EXPAND", "NVTE_GG_CONTIGUOUS_N_EXPAND_K_MAX"):
                env.pop(name, None)
            if case["enable_state"] != "unset":
                env["NVTE_GG_CONTIGUOUS"] = case["enable_state"]
            env["NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM"] = "0"
            env["NVTE_USE_CUTLASS_GROUPED_GEMM"] = "1" if case.get("ordinary") and case["api"] == "gemm" else "0"
            env["NVTE_USE_CUTLASS_GROUPED_GEMM_RTC"] = "1" if case.get("ordinary") and case["api"] == "gemm" else "0"
            env["NVTE_GG_ALLOW_FP32_ACCUM"] = "1" if case.get("ordinary") and case["api"] == "gemm" else "0"
        else:
            for name in ("NVTE_USE_CUTLASS_GROUPED_GEMM", "NVTE_USE_CUTLASS_GROUPED_GEMM_RTC",
                         "NVTE_GG_ALLOW_FP32_ACCUM", "NVTE_FORCE_RTC_NO_FALLBACK",
                         "NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK"):
                env[name] = "1"
        env["NVTE_CUTLASS_GROUPED_GEMM_RTC_VERBOSE"] = "1"
        grouped = case.get("grouped", case["api"] != "ops")
        env["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "1" if grouped else "0"
        (dest / "command.json").write_text(json.dumps({"argv": command,
            "environment": {k: v for k, v in env.items() if k.startswith("NVTE_")}}, indent=2) + "\n")
        with (dest / "stdout.log").open("w") as stream:
            try:
                process = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                         timeout=args.contract_timeout, check=False)
                code = process.returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
        evidence_path = dest / "evidence.json"
        evidence = json.loads(evidence_path.read_text()) if evidence_path.exists() else {
            "status": "FAIL", "error": "Worker did not complete; inspect stdout.log"}
        log = (dest / "stdout.log").read_text()
        launches = [line for line in log.splitlines() if "NVTE_GG_CONTIGUOUS LAUNCHED" in line]
        if "forward_setup" in evidence:
            setup_count = len(evidence["forward_setup"]["rtc_kernels"])
            evidence["setup_host_launch_receipts"] = launches[:setup_count]
            launches = launches[setup_count:]
        evidence.update(returncode=code, stdout=str((dest / "stdout.log").resolve()),
                        host_launch_receipts=launches,
                        host_no_work_receipts=[line for line in log.splitlines()
                                               if "NVTE_GG_CONTIGUOUS NO_WORK" in line])
        evidence["host_receipt_fields"] = [dict(re.findall(r"\b(\w+)=([^\s]+)", line))
                                            for line in launches]
        if not case.get("ordinary"):
            measured = evidence.get("profile", {}).get("rtc_kernels", [])
            if case.get("bad"):
                if launches or measured:
                    evidence.update(status="FAIL", host_error="Rejected probe launched RTC compute")
            elif evidence.get("status") == "PASS":
                if len(launches) != len(measured) or (
                        not launches and not (case.get("zero_k") or case.get("no_work") or case.get("linear_empty"))):
                    evidence.update(status="FAIL", host_error="Missing/mismatched actual host receipts")
                elif case.get("no_work") and (launches or not evidence["host_no_work_receipts"]):
                    evidence.update(status="FAIL", host_error="Empty output did not report host NO_WORK")
                elif any(not any(f"symbol={name} " in line for line in launches) for name in measured):
                    evidence.update(status="FAIL", host_error="Profiler symbol not in actual host receipts")
                else:
                    expected = evidence.get("expected_launches", [])
                    fields = evidence["host_receipt_fields"]
                    matches = len(fields) == len(expected) and all(
                        all(got.get(key) == str(want[key]) for key in
                            ("work", "effective_ne", "eligible_groups", "effective_kmax"))
                        and got.get("params_bytes") == "80"
                        and all(fragment in got.get("symbol", "") for fragment in want["symbol_fragments"])
                        for got, want in zip(fields, expected))
                    evidence["host_admission_work_matches"] = matches
                    if not matches:
                        evidence.update(status="FAIL", host_error="Actual symbol/admission/work differs from per-group workload")
        if code != 0:
            evidence["status"] = "FAIL"
        evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
        row = {"id": case["id"], "status": evidence["status"], "path": str(dest.resolve()),
               "error": evidence.get("error", evidence.get("host_error"))}
        result["cases"].append(row)
        receipt.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"contiguous_contract_case": row}), flush=True)
    result["status"] = "PASS" if all(c["status"] == "PASS" for c in result["cases"]) else "FAIL"
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"contiguous_contract": result}), flush=True)
    raise SystemExit(0 if result["status"] == "PASS" else 1)


def contiguous_contract_worker(args):
    """Exercise ONLY package exports, public constructors, modules and the tex boundary."""
    import hashlib
    import math
    import traceback
    import torch
    import torch.nn.functional as F
    import transformer_engine.pytorch as te
    import transformer_engine_torch as native
    from transformer_engine.pytorch import cpp_extensions as tex
    from transformer_engine.pytorch import ops
    from transformer_engine.pytorch.tensor import GroupedTensorStorage

    case = json.loads((args.dump / "case.json").read_text())
    evidence = {"case": case, "status": "RUNNING", "environment": {
        k: v for k, v in os.environ.items() if k.startswith("NVTE_")}}
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    guarded = []
    evidence["libraries"] = {kind: {"path": str(_get_shared_object_file(kind).resolve()),
        "sha256": hashlib.sha256(_get_shared_object_file(kind).read_bytes()).hexdigest()}
        for kind in ("core", "torch")}
    evidence["rtc_include"] = os.environ.get("NVTE_CUTLASS3_INCLUDE_DIR")
    source_root = Path(__file__).resolve().parents[2]
    evidence["python_sources"] = {
        relative: hashlib.sha256((source_root / relative).read_bytes()).hexdigest()
        for relative in (
            "profile/gg_fp32_accum/repro.py",
            "transformer_engine/pytorch/cpp_extensions/gemm.py",
            "transformer_engine/pytorch/module/grouped_linear.py",
            "transformer_engine/pytorch/ops/basic/grouped_linear.py",
        )
    }

    def launch_expectation(shapes, beta, layout, output_dtype=None):
        bn = 128 if case.get("default_backend") or beta or case.get("main_grad") else 256
        kmax = 512 if case.get("default_backend") else case.get("kmax", 512)
        eligible = sum(gm > 0 and 0 < gk <= kmax and gn % 1024 == 0
                       for gm, gn, gk in shapes) if not beta and bn == 256 and args.contract_ne == 4 else 0
        ne = 4 if eligible else 1
        dtype = output_dtype or case.get("dtype", "bf16")
        matrix_layout = {"TN": "_LA_RM_LB_CM_LC_RM_", "NN": "_LA_RM_LB_RM_LC_RM_",
                         "NT": "_LA_CM_LB_RM_LC_RM_"}[layout]
        symbol = f"_tb256x{bn}x64_s3_C{dtype if beta else 'void'}_contiguous{2 if ne == 4 else 1}"
        if ne == 4 and kmax != 512:
            symbol += f"_kmax{kmax}"
        ending = "_fulltile"
        return {"layout": layout, "effective_ne": ne, "eligible_groups": eligible,
                "effective_kmax": kmax if ne == 4 else 512,
                "work": sum(gm // 256 * (gn // bn // ne) for gm, gn, _ in shapes),
                "symbol_fragments": [f"_bf16_bf16_{dtype}_acc_fp32_", matrix_layout, symbol, ending]}

    def bits(t):
        t = t.detach()
        if hasattr(t, "rowwise_data"):
            t = t.rowwise_data
        return t.resolve_neg().resolve_conj().contiguous().view(torch.uint8)

    def sha(t):
        return hashlib.sha256(bits(t).cpu().numpy().tobytes()).hexdigest()

    def same(x, y):
        return torch.equal(bits(x), bits(y))

    def packed(shapes, dtype, random=False, misalign=False):
        counts = [math.prod(s) for s in shapes]
        start = 33 if misalign else 32
        buf = torch.full((sum(counts) + start + 32,), 97, device="cuda", dtype=dtype)
        views, offset = [], start
        for shape, count in zip(shapes, counts):
            view = buf[offset:offset + count].view(shape)
            if random:
                view.copy_(torch.randn(shape, device="cuda", dtype=dtype) * 0.125)
            else:
                view.fill_(float("nan"))
            views.append(view)
            offset += count
        guarded.append((buf, start, offset))
        return views, buf

    def wrap(views, backing, start=32):
        shapes = [tuple(v.shape) for v in views]
        assert len({s[1] for s in shapes}) == 1
        feature = shapes[0][1]
        first = torch.tensor([s[0] for s in shapes], device="cuda", dtype=torch.int64)
        offsets = torch.cat((torch.zeros(1, device="cuda", dtype=torch.int64),
                             first.cumsum(0) * feature))
        data = backing.narrow(0, start, sum(v.numel() for v in views))
        value = GroupedTensorStorage(shape=(sum(s[0] for s in shapes), feature),
            dtype=data.dtype, num_tensors=len(views), shapes=shapes, data=data,
            first_dims=first, tensor_offsets=offsets, offsets=offsets.tolist())
        assert value.rowwise_data is data and value.rowwise_data.data_ptr() == data.data_ptr()
        evidence.setdefault("native_metadata", []).append({
            "shapes": shapes, "first_dims": first.tolist(), "first_dims_dtype": str(first.dtype),
            "metadata_device": str(first.device), "tensor_offsets": offsets.tolist(),
            "no_copy": True, "payload_ptr": data.data_ptr(), "storage_offset": data.storage_offset()})
        return value

    def compare(actual, reference):
        assert actual.shape == reference.shape
        assert torch.isfinite(actual).all()
        if not actual.numel():
            return {"empty": True, "rel_error": 0.0, "max_abs": 0.0}
        delta = actual.float() - reference.float()
        rel = float(torch.linalg.vector_norm(delta) /
                    torch.linalg.vector_norm(reference.float()).clamp_min(1e-30))
        bf16 = actual.dtype == torch.bfloat16
        assert rel < (0.006 if bf16 else 2e-5), (rel, actual.dtype)
        torch.testing.assert_close(actual.float(), reference.float(),
                                   rtol=0.05 if bf16 else 2e-4,
                                   atol=0.05 if bf16 else 0.002)
        return {"rel_error": rel, "max_abs": float(delta.abs().max())}

    def observe(call, tag):
        error, returned = None, None
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            try:
                with torch.profiler.record_function(tag):
                    returned = call()
                    torch.cuda.synchronize()
            except Exception as exc:  # Expected failures are still fully profiled.
                error = {"type": type(exc).__name__, "message": str(exc),
                         "traceback": traceback.format_exc()}
        prof.export_chrome_trace(str(args.dump / f"{tag}.trace.json"))
        events = [{"name": e.name, "device_type": str(e.device_type),
                   "cpu_time_us": e.cpu_time_total, "id": e.id,
                   "kernels": [{"name": k.name, "duration_us": k.duration}
                               for k in e.kernels]} for e in prof.events()]
        cuda = [e["name"] for e in events if e["device_type"].endswith("CUDA")]
        record = {"cuda_kernels": cuda,
                  "rtc_kernels": [name for name in cuda if name.startswith("nvte_grouped_gemm3_")],
                  "launch_host_events": [e for e in events if "launch" in e["name"].lower()],
                  "events": events, "trace": str((args.dump / f"{tag}.trace.json").resolve())}
        evidence["profile" if tag == "public_call" else tag] = record
        if tag == "public_call":
            evidence["observed_error"] = error
        return returned, error

    def reject(error, before):
        evidence["observed_error"] = error
        unchanged = all(same(buf, old) for buf, old in before)
        evidence["full_D_backing_unchanged"] = unchanged
        compute = [name for name in evidence["profile"]["cuda_kernels"]
                   if any(token in name.lower() for token in ("gemm", "matmul", "cublas", "cutlass"))]
        evidence["rejected_compute_kernels"] = compute
        evidence["no_gemm_or_fallback_observed"] = not compute
        assert error is not None, "Public API unexpectedly accepted invalid input"
        assert unchanged, "Rejected call changed D/grad/parameter backing"
        assert not compute, "Rejected call launched GEMM or fallback compute"
        assert not evidence["profile"]["rtc_kernels"], "Rejected call launched an RTC kernel"
        stage = case.get("rejection_stage", "contiguous-contract")
        if stage == "ops-pre-fuser-weight-consistency":
            assert error["type"] == "RuntimeError", error
            assert error["message"] == (
                "Weight 1 has invalid dtype (expected torch.float32, got torch.bfloat16)."
            ), error
            assert "pre_first_fuser_forward" in error["traceback"], error
            assert not evidence["profile"]["cuda_kernels"], "Pre-fuser rejection launched a CUDA kernel"
        elif stage == "legacy-grouped-tensor-guard":
            assert error["type"] == "RuntimeError", error
            if "nvte_get_grouped_gemm_setup_workspace_size requires cuBLAS 13.3+" in error["message"]:
                assert "_get_grouped_gemm_setup_workspace" in error["traceback"], error
                stage = "legacy-cublas-setup-version-guard"
            else:
                assert (
                    "NVTE_GG_CONTIGUOUS requires the validated packed-list adapter, "
                    "not the cuBLAS GroupedTensor extension entry."
                ) in error["message"], error
                stage = "legacy-grouped-tensor-native-entry-guard"
        else:
            assert "NVTE_GG_CONTIGUOUS" in error["message"], error
        evidence["verified_rejection_stage"] = stage

    def corrupt_splits(splits, bad):
        return {"split-count": splits[:-1], "split-mismatch": [512, 512, 0, 512],
                "split-negative": [256, 768, -256, 768],
                "split-fractional": [256.5, 767.5, 0, 512], "split-bool": [True] + splits[1:],
                "split-order": [768, 256, 0, 512], "split-sum": [256, 768, 0, 256],
                "M": [128, 896, 0, 512]}.get(bad, splits)

    try:
        api, bad = case["api"], case.get("bad")
        if case.get("legacy_feature"):
            fin, fout, splits = case["fin"], case["fout"], case["splits"]
            layer = (te.GroupedLinear(len(splits), fin, fout, bias=True,
                        params_dtype=torch.bfloat16, single_grouped_weight=False) if api == "module"
                     else ops.GroupedLinear(len(splits), fin, fout, bias=True,
                        dtype=torch.bfloat16, device="cuda", single_grouped_weight=False))
            container = ops.Sequential(layer) if api == "ops" else layer
            x = torch.randn(sum(splits), fin, device="cuda", dtype=torch.bfloat16).requires_grad_(True)
            weights = [getattr(layer, f"weight{i}") for i in range(len(splits))]
            biases = [getattr(layer, f"bias{i}") for i in range(len(splits))]
            refx = x.detach().clone().requires_grad_(True)
            refw = [w.detach().clone().requires_grad_(True) for w in weights]
            refb = [b.detach().clone().requires_grad_(True) for b in biases]
            reference = torch.cat([F.linear(xx, w, b) for xx, w, b in zip(torch.split(refx, splits), refw, refb)])
            dy = torch.randn_like(reference)
            reference.backward(dy)
            def legacy_call():
                output = container(x, torch.tensor(splits, device="cuda", dtype=torch.int64))
                output.backward(dy)
                return output
            output, error = observe(legacy_call, "public_call")
            assert error is None, error
            evidence["numeric_comparisons"] = {"output": compare(output, reference),
                "input_grad": compare(x.grad, refx.grad),
                "weight_grad": [compare(w.grad, r.grad) for w, r in zip(weights, refw)],
                "bias_grad": [compare(b.grad, r.grad) for b, r in zip(biases, refb)]}
            assert not any("_contiguous" in name for name in evidence["profile"]["cuda_kernels"])
            evidence["unsupported_feature_legacy_observed"] = True
        elif case.get("large_output"):
            m, n, k = case["splits"][0], case["n"], case["k"]
            A, abuf = packed([(n, k)], torch.bfloat16)
            B, bbuf = packed([(m, k)], torch.bfloat16)
            D, dbuf = packed([(m, n)], torch.bfloat16)
            evidence["per_group_MNK"] = [(m, n, k)]
            evidence["splits"] = case["splits"]
            evidence["output_elements"] = D[0].numel()
            assert D[0].numel() > 2 ** 31
            evidence["input_sha256"] = [sha(abuf), sha(bbuf)]
            def large_output_call():
                for _ in range(case["repeats"]):
                    D[0].fill_(float("nan"))
                    tex.general_grouped_gemm(A, B, D, [None], torch.bfloat16,
                        layout="TN", m_splits=case["splits"], accumulate=False)
            _, error = observe(large_output_call, "public_call")
            assert error is None, error
            evidence["all_D_bits_positive_zero"] = int(torch.count_nonzero(D[0].view(torch.int16))) == 0
            assert evidence["all_D_bits_positive_zero"], "Large zero-K output was not fully overwritten"
            evidence["expected_launches"] = [launch_expectation([(m, n, k)], 0, "TN")] * case["repeats"]
        elif api in ("gemm", "native", "direct"):
            layout, splits = case["layout"], case["splits"]
            n = case.get("n", 256)
            shapes = [(case.get("m", 256), n, s) if layout == "NT" else
                      (s, n, case.get("k", 256)) for s in splits]
            if case.get("ordinary"):
                shapes = [(s, 32 + i * 8, 24 + i * 8) for i, s in enumerate(splits)]
                splits = [k if layout == "NT" else m for m, _, k in shapes]
            if bad == "M":
                shapes = [(128, n, 256), (384, n, 256), (256, n, 256), (0, n, 256)]
                splits = [s[0] for s in shapes]
            elif bad == "N":
                shapes = [(256, 128, 256), (256, 384, 256), (0, 256, 256), (512, 256, 256)]
                splits = [s[0] for s in shapes]
            elif bad == "K":
                shapes = [(256, n, s) for s in (128, 384, 0, 256)]
                splits = [s[2] for s in shapes]
            dtype = torch.float32 if case["dtype"] == "fp32" else torch.bfloat16
            A, abuf = packed([(n, k) if layout == "TN" else (k, n) for m, n, k in shapes],
                             torch.bfloat16, True, bad == "ptr16-A")
            B, bbuf = packed([(k, m) if layout == "NT" else (m, k) for m, n, k in shapes],
                             torch.bfloat16, True)
            D, dbuf = packed([(m, n) for m, n, k in shapes], dtype, False, bad == "ptr16-D")
            if case["beta"]:
                for d in D:
                    d.copy_(torch.randn_like(d))
            if case.get("zero_k"):
                # Include signed zero, infinities and NaN payloads; WithC must preserve ALL bits.
                patterns = ([0, -2147483648, 2143289345, 2139095040] if dtype == torch.float32
                            else [0, -32768, 32705, 32640])
                integer = torch.int32 if dtype == torch.float32 else torch.int16
                for d in D:
                    d.view(integer).view(-1).copy_(torch.tensor(patterns, device="cuda", dtype=integer)
                        .repeat(d.numel() // len(patterns)))
            if bad == "dtype-A": A = [a.float() for a in A]
            elif bad == "dtype-D": D = [d.to(torch.float16) for d in D]
            elif bad == "output-alias": D, dbuf = B, bbuf
            initial = [d.clone() for d in D]
            evidence["initial_D_sha256"] = sha(dbuf)
            products = [(y.float().T if layout == "NT" else y.float()) @
                        (x.float().T if layout == "TN" else x.float()) for x, y in zip(A, B)]
            if bad in ("stride-A", "stride-D"):
                target = A if bad == "stride-A" else D
                target[0] = target[0].as_strided(target[0].shape, (1, target[0].shape[0]))
            if bad in ("separate-A", "separate-D"):
                target = A if bad == "separate-A" else D
                target[0] = target[0].clone()
            if bad and bad.startswith("lazy-negative-") and api != "native":
                target = {"A": A, "B": B, "D": D}[bad.rsplit("-", 1)[1]]
                target[0] = torch._neg_view(target[0])
            call_splits = corrupt_splits(splits, bad) if bad != "M" else splits
            if bad == "split-backing":
                split_storage = torch.tensor(call_splits, device="cuda", dtype=torch.int64)
                split_storage.untyped_storage().resize_(8)
                call_splits = split_storage
            evidence.update(per_group_MNK=shapes, splits=splits if bad == "split-backing" else call_splits,
                operands={key: [{"shape": list(t.shape), "stride": list(t.stride()),
                                "ptr_mod16": t.data_ptr() % 16, "storage_offset": t.storage_offset()}
                               for t in values] for key, values in (("A", A), ("B", B), ("D", D))},
                input_sha256={"A": sha(abuf), "B": sha(bbuf)})

            def list_call():
                return tex.general_grouped_gemm(A, B, D, [None] * len(A), dtype,
                    layout=layout, m_splits=call_splits, grad=layout != "TN",
                    accumulate=bool(case["beta"]), single_output=False)

            call = list_call
            if api == "native":
                ga, gb = wrap(A, abuf), wrap(B, bbuf)
                gd = wrap(D, dbuf, 33 if bad == "ptr16-D" else 32)
                if bad == "offset-live": gd.tensor_offsets[1] += 1
                elif bad == "offset-cache": gd.offsets[1] += 1
                elif bad == "offset-order": gd.tensor_offsets[2] = -1
                elif bad == "offset-end": gd.tensor_offsets[-1] += 256
                elif bad == "shape-cache": gd.tensor_shapes[0] = (512, shapes[0][1])
                elif bad == "first-live-cache": gd.first_dims[0] += 256
                elif bad == "first-negative": gd.first_dims[0] = -256
                elif bad == "first-count": gd.first_dims = gd.first_dims[:-1]
                elif bad == "first-dtype": gd.first_dims = gd.first_dims.float()
                elif bad in ("metadata-backing", "metadata-lazy"):
                    field = case["field"]
                    value = getattr(gd, field)
                    if value is None:
                        value = torch.full((len(splits),), shapes[0][1], device="cuda", dtype=torch.int64)
                    value = value.clone()
                    if bad == "metadata-backing": value.untyped_storage().resize_(8)
                    else: value = torch._neg_view(value)
                    setattr(gd, field, value)
                elif bad == "rowwise-backing":
                    value = torch.empty(gd.rowwise_data.numel() + 1, device="cuda", dtype=dtype)[1:]
                    value.untyped_storage().resize_(0)
                    gd.rowwise_data = value
                elif bad and bad.startswith("lazy-negative-"):
                    target = {"A": ga, "B": gb, "D": gd}[bad.rsplit("-", 1)[1]]
                    target.rowwise_data = torch._neg_view(target.rowwise_data)
                if case.get("representation") == "discrete-in": ga = A
                elif case.get("representation") == "discrete-out": gd = D
                api_options = {}
                if "intent" in case:
                    api_options["use_contiguous_grouped_gemm"] = case["intent"]
                if bad == "false-guard":
                    os.environ["NVTE_GG_CONTIGUOUS"] = "1"
                if bad == "api-option":
                    api_options[case["option"]] = True if case["option"] == "grad" else \
                        torch.ones(1, device="cuda", dtype=torch.float32)
                call = lambda: tex.general_grouped_gemm_for_grouped_tensor(
                    ga, gb, gd, layout=layout, accumulate=bool(case["beta"]), **api_options)
            elif api == "direct":
                workspace = torch.empty(4 * 1024 * 1024, device="cuda", dtype=torch.uint8)
                if bad == "workspace-alignment": workspace = workspace[1:]
                elif bad == "workspace-backing": workspace.untyped_storage().resize_(128)
                elif bad == "workspace-short": workspace = workspace[:256]
                elif bad == "workspace-dtype": workspace = workspace.float()
                elif bad == "workspace-rank": workspace = workspace.view(2, -1)
                elif bad == "workspace-cpu": workspace = workspace.cpu()
                elif bad == "workspace-strided": workspace = workspace[::2]
                elif bad == "workspace-alias": workspace = D[0].view(torch.uint8).view(-1)
                capacity = workspace.numel() + int(bad == "workspace-capacity")
                workspace_list = [] if bad == "workspace-missing" else [workspace]
                if bad == "lists": B = []
                empty = [torch.empty(0, device="cuda", dtype=torch.bfloat16)] * len(A)
                call = lambda: native.te_general_grouped_gemm(A, layout[0] == "T", B,
                    True if bad == "layout" else layout[1] == "T", D,
                    native.DType.kFloat32 if dtype == torch.float32 else native.DType.kBFloat16,
                    call_splits, empty, native.DType.kBFloat16, False, empty, layout != "TN",
                    workspace_list, capacity, bool(case["beta"]), False, 0)
                evidence["workspace"] = {"shape": list(workspace.shape), "dtype": str(workspace.dtype),
                                         "storage_bytes": workspace.untyped_storage().nbytes(),
                                         "capacity_argument": capacity, "ptr_mod128": workspace.data_ptr() % 128}
                if case.get("default_backend"):
                    call = lambda: native.te_contiguous_grouped_gemm(
                        A, layout[0] == "T", B, True if bad == "layout" else layout[1] == "T",
                        D, call_splits, torch.Tensor() if bad == "workspace-missing" else workspace,
                        bool(case["beta"]), 0)
                elif bad == "layout":
                    call = lambda: native.te_general_grouped_gemm(A, True, B, True, D,
                        native.DType.kBFloat16, call_splits, empty, native.DType.kBFloat16,
                        False, empty, False, [workspace], workspace.numel(), False, False, 0)
                elif bad in ("grouped-entry", "discrete-in-entry", "discrete-out-entry"):
                    ga, gb, gd = wrap(A, abuf), wrap(B, bbuf), wrap(D, dbuf)
                    entry = {"grouped-entry": native.te_general_grouped_gemm_for_grouped_tensor,
                             "discrete-in-entry": native.te_general_grouped_gemm_for_discrete_in,
                             "discrete-out-entry": native.te_general_grouped_gemm_for_discrete_out}[bad]
                    raw_a, raw_d = (A if bad == "discrete-in-entry" else ga,
                                    D if bad == "discrete-out-entry" else gd)
                    alpha = torch.ones(1, device="cuda", dtype=torch.float32)
                    beta = torch.zeros_like(alpha)
                    call = lambda: entry(raw_a, layout[0] == "T", gb, layout[1] == "T",
                                         raw_d, None, None, alpha, beta, workspace, workspace, False, 0)
            if bad:
                if bad == "rtc-option":
                    os.environ[case["option"]] = "bad"
                elif bad == "capture":
                    stream = torch.cuda.Stream()
                    with torch.cuda.stream(stream):
                        _, warmup_error = observe(call, "forward_setup")
                    assert warmup_error is None, warmup_error
                    stream.synchronize()
                    for d, old in zip(D, initial): d.copy_(old)
                    graph = torch.cuda.CUDAGraph()
                    capture_entry = call
                    def capture_call():
                        with torch.cuda.graph(graph, stream=stream):
                            return capture_entry()
                    call = capture_call
                before = [(buf, buf.clone()) for buf in [dbuf] + D]
                check_workspace = api == "direct" and bad in (
                    "rtc-option", "capture", "workspace-short", "workspace-dtype",
                    "workspace-rank", "workspace-cpu", "workspace-strided", "workspace-missing", "lists")
                if check_workspace:
                    workspace.fill_(73)
                    workspace_before = workspace.clone()
                _, error = observe(call, "public_call")
                if check_workspace:
                    evidence["workspace_unchanged_before_rejection"] = same(workspace, workspace_before)
                reject(error, before)
                if bad == "capture":
                    assert "graph capture" in error["message"], error
                if check_workspace:
                    assert evidence["workspace_unchanged_before_rejection"], "Rejected call uploaded workspace descriptors"
            else:
                baseline = None
                if not (case.get("zero_k") or case.get("no_work") or case.get("ordinary")):
                    baseline_flags = ("NVTE_GG_CONTIGUOUS", "NVTE_USE_CUTLASS_GROUPED_GEMM",
                                      "NVTE_USE_CUTLASS_GROUPED_GEMM_RTC", "NVTE_GG_ALLOW_FP32_ACCUM")
                    saved_flags = {name: os.environ.get(name) for name in baseline_flags}
                    os.environ.update(NVTE_GG_CONTIGUOUS="0", NVTE_USE_CUTLASS_GROUPED_GEMM="1",
                                      NVTE_USE_CUTLASS_GROUPED_GEMM_RTC="1", NVTE_GG_ALLOW_FP32_ACCUM="1")
                    _, error = observe(list_call, "ordinary_baseline")
                    assert error is None, error
                    baseline = [d.clone() for d in D]
                    evidence["baseline_output_sha256"] = sha(dbuf)
                    if case.get("default_backend"):
                        assert all("_contiguous" not in name for name in evidence["ordinary_baseline"]["rtc_kernels"])
                        assert evidence["ordinary_baseline"]["rtc_kernels"], "Ordinary list did not execute baseline RTC"
                    for name, value in saved_flags.items():
                        if value is None: os.environ.pop(name, None)
                        else: os.environ[name] = value
                    for d, c in zip(D, initial): d.copy_(c)
                comparisons = []
                def successful_calls():
                    if case.get("default_backend"):
                        evidence["public_call_enable_state"] = os.environ.get("NVTE_GG_CONTIGUOUS")
                        assert evidence["public_call_enable_state"] == (None if case["enable_state"] == "unset" else "0")
                        assert "NVTE_GG_CONTIGUOUS_BN" not in os.environ
                        assert "NVTE_GG_CONTIGUOUS_N_EXPAND" not in os.environ
                        evidence["public_call_backend_flags"] = {name: os.environ.get(name) for name in (
                            "NVTE_USE_CUTLASS_GROUPED_GEMM", "NVTE_USE_CUTLASS_GROUPED_GEMM_RTC",
                            "NVTE_GG_ALLOW_FP32_ACCUM")}
                        if not case.get("ordinary"):
                            assert all(value == "0" for value in evidence["public_call_backend_flags"].values())
                    for iteration in range(case["repeats"]):
                        sources = [d.float().clone() for d in D] if case["beta"] else None
                        returned = call()
                        if api == "native": assert returned is gd, "Native adapter replaced output object"
                        torch.cuda.synchronize()
                        if case.get("zero_k"):
                            assert all(same(d, c) if case["beta"] else
                                       not torch.count_nonzero(bits(d)) for d, c in zip(D, initial))
                            comparisons.append({"zeroK_bits": "preserved" if case["beta"] else "positive_zero"})
                        else:
                            comparisons.append({"iteration": iteration, "oracle": [
                                compare(d, product + sources[i] if sources else product)
                                for i, (d, product) in enumerate(zip(D, products))]})
                            if baseline is not None and iteration == 0:
                                evidence["baseline_comparison"] = [compare(d, old) for d, old in zip(D, baseline)]
                    return D
                _, error = observe(successful_calls, "public_call")
                assert error is None, error
                evidence["numeric_comparisons"] = comparisons
                evidence["output_sha256"] = sha(dbuf)
                measured_count = len(evidence["profile"]["rtc_kernels"])
                if case.get("no_work"):
                    assert measured_count == 0, evidence["profile"]["rtc_kernels"]
                    assert all(same(d, c) for d, c in zip(D, initial))
                    evidence["full_D_backing_unchanged"] = evidence["initial_D_sha256"] == sha(dbuf)
                    assert evidence["full_D_backing_unchanged"]
                elif case.get("zero_k"):
                    assert measured_count in (0, case["repeats"]), evidence["profile"]["rtc_kernels"]
                    evidence["zeroK_rtc_compute_observed"] = bool(measured_count)
                else:
                    assert measured_count == case["repeats"], evidence["profile"]["rtc_kernels"]
                if not case.get("ordinary"):
                    evidence["expected_launches"] = [launch_expectation(shapes, case["beta"], layout)] * measured_count
                    assert all("_contiguous" in s for s in evidence["profile"]["rtc_kernels"])
                if case.get("default_backend") and baseline is not None:
                    final_outputs = [d.clone() for d in D]
                    saved_flags = {name: os.environ.get(name) for name in baseline_flags}
                    os.environ.update(NVTE_GG_CONTIGUOUS="0", NVTE_USE_CUTLASS_GROUPED_GEMM="1",
                                      NVTE_USE_CUTLASS_GROUPED_GEMM_RTC="1", NVTE_GG_ALLOW_FP32_ACCUM="1")
                    for d, c in zip(D, initial): d.copy_(c)
                    _, error = observe(list_call, "ordinary_after")
                    assert error is None, error
                    symbols = evidence["ordinary_after"]["rtc_kernels"]
                    assert symbols and all("_contiguous" not in name for name in symbols)
                    evidence["ordinary_after_comparison"] = [compare(d, old) for d, old in zip(D, baseline)]
                    for name, value in saved_flags.items():
                        if value is None: os.environ.pop(name, None)
                        else: os.environ[name] = value
                    for d, old in zip(D, final_outputs): d.copy_(old)
                    evidence["ordinary_default_ordinary_isolation"] = True
                assert evidence["input_sha256"] == {"A": sha(abuf), "B": sha(bbuf)}
        else:
            fin, fout, splits = case["fin"], case["fout"], case["splits"]
            grouped = case.get("grouped", api == "module")
            fused, delayed = case.get("main_grad", False), case.get("delay", False)
            layer = (te.GroupedLinear(len(splits), fin, fout, bias=False,
                        params_dtype=torch.bfloat16, single_grouped_weight=grouped,
                        fuse_wgrad_accumulation=fused, delay_wgrad_compute=delayed) if api == "module"
                     else ops.GroupedLinear(len(splits), fin, fout, bias=False,
                        dtype=torch.bfloat16, device="cuda", single_grouped_weight=grouped,
                        accumulate_into_main_grad=fused, delay_wgrad_compute=delayed))
            weights = (list(layer.weight.quantized_tensors) if grouped else
                       [getattr(layer, f"weight{i}") for i in range(len(splits))])
            if api == "module" and not grouped:
                wp, _ = packed([(fout, fin)] * len(splits), torch.bfloat16)
                with torch.no_grad():
                    for w, view in zip(weights, wp):
                        view.copy_(w)
                        w.data = view
            if bad == "separate-A":
                if grouped: layer.weight.quantized_tensors[0] = torch.nn.Parameter(weights[0].detach().clone())
                else: layer.weight0 = torch.nn.Parameter(weights[0].detach().clone())
            elif bad == "stride-A":
                weights[0].data = weights[0].detach().as_strided(weights[0].shape, (1, fout))
            elif bad == "parameter-dtype":
                if grouped: layer.weight.rowwise_data = layer.weight.rowwise_data.float()
                else: layer.weight0 = torch.nn.Parameter(weights[0].detach().float())
            elif bad == "weight-shape-live":
                weight = layer.weight
                rows = [2 * fout, 0] + [fout] * (len(splits) - 2)
                weight.first_dims = torch.tensor(rows, device="cuda", dtype=torch.int64)
                weight.tensor_offsets = torch.cat((torch.zeros(1, device="cuda", dtype=torch.int64),
                                                   weight.first_dims.cumsum(0) * fin))
                weight.offsets = weight.tensor_offsets.tolist()
                weight.tensor_shapes = [(r, fin) for r in rows]
                weight.quantized_tensors = [torch.nn.Parameter(weight.rowwise_data.narrow(
                    0, weight.offsets[i], r * fin).view(r, fin)) for i, r in enumerate(rows)]
            params = [layer.weight] if grouped else [getattr(layer, f"weight{i}") for i in range(len(splits))]
            parameter_ids = [id(p) for p in layer.parameters()]
            replacement_params, replacement_before = None, None
            if case.get("replace_before_backward") or case.get("replace_before_wgrad"):
                replacement_layer = (te.GroupedLinear(len(splits), fin, fout, bias=False,
                    params_dtype=torch.bfloat16, single_grouped_weight=grouped,
                    fuse_wgrad_accumulation=True) if api == "module" else
                    ops.GroupedLinear(len(splits), fin, fout, bias=False, dtype=torch.bfloat16,
                    device="cuda", single_grouped_weight=grouped, accumulate_into_main_grad=True))
                replacement_params = ([replacement_layer.weight] if grouped else
                    [getattr(replacement_layer, f"weight{i}") for i in range(len(splits))])
                for param in replacement_params:
                    param.main_grad = torch.full((len(splits), fout, fin) if grouped else (fout, fin),
                                                 0.25, device="cuda", dtype=torch.float32)
                    param.grad_added_to_main_grad = False
                    param.zero_out_wgrad = True
                replacement_before = [p.main_grad.clone() for p in replacement_params]
            main_views, main_buffer = None, None
            if fused:
                main_dtype = torch.bfloat16 if case.get("main_grad_dtype") == "bf16" else torch.float32
                main_views, main_buffer = packed([(fout, fin)] * len(splits), main_dtype)
                for view in main_views: view.fill_(0.75)
                if grouped:
                    layer.weight.main_grad = main_buffer[32:-32].view(len(splits), fout, fin)
                else:
                    for param, view in zip(params, main_views): param.main_grad = view
                if case.get("dummy"):
                    for param in params:
                        param.grad_added_to_main_grad = False
                        param.zero_out_wgrad = True
                if bad == "main-grad-stride":
                    strided_backing = torch.full((2 * main_buffer.numel(),), 97,
                                                 device="cuda", dtype=torch.float32)
                    strided_backing[::2].copy_(main_buffer)
                    main_buffer = strided_backing
                    main = main_buffer[64:-64:2].view(len(splits), fout, fin)
                    if grouped: layer.weight.main_grad = main
                    else:
                        for param, view in zip(params, main.unbind()): param.main_grad = view
                    evidence["main_grad_stride"] = list(main.stride())
                evidence["initial_main_grad_sha256"] = sha(main_buffer)
            xs, xbuf = packed([(sum(splits), fin)] * (2 if case.get("input_dgrad_shared") else 1),
                              torch.bfloat16, True)
            x = xs[0].requires_grad_(True)
            ys, ybuf = packed([(sum(splits), fout)], torch.bfloat16)
            dxs, dxbuf = packed([(sum(splits), fin)], torch.bfloat16)
            if bad == "dgrad-input-alias":
                dxs, dxbuf = [x.detach()], xbuf
            elif case.get("input_dgrad_shared"):
                dxs, dxbuf = [xs[1]], xbuf
            dy = torch.randn_like(ys[0]) * 0.125
            split_values = corrupt_splits(splits, bad)
            split_tensor = torch.tensor(split_values, device="cpu" if case.get("cpu_splits") else "cuda",
                                        dtype=torch.float32 if bad == "split-fractional" else torch.int64)
            if bad == "split-backing": split_tensor.untyped_storage().resize_(8)
            container = ops.Sequential(layer) if api == "ops" else layer
            comparisons = []
            def linear_call():
                iterations = 1 if bad else case.get("repeats", 1)
                for iteration in range(iterations):
                    x.grad = None
                    for param in params:
                        if not case.get("accumulate_leaf") or iteration == 0:
                            param.grad = None
                        param.overwrite_main_grad = bool(case.get("overwrite") and iteration == 0)
                    source = [g.clone() for g in main_views] if fused else None
                    preserve_main = fused and not any(splits) and not (
                        case.get("overwrite") and iteration == 0)
                    before_zero_k_main = main_buffer.clone() if preserve_main else None
                    output_buffer = ys[0].detach()
                    if case.get("default_backend"):
                        evidence["public_call_backend_flags"] = {name: os.environ.get(name) for name in (
                            "NVTE_USE_CUTLASS_GROUPED_GEMM", "NVTE_USE_CUTLASS_GROUPED_GEMM_RTC",
                            "NVTE_GG_ALLOW_FP32_ACCUM", "NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM")}
                        assert all(value == "0" for value in evidence["public_call_backend_flags"].values())
                    if case.get("default_backend") and not case.get("change_enable_after_forward"):
                        evidence["public_call_enable_state"] = os.environ.get("NVTE_GG_CONTIGUOUS")
                        assert evidence["public_call_enable_state"] == (None if case["enable_state"] == "unset" else "0")
                    if case.get("change_enable_after_forward"):
                        os.environ["NVTE_GG_CONTIGUOUS"] = "1"
                    y = (container(x, split_tensor, out=output_buffer, dgrad_out=dxs[0],
                            is_first_microbatch=True if case.get("overwrite") and iteration == 0 else
                            False if case.get("overwrite") else None) if api == "module"
                         else container(x, split_tensor, op_kwargs={0: {"output": output_buffer, "grad_input": dxs[0]}}))
                    if bad:
                        return y
                    assert y.data_ptr() == ys[0].data_ptr(), "Public boundary ignored supplied D"
                    if case.get("replace_before_backward"):
                        for i, param in enumerate(replacement_params):
                            setattr(layer, "weight" if grouped else f"weight{i}", param)
                        evidence["original_parameter_ids"] = parameter_ids
                        evidence["replacement_parameter_ids"] = [id(p) for p in layer.parameters()]
                    before_delayed = main_buffer.clone() if fused and delayed else None
                    before_leaf = ([None if p.grad is None else p.grad.clone() for p in params]
                                   if delayed and not fused else None)
                    if case.get("change_enable_after_forward"):
                        os.environ["NVTE_GG_CONTIGUOUS"] = "0"
                        evidence["backward_enable_state"] = os.environ["NVTE_GG_CONTIGUOUS"]
                    y.backward(dy)
                    if delayed:
                        phase = {"iteration": iteration, "owner_ids": [id(p) for p in params],
                                 "enable_state": os.environ.get("NVTE_GG_CONTIGUOUS")}
                        if fused:
                            phase["before_backward_sha256"] = sha(before_delayed)
                            phase["before_callback_sha256"] = sha(main_buffer)
                            phase["no_early_write"] = same(main_buffer, before_delayed)
                        else:
                            phase["no_early_write"] = all(
                                p.grad is None if old is None else p.grad is not None and same(p.grad, old)
                                for p, old in zip(params, before_leaf))
                        assert phase["no_early_write"], "Delayed wgrad wrote early"
                        if not case.get("dummy") and (not case.get("accumulate_leaf") or iteration == 0):
                            assert all(p.grad is None for p in params), "Delayed wgrad returned unwritten grads"
                        if case.get("replace_before_wgrad"):
                            for i, param in enumerate(replacement_params):
                                setattr(layer, "weight" if grouped else f"weight{i}", param)
                            evidence["original_parameter_ids"] = parameter_ids
                            evidence["replacement_parameter_ids"] = [id(p) for p in layer.parameters()]
                        with torch.profiler.record_function(f"delayed_wgrad_{iteration}"):
                            layer.backward_dw()
                        torch.cuda.synchronize()
                        phase["after_callback_sha256"] = (sha(main_buffer) if fused else
                            [sha(p.grad) for p in params])
                        evidence.setdefault("delayed_wgrad_lifecycle", []).append(phase)
                    torch.cuda.synchronize()
                    if fused:
                        wgrads = main_views
                        references = [g if case.get("overwrite") and iteration == 0 else g + old
                                      for g, old in zip(ref_fp32_wgrad, source)]
                        if case.get("dummy"):
                            assert all(p.grad_added_to_main_grad for p in params)
                            assert all(p.grad is not None and not torch.count_nonzero(p.grad) for p in params)
                        else:
                            assert all(p.grad is None for p in params), "Fusion returned nondummy parameter grads"
                        if grouped: assert layer.weight.main_grad.numel() == len(splits) * fout * fin
                    elif grouped:
                        grad = params[0].grad
                        if hasattr(grad, "rowwise_data"): grad = grad.rowwise_data
                        assert grad is not None, "Missing grouped leaf gradient"
                        wgrads = list(grad.view(len(splits), fout, fin).unbind())
                        references = [r.grad for r in refw]
                    else:
                        wgrads = [w.grad for w in params]
                        references = [r.grad for r in refw]
                    if case.get("accumulate_leaf"):
                        references = [r * (iteration + 1) for r in references]
                    assert x.grad is not None and all(w is not None for w in wgrads), "Missing public autograd gradients"
                    comparisons.append({"iteration": iteration, "output": compare(y, refy),
                        "input_grad": compare(x.grad, refx.grad),
                        "weight_grad": [compare(g, r) for g, r in zip(wgrads, references)],
                        "empty_weight_grad_matches": all(same(g, r.to(g.dtype)) for s, g, r in
                                                         zip(splits, wgrads, references) if s == 0)})
                    assert comparisons[-1]["empty_weight_grad_matches"]
                    if preserve_main:
                        comparisons[-1]["zeroK_main_grad_backing_bits_preserved"] = same(
                            main_buffer, before_zero_k_main)
                        assert comparisons[-1]["zeroK_main_grad_backing_bits_preserved"], \
                            "All-empty WithC wgrad changed main_grad backing"
                    assert same(x.grad, dxs[0]), "Caller dgrad buffer differs from autograd gradient"
                    if replacement_params is not None:
                        assert evidence["replacement_parameter_ids"] == [id(p) for p in layer.parameters()]
                        assert all(same(p.main_grad, old) and not p.grad_added_to_main_grad
                                   and p.grad is None for p, old in zip(replacement_params, replacement_before))
                        evidence["replacement_parameters_untouched"] = True
                    else:
                        assert parameter_ids == [id(p) for p in layer.parameters()], "Registered parameters changed"
                return y
            if bad:
                if bad == "capture":
                    stream = torch.cuda.Stream()
                    with torch.cuda.stream(stream):
                        _, warmup_error = observe(linear_call, "forward_setup")
                    assert warmup_error is None, warmup_error
                    stream.synchronize()
                    before = [(buf, buf.clone()) for buf in (ybuf, dxbuf, xbuf)]
                    graph = torch.cuda.CUDAGraph()
                    def capture_linear():
                        with torch.cuda.graph(graph, stream=stream):
                            return linear_call()
                    _, error = observe(capture_linear, "public_call")
                    assert error is not None and "graph capture" in error["message"], error
                elif bad in ("main-grad-stride", "dgrad-input-alias"):
                    y, error = observe(linear_call, "forward_setup")
                    assert error is None, error
                    before = [(buf, buf.clone()) for buf in (ybuf, dxbuf, xbuf, main_buffer)
                              if buf is not None]
                    _, error = observe(lambda: y.backward(dy), "public_call")
                else:
                    before = [(buf, buf.clone()) for buf in (ybuf, dxbuf, xbuf)]
                    _, error = observe(linear_call, "public_call")
                reject(error, before)
            else:
                evidence["initial_input_sha256"] = sha(x)
                evidence["initial_weight_sha256"] = [sha(w) for w in weights]
                evidence["original_owner_ids"] = [id(p) for p in params]
                refx = x.detach().clone().requires_grad_(True)
                refw = [w.detach().clone().requires_grad_(True) for w in weights]
                refy = torch.cat([F.linear(xx, w) for xx, w in zip(torch.split(refx, splits), refw)])
                refy.backward(dy)
                ref_fp32_wgrad = [yy.float().T @ xx.float() for xx, yy in
                                 zip(torch.split(x.detach(), splits), torch.split(dy, splits))]
                y, error = observe(linear_call, "public_call")
                assert error is None, error
                evidence["numeric_comparisons"] = comparisons
                evidence["expected_launches"] = []
                for iteration in range(case.get("repeats", 1)):
                    accumulate = fused and not (case.get("overwrite") and iteration == 0)
                    forward = launch_expectation([(s, fout, fin) for s in splits], 0, "TN")
                    dgrad = launch_expectation([(s, fin, fout) for s in splits], 0, "NN")
                    wgrad = launch_expectation([(fout, fin, s) for s in splits], int(accumulate), "NT",
                                              case.get("main_grad_dtype", "fp32") if fused else "bf16")
                    evidence["expected_launches"].extend(e for e in (forward, dgrad) if e["work"])
                    if wgrad["work"]:
                        evidence["expected_launches"].append(wgrad)
                assert len(evidence["profile"]["rtc_kernels"]) == len(evidence["expected_launches"]), evidence["profile"]["rtc_kernels"]
                evidence["output_sha256"] = sha(ybuf)
                evidence["input_unchanged"] = evidence["initial_input_sha256"] == sha(x)
                assert evidence["input_unchanged"], "GroupedLinear modified saved input"
                evidence["weights_unchanged"] = evidence["initial_weight_sha256"] == [sha(w) for w in weights]
                evidence["original_owners_still_leaf"] = all(p.is_leaf for p in params)
                assert evidence["weights_unchanged"] and evidence["original_owners_still_leaf"]
                if fused: evidence["main_grad_sha256"] = sha(main_buffer)
            evidence["parameter_storage"] = [{"shape": list(w.shape), "stride": list(w.stride()),
                "dtype": str(w.dtype), "storage_ptr": w.untyped_storage().data_ptr(),
                "storage_offset": w.storage_offset()} for w in weights]
        guards_ok = all(torch.all(buf[:start] == 97) and torch.all(buf[end:] == 97)
                        for buf, start, end in guarded)
        evidence["full_backing_guards"] = bool(guards_ok)
        assert guards_ok, "A/B/D/input/dgrad full-backing guard corruption"
        evidence["status"] = "PASS"
    except Exception as exc:
        evidence.update(status="FAIL", error=str(exc), traceback=traceback.format_exc())
    evidence["runtime_artifacts"] = [{"path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for pattern in ("*.cubin", "*.cu", "*.options.txt") for path in sorted(args.dump.glob(pattern))]
    evidence["runtime_compile_options"] = {path.name: path.read_text()
        for path in sorted(args.dump.glob("*.options.txt"))}
    (args.dump / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"public_runtime_observation": evidence}), flush=True)
    raise SystemExit(0 if evidence["status"] == "PASS" else 1)


p = argparse.ArgumentParser()
p.add_argument("--groups", type=int, default=1)
p.add_argument("--layout", choices=("TN", "NN", "NT"), default="NT")
p.add_argument("--nonuniform", action="store_true", help="各 group 的 M/N/K 使用不同的尾块")
p.add_argument("--kernel-names", action="store_true", help="检查单次实际 RTC launch、tile 和输出 dtype")
p.add_argument("--persistent-repeat", action="store_true", help="repeat 不恢复 D，验证连续累加")
p.add_argument("--mpe", type=int, default=256)
p.add_argument("--rows", type=int, default=512)
p.add_argument("--cols", type=int, default=1024)
p.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
p.add_argument("--backend", choices=("rtc", "cublas"), default="rtc",
               help="选择实测后端；默认保持 RTC")
p.add_argument("--source", choices=("randn", "zero", "one", "arange"), default="randn")
p.add_argument("--zero-a", action="store_true")
p.add_argument("--accumulate", type=int, choices=(0, 1), default=1)
p.add_argument("--repeat", type=int, default=1)
p.add_argument("--iters", type=int, default=0)
p.add_argument("--save", type=Path)
p.add_argument("--compare", type=Path)
p.add_argument("--dump", type=Path)
p.add_argument("--resources", type=Path, help="仅查询目录内 RTC cubin 的 driver 资源属性")
p.add_argument("--resource-smem", type=int, default=196608)
p.add_argument("--resource-threads", type=int, default=512)
p.add_argument("--check", action="store_true", help="按 fp32/bf16 阈值执行验收断言")
p.add_argument("--pair-c", action="store_true", help="同输入、同 dtype 配对测量 C/noC")
p.add_argument("--pair-warmup", type=int, default=8)
p.add_argument("--pair-iters", type=int, default=10)
p.add_argument("--reverse-pair", action="store_true")
p.add_argument("--contiguous-contract", action="store_true",
               help="Observe actual public RTC/library boundaries; --dump is required, no pytest/ACU")
p.add_argument("--contract-default-backend", action="store_true",
               help="Observe PPU default routing with old enable unset/0 and no tuning overrides")
p.add_argument("--contract-scenarios", nargs="+", choices=("gemm", "native", "invalid", "direct", "linear", "ordinary"),
               default=["gemm", "native", "invalid", "direct", "linear", "ordinary"])
p.add_argument("--contract-case", nargs="+", help="Selected case IDs or shell-style globs; use a new dump directory")
p.add_argument("--contract-worker", help=argparse.SUPPRESS)
p.add_argument("--contract-timeout", type=int, default=180, help="Per-case child timeout in seconds")
p.add_argument("--contract-many-groups", type=int, default=128)
p.add_argument("--contract-ne", type=int, choices=(1, 4), default=1)
p.add_argument("--packed-study-arm", choices=("production", "contiguous"))
p.add_argument("--packed-study-splits", help="Comma-separated true ragged M (TN/NN) or K (NT) sizes")
p.add_argument("--packed-study-ne", type=int, choices=(1, 4, 8), default=1)
p.add_argument("--packed-study-kmax", type=int, choices=(512, 1024, 2048), default=512)
p.add_argument("--packed-study-profile", action="store_true")
p.add_argument("--packed-study-cubins", type=Path)
p.add_argument("--smallk-arms", nargs="+",
               choices=("production", "matched", "contiguous", "nexpand", "contiguous256", "nexpand256"),
               help="fwd noC K512 的 packed ragged 对照；tile缓存按进程初始化，每进程一臂")
p.add_argument("--smallk-profile", action="store_true",
               help="避免嵌套 torch.profiler；匹配 launch 为正确性1次+预热3次+iters次")
p.add_argument("--smallk-cubins", type=Path,
               help="按臂加载已校验的实际 RTC cubin，仅补源码调试元数据，不重编")
p.add_argument("--pair-names", action="store_true", help="只验证配对 kernel 数与 dtype")
p.add_argument("--compile-only", action="store_true", help="只用 NVRTC 编译，不初始化或运行 GPU")
p.add_argument("--compile-tile", type=int, nargs=3, default=(256, 256, 64))
p.add_argument("--compile-stages", type=int, default=3)
p.add_argument("--compile-contiguous", type=int, choices=(0, 1, 2), default=0,
               help="仅离线编译验收：0=通用，1=连续普通，2=noC NExpand4；非性能替身")
p.add_argument("--compile-kmax", type=int, choices=(512, 1024, 2048), default=512,
               help="离线扩展kernel的K上限；普通/通用模式归一化512")
p.add_argument("--compile-arch", type=int, choices=(80, 89), default=89)
p.add_argument("--compile-ab-dtype", choices=("bf16", "fp16"), default="bf16")
p.add_argument("--compile-with-c", type=int, choices=(0, 1), default=1,
               help="离线编译的 NVTE_GG_WITH_C；用于核对实验宏在无 C cubin 中的作用域")
p.add_argument("--inspect-codegen", action="store_true", help="离线保存 ISA、资源与原子指令计数")
p.add_argument("--compile-lineinfo", action="store_true", help="离线增加源码行号，保存可复核的 PASM 交接映射")
p.add_argument("--compile-baseline-head", action="store_true", help="用 HEAD 的三个 overlay 头建立全关回归基线快照")
p.add_argument("--compile-canonical", action="store_true", help="在完整 include 快照中编译 canonical 头；不修改正在构建的 effective 树")
p.add_argument("--compare-codegen", type=Path, help="compile-only 与已有目录逐字节对比 cubin/ISA/资源并落盘")
a = p.parse_args()
if a.contiguous_contract:
    assert a.dump and not (a.compile_only or a.packed_study_arm or a.smallk_arms or a.pair_c), \
        "--contiguous-contract requires --dump and an independent runtime invocation"
    assert a.backend == "rtc" and a.contract_timeout > 0 and a.contract_many_groups > 0
    a.dump = a.dump.resolve()
    a.dump.mkdir(parents=True, exist_ok=True)
    if not a.contract_worker:
        run_contiguous_contract(a)
if a.packed_study_splits:
    assert a.packed_study_arm, "--packed-study-splits requires --packed-study-arm"
    a.packed_study_splits = [int(s.strip()) for s in a.packed_study_splits.split(",")]
    assert len(a.packed_study_splits) == a.groups and min(a.packed_study_splits) >= 0
if a.compare_codegen:
    assert a.compile_only and a.dump and a.inspect_codegen, "codegen 对照需要 compile-only/dump/inspect-codegen"
# ACU 的争用告警会连被测进程自己一并列出；上报自身 pid 以便采集侧只认外部占用。
print(json.dumps({"target_pid": os.getpid()}), flush=True)
if not a.contract_worker:
    for name in ("NVTE_USE_CUTLASS_GROUPED_GEMM", "NVTE_USE_CUTLASS_GROUPED_GEMM_RTC",
                 "NVTE_GG_ALLOW_FP32_ACCUM", "NVTE_FORCE_RTC_NO_FALLBACK",
                 "NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK"):
        os.environ[name] = "1" if a.backend == "rtc" else "0"
os.environ.setdefault("NVTE_CUTLASS_GROUPED_GEMM_RTC_VERBOSE", "1" if a.backend == "rtc" else "0")
if a.dump:
    a.dump.mkdir(parents=True, exist_ok=True)
    os.environ["NVTE_CUTLASS_RTC_DUMP_CUBIN_DIR"] = str(a.dump.resolve())

if a.compile_only:
    import ctypes

    assert a.backend == "rtc"
    root = Path(__file__).resolve().parents[2]
    include = Path(os.environ.get("NVTE_CUTLASS3_INCLUDE_DIR",
                                 str(root / "build/cmake/actlize_effective/include")))
    if a.compile_baseline_head or a.compile_canonical:
        import shutil
        import subprocess

        assert a.dump and not (a.compile_baseline_head and a.compile_canonical)
        snapshot = a.dump / ("head_include" if a.compile_baseline_head else "canonical_include")
        assert not snapshot.exists(), "不可覆盖已有 include 快照"
        shutil.copytree(include, snapshot, symlinks=False)
        relatives = ("cutlass/gemm/kernel/ppu_aiu_gemm_array_group.hpp",
                     "cutlass/gemm/collective/ppu_mma_aiu_multistage_batch_array.hpp",
                     "cutlass/epilogue/collective/ppu_epilogue_vectorized_evt.hpp")
        if a.compile_contiguous:
            assert a.compile_canonical, "专用编译验收不使用 HEAD 实验基线"
            relatives += ("cutlass/gemm/kernel/nvte_contiguous_grouped.hpp",
                          "cutlass/gemm/kernel/nvte_contiguous_grouped_params.hpp")
        for relative in relatives:
            path = f"transformer_engine/common/gemm/actlize_overlay/include/{relative}"
            contents = (subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=root)
                        if a.compile_baseline_head else (root / path).read_bytes())
            (snapshot / relative).write_bytes(contents)
        include = snapshot
    cuda = Path("/usr/local/PPU_SDK/CUDA_SDK")
    symbol = f"nvte_grouped_gemm3_compile_{a.dtype}"
    source = (f"#undef NVTE_GG_KERNEL_NAME\n#define NVTE_GG_KERNEL_NAME {symbol}\n"
              + (root / "transformer_engine/common/gemm/rtc/cutlass_grouped_gemm.cu").read_text())
    ta, tb = {"NT": (1, 0), "NN": (0, 0), "TN": (0, 1)}[a.layout]
    defines = {"NVTE_GG_TRANS_A": ta, "NVTE_GG_TRANS_B": tb, "NVTE_GG_ARCH": a.compile_arch,
               "NVTE_GG_DTYPE": int(a.compile_ab_dtype == "fp16"), "NVTE_GG_OUT_FP32": int(a.dtype == "fp32"),
               "NVTE_GG_WITH_C": a.compile_with_c,
               "NVTE_GG_CONTIGUOUS_MODE": a.compile_contiguous,
               "NVTE_GG_PRELOAD_C": int(a.dtype == "fp32"),
               "NVTE_GG_TILE_M": a.compile_tile[0], "NVTE_GG_TILE_N": a.compile_tile[1],
               "NVTE_GG_TILE_K": a.compile_tile[2], "NVTE_GG_STAGES": a.compile_stages,
               "CUTLASS_ENABLE_TENSOR_CORE_MMA": 1, "__HGGC_FP16_H__": 1,
               "_HGGC_FP16_H_": 1, "HGGC_FP16_H": 1, "__HGGC_BF16_H__": 1,
               "__HGGC_BF16_HPP__": 1, "__BF16_DEFINED__": 1, "NDEBUG": 1, "USE_SAIL": 1}
    if a.compile_contiguous > 1 and a.compile_kmax != 512:
        defines["NVTE_GG_CONTIGUOUS_N_EXPAND_K_MAX"] = a.compile_kmax
    options = ["--no-cache", "--std=c++17", f"--gpu-architecture=sm_{a.compile_arch}",
               f"-I{cuda / 'include'}", f"-I{cuda / 'include/cccl'}", f"-I{include}",
               "-I/usr/local/PPU_SDK/targets/x86_64-linux/include"]
    options += [f"-D{k}={v}" for k, v in defines.items()]
    if a.compile_lineinfo:
        options.append("--generate-line-info")
    rtc = ctypes.CDLL(str(cuda / "lib64/libnvrtc.so"))
    program = ctypes.c_void_p()
    assert rtc.nvrtcCreateProgram(ctypes.byref(program), source.encode(),
                                  b"cutlass_grouped_gemm_rtc.cu", 0, None, None) == 0
    try:
        opts = (ctypes.c_char_p * len(options))(*(s.encode() for s in options))
        status = rtc.nvrtcCompileProgram(program, len(options), opts)
        size = ctypes.c_size_t()
        assert rtc.nvrtcGetProgramLogSize(program, ctypes.byref(size)) == 0
        log = ctypes.create_string_buffer(size.value)
        assert rtc.nvrtcGetProgramLog(program, log) == 0
        if a.dump:
            (a.dump / f"{symbol}.log").write_bytes(log.value)
        assert status == 0, log.value.decode(errors="replace")
        assert rtc.nvrtcGetCUBINSize(program, ctypes.byref(size)) == 0
        cubin = ctypes.create_string_buffer(size.value)
        assert rtc.nvrtcGetCUBIN(program, cubin) == 0
        result = {"compile_only": "PASS", "gpu_execution": False,
                  "cubin_bytes": size.value, "options": options}
        if a.dump:
            path = a.dump / f"{symbol}.cubin"
            path.write_bytes(cubin.raw)
            if a.inspect_codegen:
                import re
                import subprocess

                tool = "/usr/local/PPU_SDK/bin/hgobjdump"
                isa = subprocess.check_output([tool, "--dump-isa", str(path)], text=True)
                resource = subprocess.check_output(
                    [tool, "--dump-resource-usage=all", str(path)], text=True)
                (a.dump / f"{symbol}.isa").write_text(isa)
                if a.compile_lineinfo:
                    mapped = subprocess.check_output([tool, "--line-numbers", str(path)], text=True)
                    (a.dump / f"{symbol}.isa.lines").write_text(mapped)
                    # hgobjdump 的行号模式在末尾输出 DWARF 表，而非内联 file:line。
                    from bisect import bisect_right
                    files = {int(index): name for index, name in re.findall(
                        r'file_names\[\s*(\d+)\]:\s*name:\s*"([^"]+)"', mapped)}
                    rows = [(int(pc, 16), int(line), int(column), int(file))
                            for pc, line, column, file in re.findall(
                                r"^0x([0-9a-f]+)\s+(\d+)\s+(\d+)\s+(\d+)\s", mapped, re.M)]
                    pcs = [row[0] for row in rows]
                    assert pcs == sorted(pcs) and rows, "无法解析 DWARF 行号表"
                    annotations = []
                    for instruction in isa.splitlines():
                        match = re.match(r"\s*([0-9a-f]+):\s+(?:[0-9a-f]{2}\s+){8}(.*)", instruction)
                        if not match:
                            continue
                        pc = int(match[1], 16)
                        index = bisect_right(pcs, pc) - 1
                        if index < 0:
                            continue
                        _, line, column, file = rows[index]
                        annotations.append({"address": hex(pc), "instruction": match[2],
                                            "file": files.get(file), "line": line, "column": column})
                    result["source_line_mapping"] = any(
                        row["file"] == "ppu_epilogue_vectorized_evt.hpp" and row["line"]
                        for row in annotations)
                    assert result["source_line_mapping"], "编译未提供 overlay 源码行号"
                    (a.dump / f"{symbol}.pasm_map.json").write_text(
                        json.dumps(annotations, indent=2) + "\n")
                (a.dump / f"{symbol}.resources").write_text(resource)
                result["vreg_number"] = int(re.search(r"vreg_number:(\d+)", resource)[1])
                result["stack_size"] = int(re.search(r"STACK SIZE:(\d+)", resource)[1])
                atomic_lines = [line.strip() for line in isa.splitlines()
                                if re.search(r"\bvmem\.atm\.add\.f32\b", line)]
                result["atomic_instruction_count"] = len(atomic_lines)
                result["atomic_instruction_samples"] = atomic_lines[:4]
            if a.compare_codegen:
                import hashlib
                comparison = {"baseline": str(a.compare_codegen.resolve()), "files": {}}
                for suffix in ("cubin", "isa", "resources"):
                    current = (a.dump / f"{symbol}.{suffix}").read_bytes()
                    old = (a.compare_codegen / f"{symbol}.{suffix}").read_bytes()
                    comparison["files"][suffix] = {
                        "identical": current == old, "sha256": hashlib.sha256(current).hexdigest(),
                        "baseline_sha256": hashlib.sha256(old).hexdigest()}
                comparison["status"] = "PASS" if all(
                    v["identical"] for v in comparison["files"].values()) else "FAIL"
                (a.dump / "codegen_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
                assert comparison["status"] == "PASS", comparison
                result["codegen_comparison"] = comparison
            (a.dump / f"{symbol}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        assert rtc.nvrtcDestroyProgram(ctypes.byref(program)) == 0
    raise SystemExit(0)

import torch
from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm
from transformer_engine.common import _get_shared_object_file
print(json.dumps({"core_library": str(_get_shared_object_file("core").resolve()),
                  "torch_library": str(_get_shared_object_file("torch").resolve()),
                  "rtc_include": os.environ.get("NVTE_CUTLASS3_INCLUDE_DIR", "")}), flush=True)

if a.contiguous_contract:
    contiguous_contract_worker(a)

if a.resources:
    import ctypes

    # CUDA lazy init 尚不保证 driver 的当前 context，先实际分配一个设备元素。
    context_probe = torch.empty(1, device="cuda")
    driver = ctypes.CDLL("libcuda.so.1")
    cubins = sorted(a.resources.glob("nvte_grouped_gemm3_*.cubin"))
    assert cubins, a.resources
    for path in cubins:
        module, function = ctypes.c_void_p(), ctypes.c_void_p()
        image = ctypes.create_string_buffer(path.read_bytes())
        status = driver.cuModuleLoadData(ctypes.byref(module), image)
        assert status == 0, (path, status)
        try:
            assert driver.cuModuleGetFunction(ctypes.byref(function), module,
                                              path.stem.encode()) == 0, path
            resources = {}
            for name, attribute in (("registers_per_thread", 4), ("local_bytes_per_thread", 3),
                                    ("static_smem_bytes", 1)):
                value = ctypes.c_int()
                assert driver.cuFuncGetAttribute(ctypes.byref(value), attribute, function) == 0
                resources[name] = value.value
            assert driver.cuFuncSetAttribute(function, 8, a.resource_smem) == 0
            active = ctypes.c_int()
            assert driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                ctypes.byref(active), function, a.resource_threads,
                ctypes.c_size_t(a.resource_smem)) == 0
            resources.update(dynamic_smem_bytes=a.resource_smem, threads=a.resource_threads,
                             active_blocks_per_cu=active.value,
                             theoretical_thread_occupancy=active.value * a.resource_threads /
                                 torch.cuda.get_device_properties(0).max_threads_per_multi_processor)
            result = {"cubin": str(path), **resources}
            (path.parent / f"{path.stem}.driver_resources.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
        finally:
            assert driver.cuModuleUnload(module) == 0
    raise SystemExit(0)

if a.packed_study_arm:
    import hashlib
    import math
    import shutil

    assert a.backend == "rtc" and a.dump and not a.smallk_arms and not a.pair_c
    assert not a.nonuniform and torch.cuda.get_device_capability() == (8, 9)
    assert a.dtype == ("fp32" if a.accumulate else "bf16")
    config = packed_study_config(a.packed_study_arm, a.layout, a.accumulate,
                                 a.packed_study_ne, a.packed_study_kmax,
                                 a.rows, a.cols, a.mpe, a.groups, a.packed_study_splits)
    env = packed_study_environment(os.environ, config, (a.dump / "cache").resolve())
    for key in list(os.environ):
        if key.startswith("NVTE_GG_") and key not in env:
            del os.environ[key]
    os.environ.update(env)
    loaded_cubin = None
    if a.packed_study_cubins:
        cubins = list(a.packed_study_cubins.glob("nvte_grouped_gemm3_*.cubin"))
        assert len(cubins) == 1
        loaded_cubin = cubins[0]
        for suffix in (".cubin", ".cu"):
            shutil.copyfile(loaded_cubin.with_suffix(suffix), a.dump / loaded_cubin.with_suffix(suffix).name)
        os.environ["NVTE_GG_RTC_LOAD_CUBIN_DIR"] = str(a.packed_study_cubins.resolve())
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = torch.float32 if a.accumulate else torch.bfloat16

    def packed_operand(shapes, element_dtype, random=False):
        counts = [math.prod(shape) for shape in shapes]
        backing = torch.full((sum(counts) + 64,), 97, device="cuda", dtype=element_dtype)
        views, offset = [], 32
        for shape, count in zip(shapes, counts):
            view = backing[offset:offset + count].view(shape)
            view.copy_(torch.randn(shape, device="cuda", dtype=element_dtype)) if random else view.fill_(float("nan"))
            views.append(view)
            offset += count
        return views, backing

    shapes = config["per_group_MNK"]
    A, abacking = packed_operand([(n, k) if a.layout == "TN" else (k, n) for m, n, k in shapes], torch.bfloat16, True)
    B, bbacking = packed_operand([(k, m) if a.layout == "NT" else (m, k) for m, n, k in shapes], torch.bfloat16, True)
    D, dbacking = packed_operand([(m, n) for m, n, k in shapes], dtype)
    C = [torch.randn_like(d) for d in D] if a.accumulate else []
    splits = config["splits"]

    def bits_hash(tensors):
        digest = hashlib.sha256()
        for tensor in tensors:
            integer = torch.int32 if tensor.dtype == torch.float32 else torch.int16
            digest.update(tensor.detach().cpu().view(integer).numpy().tobytes())
        return digest.hexdigest()

    input_hashes = {"A": bits_hash([abacking]), "B": bits_hash([bbacking])}
    if C:
        input_hashes["initial_C"] = bits_hash(C)

    def launch_packed():
        # 每次有C launch恢复同一C；copy不匹配grouped GEMM的profiler过滤器。
        if C:
            for d, c in zip(D, C):
                d.copy_(c)
        general_grouped_gemm(A, B, D, [None] * a.groups, torch.bfloat16,
                             layout=a.layout, m_splits=splits, grad=a.layout != "TN",
                             accumulate=bool(a.accumulate), single_output=False)

    launch_packed()
    torch.cuda.synchronize()
    errors = []
    for i, (x, y, d) in enumerate(zip(A, B, D)):
        ref = (y.float().t() if a.layout == "NT" else y.float()) @ \
              (x.float().t() if a.layout == "TN" else x.float())
        if C:
            ref += C[i].float()
        assert torch.isfinite(d).all()
        rel = float(torch.linalg.vector_norm(d.float() - ref) / torch.linalg.vector_norm(ref).clamp_min(1e-30))
        group_k = shapes[i][2]
        limit = max(1e-6, 1e-6 * group_k / 256) if a.accumulate else 4e-3
        assert rel < limit, (i, rel, limit)
        torch.testing.assert_close(d.float(), ref, rtol=1e-4 if a.accumulate else 0.05,
                                   atol=5e-4 * max(1, group_k / 256) if a.accumulate else 0.05)
        errors.append(rel)
    output_hash = bits_hash(D)
    output = [d.cpu() for d in D]
    if a.save:
        assert not a.save.exists()
        torch.save(output, a.save)
    if a.compare:
        old = torch.load(a.compare, weights_only=True)
        integer = torch.int32 if a.accumulate else torch.int16
        assert len(old) == len(output) and all(
            d.dtype == b.dtype and d.shape == b.shape and torch.equal(d.view(integer), b.view(integer))
            for d, b in zip(output, old)), "与同输入基线不是逐位相同"
    kernels = []
    if not a.packed_study_profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            launch_packed()
            torch.cuda.synchronize()
        kernels = [e.name for e in prof.events() if e.name.startswith("nvte_grouped_gemm3_")]
        assert len(kernels) == 1 and f"bf16_bf16_{a.dtype}_acc_fp32" in kernels[0], kernels
        if config["arm"] == "contiguous":
            assert f"tb256x{config['tile_n']}x64_s3" in kernels[0], kernels
            source = "fp32" if a.accumulate else "void"
            ending = f"_C{source}_contiguous{config['mode']}"
            if config["effective_ne"] > 1 and config["kmax"] != 512:
                ending += f"_kmax{config['kmax']}"
            ending += "_fulltile"
            assert kernels[0].endswith(ending), (kernels, config)
        else:
            assert "_contiguous" not in kernels[0] and kernels[0].endswith(f"_C{a.dtype}"), kernels
    for backing in (abacking, bbacking, dbacking):
        assert torch.all(backing[:32] == 97) and torch.all(backing[-32:] == 97)
    record = {"correctness": "PASS", "config": config, "MNK": [a.rows, a.cols, a.mpe],
              "layout": a.layout, "groups": a.groups, "splits": splits, "seed": 1234,
              "per_group_MNK": shapes, "split_axis": "K" if a.layout == "NT" else "M",
              "per_group_operand_shapes": {name: [list(t.shape) for t in values]
                                           for name, values in (("A", A), ("B", B), ("D", D))},
              "per_group_operand_strides": {name: [list(t.stride()) for t in values]
                                            for name, values in (("A", A), ("B", B), ("D", D))},
              "dtype": a.dtype, "alpha": 1, "beta": a.accumulate,
              "input_sha256": input_hashes, "output_sha256": output_hash,
              "max_rel_error": max(errors), "kernels": kernels,
              "operand_shapes": {"A": list(A[0].shape), "B": list(B[0].shape), "D": list(D[0].shape)},
              "operand_strides": {"A": list(A[0].stride()), "B": list(B[0].stride()), "D": list(D[0].stride())},
              "environment": {key: value for key, value in os.environ.items() if key.startswith("NVTE_")},
              "core_library": str(_get_shared_object_file("core").resolve()),
              "torch_library": str(_get_shared_object_file("torch").resolve()),
              "rtc_include": os.environ.get("NVTE_CUTLASS3_INCLUDE_DIR"),
              "skip_matching": 4 if a.packed_study_profile else None}
    if config["arm"] == "contiguous":
        record["contract"] = "strict256_packed"
    if loaded_cubin:
        record.update(loaded_cubin=str(loaded_cubin),
                      loaded_cubin_sha256=hashlib.sha256(loaded_cubin.read_bytes()).hexdigest())
    (a.dump / "packed_study.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"packed_study": record}), flush=True)
    if a.iters:
        assert a.packed_study_profile and a.iters == 3
        for _ in range(3):
            launch_packed()
        torch.cuda.synchronize()
        for _ in range(a.iters):
            launch_packed()
        torch.cuda.synchronize()
    raise SystemExit(0)


if a.smallk_arms:
    import hashlib
    import math

    assert (a.layout, a.rows, a.cols, a.mpe, a.groups, a.dtype, a.accumulate) == \
        ("TN", 4096, 4096, 512, 16, "bf16", 0), "本次固定 fwd noC MNK4096x4096x512 groups16"
    assert a.backend == "rtc" and a.dump and not a.pair_c
    assert torch.cuda.get_device_capability() == (8, 9)
    assert len(a.smallk_arms) == 1, "tile/withC为进程级缓存；四臂须独立进程并校验输入摘要与输出逐位相等"
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    splits = [a.rows - (0, 1, 17, 63)[i % 4] for i in range(a.groups)]

    def packed(shapes, random):
        counts = [math.prod(s) for s in shapes]
        storage = torch.full((sum(counts) + 64,), 97, device="cuda", dtype=torch.bfloat16)
        views, offset = [], 32
        for shape, count in zip(shapes, counts):
            view = storage[offset:offset + count].view(shape)
            view.copy_(torch.randn(shape, device="cuda", dtype=torch.bfloat16)) if random else \
                view.fill_(float("nan"))
            views.append(view)
            offset += count
        return views, storage

    A, abacking = packed([(a.cols, a.mpe)] * a.groups, True)
    B, bbacking = packed([(m, a.mpe) for m in splits], True)
    D, dbacking = packed([(m, a.cols) for m in splits], False)
    # seed 与完整输入位模式摘要确保各独立 profiler 进程也使用相同输入。
    input_hashes = {name: hashlib.sha256(buf.cpu().numpy().tobytes()).hexdigest()
                    for name, buf in (("A", abacking.view(torch.int16)),
                                      ("B", bbacking.view(torch.int16)))}
    references = [y.float() @ x.float().t() for x, y in zip(A, B)]
    record = {"case": "fwd_noC", "MNK": [a.rows, a.cols, a.mpe], "splits": splits,
              "groups": a.groups, "seed": 1234, "input_sha256": input_hashes, "arms": {},
              "core_library": str(_get_shared_object_file("core").resolve()),
              "torch_library": str(_get_shared_object_file("torch").resolve()),
              "rtc_include": os.environ.get("NVTE_CUTLASS3_INCLUDE_DIR"),
              "skip_matching": 4 if a.smallk_profile and a.iters else None}
    baseline = None
    for arm in a.smallk_arms:
        dest = a.dump / arm
        dest.mkdir(exist_ok=False)
        os.environ["NVTE_CUTLASS_RTC_DUMP_CUBIN_DIR"] = str(dest.resolve())
        os.environ["NVTE_GG_RTC_CACHE_DIR"] = str((dest / "cache").resolve())
        for key in ("NVTE_GG_N_EXPAND", "NVTE_GG_FORCE_TILE", "NVTE_GG_RTC_LOAD_CUBIN_DIR"):
            os.environ.pop(key, None)
        contiguous = arm in ("contiguous", "nexpand", "contiguous256", "nexpand256")
        bn = 256 if arm in ("production", "contiguous256", "nexpand256") else 128
        ne = 4 if arm in ("nexpand", "nexpand256") else 1
        os.environ["NVTE_GG_CONTIGUOUS"] = "1" if contiguous else "0"
        os.environ["NVTE_GG_CONTIGUOUS_BN"] = str(bn)
        os.environ["NVTE_GG_CONTIGUOUS_N_EXPAND"] = str(ne)
        if arm != "production":
            os.environ["NVTE_GG_FORCE_TILE"] = f"256,{bn},64,3"
        loaded_cubin = None
        if a.smallk_cubins:
            import shutil
            load_dir = (a.smallk_cubins / arm).resolve()
            cubins = list(load_dir.glob("nvte_grouped_gemm3_*.cubin"))
            assert len(cubins) == 1, load_dir
            loaded_cubin = cubins[0]
            assert loaded_cubin.with_suffix(".cu").is_file()
            # LOAD 分支不再 dump TU/cubin，保存确切加载输入；options 仍由 host 实际生成。
            for suffix in (".cubin", ".cu"):
                shutil.copyfile(loaded_cubin.with_suffix(suffix), dest / loaded_cubin.with_suffix(suffix).name)
            os.environ["NVTE_GG_RTC_LOAD_CUBIN_DIR"] = str(load_dir)

        def launch_smallk():
            general_grouped_gemm(A, B, D, [None] * a.groups, torch.bfloat16,
                                 layout="TN", m_splits=splits, grad=False,
                                 accumulate=False, single_output=False)

        for d in D:
            d.fill_(float("nan"))
        launch_smallk()
        torch.cuda.synchronize()
        rel_errors = []
        for actual, reference in zip(D, references):
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual.float(), reference, rtol=0.005, atol=0.005)
            rel_errors.append(float(torch.linalg.vector_norm(actual.float() - reference.float()) /
                                    torch.linalg.vector_norm(reference.float()).clamp_min(1e-30)))
        output = [d.cpu() for d in D]
        if baseline is None:
            baseline = output
        else:
            assert all(torch.equal(d, old) for d, old in zip(output, baseline)), arm
        if a.save:
            assert len(a.smallk_arms) == 1 and not a.save.exists()
            torch.save(output, a.save)
        if a.compare:
            old = torch.load(a.compare, weights_only=True)
            assert len(old) == len(output) and all(torch.equal(x, y) for x, y in zip(output, old))
        kernels = []
        if not a.smallk_profile:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                launch_smallk()
                torch.cuda.synchronize()
            kernels = [e.name for e in prof.events() if e.name.startswith("nvte_grouped_gemm3_")]
            assert len(kernels) == 1 and "bf16_bf16_bf16_acc_fp32" in kernels[0], kernels
            if arm in ("production", "matched"):
                # 生产 beta=0 的 noC 语义使用 typed C 的运行时短路，不强制实验 Cvoid。
                assert kernels[0].endswith("_Cbf16") and "_contiguous" not in kernels[0], kernels
                assert f"tb256x{bn}x64_s3" in kernels[0], kernels
            else:
                assert f"tb256x{bn}x64_s3" in kernels[0], kernels
                assert kernels[0].endswith(f"_Cvoid_contiguous{2 if ne == 4 else 1}"), kernels
        for storage in (abacking, bbacking, dbacking):
            assert torch.all(storage[:32] == 97) and torch.all(storage[-32:] == 97)
        mblocks = sum((m + 255) // 256 for m in splits)
        output_hash = hashlib.sha256()
        for tensor in output:
            output_hash.update(tensor.view(torch.int16).numpy().tobytes())
        entry = {"correctness": "PASS", "max_rel_error": max(rel_errors), "kernels": kernels,
                 "output_sha256": output_hash.hexdigest(),
                 "tile_override": os.environ.get("NVTE_GG_FORCE_TILE"),
                 "contiguous": os.environ["NVTE_GG_CONTIGUOUS"],
                 "n_expand": os.environ["NVTE_GG_CONTIGUOUS_N_EXPAND"],
                 "alpha": 1, "beta": 0, "accumulate": False,
                 "tile_n": bn,
                 "compiled_source_type": "void" if contiguous else "bf16",
                 "expected_contiguous_work_count": mblocks * a.cols // (bn * ne)
                     if contiguous else None}
        if loaded_cubin:
            entry["loaded_cubin"] = str(loaded_cubin)
            entry["loaded_cubin_sha256"] = hashlib.sha256(loaded_cubin.read_bytes()).hexdigest()
        record["arms"][arm] = entry
        print(json.dumps({"smallk_arm": arm, **entry}), flush=True)
        if a.iters:
            for _ in range(3):
                launch_smallk()
            torch.cuda.synchronize()
            for _ in range(a.iters):
                launch_smallk()
            torch.cuda.synchronize()
    (a.dump / "smallk.json").write_text(json.dumps(record, indent=2) + "\n")
    raise SystemExit(0)

torch.manual_seed(1234)
torch.backends.cuda.matmul.allow_tf32 = False
dtype = torch.float32 if a.dtype == "fp32" else torch.bfloat16
shapes = [(max(8, a.rows - 8 * (i % 3)), max(8, a.cols - 8 * (i % 5)),
           (8, 56, 64, 72, 128, 200, 2048)[i % 7]) if a.nonuniform
          else (a.rows, a.cols, a.mpe) for i in range(a.groups)]
A = [torch.randn((n, k) if a.layout == "TN" else (k, n),
                 device="cuda", dtype=torch.bfloat16) for m, n, k in shapes]
B = [torch.randn((k, m) if a.layout == "NT" else (m, k),
                 device="cuda", dtype=torch.bfloat16) for m, n, k in shapes]
splits = [k if a.layout == "NT" else m for m, n, k in shapes]
if a.zero_a:
    for x in A:
        x.zero_()
products = [(b.float().T if a.layout == "NT" else b.float()) @
            (x.float().T if a.layout == "TN" else x.float()) for x, b in zip(A, B)]
C = []
for m, n, k in shapes:
    if a.source == "randn":
        c = torch.randn(m, n, device="cuda", dtype=dtype)
    elif a.source == "arange":
        c = torch.arange(m * n, device="cuda", dtype=torch.float32).reshape(m, n).to(dtype)
    else:
        c = torch.full((m, n), float(a.source == "one"), device="cuda", dtype=dtype)
    C.append(c)
guard, sentinel = 32, 97.0
storage = [torch.full((c.numel() + 2 * guard,), sentinel, device="cuda", dtype=dtype) for c in C]
D = [buf[guard:-guard].reshape(c.shape) for buf, c in zip(storage, C)]
refs = [product + (c.float() if a.accumulate else 0) for product, c in zip(products, C)]

def launch():
    general_grouped_gemm(A, B, D, [None] * a.groups, torch.bfloat16,
                         layout=a.layout, m_splits=splits,
                         grad=a.layout != "TN", accumulate=bool(a.accumulate), single_output=False)

if a.pair_c:
    # 两臂使用相同 A/B、相同输出 dtype；唯一变化是 beta 和 ptr_C。
    outputs = {False: [c.clone() for c in C], True: D}
    def paired_launch(with_c):
        for d, c in zip(outputs[with_c], C):
            d.copy_(c)
        general_grouped_gemm(
            A, B, outputs[with_c], [None] * a.groups, torch.bfloat16,
            layout=a.layout, m_splits=splits, grad=a.layout != "TN",
            accumulate=with_c, single_output=False,
        )

    # 先数值对拍再采集；每步恢复 C，避免反复 RMW 改变待测输入。
    for with_c in (False, True):
        paired_launch(with_c)
        torch.cuda.synchronize()
        errors = []
        for d, product, source in zip(outputs[with_c], products, C):
            ref = product + source.float() if with_c else product
            rel = float(torch.linalg.vector_norm(d.float() - ref)
                        / torch.linalg.vector_norm(ref).clamp_min(1e-30))
            assert torch.isfinite(d).all()
            # 深 K 的无 C 基线已约 4.4e-6；按 K 放大累加误差预算。
            # C 路径另与同一无 C 产物加 C 比较，避免混淆 mainloop 误差。
            limit = max(1e-6, 1e-6 * a.mpe / 256) if dtype == torch.float32 else 4e-3
            assert rel < limit, (a.dtype, with_c, rel, limit)
            errors.append(rel)
        print(json.dumps({"pair_check": "with_c" if with_c else "no_c",
                          "dtype": a.dtype, "groups": a.groups, "mpe": a.mpe,
                          "gemm": [a.rows, a.cols, a.mpe],
                          "max_rel_err": max(errors), "rel_errors": errors}), flush=True)
    if dtype == torch.float32:
        pair_errors, mismatches = [], []
        for no_c, with_c, source in zip(outputs[False], outputs[True], C):
            expected = no_c + source
            rel = float(torch.linalg.vector_norm(with_c - expected)
                        / torch.linalg.vector_norm(expected).clamp_min(1e-30))
            assert rel < 1e-6, ("C 路径相对同一无 C 基线异常", rel)
            pair_errors.append(rel)
            mismatches.append(int(torch.count_nonzero(with_c != expected)))
        print(json.dumps({"pair_source_check": "PASS", "max_rel_err": max(pair_errors),
                          "mismatches": mismatches}), flush=True)

    if a.pair_names:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            paired_launch(False)
            paired_launch(True)
            torch.cuda.synchronize()
        prefix = "nvte_grouped_gemm3_" if a.backend == "rtc" else "gemm_"
        kernels = [e.name for e in prof.events() if e.name.startswith(prefix)]
        per_call = 1 if a.backend == "rtc" else a.groups
        assert len(kernels) == 2 * per_call, kernels
        if a.backend == "rtc":
            assert kernels[0] == kernels[1], kernels
            assert f"bf16_bf16_{a.dtype}_acc_fp32" in kernels[0], kernels
        else:
            assert all("dtypeBF16xBF16xFP32" in name for name in kernels), kernels
        print(json.dumps({"pair_kernels": kernels, "kernels_per_call": per_call,
                          "backend": a.backend}), flush=True)
        raise SystemExit(0)

    for _ in range(a.pair_warmup):
        paired_launch(False)
        paired_launch(True)
    torch.cuda.synchronize()
    order = []
    for sample in range(a.pair_iters):
        arms = (True, False) if (sample + int(a.reverse_pair)) % 2 else (False, True)
        for with_c in arms:
            order.append("with_c" if with_c else "no_c")
            paired_launch(with_c)
        torch.cuda.synchronize()
    print(json.dumps({"pair_order": order, "skip_matching": 2 + 2 * a.pair_warmup}), flush=True)
    raise SystemExit(0)

for run in range(a.repeat):
    if not a.persistent_repeat or run == 0:
        for d, c in zip(D, C):
            d.copy_(c) if a.accumulate else d.fill_(float("nan"))
    if a.persistent_repeat and run > 0 and a.accumulate:
        refs = [product + d.float().clone() for product, d in zip(products, D)]
    launch()
    torch.cuda.synchronize()
    for buf in storage:
        assert torch.all(buf[:guard] == sentinel) and torch.all(buf[-guard:] == sentinel)
    for group, (d, ref) in enumerate(zip(D, refs)):
        err = d.float() - ref
        rel = float(torch.linalg.vector_norm(err) / torch.linalg.vector_norm(ref).clamp_min(1e-30))
        print(json.dumps({"run": run, "group": group, "dtype": a.dtype,
                          "mpe": a.mpe, "rows": a.rows, "accumulate": a.accumulate,
                          "source": a.source, "rel_err": rel,
                          "max_abs": float(err.abs().max()),
                          "lo_abs": float(err[:256].abs().max()),
                          "hi_abs": float(err[256:].abs().max()) if a.rows > 256 else None,
                          "d00": float(d[0, 0]), "ref00": float(ref[0, 0]),
                          "mismatch": int(torch.count_nonzero(err))}), flush=True)
        if a.check:
            assert torch.isfinite(d).all()
            if a.zero_a and a.accumulate:
                assert torch.equal(d, C[group]), "C 地址映射不是恒等"
            else:
                limit = max(1e-6, 1e-6 * shapes[group][2] / 256) if dtype == torch.float32 else 4e-3
                assert rel < limit, rel
                torch.testing.assert_close(d.float(), ref, rtol=0.05 if dtype == torch.bfloat16 else 1e-4,
                                           atol=0.05 if dtype == torch.bfloat16 else 5e-4 * max(1, shapes[group][2] / 256))
if a.kernel_names:
    saved_d = [d.clone() for d in D]
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        launch()
        torch.cuda.synchronize()
    kernels = [e.name for e in prof.events() if e.name.startswith("nvte_grouped_gemm3_")]
    assert len(kernels) == 1, kernels
    assert f"bf16_bf16_{a.dtype}_acc_fp32" in kernels[0], kernels
    tile = os.environ.get("NVTE_GG_FORCE_TILE", "256,256,64,3").split(",")
    assert f"tb{'x'.join(tile[:3])}_s{tile[3]}" in kernels[0], kernels
    print(json.dumps({"kernels": kernels, "shapes": shapes}), flush=True)
    for d, saved in zip(D, saved_d):
        d.copy_(saved)
if a.compare:
    baseline = torch.load(a.compare, weights_only=True)
    assert len(D) == len(baseline), "基线 group 数不一致"
    bits_dtype = torch.int32 if dtype == torch.float32 else torch.int16
    for group, (d, old) in enumerate(zip(D, baseline)):
        assert d.dtype == old.dtype and d.shape == old.shape, f"group {group} 基线 dtype/shape 不一致"
        assert torch.equal(d.cpu().view(bits_dtype), old.view(bits_dtype)), f"group {group} 与基线不是逐位相同"
    print("BITWISE_BASELINE_OK", flush=True)
if a.save:
    a.save.parent.mkdir(parents=True, exist_ok=True)
    torch.save([d.cpu() for d in D], a.save)
if a.iters:
    # 性能只用于 beta=0；重复 in-place 累加会改变输入 C 的值。
    assert not a.accumulate
    for _ in range(3):
        launch()
    torch.cuda.synchronize()
    for _ in range(a.iters):
        launch()
    torch.cuda.synchronize()
