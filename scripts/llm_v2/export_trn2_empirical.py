#!/usr/bin/env python
"""Write tilebench/data/peak_performance/empirical/Trainium2.json (tilearena-empirical-peak/1, the layout of the
B200 / GH200 / MI300X files) from a finalized Trn2 calibration directory. Data copy only; status candidate.

    python scripts/llm_v2/export_trn2_empirical.py artifacts/llm_v2/calibration/Trn2/<calibration_id>
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_public(out: Path):
    """tilebench/data/peak_performance/empirical/Trainium2.json in the tilearena-empirical-peak/1 layout of the
    B200 / GH200 / MI300X files: exact profile values, source path + file sha256 + embedded seal, scope, status."""
    prof_path = out / "profile.json"
    prof = json.loads(prof_path.read_text())
    git = lambda *x: subprocess.run(["git", *x], cwd=REPO, capture_output=True, text=True).stdout.strip()
    env = json.loads((out / "environment.json").read_text())
    scale = {"byte/s": (1e9, "GB/s"), "FLOP/s": (1e12, "TFLOP/s"), "OP/s": (1e12, "TOP/s")}
    modes, missing = {}, {}
    for name, m in sorted(prof["modes"].items()):
        if m["status"] == "calibrated":
            f, u = scale[m["unit"]]
            e = {"value": m["value"], "unit": m["unit"], "value_scaled": m["value"] / f, "unit_scaled": u,
                 "status": "calibrated", "role": m["role"], "source_kind": "measured",
                 "selected_point": m.get("selected_point"), "flags": m.get("flags", []), "meaning": m.get("meaning"),
                 "resource_scope": "per logical NeuronCore (LNC=2: two physical NeuronCore-v3)",
                 "limitations": m.get("limitations")}
            if m.get("per_physical_core_diagnostic"):
                e["per_physical_core_diagnostic"] = [{"id": d["id"], "value": d["throughput"],
                                                      "value_scaled": d["throughput"] / f if d["throughput"] else None,
                                                      "unit_scaled": u, "valid": d["valid"]} for d in m["per_physical_core_diagnostic"]]
            if m.get("precision_path"):
                e["precision_path"] = m["precision_path"]
            modes[name] = e
        else:
            missing[name] = {"status": m["status"], "unit": m["unit"], "reason": m.get("reason")}
    doc = {
        "schema": "tilearena-empirical-peak/1", "device": "Trn2", "architecture": "AWS Trainium2 (NeuronCore-v3)",
        "measurement_type": "empirical_sustained_calibration", "ceiling_basis": "empirical",
        "description": "Empirically measured sustained throughput and bandwidth on the Trn2 campaign device (one logical "
                       "NeuronCore, LNC=2), copied verbatim from the TileArena calibration profile (not an AWS datasheet "
                       "value). Intended as the Trn2 P_emp / BW_emp of the empirical SOL T_SOL = max(F / P_emp[mode], Q / BW_emp).",
        "calibration_id": prof["calibration_id"],
        "calibration_status": "candidate",
        "frozen_by": None,
        "registered_at": None,
        "source": {"profile_path": str(prof_path.relative_to(REPO)), "profile_file_sha256": _sha(prof_path),
                   "profile_embedded_seal_sha256": prof["profile_sha256"],
                   "environment_path": str((out / "environment.json").relative_to(REPO)),
                   "environment_file_sha256": _sha(out / "environment.json"),
                   "branch": "exp/llm-trn2", "commit": prof["code"]["git_sha"],
                   "calibration_manifest_entry": "not registered (candidate; tilebench/llm/v2/manifests/calibration.yaml "
                                                 "devices.Trn2 stays status none until the owner decides)"},
        "protocol": prof["protocol"],
        "measured_with_code": prof["code"],
        "scope": prof["scope"],
        "environment": {"instance_type": env.get("instance_type"), "logical_neuroncore_config": env.get("logical_neuroncore_config"),
                        "packages": env.get("packages"), "aws_neuronx_debs": env.get("aws_neuronx_debs"),
                        "neuron_driver": env.get("neuron_driver"), "device_memory": env.get("device_memory")},
        "modes": modes,
        "modes_not_calibrated": missing,
        "notes": [
            "value is the exact number stored in the profile (unit as stored: byte/s, FLOP/s, OP/s); value_scaled = value / 1e9 "
            "(GB/s) or value / 1e12 (TFLOP/s) for readability only.",
            "All values are per logical NeuronCore (LNC=2, two physical NeuronCore-v3, NEURON_RT_NUM_CORES=1), the scope on which "
            "NKI kernels are timed. per_physical_core_diagnostic entries were measured with one of the two cores active and are "
            "not scoring values; whole-device rates (4 logical / 8 physical cores) were not measured and must not be derived by "
            "multiplying.",
            "Timing: device busy time per execution from the Neuron runtime trace (the formal TileBench NKI method), no host "
            "interval; compilation and warmup outside the timed sessions.",
            "mma_tf32_f32acc uses neuronx-cc --auto-cast matmult --auto-cast-type tf32 on fp32 operands; the default eager fp32 "
            "matmul is IEEE (gemm_fp32_ieee, control only).",
            "fp32_fma_vector / fp16x2_fma_vector / bf16x2_fma_vector are NKI multiply-add chains on VectorE, ScalarE and GpSimd "
            "(Trn2 has no packed-pair FMA; the 16-bit modes are FMA on fp16/bf16 tiles).",
            "mma_int8_i32acc and mma_fp8_e4m3_f32acc are unsupported on this device path (see modes_not_calibrated).",
            "The legacy tilebench/data/peak_performance/Trainium2.json is unchanged; its 577.7 GB/s is an older torch-xla "
            "measurement and its compute entries are not empirical.",
            "This file is a data-only copy; the scoring path reads the profile through manifests/calibration.yaml, not this file.",
        ],
    }
    dst = REPO / "tilebench/data/peak_performance/empirical/Trainium2.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n")
    print(str(dst.relative_to(REPO)), _sha(dst))



if __name__ == "__main__":
    export_public(Path(sys.argv[1]).resolve())
