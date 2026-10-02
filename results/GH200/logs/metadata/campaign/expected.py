import json, yaml, collections
from pathlib import Path
from tilebench.paths import OPERATOR_ROOT
from tilebench.data.tensors import expand_cases
exp = {}
for d in sorted(Path(OPERATOR_ROOT).iterdir()):
    if not (d / "config.yaml").exists() or d.name.startswith("_"): continue
    cases = expand_cases(d.name, yaml.safe_load(open(d / "config.yaml")))
    exp[d.name] = {"n": len(cases), "per_dtype": dict(collections.Counter(c.get("dtype", "fp32") for c in cases))}
json.dump(exp, open("/tmp/tilebench-gh200-campaign/expected.json", "w"), indent=1)
print(len(exp), "operators;", sum(v["n"] for v in exp.values()), "cases")
