"""Regenerate all paper-figure drafts from artifacts/paper_figures/combined/ (CPU only) and validate them.

  CUDA_VISIBLE_DEVICES= PYTHONPATH=.:scripts/paper_figures python scripts/paper_figures/build_all_figures.py

Order: figure evidence -> RQ1, RQ2, RQ3 -> appendix A1-A5 -> plots/plot_manifest.json -> validate_plots.py.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figure_data as FD  # noqa: E402
import plot_appendix  # noqa: E402
import plot_rq1  # noqa: E402
import plot_rq2  # noqa: E402
import plot_rq3  # noqa: E402

SLOTS = [  # (id, paper slot, figure name, placement)
    ("fig2_rq1", "Figure 2", "fig_rq1_cross_accelerator", "main"),
    ("fig3_rq2", "Figure 3", "fig_rq2_cross_device_diagnosis", "main"),
    ("fig4_rq3", "Figure 4", "fig_rq3_within_device_dsl", "main"),
    ("figA1", "Appendix Figure A1", "fig_a1_performance_atlas", "appendix"),
    ("figA2", "Appendix Figure A2", "fig_a2_shape_dtype", "appendix"),
    ("figA3", "Appendix Figure A3", "fig_a3_execution_paths", "appendix"),
    ("figA4", "Appendix Figure A4", "fig_a4_within_device_matrix", "appendix"),
    ("figA5", "Appendix Figure A5", "fig_a5_profiling_evidence", "appendix"),
]
RQ4_BLOCKED = [
    "no finalized LLM-generation results (protocol v2 formal campaigns) are part of artifacts/paper_figures/",
    "no SOL-efficiency curves, LLM generation costs or human development times are available as validated inputs; none may be fabricated",
    "no placeholder numerical figure is produced",
]
TODO = [
    {"id": "fig5_rq4_llm", "paper_slot": "Figure 5", "rq": "RQ4", "status": "todo", "blocked_by": RQ4_BLOCKED},
    {"id": "appendix_llm", "paper_slot": "Appendix LLM figure", "rq": "RQ4", "status": "todo", "blocked_by": RQ4_BLOCKED},
]
EXCLUDED = {"NKI / Trainium": "no finalized NKI results; not shown in any figure",
            "skill-transfer figures": "out of scope for this draft set"}


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=FD.REPO)


def main():
    run([sys.executable, str(HERE / "build_figure_evidence.py"), "--repo", str(FD.REPO)])
    plot_rq1.main()
    plot_rq2.main()
    plot_rq3.main()
    plot_appendix.main()
    figs = []
    for fid, slot, name, place in SLOTS:
        m = json.load(open(FD.PLOTS / "manifests" / f"{name}.json"))
        figs.append({"id": fid, "paper_slot": slot, "figure": name, "rq": m["rq"], "status": "generated", "placement": place,
                     "width_in": m["layout"]["width_in"], "height_in": m["layout"]["height_in"],
                     "manifest": f"manifests/{name}.json", "outputs": m["outputs"]})
    qa = json.load(open(FD.COMBINED / "qa_combined.json"))
    top = {"schema": "tilearena-plot-manifest/1", "source_git_commit": FD.git_head(), "combined_qa_status": qa["status"],
           "inputs": ("plot scripts read artifacts/paper_figures/combined/ only; build_figure_evidence.py reads the NVIDIA/AMD packages and "
                      "writes combined/{figure_evidence.csv,execution_path_matrix.csv,rq2_case_selection.json}"),
           "figures": figs + TODO, "excluded": EXCLUDED}
    json.dump(top, open(FD.PLOTS / "plot_manifest.json", "w"), indent=1)
    run([sys.executable, str(HERE / "validate_plots.py")])


if __name__ == "__main__":
    main()
