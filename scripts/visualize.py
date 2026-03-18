import json

import matplotlib.pyplot as plt
import numpy as np


def plot_results(json_path, output_png):
    with open(json_path, "r") as f:
        results = json.load(f)

    def format_params(params):
        return ", ".join([f"{k}={v}" for k, v in params.items()])

    labels = [f"{format_params(r['params'])}\n{r['dtype']}" for r in results]
    torch_times = [r["torch_ms"] for r in results]
    triton_times = [r["triton_ms"] for r in results]
    cutile_times = [r["cutile_ms"] for r in results]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width, torch_times, width, label="PyTorch")
    ax.bar(x, triton_times, width, label="Triton")
    ax.bar(x + width, cutile_times, width, label="cuTile")

    ax.set_ylabel("Time (ms)")
    ax.set_title("Performance Comparison: PyTorch vs Triton vs cuTile")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.set_yscale("log")  # Use log scale because of large differences

    plt.tight_layout()
    plt.savefig(output_png)
    print(f"Plot saved to {output_png}")


if __name__ == "__main__":
    plot_results("results/logs/results.json", "results/comparison.png")
