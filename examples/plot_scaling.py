import csv
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

CSV_PATH = RESULTS_DIR / "scaling_sweep.csv"
PLOT_PATH = RESULTS_DIR / "scaling_curve.png"


def run_scaling_sweep() -> None:
    output = subprocess.check_output(
        [sys.executable, "examples/scaling_sweep.py"],
        text=True,
    )

    CSV_PATH.write_text(output)


def plot_scaling_curve() -> None:
    workers = []
    step_times = []
    communication_ratios = []

    with CSV_PATH.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            workers.append(int(row["workers"]))
            step_times.append(float(row["total_step_ms"]))
            communication_ratios.append(float(row["communication_ratio"]))

    plt.figure()
    plt.plot(workers, step_times, marker="o")
    plt.xlabel("Workers")
    plt.ylabel("Total Step Time (ms)")
    plt.title("Distributed Training Scaling Curve")
    plt.grid(True)
    plt.savefig(PLOT_PATH, bbox_inches="tight")

    print(f"Saved CSV: {CSV_PATH}")
    print(f"Saved plot: {PLOT_PATH}")


def main() -> None:
    run_scaling_sweep()
    plot_scaling_curve()


if __name__ == "__main__":
    main()
