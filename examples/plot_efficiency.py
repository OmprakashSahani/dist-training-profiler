import csv
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

CSV_PATH = RESULTS_DIR / "scaling_sweep.csv"
PLOT_PATH = RESULTS_DIR / "scaling_efficiency.png"


def run_scaling_sweep() -> None:
    output = subprocess.check_output(
        [sys.executable, "examples/scaling_sweep.py"],
        text=True,
    )

    CSV_PATH.write_text(output)


def plot_efficiency_curve() -> None:
    workers = []
    efficiencies = []

    with CSV_PATH.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                workers.append(int(row["workers"]))
                efficiencies.append(float(row["scaling_efficiency"]))
            except (ValueError, KeyError):
                continue

    plt.figure()
    plt.plot(workers, efficiencies, marker="o")
    plt.xlabel("Workers")
    plt.ylabel("Scaling Efficiency")
    plt.title("Distributed Training Scaling Efficiency")
    plt.grid(True)
    plt.savefig(PLOT_PATH, bbox_inches="tight")

    print(f"Saved plot: {PLOT_PATH}")


def main() -> None:
    run_scaling_sweep()
    plot_efficiency_curve()


if __name__ == "__main__":
    main()
