import csv
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

CSV_PATH = RESULTS_DIR / "overlap_sweep.csv"
PLOT_PATH = RESULTS_DIR / "overlap_curve.png"


def run_overlap_sweep() -> None:
    output = subprocess.check_output(
        [sys.executable, "examples/overlap_sweep.py"],
        text=True,
    )

    CSV_PATH.write_text(output)


def plot_overlap_curve() -> None:
    overlaps = []
    step_times = []

    with CSV_PATH.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            overlaps.append(float(row["overlap"]))
            step_times.append(float(row["total_step_ms"]))

    plt.figure()
    plt.plot(overlaps, step_times, marker="o")
    plt.xlabel("Overlap Factor")
    plt.ylabel("Total Step Time (ms)")
    plt.title("Communication Overlap Impact")
    plt.grid(True)
    plt.savefig(PLOT_PATH, bbox_inches="tight")

    print(f"Saved plot: {PLOT_PATH}")


def main() -> None:
    run_overlap_sweep()
    plot_overlap_curve()


if __name__ == "__main__":
    main()
