from dist_profiler.analysis.bottleneck import analyze_bottleneck_transition
from dist_profiler.simulation.training_step import simulate_training_step


WORKERS = [1, 2, 4, 8, 16, 32]


def run_sweep() -> list[dict]:
    results = []

    baseline_step_time = None

    for workers in WORKERS:
        result = simulate_training_step(
            num_workers=workers,
            tensor_size_mb=1000,
            bandwidth_gbps=50,
            latency_ms=1,
            compute_time_ms=120,
            overlap_factor=0.3,
        )

        result["workers"] = workers

        if workers == 1:
            baseline_step_time = result["total_step_time_ms"]

        speedup = baseline_step_time / result["total_step_time_ms"]
        scaling_efficiency = speedup / workers

        result["speedup"] = speedup
        result["scaling_efficiency"] = scaling_efficiency

        results.append(result)

    return results


def main():
    results = run_sweep()

    print(
        "workers,total_step_ms,communication_ms,"
        "communication_ratio,speedup,scaling_efficiency,bottleneck"
    )

    for result in results:
        print(
            f"{result['workers']},"
            f"{result['total_step_time_ms']:.2f},"
            f"{result['communication_time_ms']:.2f},"
            f"{result['communication_ratio']:.2f},"
            f"{result['speedup']:.2f},"
            f"{result['scaling_efficiency']:.2f},"
            f"{result['bottleneck']}"
        )

    print()
    print(analyze_bottleneck_transition(results))


if __name__ == "__main__":
    main()