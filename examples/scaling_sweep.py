from dist_profiler.analysis.bottleneck import analyze_bottleneck_transition
from dist_profiler.simulation.training_step import simulate_training_step


WORKERS = [1, 2, 4, 8, 16, 32]


def run_sweep() -> list[dict]:
    results = []

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
        results.append(result)

    return results


def main():
    results = run_sweep()

    print("workers,total_step_ms,communication_ms,communication_ratio,bottleneck")

    for result in results:
        print(
            f"{result['workers']},"
            f"{result['total_step_time_ms']:.2f},"
            f"{result['communication_time_ms']:.2f},"
            f"{result['communication_ratio']:.2f},"
            f"{result['bottleneck']}"
        )

    print()
    print(analyze_bottleneck_transition(results))


if __name__ == "__main__":
    main()