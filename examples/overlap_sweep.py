from dist_profiler.simulation.training_step import simulate_training_step


OVERLAP_FACTORS = [0.0, 0.3, 0.6, 0.9]


def main() -> None:
    print("overlap,total_step_ms,effective_communication_ms,communication_ratio,bottleneck")

    for overlap in OVERLAP_FACTORS:
        result = simulate_training_step(
            num_workers=16,
            tensor_size_mb=1000,
            bandwidth_gbps=50,
            latency_ms=1,
            compute_time_ms=120,
            overlap_factor=overlap,
        )

        print(
            f"{overlap:.1f},"
            f"{result['total_step_time_ms']:.2f},"
            f"{result['effective_communication_ms']:.2f},"
            f"{result['communication_ratio']:.2f},"
            f"{result['bottleneck']}"
        )


if __name__ == "__main__":
    main()
