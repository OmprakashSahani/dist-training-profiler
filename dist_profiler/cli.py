import argparse

from dist_profiler.simulation.training_step import simulate_training_step


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dist-profiler",
        description="Distributed training profiler for ML systems analysis.",
    )

    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tensor-size", type=float, default=500)
    parser.add_argument("--bandwidth", type=float, default=50)
    parser.add_argument("--latency", type=float, default=1)
    parser.add_argument("--compute-time", type=float, default=120)
    parser.add_argument("--overlap", type=float, default=0.3)

    args = parser.parse_args()

    result = simulate_training_step(
        num_workers=args.workers,
        tensor_size_mb=args.tensor_size,
        bandwidth_gbps=args.bandwidth,
        latency_ms=args.latency,
        compute_time_ms=args.compute_time,
        overlap_factor=args.overlap,
    )

    print("Distributed Training Profiler")
    print(f"Workers: {args.workers}")
    print()

    print(f"Compute Time: {result['compute_time_ms']:.2f} ms")
    print(f"Communication Time: {result['communication_time_ms']:.2f} ms")
    print(f"Effective Communication: {result['effective_communication_ms']:.2f} ms")
    print(f"Total Step Time: {result['total_step_time_ms']:.2f} ms")
    print(f"Communication Ratio: {result['communication_ratio']:.2f}")
    print(f"Bottleneck: {result['bottleneck']}")


if __name__ == "__main__":
    main()