import argparse

from dist_profiler.simulation.all_reduce import simulate_ring_all_reduce


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dist-profiler",
        description="Distributed training profiler for ML systems analysis.",
    )

    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tensor-size", type=float, default=500)
    parser.add_argument("--bandwidth", type=float, default=50)
    parser.add_argument("--latency", type=float, default=1)

    args = parser.parse_args()

    communication_time = simulate_ring_all_reduce(
        num_workers=args.workers,
        tensor_size_mb=args.tensor_size,
        bandwidth_gbps=args.bandwidth,
        latency_ms=args.latency,
    )

    print("Distributed Training Profiler")
    print(f"Workers: {args.workers}")
    print(f"Tensor Size: {args.tensor_size} MB")
    print(f"Communication Time: {communication_time:.2f} ms")


if __name__ == "__main__":
    main()
