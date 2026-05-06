import argparse

from dist_profiler.analysis.memory import (
    analyze_gpu_fit,
    estimate_memory_usage,
)
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

    parser.add_argument("--model-size", type=float, default=7)
    parser.add_argument("--bytes-per-param", type=int, default=2)
    parser.add_argument("--activation-multiplier", type=float, default=1.5)
    parser.add_argument("--optimizer-multiplier", type=float, default=2.0)
    parser.add_argument("--gpu-memory", type=float, default=80)

    args = parser.parse_args()

    training_result = simulate_training_step(
        num_workers=args.workers,
        tensor_size_mb=args.tensor_size,
        bandwidth_gbps=args.bandwidth,
        latency_ms=args.latency,
        compute_time_ms=args.compute_time,
        overlap_factor=args.overlap,
    )

    memory_result = estimate_memory_usage(
        num_parameters_billion=args.model_size,
        bytes_per_param=args.bytes_per_param,
        activation_multiplier=args.activation_multiplier,
        optimizer_multiplier=args.optimizer_multiplier,
    )

    gpu_fit = analyze_gpu_fit(
        total_memory_gb=memory_result["total_memory_gb"],
        gpu_memory_gb=args.gpu_memory,
    )

    print("Distributed Training Profiler")
    print("=" * 40)

    print("\nTraining Step Analysis")
    print("-" * 40)

    print(f"Workers: {args.workers}")
    print(f"Compute Time: {training_result['compute_time_ms']:.2f} ms")
    print(f"Communication Time: {training_result['communication_time_ms']:.2f} ms")
    print(f"Total Step Time: {training_result['total_step_time_ms']:.2f} ms")
    print(f"Communication Ratio: {training_result['communication_ratio']:.2f}")
    print(f"Bottleneck: {training_result['bottleneck']}")

    print("\nMemory Analysis")
    print("-" * 40)

    print(f"Model Size: {args.model_size}B parameters")
    print(f"Parameter Memory: {memory_result['parameter_memory_gb']:.2f} GB")
    print(f"Gradient Memory: {memory_result['gradient_memory_gb']:.2f} GB")
    print(f"Optimizer Memory: {memory_result['optimizer_memory_gb']:.2f} GB")
    print(f"Activation Memory: {memory_result['activation_memory_gb']:.2f} GB")
    print(f"Total Memory: {memory_result['total_memory_gb']:.2f} GB")

    print("\nGPU Fit Analysis")
    print("-" * 40)

    fit_status = "YES" if gpu_fit["fits"] else "NO"

    print(f"GPU Memory: {args.gpu_memory:.2f} GB")
    print(f"Fits on GPU: {fit_status}")
    print(f"Memory Utilization: {gpu_fit['utilization']:.2f}x")


if __name__ == "__main__":
    main()