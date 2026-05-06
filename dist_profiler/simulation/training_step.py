from dist_profiler.simulation.all_reduce import simulate_ring_all_reduce


def simulate_training_step(
    num_workers: int,
    tensor_size_mb: float,
    bandwidth_gbps: float,
    latency_ms: float,
    compute_time_ms: float,
    overlap_factor: float,
) -> dict:
    """
    Simulate distributed training step.

    overlap_factor:
        0.0 = no overlap
        1.0 = perfect overlap
    """

    communication_time_ms = simulate_ring_all_reduce(
        num_workers=num_workers,
        tensor_size_mb=tensor_size_mb,
        bandwidth_gbps=bandwidth_gbps,
        latency_ms=latency_ms,
    )

    effective_communication = communication_time_ms * (1 - overlap_factor)

    total_step_time = compute_time_ms + effective_communication

    communication_ratio = effective_communication / total_step_time

    bottleneck = (
        "communication-bound"
        if communication_ratio > 0.5
        else "compute-bound"
    )

    return {
        "compute_time_ms": compute_time_ms,
        "communication_time_ms": communication_time_ms,
        "effective_communication_ms": effective_communication,
        "total_step_time_ms": total_step_time,
        "communication_ratio": communication_ratio,
        "bottleneck": bottleneck,
    }
