import math


def simulate_ring_all_reduce(
    num_workers: int,
    tensor_size_mb: float,
    bandwidth_gbps: float,
    latency_ms: float,
) -> float:
    """
    Simulate ring all-reduce communication time.

    Returns:
        communication_time_ms
    """

    tensor_size_gb = tensor_size_mb / 1024

    communication_time = (
        2 * (num_workers - 1) / num_workers
    ) * (tensor_size_gb / bandwidth_gbps)

    communication_time_ms = communication_time * 1000

    total_latency = 2 * (num_workers - 1) * latency_ms

    return communication_time_ms + total_latency
