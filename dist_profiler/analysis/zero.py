def simulate_zero_memory(
    parameter_memory_gb: float,
    gradient_memory_gb: float,
    optimizer_memory_gb: float,
    activation_memory_gb: float,
    num_workers: int,
) -> dict:
    """
    Simulate ZeRO memory optimization stages.
    """

    # Baseline
    baseline = (
        parameter_memory_gb
        + gradient_memory_gb
        + optimizer_memory_gb
        + activation_memory_gb
    )

    # ZeRO-1: shard optimizer states
    zero1 = (
        parameter_memory_gb
        + gradient_memory_gb
        + (optimizer_memory_gb / num_workers)
        + activation_memory_gb
    )

    # ZeRO-2: shard optimizer + gradients
    zero2 = (
        parameter_memory_gb
        + (gradient_memory_gb / num_workers)
        + (optimizer_memory_gb / num_workers)
        + activation_memory_gb
    )

    # ZeRO-3: shard everything
    zero3 = (
        (parameter_memory_gb / num_workers)
        + (gradient_memory_gb / num_workers)
        + (optimizer_memory_gb / num_workers)
        + activation_memory_gb
    )

    return {
        "baseline_gb": baseline,
        "zero1_gb": zero1,
        "zero2_gb": zero2,
        "zero3_gb": zero3,
    }
