def estimate_memory_usage(
    num_parameters_billion: float,
    bytes_per_param: int,
    activation_multiplier: float,
    optimizer_multiplier: float,
) -> dict:
    """
    Estimate training memory usage.

    Returns memory in GB.
    """

    num_parameters = num_parameters_billion * 1e9

    parameter_memory_gb = (
        num_parameters * bytes_per_param
    ) / (1024**3)

    gradient_memory_gb = parameter_memory_gb

    optimizer_memory_gb = (
        parameter_memory_gb * optimizer_multiplier
    )

    activation_memory_gb = (
        parameter_memory_gb * activation_multiplier
    )

    total_memory_gb = (
        parameter_memory_gb
        + gradient_memory_gb
        + optimizer_memory_gb
        + activation_memory_gb
    )

    return {
        "parameter_memory_gb": parameter_memory_gb,
        "gradient_memory_gb": gradient_memory_gb,
        "optimizer_memory_gb": optimizer_memory_gb,
        "activation_memory_gb": activation_memory_gb,
        "total_memory_gb": total_memory_gb,
    }
