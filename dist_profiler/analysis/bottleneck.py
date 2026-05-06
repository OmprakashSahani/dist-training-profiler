def analyze_bottleneck_transition(results: list[dict]) -> str:
    """
    Detect where training becomes communication-bound.
    """

    for result in results:
        if result["bottleneck"] == "communication-bound":
            return (
                f"Training becomes communication-bound at "
                f"{result['workers']} workers."
            )

    return "Training remains compute-bound across tested worker counts."
