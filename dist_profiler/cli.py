import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dist-profiler",
        description="Distributed training profiler for ML systems analysis.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of distributed workers",
    )

    args = parser.parse_args()

    print("Distributed Training Profiler")
    print(f"Workers: {args.workers}")


if __name__ == "__main__":
    main()
