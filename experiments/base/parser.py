import argparse


def addparse(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-e",
        "--experiment_name",
        help="Experiment name.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-b",
        "--bellman_iterations_scope",
        help="Number of Bellman iterations taken into account at once.",
        type=int,
    )
    parser.add_argument(
        "-s",
        "--seed",
        help="Seed of the training.",
        type=int,
        required=True,
    )
