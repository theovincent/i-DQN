import subprocess
import sys
import argparse


def run_cli(argvs=sys.argv[1:]):
    parser = argparse.ArgumentParser("Parallel launcher.")
    parser.add_argument(
        "-c",
        "--command",
        help="Command the launch.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-s",
        "--seed",
        help="Unit seed.",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-ns",
        "--n_parallel_seeds",
        help="Number of seeds to be launched in parallel.",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--stdout",
        help="File where the standard output can will be stored.",
        type=str,
        required=True,
    )
    args = parser.parse_args(argvs)

    processes = []

    for dozen in range(1, args.n_parallel_seeds + 1):
        with open(f"{args.stdout}_{dozen}{args.seed}.out", "w") as stdout_file:
            processes.append(
                subprocess.Popen(
                    args.command + f" -s {dozen}{args.seed}", shell=True, stdout=stdout_file, stderr=stdout_file
                )
            )

    for process in processes:
        process.communicate()
