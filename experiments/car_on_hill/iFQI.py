import sys
import argparse
import json

from experiments.base.parser import addparse
from experiments.base.print import print_info


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train IFQI on Car-On-Hill.")
    addparse(parser)
    args = parser.parse_args(argvs)
    print_info(args.experiment_name, "IFQI", "Car-On-Hill", args.bellman_iterations_scope, args.seed)
    p = json.load(open(f"experiments/car_on_hill/figures/{args.experiment_name}/parameters.json"))  # p for parameters

    from experiments.car_on_hill.utils import define_multi_q, define_data_loader_samples, generate_keys
    from experiments.base.iFQI import train

    shuffle_key, q_key = generate_keys(args.seed)

    data_loader_samples = define_data_loader_samples(
        p["n_samples"], args.experiment_name, p["batch_size_samples"], shuffle_key
    )
    q = define_multi_q(
        args.bellman_iterations_scope + 1, p["gamma"], q_key, p["layers_dimension"], learning_rate=p["learning_rate"]
    )

    train("car_on_hill", args, q, p, data_loader_samples)
