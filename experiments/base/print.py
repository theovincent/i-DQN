def print_info(
    experiment_name: str,
    algorithm: str,
    environment_name: str,
    bellman_iteration_scope: int,
    seed: int,
    train: bool = True,
):
    print(f"-------- {experiment_name} --------")
    if train:
        print(
            f"Training {algorithm} on {environment_name} with {bellman_iteration_scope} Bellman iterations at a time and seed {seed}..."
        )
    else:
        print(
            f"Evaluating {algorithm} on {environment_name} with {bellman_iteration_scope} Bellman iterations at a time and seed {seed}..."
        )
