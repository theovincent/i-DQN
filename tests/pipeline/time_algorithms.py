import os
import shutil
from time import time
import subprocess
import numpy as np


def run_cli():
    time_atari_env = TimeAtariPipeline()

    time_dqn = time_atari_env.time_dqn()
    time_iqn = time_atari_env.time_iqn()
    time_rem = time_atari_env.time_rem()
    time_idqn = time_atari_env.time_idqn()
    time_iiqn = time_atari_env.time_iiqn()
    time_irem = time_atari_env.time_irem()

    print("\n\n")
    print("Time DQN: ", time_dqn)
    print("Time IQN: ", time_iqn)
    print("Time REM: ", time_rem)
    print("Time iDQN: ", time_idqn)
    print("Time iIQN: ", time_iiqn)
    print("Time iREM: ", time_irem)


class TimeAtariPipeline:
    def __init__(self) -> None:
        self.random_seed = np.random.randint(100)
        print(f"random seed {self.random_seed}")
        self.n_runs = 2
        self.game = "Breakout"
        self.experiment_name = "time_pipeline"

        if not os.path.exists(f"experiments/atari/figures/{self.experiment_name}"):
            os.makedirs(f"experiments/atari/figures/{self.experiment_name}")
        shutil.copyfile(
            "tests/pipeline/parameters_time.json", f"experiments/atari/figures/{self.experiment_name}/parameters.json"
        )

    def time_algorithm(self, run_cli, algorithm, *args) -> float:
        if not os.path.exists(f"experiments/atari/figures/{self.experiment_name}/{self.game}/{algorithm}"):
            os.makedirs(f"experiments/atari/figures/{self.experiment_name}/{self.game}/{algorithm}")

        t_begin = time()

        for _ in range(self.n_runs):
            run_cli(["-e", f"{self.experiment_name}/{self.game}", "-s", str(self.random_seed), *args])

        time_compute = (time() - t_begin) / self.n_runs

        subprocess.run(["rm", "-r", f"experiments/atari/figures/{self.experiment_name}/{self.game}/{algorithm}"])

        return time_compute

    def time_dqn(self) -> float:
        from experiments.atari.DQN import run_cli

        return self.time_algorithm(run_cli, "DQN")

    def time_iqn(self) -> float:
        from experiments.atari.IQN import run_cli

        return self.time_algorithm(run_cli, "IQN")

    def time_rem(self) -> float:
        from experiments.atari.REM import run_cli

        return self.time_algorithm(run_cli, "REM")

    def time_idqn(self) -> float:
        from experiments.atari.iDQN import run_cli

        return self.time_algorithm(run_cli, "iDQN", "-b", "5")

    def time_iiqn(self) -> float:
        from experiments.atari.iIQN import run_cli

        return self.time_algorithm(run_cli, "iIQN", "-b", "5")

    def time_irem(self) -> float:
        from experiments.atari.iREM import run_cli

        return self.time_algorithm(run_cli, "iREM", "-b", "5")
