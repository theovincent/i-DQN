import unittest
import os
import shutil
import subprocess
import numpy as np


class TestPipeline(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = np.random.randint(1000)
        print(f"random seed {self.random_seed}")
        self.game = "Pong"

    def test_dqn_pipeline(self) -> None:
        experiment_name = "test_pipeline"

        if not os.path.exists(f"experiments/atari/figures/{experiment_name}"):
            os.makedirs(f"experiments/atari/figures/{experiment_name}")
        shutil.copyfile(
            "tests/pipeline/parameters.json", f"experiments/atari/figures/{experiment_name}/parameters.json"
        )

        subprocess.run(["launch_job/create_tmux.sh"])

        output = subprocess.run(
            [
                "launch_job/atari/launch_local_dqn.sh",
                "-e",
                f"{experiment_name}/{self.game}",
                "-fs",
                str(self.random_seed),
                "-ls",
                str(self.random_seed),
                "-ns",
                "1",
            ]
        )

        subprocess.run(["rm", "-r", f"experiments/atari/figures/{experiment_name}/{self.game}/DQN"])
        subprocess.run(["rm", "-r", f"out/atari/{experiment_name}/{self.game}/train_dqn_{str(self.random_seed)}"])
        self.assertEqual(output, 0)
