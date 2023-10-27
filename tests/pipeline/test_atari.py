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

    def test_dqn_pipeline(self) -> None:
        experiment_name = f"test_dqn"

        os.makedirs(f"experiments/atari/figures/{experiment_name}")
        shutil.copyfile("tests/pipeline/parameters.json", f"experiments/atari/figures/{experiment_name}/")

        output = subprocess.run(
            [
                "launch_job/atari/launch_local_dqn.sh",
                "-e",
                f"{experiement_name}/{self.game}",
                "-fs",
                self.random_seed,
                "-ls",
                self.random_seed,
                "-ns",
                1,
            ]
        )

        os.remove(f"experiments/atari/figures/{experiment_name}/*")
        os.remove(f"out/atari/{experiment_name}/*")

        self.assertEqual(output, 0)
