import unittest
import os
import shutil
import time
import subprocess
import numpy as np


class TestPipeline(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = np.random.randint(1000)
        print(f"random seed {self.random_seed}")
        self.game = "Pong"
        self.experiment_name = "test_pipeline"

        if not os.path.exists(f"experiments/atari/figures/{self.experiment_name}"):
            os.makedirs(f"experiments/atari/figures/{self.experiment_name}")
        shutil.copyfile(
            "tests/pipeline/parameters_test.json", f"experiments/atari/figures/{self.experiment_name}/parameters.json"
        )

        subprocess.run(["launch_job/create_tmux.sh"])

    def run_pipeline(self, algorithm: str, iterated: bool = False):
        if iterated:
            args = ("-lb", "3")
            output_file = "3_"
        else:
            args = ()
            output_file = ""

        output = subprocess.run(
            [
                f"launch_job/atari/launch_local_{algorithm.lower()}.sh",
                "-e",
                f"{self.experiment_name}/{self.game}",
                "-fs",
                str(self.random_seed),
                "-ls",
                str(self.random_seed),
                "-ns",
                "1",
                *args,
            ]
        )
        self.assertEqual(output.returncode, 0)

        time.sleep(10)

        output = subprocess.run(
            ["rm", "-r", f"experiments/atari/figures/{self.experiment_name}/{self.game}/{algorithm}"]
        )
        self.assertEqual(output.returncode, 0)

        output = subprocess.run(
            [
                "mv",
                f"out/atari/{self.experiment_name}/{self.game}/{output_file}train_{algorithm.lower()}_1{str(self.random_seed)}.out",
                f"out/atari/{self.experiment_name}/{self.game}/{output_file}train_{algorithm.lower()}.out",
            ]
        )
        self.assertEqual(output.returncode, 0)

    def test_dqn_pipeline(self) -> None:
        self.run_pipeline("DQN")

    def test_iqn_pipeline(self) -> None:
        self.run_pipeline("IQN")

    def test_rem_pipeline(self) -> None:
        self.run_pipeline("REM")

    def test_idqn_pipeline(self) -> None:
        self.run_pipeline("iDQN", iterated=True)

    def test_iiqn_pipeline(self) -> None:
        self.run_pipeline("iIQN", iterated=True)

    def test_irem_pipeline(self) -> None:
        self.run_pipeline("iREM", iterated=True)
