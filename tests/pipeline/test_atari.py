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
            "tests/pipeline/parameters.json", f"experiments/atari/figures/{self.experiment_name}/parameters.json"
        )

        subprocess.run(["launch_job/create_tmux.sh"])

    def test_dqn_pipeline(self) -> None:
        output = subprocess.run(
            [
                "launch_job/atari/launch_local_dqn.sh",
                "-e",
                f"{self.experiment_name}/{self.game}",
                "-fs",
                str(self.random_seed),
                "-ls",
                str(self.random_seed),
                "-ns",
                "1",
            ]
        )
        self.assertEqual(output.returncode, 0)

        time.sleep(10)

        output = subprocess.run(["rm", "-r", f"experiments/atari/figures/{self.experiment_name}/{self.game}/DQN"])
        self.assertEqual(output.returncode, 0)

        output = subprocess.run(
            [
                "mv",
                f"out/atari/{self.experiment_name}/{self.game}/train_dqn_1{str(self.random_seed)}.out",
                f"out/atari/{self.experiment_name}/{self.game}/train_dqn.out",
            ]
        )
        self.assertEqual(output.returncode, 0)

    def test_iqn_pipeline(self) -> None:
        output = subprocess.run(
            [
                "launch_job/atari/launch_local_iqn.sh",
                "-e",
                f"{self.experiment_name}/{self.game}",
                "-fs",
                str(self.random_seed),
                "-ls",
                str(self.random_seed),
                "-ns",
                "1",
            ]
        )
        self.assertEqual(output.returncode, 0)

        time.sleep(10)

        subprocess.run(["rm", "-r", f"experiments/atari/figures/{self.experiment_name}/{self.game}/IQN"])
        self.assertEqual(output.returncode, 0)

        subprocess.run(
            [
                "mv",
                f"out/atari/{self.experiment_name}/{self.game}/train_iqn_1{str(self.random_seed)}.out",
                f"out/atari/{self.experiment_name}/{self.game}/train_iqn.out",
            ]
        )
        self.assertEqual(output.returncode, 0)

    def test_rem_pipeline(self) -> None:
        output = subprocess.run(
            [
                "launch_job/atari/launch_local_rem.sh",
                "-e",
                f"{self.experiment_name}/{self.game}",
                "-fs",
                str(self.random_seed),
                "-ls",
                str(self.random_seed),
                "-ns",
                "1",
            ]
        )
        self.assertEqual(output.returncode, 0)

        time.sleep(10)

        subprocess.run(["rm", "-r", f"experiments/atari/figures/{self.experiment_name}/{self.game}/REM"])
        self.assertEqual(output.returncode, 0)

        subprocess.run(
            [
                "mv",
                f"out/atari/{self.experiment_name}/{self.game}/train_rem_1{str(self.random_seed)}.out",
                f"out/atari/{self.experiment_name}/{self.game}/train_rem.out",
            ]
        )
        self.assertEqual(output.returncode, 0)

    def test_idqn_pipeline(self) -> None:
        output = subprocess.run(
            [
                "launch_job/atari/launch_local_idqn.sh",
                "-e",
                f"{self.experiment_name}/{self.game}",
                "-fs",
                str(self.random_seed),
                "-ls",
                str(self.random_seed),
                "-ns",
                "1",
                "-lb",
                "3",
            ]
        )
        self.assertEqual(output.returncode, 0)

        time.sleep(10)

        subprocess.run(["rm", "-r", f"experiments/atari/figures/{self.experiment_name}/{self.game}/iDQN"])
        self.assertEqual(output.returncode, 0)

        subprocess.run(
            [
                "mv",
                f"out/atari/{self.experiment_name}/{self.game}/3_train_idqn_1{str(self.random_seed)}.out",
                f"out/atari/{self.experiment_name}/{self.game}/3_train_idqn.out",
            ]
        )
        self.assertEqual(output.returncode, 0)

    def test_iiqn_pipeline(self) -> None:
        output = subprocess.run(
            [
                "launch_job/atari/launch_local_iiqn.sh",
                "-e",
                f"{self.experiment_name}/{self.game}",
                "-fs",
                str(self.random_seed),
                "-ls",
                str(self.random_seed),
                "-ns",
                "1",
                "-lb",
                "3",
            ]
        )
        self.assertEqual(output.returncode, 0)

        time.sleep(10)

        subprocess.run(["rm", "-r", f"experiments/atari/figures/{self.experiment_name}/{self.game}/iIQN"])
        self.assertEqual(output.returncode, 0)

        subprocess.run(
            [
                "mv",
                f"out/atari/{self.experiment_name}/{self.game}/3_train_iiqn_1{str(self.random_seed)}.out",
                f"out/atari/{self.experiment_name}/{self.game}/3_train_iiqn.out",
            ]
        )
        self.assertEqual(output.returncode, 0)

    def test_irem_pipeline(self) -> None:
        output = subprocess.run(
            [
                "launch_job/atari/launch_local_irem.sh",
                "-e",
                f"{self.experiment_name}/{self.game}",
                "-fs",
                str(self.random_seed),
                "-ls",
                str(self.random_seed),
                "-ns",
                "1",
                "-lb",
                "3",
            ]
        )
        self.assertEqual(output.returncode, 0)

        time.sleep(10)

        subprocess.run(["rm", "-r", f"experiments/atari/figures/{self.experiment_name}/{self.game}/iREM"])
        self.assertEqual(output.returncode, 0)

        subprocess.run(
            [
                "mv",
                f"out/atari/{self.experiment_name}/{self.game}/3_train_irem_1{str(self.random_seed)}.out",
                f"out/atari/{self.experiment_name}/{self.game}/3_train_irem.out",
            ]
        )
        self.assertEqual(output.returncode, 0)
