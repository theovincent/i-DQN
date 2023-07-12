# Iterated Deep Q-Network: Efficient Learning of Bellman Iterations for Deep Reinforcement Learning

## User installation
We recommend using Python 3.9|3.10.
A GPU is needed to run the experiments. In the folder where the code is, create a Python virtual environment, activate it, updae pip and install the package and its dependencies in editable mode:
```bash
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -e .
```

## Run the experiments
All the Atari games can be ran the same way by simply replacing the Atari game name, here is an example for Asteroids.

The following command line runs the training in a tmux terminal:
```Bash
tmux new -s train -d
launch_job/atari/launch_local_idqn.sh -e first_try/Asteroids -lb 5 -fs 1 -ls 1 -ns 1
```
The expected time to finish the run is around 60 hours.

To monitor the current state of the training, you can have a look to the logs at:
```Bash
cat out/atari/first_try/Asteroids/5_train_idqn_11.out
```

At any time during the training, you can generate the figures shown in the paper by running the jupyter notebook file located at *experiments/atari/plots.ipynb*. In the first cell of the notebook, please make sure to change the entries according to what you have been running. You can also have a look at the loss of the training thought the jupyter notebook under *experiments/atari/plots_loss.ipynb*.

## Run the tests
Run all tests with
```Bash
pytest
```
The tests should take around 1 minute to run.

## Baseline scores
Get the google bucket provided in https://github.com/google-research/rliable to have the scores of the baselines. For that you might need to install the google cloud SDK https://cloud.google.com/sdk/docs/downloads-interactive?hl=en#linux-mac and run:
```bash
gsutil -m cp -R gs://rl-benchmark-data/ALE experiments/atari/baselines_scores/
```
The file *atari_200_iters_scores.npy* is the one used to plot the figures. Please bring this file to the *experiments/atari/baselines_scores/* folder:
```bash
cp experiments/atari/baselines_scores/ALE/atari_200_iters_scores.npy experiments/atari/baselines_scores/
```

vThe wrapped environment is build on Gymnasium with no frame kipping, with 25% of probability that the previous action is played instead of the current one and with a reduced subset of actions. 
One step of the wrapped environment is composed of:
- 4 steps of the gymnasium environment.
- Max pooling over the 2 last greyscale frames.
- Converting to a greyscale image with OpenCV.
- Downscaling to 84 x 84 with OpenCV using linear interpolation.
- Outputting the resulting frame along with the resulting frames of the 3 last steps. 

Each episode ends when the _game over_ signal is sent.

## Potential issues
If JAX cannot access the GPU, we recomment using docker. A [Dockerfile](Dockerfile) has been developped for that purpose.

Restraining the GPU memory pre allocation by setting ```XLA_PYTHON_CLIENT_MEM_FRACTION``` to ```0.4``` in line 15 of file *launch_job/atari/train_idqn.sh* might solve the issue as well.

Now you can go back to the [user installation](#user-installation) guidelines.