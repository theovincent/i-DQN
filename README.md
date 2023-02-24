# iDQN
Iterated Deep-Q Networks


You will need to install _swig_ to be able to install the library _box2d-py_ with the command line:
```bash
sudo apt-get install swig
```
or with conda
```Bash
conda install -c conda-forge swig
```

For GPU usage:
```Bash
pip install -U jax[cuda11_cudnn82]==0.4.2 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

conda install -c conda-forge cudatoolkit-dev=11.6
conda install -c conda-forge cudnn=8.4
Set export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5


Atari implementation:
The wrapped environment is build on gymnasium with no frame kipping, with 25% of probability that the previous action is played instead of the current one and with a reduced subset of actions. 
One step of the wrapped environment is composed of:
- 4 steps of the gymnasium environment.
- Max pooling over those 4 frames.
- Converting to a grayscale image with OpenCV.
- Downscaling to 84 x 84 with OpenCV using linear interpolation.
- Outputing the resulting frame along with the resulting frames of the 3 last steps. 

Each episode each started with a random number of no-op actions that can go up to 30 single steps. Each eposide ends when a life is lost. The reward of the wrapped environment is the sign of the reward from gymnasium. 