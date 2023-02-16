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