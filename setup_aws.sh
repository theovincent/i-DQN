# git clone git@github.com:theovincent/iDQN.git
# cd iDQN
# git checkout dopamine_replay
docker pull theovincent/idqn
docker tag theovincent/idqn idqn
docker run -it --gpus all --mount type=bind,src=/home/$USER/iDQN/,dst=/home/$USER/iDQN/ idqn bash -c "
    cd /home/ubuntu/iDQN/ &&
    python3.10 -m venv env &&
    source env/bin/activate &&
    pip install --upgrade pip &&
    pip install --upgrade "jax[cuda12_pip]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html &&
    pip install -e ."