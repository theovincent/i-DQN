#!/bin/bash

docker run -it --gpus all --rm --mount type=bind,src=/home/$USER/iDQN/,dst=/home/$USER/iDQN/ idqn bash -c cd /home/$USER/iDQN && $@