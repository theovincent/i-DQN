FROM nvidia/cuda:12.0.0-cudnn8-devel-ubuntu20.04

RUN apt update
# tzdata asks questions which we want to avoid.
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata
RUN apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa   
RUN apt update
RUN apt upgrade -y
RUN apt install -y python3.10-dev python3.10-venv
# For opencv
RUN apt-get install -y ffmpeg libsm6 libxext6

# docker build -t idqn . 
# docker run -it --rm --mount type=bind,src=/home/$USER/iDQN/,dst=/home/$USER/iDQN/ idqn
# srun --gres gpu --job-name "requirements" --cpus-per-task 4 --mem-per-cpu 4000 --time 24:00:00 --pty bash
# On a cloud compute provider
# docker run -it --gpus all --mount type=bind,src=/home/$USER/iDQN/,dst=/home/$USER/iDQN/ idqn