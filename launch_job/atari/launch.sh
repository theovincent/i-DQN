#!/bin/bash

# ---- The weigths and performances will be logged in experiments/atari/figures/hackatari/$GAME/iIQN/ ---- #
# 3_J_11.npy contains the list of the performances: average return for each epoch.
# L_11.npy is the loss for each gradient step.
# Q_11_$N_BEST_EPOCH\_best_online_params in the weights of the best online network seen so far.

# The index of the 5 seeds is 11, 21, 12, 22, 13.

for GAME in Amidar Boxing Freeway Kangaroo MsPacman Pong SpaceInvaders Tennis
do
    launch_job/atari/launch_iiqn.sh --experiment_name hackatari/$GAME --first_seed 1 --last_seed 2 \
        --list_bellman_iterations_scope 3 --n_parallel_seeds 2 
    launch_job/atari/launch_iiqn.sh --experiment_name hackatari/$GAME --first_seed 3 --last_seed 3 \
        --list_bellman_iterations_scope 3 --n_parallel_seeds 1
done