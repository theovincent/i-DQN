#!/bin/bash

for GAME in Amidar Boxing Freeway Kangaroo MsPacman Pong SpaceInvaders Tennis
do
    launch_job/atari/launch_iiqn.sh --experiment_name iIQN/$GAME --first_seed 1 --last_seed 2 \
        --list_bellman_iterations_scope 3 --n_parallel_seeds 2 
    launch_job/atari/launch_iiqn.sh --experiment_name iIQN/$GAME --first_seed 3 --last_seed 3 \
        --list_bellman_iterations_scope 3 --n_parallel_seeds 1
done