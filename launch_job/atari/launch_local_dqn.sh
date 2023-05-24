#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

IFS="/" read -ra split_experiment_name <<< $EXPERIMENT_NAME
EXPERIMENT_GENERAL_NAME=${split_experiment_name[0]}

[ -d out/atari/$EXPERIMENT_NAME ] || mkdir -p out/atari/$EXPERIMENT_NAME

[ -d experiments/atari/figures/$EXPERIMENT_NAME ] || mkdir -p experiments/atari/figures/$EXPERIMENT_NAME
[ -f experiments/atari/figures/$EXPERIMENT_GENERAL_NAME/parameters.json ] || cp experiments/atari/parameters.json experiments/atari/figures/$EXPERIMENT_GENERAL_NAME/parameters.json
[ -d experiments/atari/figures/$EXPERIMENT_NAME/DQN ] || mkdir experiments/atari/figures/$EXPERIMENT_NAME/DQN


for (( seed=$FIRST_SEED; seed<=$LAST_SEED; seed++ ))
do
    seed_command="export SLURM_ARRAY_TASK_ID=$seed"

    # DQN
    echo "launch train idqn"
    train_command="launch_job/atari/train_dqn.sh -e $EXPERIMENT_NAME -g -ns $N_PARALLEL_SEEDS"
    tmux send-keys -t train "$seed_command" ENTER "$train_command" ENTER
done