#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

IFS="/" read -ra split_experiment_name <<< $EXPERIMENT_NAME
EXPERIMENT_GENERAL_NAME=${split_experiment_name[0]}

[ -d out/atari/$EXPERIMENT_NAME ] || mkdir -p out/atari/$EXPERIMENT_NAME

[ -d experiments/atari/figures/$EXPERIMENT_NAME ] || mkdir -p experiments/atari/figures/$EXPERIMENT_NAME
[ -f experiments/atari/figures/$EXPERIMENT_GENERAL_NAME/parameters.json ] || cp experiments/atari/parameters.json experiments/atari/figures/$EXPERIMENT_GENERAL_NAME/parameters.json
[ -d experiments/atari/figures/$EXPERIMENT_NAME/iDQN ] || mkdir experiments/atari/figures/$EXPERIMENT_NAME/iDQN


seed_command="export SLURM_ARRAY_TASK_ID=$FIRST_SEED"
train_command="launch_job/atari/train_idqn.sh -e $EXPERIMENT_NAME -b ${LIST_BELLMAN_ITERATIONS_SCOPE[0]} -ns $N_PARALLEL_SEEDS"

# iDQN
echo "launch train idqn"
if [[ $USE_DOCKER ]]
then
    launch_job/docker_launcher.sh "$seed_command && $train_command"
else
    tmux send-keys -t train "$seed_command" ENTER "$train_command" ENTER
fi