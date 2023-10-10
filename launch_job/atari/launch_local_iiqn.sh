#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

IFS="/" read -ra split_experiment_name <<< $EXPERIMENT_NAME
EXPERIMENT_GENERAL_NAME=${split_experiment_name[0]}

[ -d out/atari/$EXPERIMENT_NAME ] || mkdir -p out/atari/$EXPERIMENT_NAME

[ -d experiments/atari/figures/$EXPERIMENT_NAME ] || mkdir -p experiments/atari/figures/$EXPERIMENT_NAME
[ -f experiments/atari/figures/$EXPERIMENT_GENERAL_NAME/parameters.json ] || cp experiments/atari/parameters.json experiments/atari/figures/$EXPERIMENT_GENERAL_NAME/parameters.json
[ -d experiments/atari/figures/$EXPERIMENT_NAME/iIQN ] || mkdir experiments/atari/figures/$EXPERIMENT_NAME/iIQN


seed_command="export SLURM_ARRAY_TASK_ID=$FIRST_SEED"

# iIQN
echo "launch train iiqn"
train_command="launch_job/atari/train_iiqn.sh -e $EXPERIMENT_NAME -b ${LIST_BELLMAN_ITERATIONS_SCOPE[0]} -ns $N_PARALLEL_SEEDS"
tmux_command='tmux send-keys -t train "$seed_command" ENTER "$train_command" ENTER'
if [[ $USE_DOCKER ]]
then
    launch_job/docker_launcher.sh $tmux_command
else
    $tmux_command
fi