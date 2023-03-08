#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

IFS="/" read -ra split_experiment_name <<< $EXPERIMENT_NAME
EXPERIMENT_GENERAL_NAME=${split_experiment_name[0]}

[ -d out/atari/$EXPERIMENT_NAME ] || mkdir -p out/atari/$EXPERIMENT_NAME

[ -d experiments/atari/figures/$EXPERIMENT_NAME ] || mkdir -p experiments/atari/figures/$EXPERIMENT_NAME
[ -f experiments/atari/figures/$EXPERIMENT_GENERAL_NAME/parameters.json ] || cp experiments/atari/parameters.json experiments/atari/figures/$EXPERIMENT_GENERAL_NAME/parameters.json
[ -d experiments/atari/figures/$EXPERIMENT_NAME/iDQN ] || mkdir experiments/atari/figures/$EXPERIMENT_NAME/iDQN


for (( seed=$FIRST_SEED; seed<=$LAST_SEED; seed++ ))
do
    seed_command="export SLURM_ARRAY_TASK_ID=$seed"
    for bellman_iterations_scope in "${LIST_BELLMAN_ITERATIONS_SCOPE[@]}"
    do
        # iDQN
        echo "launch train idqn"
        train_command="launch_job/atari/train_idqn.sh -e $EXPERIMENT_NAME -b $bellman_iterations_scope -g -ns $N_PARALLEL_SEEDS"
        tmux send-keys -t train "$seed_command" ENTER "$train_command" ENTER "$train_command -r" ENTER

        echo "launch evaluate idqn"
        evaluate_command="launch_job/atari/evaluate_idqn.sh -e $EXPERIMENT_NAME -b $bellman_iterations_scope -ns $N_PARALLEL_SEEDS"
        tmux send-keys -t evaluate "$seed_command" ENTER "$evaluate_command" ENTER "$evaluate_command -r" ENTER
    done
done