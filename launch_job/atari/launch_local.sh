#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

IFS=" " read -ra split_experiment_name <<< $EXPERIMENT_NAME
EXPERIMENT_GENERAL_NAME=${split_experiment_name[0]}

[ -d experiments/atari/figures/$EXPERIMENT_NAME ] || mkdir -p experiments/atari/figures/$EXPERIMENT_NAME
[ -f experiments/atari/figures/$EXPERIMENT_GENERAL_NAME/parameters.json ] || cp experiments/atari/parameters.json experiments/atari/figures/$EXPERIMENT_GENERAL_NAME/parameters.json
[ -d experiments/atari/figures/$EXPERIMENT_NAME/iDQN ] || mkdir experiments/atari/figures/$EXPERIMENT_NAME/iDQN



for (( seed=$FIRST_SEED; seed<=$LAST_SEED; seed++ ))
do
    for bellman_iterations_scope in "${LIST_BELLMAN_ITERATIONS_SCOPE[@]}"
    do
        #Create TMUX
        # iDQN
        echo "launch train idqn"
        atari_idqn -e $EXPERIMENT_NAME -b $bellman_iterations_scope -s $seed
        atari_idqn -e $EXPERIMENT_NAME -b $bellman_iterations_scope -s $seed -r

        #Create another TMUX
        echo "launch evaluate idqn"
        atari_idqn_evaluate -e $EXPERIMENT_NAME -b $bellman_iterations_scope -s $seed
        atari_idqn_evaluate -e $EXPERIMENT_NAME -b $bellman_iterations_scope -s $seed -r
    done
done