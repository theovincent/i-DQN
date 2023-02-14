#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

[ -d experiments/lunar_lander/figures/$EXPERIMENT_NAME ] || mkdir -p experiments/lunar_lander/figures/$EXPERIMENT_NAME
[ -f experiments/lunar_lander/figures/$EXPERIMENT_NAME/parameters.json ] || cp experiments/lunar_lander/parameters.json experiments/lunar_lander/figures/$EXPERIMENT_NAME/parameters.json
[ -d experiments/lunar_lander/figures/$EXPERIMENT_NAME/iDQN ] || mkdir experiments/lunar_lander/figures/$EXPERIMENT_NAME/iDQN


for (( seed=$FIRST_SEED; seed<=$LAST_SEED; seed++ ))
do
    for bellman_iterations_scope in "${LIST_BELLMAN_ITERATIONS_SCOPE[@]}"
    do
        # iDQN
        echo "launch train idqn"
        lunar_lander_idqn -e $EXPERIMENT_NAME -b $bellman_iterations_scope -s $seed

        echo "launch evaluate idqn"
        lunar_lander_idqn_evaluate -e $EXPERIMENT_NAME -b $bellman_iterations_scope -s $seed
    done
done