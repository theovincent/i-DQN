#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

[ -d experiments/car_on_hill/figures/$EXPERIMENT_NAME ] || mkdir -p experiments/car_on_hill/figures/$EXPERIMENT_NAME
[ -f experiments/car_on_hill/figures/$EXPERIMENT_NAME/parameters.json ] || cp experiments/car_on_hill/parameters.json experiments/car_on_hill/figures/$EXPERIMENT_NAME/parameters.json
[ -d experiments/car_on_hill/figures/$EXPERIMENT_NAME/iFQI ] || mkdir experiments/car_on_hill/figures/$EXPERIMENT_NAME/iFQI


# Collect data
echo "launch collect sample"
car_on_hill_sample -e $EXPERIMENT_NAME

for (( seed=$FIRST_SEED; seed<=$LAST_SEED; seed++ ))
do
    for bellman_iterations_scope in "${LIST_BELLMAN_ITERATIONS_SCOPE[@]}"
    do
        # iFQI
        echo "launch train ifqi"
        car_on_hill_ifqi -e $EXPERIMENT_NAME -b $bellman_iterations_scope -s $seed

        echo "launch evaluate ifqi"
        car_on_hill_ifqi_evaluate -e $EXPERIMENT_NAME -b $bellman_iterations_scope -s $seed
    done
done