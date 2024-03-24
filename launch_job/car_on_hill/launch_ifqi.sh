#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

IFS="/" read -ra split_experiment_name <<< $EXPERIMENT_NAME
EXPERIMENT_GENERAL_NAME=${split_experiment_name[0]}

[ -d out/car_on_hill/$EXPERIMENT_NAME ] || mkdir -p out/car_on_hill/$EXPERIMENT_NAME

[ -d experiments/car_on_hill/figures/$EXPERIMENT_NAME ] || mkdir -p experiments/car_on_hill/figures/$EXPERIMENT_NAME
[ -f experiments/car_on_hill/figures/$EXPERIMENT_GENERAL_NAME/parameters.json ] || cp experiments/car_on_hill/parameters.json experiments/car_on_hill/figures/$EXPERIMENT_GENERAL_NAME/parameters.json
[ -d experiments/car_on_hill/figures/$EXPERIMENT_NAME/iFQI ] || mkdir experiments/car_on_hill/figures/$EXPERIMENT_NAME/iFQI


for bellman_iterations_scope in "${LIST_BELLMAN_ITERATIONS_SCOPE[@]}"
do
    # iFQI
    echo "launch train ifqi"
    submission_train_ifqi_1=$(sbatch -J $EXPERIMENT_NAME --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=3 --mem-per-cpu=500M --time=40:00 --output=/dev/null launch_job/car_on_hill/train_ifqi.sh -e $EXPERIMENT_NAME -b $bellman_iterations_scope -ns $N_PARALLEL_SEEDS)
done