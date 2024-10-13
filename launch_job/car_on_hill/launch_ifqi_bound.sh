#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

IFS="/" read -ra split_experiment_name <<< $EXPERIMENT_NAME
EXPERIMENT_GENERAL_NAME=${split_experiment_name[0]}

[ -d out/car_on_hill/$EXPERIMENT_NAME ] || mkdir -p out/car_on_hill/$EXPERIMENT_NAME

[ -d experiments/car_on_hill/figures/$EXPERIMENT_NAME ] || mkdir -p experiments/car_on_hill/figures/$EXPERIMENT_NAME
[ -f experiments/car_on_hill/figures/$EXPERIMENT_GENERAL_NAME/parameters.json ] || cp experiments/car_on_hill/parameters.json experiments/car_on_hill/figures/$EXPERIMENT_GENERAL_NAME/parameters.json
[ -d experiments/car_on_hill/figures/$EXPERIMENT_NAME/iFQI_bound ] || mkdir experiments/car_on_hill/figures/$EXPERIMENT_NAME/iFQI_bound


for bellman_iterations_scope in "${LIST_BELLMAN_ITERATIONS_SCOPE[@]}"
do
    # iFQI_bound
    echo "launch train ifqi bound"
    # --gres=gpu:1 -p gpu
    submission_train_ifqi_bound_1=$(sbatch -J $EXPERIMENT_NAME\_$bellman_iterations_scope --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=3 --mem-per-cpu=500M --time=5:00:00 -p amd,amd2,amd3 --output=/dev/null launch_job/car_on_hill/train_ifqi_bound.sh -e $EXPERIMENT_NAME -b $bellman_iterations_scope -ns $N_PARALLEL_SEEDS)
done