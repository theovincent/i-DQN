#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

[ -d out/car_on_hill/$EXPERIMENT_NAME ] || mkdir -p out/car_on_hill/$EXPERIMENT_NAME

[ -d experiments/car_on_hill/figures/$EXPERIMENT_NAME ] || mkdir -p experiments/car_on_hill/figures/$EXPERIMENT_NAME
[ -f experiments/car_on_hill/figures/$EXPERIMENT_NAME/parameters.json ] || cp experiments/car_on_hill/parameters.json experiments/car_on_hill/figures/$EXPERIMENT_NAME/parameters.json
[ -d experiments/car_on_hill/figures/$EXPERIMENT_NAME/iFQI ] || mkdir experiments/car_on_hill/figures/$EXPERIMENT_NAME/iFQI


# Collect data
echo "launch collect sample"
submission_collect_sample=$(sbatch -J C_collect_sample --cpus-per-task=3 --mem-per-cpu=500Mc --time=50:00 --output=out/car_on_hill/$EXPERIMENT_NAME/collect_sample.out -p amd,amd2 launch_job/car_on_hill/collect_sample.sh -e $EXPERIMENT_NAME)

IFS=" " read -ra split_submission_collect_sample <<< $submission_collect_sample
submission_id_collect_sample=${split_submission_collect_sample[-1]}


for bellman_iterations_scope in "${LIST_BELLMAN_ITERATIONS_SCOPE[@]}"
do
    # iFQI
    echo "launch train ifqi"
    submission_train_ifqi=$(sbatch -J C_train_ifqi --dependency=afterok:$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=3 --mem-per-cpu=750Mc --time=3:30:00 --output=out/car_on_hill/$EXPERIMENT_NAME/$bellman_iterations_scope\_train_ifqi_%a.out -p amd,amd2,rtx,rtx2 launch_job/car_on_hill/train_ifqi.sh -e $EXPERIMENT_NAME -b $bellman_iterations_scope -g)

    IFS=" " read -ra split_submission_train_ifqi <<< $submission_train_ifqi
    submission_id_train_ifqi=${split_submission_train_ifqi[-1]}

    echo "launch evaluate ifqi"
    submission_evaluate_ifqi=$(sbatch -J C_evaluate_ifqi --dependency=afterok:$submission_id_train_ifqi,$submission_id_collect_sample --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=9 --mem-per-cpu=150Mc --time=30:00 --output=out/car_on_hill/$EXPERIMENT_NAME/$bellman_iterations_scope\_evaluate_ifqi_%a.out -p amd,amd2 launch_job/car_on_hill/evaluate_ifqi.sh -e $EXPERIMENT_NAME -b $bellman_iterations_scope)
done