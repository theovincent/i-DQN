#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

[ -d out/lunar_lander/$EXPERIMENT_NAME ] || mkdir -p out/lunar_lander/$EXPERIMENT_NAME

[ -d experiments/lunar_lander/figures/$EXPERIMENT_NAME ] || mkdir -p experiments/lunar_lander/figures/$EXPERIMENT_NAME
[ -f experiments/lunar_lander/figures/$EXPERIMENT_NAME/parameters.json ] || cp experiments/lunar_lander/parameters.json experiments/lunar_lander/figures/$EXPERIMENT_NAME/parameters.json
[ -d experiments/lunar_lander/figures/$EXPERIMENT_NAME/iDQN ] || mkdir experiments/lunar_lander/figures/$EXPERIMENT_NAME/iDQN


for bellman_iterations_scope in "${LIST_BELLMAN_ITERATIONS_SCOPE[@]}"
do
    # iDQN
    echo "launch train idqn"
    submission_train_idqn=$(sbatch -J L_train_idqn --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=3 --mem-per-cpu=1Gc --time=15:30:00 --output=out/lunar_lander/$EXPERIMENT_NAME/$bellman_iterations_scope\_train_idqn_%a.out --gres=gpu:1 -p amd,amd2,rtx,rtx2,dgx launch_job/lunar_lander/train_idqn.sh -e $EXPERIMENT_NAME -b $bellman_iterations_scope -g)

    IFS=" " read -ra split_submission_train_idqn <<< $submission_train_idqn
    submission_id_train_idqn=${split_submission_train_idqn[-1]}

    echo "launch evaluate idqn"
    submission_evaluate_idqn=$(sbatch -J L_evaluate_idqn --dependency=after:$submission_id_train_idqn --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=1 --mem-per-cpu=2Gc --time=15:30:00 --output=out/lunar_lander/$EXPERIMENT_NAME/$bellman_iterations_scope\_evaluate_idqn_%a.out -p amd,amd2 launch_job/lunar_lander/evaluate_idqn.sh -e $EXPERIMENT_NAME -b $bellman_iterations_scope)
done