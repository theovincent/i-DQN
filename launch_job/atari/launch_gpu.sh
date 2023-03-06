#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

IFS=" " read -ra split_experiment_name <<< $EXPERIMENT_NAME
EXPERIMENT_GENERAL_NAME=${split_experiment_name[0]}

[ -d out/atari/$EXPERIMENT_NAME ] || mkdir --gres=gpu:1 -pout/atari/$EXPERIMENT_NAME

[ -d experiments/atari/figures/$EXPERIMENT_NAME ] || mkdir --gres=gpu:1 -pexperiments/atari/figures/$EXPERIMENT_NAME
[ -f experiments/atari/figures/$EXPERIMENT_GENERAL_NAME/parameters.json ] || cp experiments/atari/parameters.json experiments/atari/figures/$EXPERIMENT_GENERAL_NAME/parameters.json
[ -d experiments/atari/figures/$EXPERIMENT_NAME/iDQN ] || mkdir experiments/atari/figures/$EXPERIMENT_NAME/iDQN


for bellman_iterations_scope in "${LIST_BELLMAN_ITERATIONS_SCOPE[@]}"
do
    # iDQN
    echo "launch train idqn 1/2"
    submission_train_idqn_1=$(sbatch -J L_train_idqn_$EXPERIMENT_NAME --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=3 --mem-per-cpu=1Gc --time=15:30:00 --output=out/atari/$EXPERIMENT_NAME/$bellman_iterations_scope\_train_idqn_%a.out --gres=gpu:1 -p amd,amd2,rtx,rtx2,dgx launch_job/atari/train_idqn.sh -e $EXPERIMENT_NAME -b $bellman_iterations_scope -g)

    IFS=" " read -ra split_submission_train_idqn_1 <<< $submission_train_idqn_1
    submission_id_train_idqn_1=${split_submission_train_idqn_1[-1]}

    echo "launch evaluate idqn 1/2"
    submission_evaluate_idqn_1=$(sbatch -J L_evaluate_idqn_$EXPERIMENT_NAME --dependency=after:$submission_id_train_idqn_1 --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=1 --mem-per-cpu=2Gc --time=15:30:00 --output=out/atari/$EXPERIMENT_NAME/$bellman_iterations_scope\_evaluate_idqn_%a.out -p amd,amd2 launch_job/atari/evaluate_idqn.sh -e $EXPERIMENT_NAME -b $bellman_iterations_scope)

    IFS=" " read -ra split_submission_evaluate_idqn_1 <<< $submission_evaluate_idqn_1
    submission_id_evaluate_idqn_1=${split_submission_evaluate_idqn_1[-1]}

    echo "launch train idqn 2/2"
    submission_train_idqn_2=$(sbatch -J L_train_idqn_$EXPERIMENT_NAME --dependency=afterok:$submission_id_train_idqn_1 --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=3 --mem-per-cpu=1Gc --time=15:30:00 --output=out/atari/$EXPERIMENT_NAME/$bellman_iterations_scope\_train_idqn_%a.out --gres=gpu:1 -p amd,amd2,rtx,rtx2,dgx launch_job/atari/train_idqn.sh -e $EXPERIMENT_NAME -b $bellman_iterations_scope -r -g)

    IFS=" " read -ra split_submission_train_idqn_2 <<< $submission_train_idqn_2
    submission_id_train_idqn_2=${split_submission_train_idqn_2[-1]}

    echo "launch evaluate idqn 2/2"
    submission_evaluate_idqn_2=$(sbatch -J L_evaluate_idqn_$EXPERIMENT_NAME --dependency=afterok:$submission_id_train_idqn_1,after:$submission_id_train_idqn_2 --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=1 --mem-per-cpu=2Gc --time=15:30:00 --output=out/atari/$EXPERIMENT_NAME/$bellman_iterations_scope\_evaluate_idqn_%a.out -p amd,amd2 launch_job/atari/evaluate_idqn.sh -e $EXPERIMENT_NAME -b $bellman_iterations_scope -r)
done