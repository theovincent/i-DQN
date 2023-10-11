#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

IFS="/" read -ra split_experiment_name <<< $EXPERIMENT_NAME
EXPERIMENT_GENERAL_NAME=${split_experiment_name[0]}

[ -d out/atari/$EXPERIMENT_NAME ] || mkdir -p out/atari/$EXPERIMENT_NAME

[ -d experiments/atari/figures/$EXPERIMENT_NAME ] || mkdir -p experiments/atari/figures/$EXPERIMENT_NAME
[ -f experiments/atari/figures/$EXPERIMENT_GENERAL_NAME/parameters.json ] || cp experiments/atari/parameters.json experiments/atari/figures/$EXPERIMENT_GENERAL_NAME/parameters.json
[ -d experiments/atari/figures/$EXPERIMENT_NAME/IQN ] || mkdir experiments/atari/figures/$EXPERIMENT_NAME/IQN


# IQN
echo "launch train iqn"
submission_train_iqn_1=$(sbatch -J $EXPERIMENT_NAME --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=$(( 4 * $N_PARALLEL_SEEDS )) --mem-per-cpu=3G --time=3-00:00:00 --gres=gpu:1 -p gpu --output=/dev/null launch_job/atari/train_iqn.sh -e $EXPERIMENT_NAME -ns $N_PARALLEL_SEEDS)