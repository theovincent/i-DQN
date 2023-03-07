#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

source env_cpu/bin/activate

if [[ $RESTART_TRAINING = true ]]
then
    training="last"
else
    training="first"
fi

parallel_launcher -c "atari_idqn_evaluate -e $EXPERIMENT_NAME -b $BELLMAN_ITERATIONS_SCOPE $RESTART_TRAINING" -s $SLURM_ARRAY_TASK_ID -ns $N_PARALLEL_SEEDS -o out/atari/$EXPERIMENT_NAME/$BELLMAN_ITERATIONS_SCOPE\_train_idqn_$training
