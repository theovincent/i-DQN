#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

if [[ $GPU = true ]]
then
    source env_gpu/bin/activate
else
    source env_cpu/bin/activate
fi 

if [[ $RESTART_TRAINING = true ]]
then
    training="last"
else
    training="first"
fi

export XLA_PYTHON_CLIENT_MEM_FRACTION=$(echo "scale=2 ; 0.9 / $N_PARALLEL_SEEDS" | bc)

parallel_launcher -c "atari_idqn -e $EXPERIMENT_NAME -b $BELLMAN_ITERATIONS_SCOPE $RESTART_TRAINING" -s $SLURM_ARRAY_TASK_ID -ns $N_PARALLEL_SEEDS -o out/atari/$EXPERIMENT_NAME/$BELLMAN_ITERATIONS_SCOPE\_train_idqn_$training