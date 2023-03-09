#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

if [[ $GPU = true ]]
then
    source env_gpu/bin/activate
else
    source env_cpu/bin/activate
fi 

if [[ $RESTART_TRAINING = "-r" ]]
then
    training="last"
else
    training="first"
fi

if [[ $N_PARALLEL_SEEDS = 1 ]]
then
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.89
elif [[ $N_PARALLEL_SEEDS = 2 ]]
then
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.44
else
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.275
fi

parallel_launcher -c "atari_idqn -e $EXPERIMENT_NAME -b $BELLMAN_ITERATIONS_SCOPE $RESTART_TRAINING" -s $SLURM_ARRAY_TASK_ID -ns $N_PARALLEL_SEEDS -o out/atari/$EXPERIMENT_NAME/$BELLMAN_ITERATIONS_SCOPE\_train_idqn_$training