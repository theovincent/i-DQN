#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

source env/bin/activate

if [[ $N_PARALLEL_SEEDS = 1 ]]
then
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.89
elif [[ $N_PARALLEL_SEEDS = 2 ]]
then
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.42
else
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.275
fi

parallel_launcher -c "atari_dqn -e $EXPERIMENT_NAME" -s $SLURM_ARRAY_TASK_ID -ns $N_PARALLEL_SEEDS -o out/atari/$EXPERIMENT_NAME/train_dqn