#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

source env/bin/activate

parallel_launcher -c "car_on_hill_ifqi -e $EXPERIMENT_NAME -b $BELLMAN_ITERATIONS_SCOPE" -s $SLURM_ARRAY_TASK_ID -ns $N_PARALLEL_SEEDS -o out/car_on_hill/$EXPERIMENT_NAME/$BELLMAN_ITERATIONS_SCOPE\_train_ifqi