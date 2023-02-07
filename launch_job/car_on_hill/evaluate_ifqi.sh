#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

source env_cpu/bin/activate 

car_on_hill_ifqi_evaluate -e $EXPERIMENT_NAME -s $SLURM_ARRAY_TASK_ID -b $BELLMAN_ITERATIONS_SCOPE