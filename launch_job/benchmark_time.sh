#!/bin/bash

launch_job/atari/launch_local_dqn.sh -e benchmark_time/Breakout -fs 1 -ls 1 -ns 1
launch_job/atari/launch_local_idqn.sh -e benchmark_time/Breakout -fs 1 -ls 1 -ns 1 -lb 2
launch_job/atari/launch_local_idqn.sh -e benchmark_time/Breakout -fs 1 -ls 1 -ns 1 -lb 5 
launch_job/atari/launch_local_idqn.sh -e benchmark_time/Breakout -fs 1 -ls 1 -ns 1 -lb 10

launch_job/atari/launch_local_iqn.sh -e benchmark_time/Breakout -fs 1 -ls 1 -ns 1
launch_job/atari/launch_local_iiqn.sh -e benchmark_time/Breakout -fs 1 -ls 1 -ns 1 -lb 2
launch_job/atari/launch_local_iiqn.sh -e benchmark_time/Breakout -fs 1 -ls 1 -ns 1 -lb 5 
launch_job/atari/launch_local_iiqn.sh -e benchmark_time/Breakout -fs 1 -ls 1 -ns 1 -lb 10