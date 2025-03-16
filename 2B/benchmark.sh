#!/bin/bash

number_of_nodes=(1 2 3 4 5 6 7 8 9 10)

for N in "${number_of_nodes}"; do
        echo "Running on $N nodes"
        time srun -N $N -n 2 --exclusive --ntasks-per-node=1 -t 00:01:00 ./point_to_point > output_N${N}.txt 2> error_N${N}.txt
done
