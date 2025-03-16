#!/bin/bash

#!/bin/bash

intervals=(100000 1000000 10000000 20000000 30000000)
thread_counts=(1 2 4 8 16)
process_counts=(1 2 4 8 16)

output_file1="benchmark_for.csv"
output_file2="benchmark_task.csv"
output_file3="benchmark_mpi.csv"
echo "Intervals, Threads, Error, Elapsed Time" > $output_file1
echo "Intervals, Threads, Error, Elapsed Time" > $output_file2
echo "Intervals, Processes, Error, Elapsed Time" > $output_file3

# OpenMP For benchmarks
for threads in "${thread_counts[@]}"
do
    for interval in "${intervals[@]}"
    do
        export OMP_NUM_THREADS=$threads
        output=$(./numint_for $interval)
        error=$(echo "$output" | grep 'Error'| awk '{print $2}')
        elapsed_time=$(echo "$output" | grep 'Elapsed time'| awk '{print $3}')
        echo "$interval, $threads, $error, $elapsed_time" >> "$output_file1"
    done
done

# OpenMP Task benchmarks
for threads in "${thread_counts[@]}"
do
    for interval in "${intervals[@]}"
    do
        export OMP_NUM_THREADS=$threads
        output=$(./numint_task $interval)
        error=$(echo "$output" | grep 'Error'| awk '{print $2}')
        elapsed_time=$(echo "$output" | grep 'Elapsed time'| awk '{print $3}')
        echo "$interval, $threads, $error, $elapsed_time" >> "$output_file2"
    done
done

# MPI benchmarks
for procs in "${process_counts[@]}"
do
    for interval in "${intervals[@]}"
    do
        output=$(srun -n $procs ./numint_mpi $interval)
        error=$(echo "$output" | grep 'Error'| awk '{print $2}')
        elapsed_time=$(echo "$output" | grep 'Elapsed time'| awk '{print $3}')
        echo "$interval, $procs, $error, $elapsed_time" >> "$output_file3"
    done
done
