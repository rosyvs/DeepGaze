#!/bin/bash
cd ../
start_seed=$3
for ((i=0;i<$1;i++)); do
    seed=$(($start_seed + $i))
    echo sbatch $2 $i $seed
done