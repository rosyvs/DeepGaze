#!/bin/bash
cd ../
for ((i=0;i<$1;i++)); do
    echo sbatch $2 $i
done