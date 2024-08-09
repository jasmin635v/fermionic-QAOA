#!/bin/bash

#SBATCH --account=def-ko1
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --array=0-11

# Define your script parameters
N_SAMPLES=400
N_LAYERS=3
N_QUBITS=4
MAX_UNLABELED_GRAPHS=6

echo "slurm array script to run jobs"
echo "NQbits" $N_QUBITS "sample" $N_SAMPLES "layers" $N_LAYERS "maxunlbl graphs" $MAX_UNLABELED_GRAPHS

# Load necessary modules and activate environment
echo "---"
echo " module load  "
echo "---"

module load python/3.12

echo "---"
echo " virtual env --nodownload maxcut "
echo "--- "

virtualenv --no-download maxcut

echo "---"
echo " source maxcut/bin/activate "
echo "---"

source maxcut/bin/activate


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "---"
echo "Running QAOA_main.py script"
echo "---"


START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
START_TIME_FILE=$(date +"%H%M%S")
OUTPUT_FILE="SlurmArray-Vertice-${N_QUBITS}_Layers-${N_LAYERS}_MaxUnlabeledGraphs-${MAX_UNLABELED_GRAPHS}_${START_TIME_FILE}.txt"


# Execute job list from /stored_jobs corresponding to parameters and save every result to individual files in stored_job_results
python QAOA_main.py --n_samples $N_SAMPLES --n_layers $N_LAYERS --n_qubits $N_QUBITS --max_unlbl_graph $MAX_UNLABELED_GRAPHS --task_id $SLURM_ARRAY_TASK_ID --job_name "job_execute_slurmarray"


# Record end time and calculate elapsed time
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
ELAPSED_TIME=$(date -u -d @$(($(date -d "$END_TIME" +%s) - $(date -d "$START_TIME" +%s))) +"%H:%M:%S")

# Append elapsed time to output file
echo "Elapsed time: (from sh) $ELAPSED_TIME" >> "$OUTPUT_FILE"
