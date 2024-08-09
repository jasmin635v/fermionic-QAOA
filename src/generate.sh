#!/bin/bash

#SBATCH --account=def-ko1    ##INSERT YOUR ACCCOUNT
#SBATCH --time=10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M


# Define your script parameters
N_QUBITS=4
MAX_UNLABELED_GRAPHS=6

module load python/3.12
virtualenv --no-download maxcut
source maxcut/bin/activate


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo " "
echo "Running Graph Generation script"


START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
START_TIME_FILE=$(date +"%H%M%S")
OUTPUT_FILE="Generate-Vertice-${N_QUBITS}_IsoNum-${MAX_ISOMORPH_NUMBER}_MaxJob-${MAX_JOB}_${START_TIME_FILE}.txt"


# Generate the jobs list in  /stored_jobs
python QAOA_main.py \
    --n_qubits $N_QUBITS \
    --max_unlbl_graph $MAX_UNLABELED_GRAPHS \  ## other options: --max_isomorph_number --max_job
    --job_name "job_generate_graphs" > "$OUTPUT_FILE" 2>&1


# Record end time and calculate elapsed time
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
ELAPSED_TIME=$(date -u -d @$(($(date -d "$END_TIME" +%s) - $(date -d "$START_TIME" +%s))) +"%H:%M:%S")

# Append elapsed time to output file
echo "Elapsed time: (from sh) $ELAPSED_TIME" >> "$OUTPUT_FILE"

