#!/bin/bash

#SBATCH --account= 
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --array=0-23

# Define your script parameters
echo "slurm array script to run jobs from job name"

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
OUTPUT_FILE="SlurmArray-Job-ExecuteJobName-${SLURM_ARRAY_TASK_ID}.txt"


# Execute job list from /stored_jobs corresponding to parameters and save every result to individual files in stored_job_results
python QAOA_main.py --task_id $SLURM_ARRAY_TASK_ID --job_name "job_execute_slurm_array_from_jobname" --stored_job_name "job-sequence-8l" > "$OUTPUT_FILE" 2>&1


# Record end time and calculate elapsed time
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
ELAPSED_TIME=$(date -u -d @$(($(date -d "$END_TIME" +%s) - $(date -d "$START_TIME" +%s))) +"%H:%M:%S")

# Append elapsed time to output file
echo "Elapsed time: (from sh) $ELAPSED_TIME" >> "$OUTPUT_FILE"
