#!/bin/bash

#SBATCH --account= #ADD ACCOUNT
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

module load python/3.12
virtualenv --no-download maxcut
source maxcut/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
START_TIME_FILE=$(date +"%H%M%S")
OUTPUT_FILE="MergeJob_${START_TIME_FILE}.txt"

# Get results from /stored_job_results corresponding to jobs in job list in /stored_jobs corresponding to parameters
# Merge results, process and save to txt in /merged_processed_results
python QAOA_main.py \
    --stored_job_name "job-sequence" \
    --job_name "job_process_results_from_jobname"

# Record end time and calculate elapsed time
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
ELAPSED_TIME=$(date -u -d @$(($(date -d "$END_TIME" +%s) - $(date -d "$START_TIME" +%s))) +"%H:%M:%S")

# Append elapsed time to output file
echo "Elapsed time: (from sh) $ELAPSED_TIME" >> "$OUTPUT_FILE"
