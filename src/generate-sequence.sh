
#!/bin/bash

#SBATCH --account=  #ADD YOUR ACCOUNT
#SBATCH --time=10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G


# Define your script parameters
START_GRAPH="010204050613141523241718" #put any graph in this format. in this case: 8 is the max edge therefore 9 vertices
STORED_JOB_NAME="job-sequence"

module load python/3.12
virtualenv --no-download maxcut
source maxcut/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
START_TIME_FILE=$(date +"%H%M%S")
OUTPUT_FILE="GraphSequence_StartGraph-${START_GRAPH}_StoredJobName_${STORED_JOB_NAME}.txt"

# Generate the jobs list in  /stored_jobs
python QAOA_main.py \
    --start_graph $START_GRAPH \
    --stored_job_name $STORED_JOB_NAME \
    --job_name "job_generate_graphs_sequence" > "$OUTPUT_FILE" 2>&1


# Record end time and calculate elapsed time
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
ELAPSED_TIME=$(date -u -d @$(($(date -d "$END_TIME" +%s) - $(date -d "$START_TIME" +%s))) +"%H:%M:%S")

# Append elapsed time to output file
echo "Elapsed time: (from sh) $ELAPSED_TIME" >> "$OUTPUT_FILE"
