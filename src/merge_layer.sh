#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

# Define your script parameters
N_SAMPLES=400
N_LAYERS="3 4 5"
N_QUBITS=4
MAX_UNLABELED_GRAPHS=6

echo "result merge script"

# Load necessary modules and activate environment
echo " --- "
echo " module load  "
echo " "

module load python/3.12

echo " ---"
echo " virtual env --nodownload maxcut "
echo " "

virtualenv --no-download maxcut

echo " --- "
echo " source maxcut/bin/activate "
echo " "

source maxcut/bin/activate


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo " "
echo "Running main script -merge layer"


START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
START_TIME_FILE=$(date +"%H%M%S")
OUTPUT_FILE="MergeLayer-${N_QUBITS}_Layers-${N_LAYERS}_IsoNum-${MAX_ISOMORPH_NUMBER}_MaxJob-${MAX_JOB}_${START_TIME_FILE}.txt"

# Get results from /stored_job_results corresponding to jobs in job list in /stored_jobs corresponding to parameters
# Merge results, process and save to txt in /merged_processed_results
python QAOA_main.py \
    --n_samples $N_SAMPLES \
    --n_layers $N_LAYERS \
    --n_qubits $N_QUBITS \
    --max_unlbl_graph $MAX_UNLABELED_GRAPHS \
    --job_name "job_process_results_layers"


# Record end time and calculate elapsed time
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
ELAPSED_TIME=$(date -u -d @$(($(date -d "$END_TIME" +%s) - $(date -d "$START_TIME" +%s))) +"%H:%M:%S")

# Append elapsed time to output file
echo "Elapsed time: (from sh) $ELAPSED_TIME" >> "$OUTPUT_FILE"
