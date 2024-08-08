from pennylane import numpy as np
import cmath
import math
import os
import time
import sys
import argparse
import subprocess
import re

matchgateOnes = np.array([[1, 0, 0, 1],
                          [0, 1, 1, 0],
                          [0, 1, 1, 0],
                          [1, 0, 0, 1]])

fSwap = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, -1]])

fRyy = 1 / np.sqrt(2) * np.array([[1, 0, 0, -1],
                                  [0, 1,  -1, 0],
                                  [0, 1, 1, 0],
                                  [1, 0, 0, 1]])


def fRyy(theta1, theta2):

    cos_half_theta1 = np.cos(theta1 / 2)
    sin_half_theta1 = np.sin(theta1 / 2)
    cos_half_theta2 = np.cos(theta2 / 2)
    sin_half_theta2 = np.sin(theta2 / 2)

    return 1 / np.sqrt(2) * np.array([[cos_half_theta1, 0, 0, -sin_half_theta1],
                                      [0, cos_half_theta2,  -sin_half_theta2, 0],
                                      [0, cos_half_theta2, sin_half_theta2, 0],
                                      [sin_half_theta1, 0, 0, cos_half_theta1]])


def parse_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_samples",
        type=int,
        default=400,
        help="the number of times the circuit is sampled after the parameters are found",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        nargs='+',  # Accept one or more integers
        default=3,
        help="The number of mixer / cost layers.",
    )
    parser.add_argument(
        "--n_qubits",
        type=int,
        default=4,
        help="The number of vertices or qbits of the circuit",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=-1,
        help="The number of steps of the layer parameter gradiebt optimizer.", #the gradient optimizer is unused
    )
    parser.add_argument(
        "--max_isomorph_number",
        type=int,
        default=0,
        help="The max number of isomorphic graphs to compute per unlabeled graph.",
    )
    parser.add_argument(
        "--max_job",
        type=int,
        default=-1,
        help="The max number of jobs / graphs to solve (first n taken)).",
    )

    parser.add_argument(
        "--job_name",
        type=str,
        default="nojob",
        help="what job to run",
    )

    parser.add_argument(
        "--stored_job_name",
        type=str,
        default="nojob",
        help="the name under which the job results are stored (used inside the script)",
    )

    parser.add_argument(
        "--task_id",
        type=int,
        default="-1",
        help="the slurm task id handled by slurm",
    )

    parser.add_argument(
        "--max_unlbl_graph",
        type=int,
        default="10",
        help="max unlabeled graphs",
    )

    parser.add_argument(
        "--start_graph",
        type=str,
        default="10",
        help="max unlabeled graphs",
    )



    return parser.parse_args()


def Rzz_matrice(gamma):
    exp_term = cmath.exp(1j * gamma / 2)
    exp_term_m = cmath.exp(-1j * gamma / 2)
    return np.array([[exp_term_m, 0, 0, 0],
                     [0, exp_term, 0, 0],
                     [0, 0, exp_term, 0],
                     [0, 0, 0, exp_term_m]])


def bitstring_to_int(bit_string_sample):
    bit_string = "".join(str(bs) for bs in bit_string_sample)
    return int(bit_string, base=2)


def bitstring_to_objective(bitstring, graph):
    # convert bitstring to a list of 0 and 1
    binary_list = [int(char) for char in bitstring]
    obj = 0
    for edge in graph:
        # objective for the MaxCut problem
        obj += 1 - (binary_list[edge[0]] * binary_list[edge[1]])
    return obj  # *0.5


def int_to_bitstring(num):
    # Convert integer to binary string (without '0b' prefix)
    bit_string = bin(num)[2:]
    return bit_string


def compute_stats(numbers):

    weighted_sum = sum([number[0]*number[1] for number in numbers])
    total_count = sum([number[1] for number in numbers])
    mean = weighted_sum/total_count

    minimum = min([number[0] for number in numbers])
    maximum = max([number[0] for number in numbers])

    # Step 2: Calculate weighted variance
    weighted_variance_sum = sum(
        number[1] * (number[0] - mean)**2 for number in numbers)
    weighted_variance = weighted_variance_sum / total_count

    # Step 3: Calculate weighted standard deviation
    weighted_stddev = math.sqrt(weighted_variance)

    sorted_numbers = sorted(numbers, key=lambda x: x[1], reverse=True)
    most_common_element = sorted_numbers[0][0]
    most_common_count = sorted_numbers[0][1]

    most_common_element_count_ratio = most_common_count / total_count
    # weighted_mean_3 = (sorted_numbers[0][0]*sorted_numbers[0][1] + sorted_numbers[1][0]*sorted_numbers[1][1] +sorted_numbers[2][0]*sorted_numbers[2][1])  / (sorted_numbers[0][1] + sorted_numbers[1][1] +sorted_numbers[2][1])

    return most_common_element, most_common_element_count_ratio, mean, maximum, weighted_stddev


def write_to_progress_file(text, start_time=None, slurm=True, fileName=None):  # UNUSED

    if start_time != None:
        end_time = time.time()
        formatted_time = format_time(int(end_time - start_time))
        text += f" time taken: {formatted_time}. Time: {end_time} "

    # with file_lock:
    #     if slurm:
    #         write_to_slurm_output(text)
    #     else:
    #         with open(filename, 'a') as f:
    #             f.write(f'{text}\n')


def format_time(seconds):
    m, s = divmod(seconds, 60)
    return f'{m:02}:{s:02}'


def write_to_slurm_output(message):  # UNUSED
    # Print to stdout or stderr
    print(message+"\n")
    # Force flush to ensure immediate output
    sys.stdout.flush()

# def plot_table_from_list_of_lists(list_of_lists, column_headers, title_text = ""):
#     footer_text = date.today()
#     fig_background_color = 'white'
#     fig_border = 'steelblue'
#     data =  list_of_lists

#     cell_text = []
#     for row in data:
#         cell_text.append([f'{str(x)}' for x in row])
#     # Get some lists of color specs for row and column headers
#     #rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
#     ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))

#     plt.figure(linewidth=2,
#             edgecolor=fig_border,
#             facecolor=fig_background_color,
#             tight_layout={'pad':1},
#             #figsize=(5,3)
#             )
#     # Add a table at the bottom of the axes
#     the_table = plt.table(cellText=cell_text,
#                         #rowLabels=row_headers,
#                         #rowColours=rcolors,
#                         rowLoc='right',
#                         colColours=ccolors,
#                         colLabels=column_headers,
#                         loc='center')
#     # Scaling is the only influence we have over top and bottom cell padding.
#     # Make the rows taller (i.e., make cell y scale larger).
#     the_table.scale(1, 1.5)
#     # Hide axes
#     ax = plt.gca()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     # Hide axes border
#     plt.box(on=None)
#     # Add title
#     plt.suptitle(title_text)
#     # Add footer
#     plt.figtext(0.95, 0.05, footer_text, horizontalalignment='right', size=6, weight='light')
#     # Force the figure to update, so backends center objects correctly within the figure.
#     # Without plt.draw() here, the title will center on the axes and not the figure.
#     plt.draw()
#     # Create image. plt.savefig ignores figure edge and face colors, so map them.
#     fig = plt.gcf()
#     plt.savefig('pyplot-table-demo.png',
#                 #bbox='tight',
#                 edgecolor=fig.get_edgecolor(),
#                 facecolor=fig.get_facecolor(),
#                 dpi=150
#                 )

def format_job_name_from_result(job_result):
    graph_string = graph_to_string(job_result[2])
    return f"{job_result[0]}_{job_result[1]}_{graph_string}.npy"

def graph_to_string(graph):
    return "_"+str(graph).replace('_','').replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace(' ', '').replace(',', '')

def string_graph_to_graph(string_graph):
    # Step 1: Extract digit pairs (ignore the leading underscore)
    pairs = [string_graph[i:i+2] for i in range(1, len(string_graph), 2)]
    
    # Step 2: Convert pairs to tuples of integers
    tuple_pairs = [tuple(map(int, pair)) for pair in pairs]
    
    return tuple_pairs

def vertice_from_graph(graph):
    str_graph = graph_to_string(graph)
    graph = string_graph_to_graph(str_graph)
    max_vertice = max(max(edge) for edge in graph)
    return max_vertice + 1

def param_to_string(graph):
    return str(graph).replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace(' ', '_').replace(',', '')


def graph_from_label_string(label):
    last_underscore_pos = label.rfind('_')
    # Extract and return the substring starting from the last underscore
    return label[last_underscore_pos:]


def load_numpy_arrays_to_list(filenames):
    # Assuming format_job_name_from_result(QAOA_result) returns unique names for each result
    # and QAOA_result is a list containing arrays
    # Example filenames (replace with your actual filenames)

    # Initialize a list to store all loaded lists
    all_results = []

    # Load each array from the files
    for filename in filenames:
        # Load the numpy array
        loaded_array = np.load(filename)

        # Convert to list if necessary
        if isinstance(loaded_array, np.ndarray):
            loaded_array = loaded_array.tolist()

        # Append to the master list
        all_results.append(loaded_array)

    # Now all_results is a list of lists containing your original lists from QAOA_result
    return all_results


def remove_npy_files(filenames):
    for filename in filenames:
        file_path = os.path.join(os.getcwd(), filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted: {filename}")
        else:
            print(f"{filename} does not exist.")

def return_slurm_array_test_script_string(account= "def-ko1"):
    script_string = f"""#!/bin/bash
    #SBATCH --account={account}
    #SBATCH --time=10:00
    #SBATCH --mem=512M
    #SBATCH --cpus-per-task=1

    N_QUBITS=3
    MAX_UNLABELED_GRAPHS=1

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
    OUTPUT_FILE="SlurmTEST-${{N_QUBITS}}__MaUnlabeledGraphs-${{MAX_UNLABELED_GRAPHS}}_${{START_TIME_FILE}}.txt"

    # Pass parameters as environment variables
    # Execute the main script
    python QAOA_main.py --n_qubits $N_QUBITS --max_unlbl_graph $MAX_UNLABELED_GRAPHS --job_name "job_generate_graphs" > "$OUTPUT_FILE" 2>&1

    # Record end time and calculate elapsed time
    END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
    ELAPSED_TIME=$(date -u -d @$(($(date -d "$END_TIME" +%s) - $(date -d "$START_TIME" +%s))) +"%H:%M:%S")

    # Append elapsed time to output file
    echo "Elapsed time: (from sh) $ELAPSED_TIME" >> "$OUTPUT_FILE"
    """
    return script_string

def return_slurm_array_test_script_job_execute_slurmarray_from_job_name_string(stored_job_name, n_layers, n_tasks, n_samples = 400, account= "def-ko1"):
    script_string = f"""#!/bin/bash
    #SBATCH --account={account}
    #SBATCH --time=8:00:00
    #SBATCH --mem=1G
    #SBATCH --cpus-per-task=2
    #SBATCH --array=0-{str(n_tasks-1)}  # Adjust based on your requirements

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
    OUTPUT_FILE="SlurmArray-Vertice-${{N_QUBITS}}_Layers-${{N_LAYERS}}_MaxUnlabeledGraphs-${{MAX_UNLABELED_GRAPHS}}_${{START_TIME_FILE}}.txt"

    # Pass parameters as environment variables
    N_SAMPLES={str(n_samples)}
    N_LAYERS={n_layers}

    # Execute the main script
    python QAOA_main.py --n_samples $N_SAMPLES --n_layers $N_LAYERS --task_id $SLURM_ARRAY_TASK_ID --job_name "job_execute_slurmarray_from_job_name" --stored_job_name "{stored_job_name}"

    # Record end time and calculate elapsed time
    END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
    ELAPSED_TIME=$(date -u -d @$(($(date -d "$END_TIME" +%s) - $(date -d "$START_TIME" +%s))) +"%H:%M:%S")

    # Append elapsed time to output file
    echo "Elapsed time: (from sh) $ELAPSED_TIME" >> "$OUTPUT_FILE"
    """
    return script_string

def submit_slurm_job(job_script):
    # Create a temporary SLURM script
    slurm_script = "slurm_temp_job.sh"
    with open(slurm_script, "w") as f:
        f.write(job_script)
        #f.write(return_slurm_array_test_script_job_execute_slurmarray_from_job_name_string(job_name, n_layers,n_tasks, n_samples))

    # Make the script executable
    subprocess.run(["chmod", "+x", slurm_script])

    # Submit the SLURM job
    subprocess.run(["sbatch", slurm_script])

    # Submit the SLURM job and capture the output
    result = subprocess.run(["sbatch", slurm_script], capture_output=True, text=True)
    print("slurm result: " + str(result))
    print("slurm result stdout: " + str(result.stdout))
    
    # Extract job ID from the output
     #   """Extracts the job ID from the sbatch output."""
    match = re.search(r'Submitted batch job (\d+)', str(result.stdout))

    print("slurm result search match: " + str(match))   

    os.remove(slurm_script)

    return match.group(1) if match else None

def check_job_id_state(job_ids, verbose = False):
     # Create a comma-separated string of job IDs
    job_ids_str = ','.join(map(str, job_ids))

    # Run the squeue command to check the status
    result = subprocess.run(['squeue', '-j', job_ids_str], capture_output=True, text=True)

        # Process the output
    lines = result.stdout.strip().split('\n')

    if verbose:
        print("subprocess result: " + result)
        print("lines result " + lines)

    job_states = []
    # Check if there are any jobs that are not completed
    if len(lines) > 1:  # The first line is the header
        # Extract job states
        for line in lines[1:]:
            columns = line.split()
            job_id = columns[0]  # First column is the job ID
            job_state = columns[4]  # State is usually in the 5th column
            job_states.append((job_id,job_state))
            if verbose:
                print("columns result: " + columns)
                print("job_id result " + job_id)
                print("job_state result: " + job_state)

    return job_states

def check_job_id_state_completed_or_failed(job_ids, verbose = False):
    
    # Create a comma-separated string of job IDs
    job_states = check_job_id_state(job_ids, verbose)

    print("job states: " + job_states)
    #check all completed
    for job_state in job_states:
        if job_state[0] not in ['CD', 'F']:  # 'CD' is Completed, 'F' is Failed
            print(f"Job {job_state[0]} is not completed. Current state: {job_state}")
            return False
    
    return True
    
def check_job_id_state_failed(job_ids, verbose = False):
        # Create a comma-separated string of job IDs
    job_states = check_job_id_state(job_ids, verbose)

    #check if one has failed
    for job_state in job_states:
        if job_state[0] in ['F']:  # 'CD' is Completed, 'F' is Failed
            print(f"Job {job_state[0]} has failed. Current state: {job_state}")
            return False
    
    return True