from pennylane import numpy as np
import cmath, math, os, threading, time, sys
from datetime import date
import argparse


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

# current_directory = os.path.dirname(os.path.realpath(__file__))
# filename = os.path.join(current_directory, 'progress.txt')
# file_lock = threading.Lock()
# default=os.path.join(os.path.dirname(__file__), "data")


def parse_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="the number of times the circuit is sampled after the parameters are found",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=4,
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
        default=20,
        help="The number of steps of the layer parameter optimizer.",
    )
    parser.add_argument(
        "--max_isomorph_number",
        type=int,
        default=3,
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
        "--task_id",
        type=int,
        default="-1",
        help="the slurm task id handled by slurm",
    )

    return parser.parse_args()

def Rzz_matrice(gamma):
    exp_term = cmath.exp(1j * gamma/ 2)
    exp_term_m = cmath.exp(-1j * gamma/ 2)
    return np.array([[exp_term_m, 0, 0, 0],
            [0, exp_term, 0, 0],
            [0, 0, exp_term, 0],
            [0, 0, 0, exp_term_m]])

def bitstring_to_int(bit_string_sample):
    bit_string = "".join(str(bs) for bs in bit_string_sample)
    return int(bit_string, base=2)

def bitstring_to_objective(bitstring, graph):
    #convert bitstring to a list of 0 and 1
    binary_list = [int(char) for char in bitstring]
    obj = 0
    for edge in graph:
        # objective for the MaxCut problem
        obj += 1 - (binary_list[edge[0]] * binary_list[edge[1]])
    return obj * 0.5

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
    weighted_variance_sum = sum(number[1]  * (number[0]  - mean)**2 for number in numbers)
    weighted_variance = weighted_variance_sum / total_count

    # Step 3: Calculate weighted standard deviation
    weighted_stddev = math.sqrt(weighted_variance)

    sorted_numbers = sorted(numbers, key=lambda x: x[1], reverse=True)
    most_common_element = sorted_numbers[0][0]
    most_common_count = sorted_numbers[0][1]

    most_common_element_count_ratio = most_common_count / total_count
    #weighted_mean_3 = (sorted_numbers[0][0]*sorted_numbers[0][1] + sorted_numbers[1][0]*sorted_numbers[1][1] +sorted_numbers[2][0]*sorted_numbers[2][1])  / (sorted_numbers[0][1] + sorted_numbers[1][1] +sorted_numbers[2][1])

    return  most_common_element, most_common_element_count_ratio, mean, maximum, weighted_stddev



def write_to_progress_file(text, start_time = None, slurm = True, fileName = None):

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
    """
    Convert seconds into a string of format 'MM:SS'.
    """
    m, s = divmod(seconds, 60)
    return f'{m:02}:{s:02}'

def write_to_slurm_output(message):
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
    return str(graph).replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace(' ', '').replace(',', '')

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
    # Construct the full path to the file in the current directory
        file_path = os.path.join(os.getcwd(), filename)
        
        # Check if the file exists before attempting to delete
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted: {filename}")
        else:
            print(f"{filename} does not exist.")