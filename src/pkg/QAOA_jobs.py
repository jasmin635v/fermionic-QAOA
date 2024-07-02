from pennylane import numpy as np
from QAOA_utils import *
from graph import *
import QAOA
from QAOA import QAOAResult
from datetime import date
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# Get the current directory path and # Define the filename
current_directory = os.path.dirname(os.path.realpath(__file__))
filename = os.path.join(current_directory, 'progress.txt')
file_lock = threading.Lock()

default=os.path.join(os.path.dirname(__file__), "data")

def plot_table_from_list_of_lists(list_of_lists, column_headers, title_text = ""):
    footer_text = date.today()
    fig_background_color = 'white'
    fig_border = 'steelblue'
    data =  list_of_lists

    cell_text = []
    for row in data:
        cell_text.append([f'{str(x)}' for x in row])
    # Get some lists of color specs for row and column headers
    #rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
    ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))

    plt.figure(linewidth=2,
            edgecolor=fig_border,
            facecolor=fig_background_color,
            tight_layout={'pad':1},
            #figsize=(5,3)
            )
    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                        #rowLabels=row_headers,
                        #rowColours=rcolors,
                        rowLoc='right',
                        colColours=ccolors,
                        colLabels=column_headers,
                        loc='center')
    # Scaling is the only influence we have over top and bottom cell padding.
    # Make the rows taller (i.e., make cell y scale larger).
    the_table.scale(1, 1.5)
    # Hide axes
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Hide axes border
    plt.box(on=None)
    # Add title
    plt.suptitle(title_text)
    # Add footer
    plt.figtext(0.95, 0.05, footer_text, horizontalalignment='right', size=6, weight='light')
    # Force the figure to update, so backends center objects correctly within the figure.
    # Without plt.draw() here, the title will center on the axes and not the figure.
    plt.draw()
    # Create image. plt.savefig ignores figure edge and face colors, so map them.
    fig = plt.gcf()
    plt.savefig('pyplot-table-demo.png',
                #bbox='tight',
                edgecolor=fig.get_edgecolor(),
                facecolor=fig.get_facecolor(),
                dpi=150
                )

def format_job_name_from_result(job_result):
    graph_string = graph_to_string(job_result[2])
    return f"{job_result[0]}_{job_result[1]}_{graph_string}.npy"

def graph_to_string(graph):
    return graph.replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace(' ', '').replace(',', '')

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

def execute_qaoa_subjob1(graph,n_vertices,cost_layer, label):
    np.random.seed(42)
    QAOA_result = extract_result(graph,n_vertices, cost_layer, label)
    current_directory = os.getcwd()
    #if append_result:
    return QAOA_result
    # else :
    #     np.save(f"{format_job_name_from_result(QAOA_result)}", np.array(QAOA_result)) 
    #     return format_job_name_from_result(QAOA_result)
    
def extract_result(graph,n_vertices, cost_layer, label):

    start_time = time.time()
    QAOA.graph = graph
    graph_results = QAOA.qaoa_maxcut(n_vertices,graph,n_vertices, cost_layer=cost_layer) #n_layer = n_vertices
    graph_results_distribution, graph_results_parameters  = graph_results.bit_strings_objectives_distribution, graph_results.parameters
    most_common_element, most_common_element_count_ratio, mean, maximum, stdev = compute_stats(graph_results_distribution)
    end_time = time.time()

    elapsed_time_seconds = end_time - start_time
    elapsed_minutes = int(elapsed_time_seconds // 60)
    elapsed_seconds = int(elapsed_time_seconds % 60)
    elapsed_time_formatted = f"{elapsed_minutes} mins {elapsed_seconds} secs"

    return [cost_layer,label, graph_to_string(graph), most_common_element, most_common_element_count_ratio, mean, maximum, stdev, str(graph_results_parameters)]

def write_to_progress_file(text, start_time = None, slurm = True):

    if start_time != None:
        end_time = time.time()
        formatted_time = format_time(int(end_time - start_time))
        text += f" time taken: {formatted_time} "

    with file_lock:
        if slurm:
            write_to_slurm_output(text)
        else:
            with open(filename, 'a') as f:
                f.write(f'{text}\n')

def format_time(seconds):
    """
    Convert seconds into a string of format 'MM:SS'.
    """
    m, s = divmod(seconds, 60)
    return f'{m:02}:{s:02}'
#job1: results for 4 vertices,3 isomorphism per graph

def write_to_slurm_output(message):
    # Print to stdout or stderr
    print(message+"\n")
    # Force flush to ensure immediate output
    sys.stdout.flush()

def execute_qaoa_job1(n_vertices, n_layers, n_steps, n_samples, n_isomorph_max, max_graph= None, parallel_task= True):
    start_time = time.time()
    write_to_progress_file(f"start of QAOA - job1")

    def run_jobs_parallel(all_jobs):

        results_list = []

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(execute_job, job) for job in all_jobs]

        # Collect results as they are completed
        for future in as_completed(futures):
            results_list.append(future.result())
                    
        return results_list

    def run_jobs(all_jobs):

        loop_count = 0
        results_list = []
        for job in all_jobs:
            loop_count += 1            
            write_to_progress_file(f"into loop {loop_count} on {isomorphic_graph_number}",start_time)
            result = execute_qaoa_subjob1(job[0],job[1],job[2], job[3])   
            results_list.append(result)                   
            write_to_progress_file(f"{loop_count} on {isomorphic_graph_number} done",start_time)
            
        return results_list

    def execute_job(job):
        graph, n_vertices, method, identifier = job
        write_to_progress_file(f"starting job:{method}_{identifier}_{graph}", start_time)
        result = execute_qaoa_subjob1(graph, n_vertices, method, identifier)
        write_to_progress_file(f"done job: {method}_{identifier}_{graph}", start_time)
        return result
    
    def generate_job_list(isomorphic_graph_lists):
        job_lists_QAOA = [[graph,n_vertices,"QAOA", "unlabeledGraph"] for graph in isomorphic_graph_lists]
        job_lists_fQAOA = [[graph,n_vertices,"fQAOA", "unlabeledGraph"] for graph in isomorphic_graph_lists]
        job_lists_iso = []
        for ii, graph in enumerate(isomorphic_graph_lists):
            isomorphic_graphs = generate_isomorphics_from_combination(graph,max_isomorphism_number=n_isomorph_max)
            for ij, isomorph_graph in enumerate(isomorphic_graphs):                   
                job_lists_iso.append([isomorph_graph,n_vertices,"QAOA", f"isomorphGraph{ii}_{graph_to_string(graph)}"])
                job_lists_iso.append([isomorph_graph,n_vertices,"fQAOA", f"isomorphGraph{ii}_{graph_to_string(graph)}"])

        all_jobs = job_lists_QAOA + job_lists_fQAOA + job_lists_iso
        return all_jobs

    #sets the seed of the random number generator provided by NumPy from reproducibility
    np.random.seed(42)

    #QAOA.n_wires = n_vertices #set equal to vertice always. Default.Qbits takes an automatic number of qbits based on circuit. circuit is based on graph and no graph has free vertices.    
    QAOA.n_layers = n_layers
    QAOA.steps = n_steps
    QAOA.n_samples = n_samples

    isomorphic_graphs = generate_all_connected_graphs(n_vertices, True)
    write_to_progress_file(f"graphs generated")

    isomorphic_graphs_graphs = [graph[0] for graph in isomorphic_graphs]
    isomorphic_graph_number = len(isomorphic_graphs_graphs)      

    if max_graph != None: #limit to amount of graph number if needed. TB Implemented: sampling according to weight
        isomorphic_graphs_graphs = isomorphic_graphs_graphs[:max_graph]

    all_jobs = generate_job_list(isomorphic_graphs_graphs)
    job_count = 0
    all__jobs_count = len(all_jobs)

    if not parallel_task:
        results_list = run_jobs(all_jobs)
    else:
        results_list = run_jobs_parallel(all_jobs)

    formatted_datatime = datetime.now().strftime("%m%d%H%M")
    #np.savetxt(f"/home/jcjcp/scratch/jcjcp/QAOA/MaxCut/src/pkg/qaoa_job1_{formatted_datatime}.txt", results_list, fmt='%s', delimiter='\t')
    np.savetxt(f"qaoa_job1_{formatted_datatime}.txt", results_list, fmt='%s', delimiter='\t')

    # if not append_result: #merge npy files and create txt
    #     results_list = load_numpy_arrays_to_list(results_list)
    #     remove_npy_files(results_list)
    #     # Get current date and time
    #     current_datetime = datetime.now()
    #     formatted_datatime = current_datetime.strftime("%m%d%H%M")
    #     np.save(f"qaoa_job1_{formatted_datatime}", np.array(results_list)) 
    #     np.savetxt(f"qaoa_job1_{formatted_datatime}.txt", results_list, fmt='%s', delimiter='\t')
    # else: # get 
    #     headers = ["cost_layer","label", "graph", "most common element", "most common element sampling proportion", "mean of distribution", "maximum of distribution", "standard dev. of distribution", "simulation optimized layer parameters", "time taken"]
    #     results_list = load_numpy_arrays_to_list(results_list)
    #     #results_list = [["cost_layer","label", "graph", "most common element", "most common element sampling proportion", "mean of distribution", "maximum of distribution", "standard dev. of distribution", "simulation optimized layer parameters", "time taken"],["cost_layer","label", "graph", "most common element", "most common element sampling proportion", "mean of distribution", "maximum of distribution", "standard dev. of distribution", "simulation optimized layer parameters", "time taken"]]
    #     plot_table_from_list_of_lists(results_list,headers)

#execute_qaoa_job1(n_vertices = 3, n_layers = 1, n_samples = 1, n_steps = 1, n_isomorph_max = 0)
execute_qaoa_job1(n_vertices = 4, n_layers = 4, n_samples = 200, n_steps = 30, n_isomorph_max = 3, parallel_task=True)