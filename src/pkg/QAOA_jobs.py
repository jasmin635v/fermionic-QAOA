from pennylane import numpy as np
import math
import cmath
from QAOA_utils import *
from graph import *
import QAOA
from QAOA import QAOAResult
from datetime import date
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime

def plot_table_from_list_of_lists(list_of_lists, column_headers, title_text = ""):
    footer_text = date.today()
    fig_background_color = 'white'
    fig_border = 'steelblue'
    data =  list_of_lists
    # Pop the headers from the data array
    #row_headers = [x.pop(0) for x in data]
    # Table data needs to be non-numeric text. Format the data
    # while I'm at it.
    cell_text = []
    for row in data:
        cell_text.append([f'{str(x)}' for x in row])
    # Get some lists of color specs for row and column headers
    #rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
    ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))
    # Create the figure. Setting a small pad on tight_layout
    # seems to better regulate white space. Sometimes experimenting
    # with an explicit figsize here can produce better outcome.
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
    graph_string = job_result[2].replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace(' ', '').replace(',', '')
    return f"{job_result[0]}_{job_result[1]}_{graph_string}.npy"

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

def execute_qaoa_job1(append_result = False):

    def run_for_isomorph_graph_job1(isomorphic_graphs_4vertices, append_result ):
        
        results_list = []
        numpy_array_name_list = []
        for graph in isomorphic_graphs_4vertices:

            results_list, numpy_array_name_list = execute_qaoa_subjob1(graph,n_vertices,"QAOA", "unlabeledGraph",append_result, results_list,numpy_array_name_list)
            results_list, numpy_array_name_list = execute_qaoa_subjob1(graph,n_vertices,"fQAOA", "unlabeledGraph",append_result, results_list,numpy_array_name_list)         

            #compute up to n isomorphic graph for each unlabeled graph
            isomorphic_graphs = generate_isomorphics_from_combination(graph,max_isomorphism_number=graph_isomorphism_number_to_compute)
            for ii, isomorph_graph in enumerate(isomorphic_graphs):            
                results_list,numpy_array_name_list = execute_qaoa_subjob1(graph,n_vertices,"QAOA", f"isomorphGraph{ii}",append_result, results_list,numpy_array_name_list)
                results_list,numpy_array_name_list = execute_qaoa_subjob1(graph,n_vertices,"fQAOA", f"isomorphGraph{ii}",append_result, results_list,numpy_array_name_list)      

        return results_list, numpy_array_name_list

    #sets the seed of the random number generator provided by NumPy from reproducibility
    np.random.seed(42)
    n_vertices = 4
    QAOA.n_wires = n_vertices #set equal to vertice always
    isomorphic_graphs_4vertices = generate_all_connected_graphs(n_vertices, True)
    graph_isomorphism_number_to_compute = 3 # 3 isomorphic graph per graph at most

    #TEST JASMIN: FIRST GRAPH ONLY, no isomorph
    #isomorphic_graphs_4vertices = [isomorphic_graphs_4vertices[0][0]]
    #graph_isomorphism_number_to_compute = 0
   
    results_list, numpy_array_name_list = run_for_isomorph_graph_job1(isomorphic_graphs_4vertices, append_result)

    if not append_result:
        results_list = load_numpy_arrays_to_list(numpy_array_name_list)
        remove_npy_files(numpy_array_name_list)
        # Get current date and time
        current_datetime = datetime.now()
        formatted_datatime = current_datetime.strftime("%m%d%H%M")
        np.save(f"qaoa_job1_{formatted_datatime}", np.array(results_list)) 
    else:
        headers = ["cost_layer","label", "graph", "most common element", "most common element sampling proportion", "mean of distribution", "maximum of distribution", "standard dev. of distribution", "simulation optimized layer parameters", "time taken"]
        results_list = [["cost_layer","label", "graph", "most common element", "most common element sampling proportion", "mean of distribution", "maximum of distribution", "standard dev. of distribution", "simulation optimized layer parameters", "time taken"],["cost_layer","label", "graph", "most common element", "most common element sampling proportion", "mean of distribution", "maximum of distribution", "standard dev. of distribution", "simulation optimized layer parameters", "time taken"]]
        plot_table_from_list_of_lists(results_list,headers)

def execute_qaoa_subjob1(graph,n_vertices,cost_layer, label, append_result, result_list = None, numpy_array_name_list = None):
    np.random.seed(42)
    QAOA_result = extract_result(graph,n_vertices, cost_layer, label)
    current_directory = os.getcwd()
    if append_result and result_list is not None :
        result_list.append(QAOA_result)
        return result_list
    elif numpy_array_name_list is not None :
        numpy_array_name_list.append(format_job_name_from_result(QAOA_result))
        np.save(f"{format_job_name_from_result(QAOA_result)}", np.array(QAOA_result)) 
    return result_list, numpy_array_name_list

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

    return [cost_layer,label, str(graph), most_common_element, most_common_element_count_ratio, mean, maximum, stdev, str(graph_results_parameters), elapsed_time_formatted]

#job1: results for 4 vertices,3 isomorphism per graph
execute_qaoa_job1()