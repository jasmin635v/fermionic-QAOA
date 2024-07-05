from QAOA_utils import *
import QAOA
from graph import *
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys, pennylane as qml, math, cmath
from pennylane import numpy as np
from QAOA_utils import *
from graph import *
from collections import Counter
from multiprocessing import Pool, cpu_count
from functools import partial


def execute_job_parallel(job):
    graph, n_vertices, n_layers, method, identifier, n_steps, n_samples = job
    start_time = time.time()
    write_to_progress_file(f"starting job:{method}_{identifier}_{n_vertices}-vertices_{n_layers}-layers")
    result = execute_qaoa_subjob1(graph, n_vertices, n_layers, method, identifier, n_steps, n_samples)
    write_to_progress_file(f"done job: {method}_{identifier}_{n_vertices}-vertices_{n_layers}-layers")
    return result

def run_jobs(all_jobs):
    loop_count = 0
    results_list = []
    start_time = time.time()
    job_count = len(all_jobs)
    for job in all_jobs:
        loop_count += 1            
        write_to_progress_file(f"into loop {loop_count} on {job_count}",start_time)
        result = execute_qaoa_subjob1(job[0],job[1],job[2], job[3], job[4], job[5], job[6])   
        results_list.append(result)                   
        write_to_progress_file(f"{loop_count} on {job_count} done",start_time)
            
        return results_list

def run_jobs_parallel(all_jobs):

    # Determine the number of processes to use
    num_processes = min(cpu_count(), len(all_jobs))  # Use all available cores or limit to number of jobs

    # Create a multiprocessing Pool with the determined number of processes
    with Pool(processes=num_processes) as pool:
        # Map jobs to the pool for parallel execution
        results = pool.map(execute_job_parallel, all_jobs)

    return results

def run_jobs_parallel_threadpoolexecutor(all_jobs):

        results_list = []

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(execute_job_parallel, job) for job in all_jobs]

        # Collect results as they are completed
        for future in as_completed(futures):
            results_list.append(future.result())
                    
        return results_list  

def calculate_add_ratios_too_results_list(results_list):
    fQAOA_list = [entry for entry in results_list if entry[0] == 'fQAOA']
    QAOA_list = [entry for entry in results_list if entry[0] == 'QAOA']

    ratios = []
    for fQAOA_entry in fQAOA_list:

        QAOA_entry = [QAOA_entry for QAOA_entry in QAOA_list if QAOA_entry[2] == fQAOA_entry[2]]
       
        if not QAOA_entry:
            continue

        QAOA_mean = QAOA_entry[0][5]

        if not QAOA_mean > 0:
            continue

        ratio =  fQAOA_entry[5] / QAOA_mean
        # [cost_layer,label, graph_to_string(graph), most_common_element, most_common_element_count_ratio, mean, maximum, stdev, str(graph_results_parameters)]
        ratios.append(["fQAOA/QAOA", "mean_ratio-"+fQAOA_entry[1], fQAOA_entry[2], "-", "-",ratio,"-","-","-"  ])

    return results_list + ratios

def execute_qaoa_subjob1(graph,n_vertices, n_layers, cost_layer, label, n_steps = 30, n_samples = 200): 
    #[isomorph_graph,n_vertices, n_layers, "QAOA", f"isomorphGraph{ii}_{graph_to_string(graph)}", n_steps, n_samples]
    np.random.seed(42)
    start_time = time.time() #(graph, n_wires, n_layers, cost_layer = "QAOA", n_steps = 30, n_samples = 200, lightning_device = True, mixer_layer = "fermionic_Ryy"):
    graph_results = QAOA.qaoa_maxcut(graph, n_vertices,n_layers, cost_layer=cost_layer , n_steps = n_steps, n_samples = n_samples) #n_layer = n_vertices
    graph_results_distribution, graph_results_parameters  = graph_results.bit_strings_objectives_distribution, graph_results.parameters
    most_common_element, most_common_element_count_ratio, mean, maximum, stdev = compute_stats(graph_results_distribution)

    elapsed_time_seconds = time.time() - start_time
    elapsed_time_formatted = f"{int(elapsed_time_seconds // 60)} mins {int(elapsed_time_seconds % 60)} secs"

    #chi squared
    return [cost_layer,label, graph_to_string(graph), most_common_element, most_common_element_count_ratio, mean, maximum, stdev, str(graph_results_parameters)]

def execute_qaoa_job1(n_vertices, n_layers, n_steps, n_samples, n_isomorph_max, max_unlabeled_graph= None, max_job = None, parallel_task= True):
    
    start_time = time.time()
    write_to_progress_file(f"start of QAOA - job1")

    def generate_job_list_job1(isomorphic_graph_lists, n_layers, n_steps, n_samples):
        job_lists_QAOA = [[graph,n_vertices, n_layers,"QAOA", f"unlabeledGraph_{graph_to_string(graph)}", n_steps, n_samples] for graph in isomorphic_graph_lists]
        job_lists_fQAOA = [[graph,n_vertices,n_layers,"fQAOA", f"unlabeledGraph_{graph_to_string(graph)}", n_steps, n_samples] for graph in isomorphic_graph_lists]
        job_lists_iso = []
        for ii, graph in enumerate(isomorphic_graph_lists):
            isomorphic_graphs = generate_isomorphics_from_combination(graph,max_isomorphism_number=n_isomorph_max)
            isomorphic_graphs = isomorphic_graphs[1:] #the first generated isomorphic graph is identity
            for ij, isomorph_graph in enumerate(isomorphic_graphs):                   
                job_lists_iso.append([isomorph_graph,n_vertices, n_layers, "QAOA", f"isomorphGraph{ii}_{graph_to_string(graph)}", n_steps, n_samples])
                job_lists_iso.append([isomorph_graph,n_vertices, n_layers, "fQAOA", f"isomorphGraph{ii}_{graph_to_string(graph)}", n_steps, n_samples])

        all_jobs = job_lists_QAOA + job_lists_fQAOA + job_lists_iso
        return all_jobs

    np.random.seed(42)

    unlabeled_graphs = generate_all_connected_graphs(n_vertices, True)
    write_to_progress_file(f"graphs generated")

    unlabeled_graphs_graphs = [graph[0] for graph in unlabeled_graphs]
    unlabeled_graph_number = len(unlabeled_graphs_graphs)      

    if max_unlabeled_graph != None: #limit to amount of unlabeled graph number if needed. TB Implemented: sampling according to weight
        unlabeled_graphs_graphs = unlabeled_graphs_graphs[:max_unlabeled_graph]

    all_jobs = generate_job_list_job1(unlabeled_graphs_graphs, n_layers, n_steps, n_samples)
    job_count = 0

    if max_job != None: #limit to amount of graph number if needed. TB Implemented: sampling according to weight
        all_jobs = all_jobs[:max_job]

    if not parallel_task:
        results_list = run_jobs(all_jobs)
    else:
        results_list = run_jobs_parallel(all_jobs)

    #results_list = calculate_ratios_from_results_list(results_list)
    results_list = calculate_add_ratios_too_results_list(results_list)

    formatted_datatime = datetime.now().strftime("%m%d%H%M")
    #np.savetxt(f"/home/jcjcp/scratch/jcjcp/QAOA/MaxCut/src/pkg/qaoa_job1_{formatted_datatime}.txt", results_list, fmt='%s', delimiter='\t')
    np.savetxt(f"qaoa_job1_{formatted_datatime}.txt", results_list, fmt='%s', delimiter='\t')

