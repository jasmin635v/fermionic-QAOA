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
    result = QAOA.execute_qaoa_subjob1(graph, n_vertices, n_layers, method, identifier, n_steps, n_samples)
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
        result = QAOA.execute_qaoa_subjob1(job[0],job[1],job[2], job[3], job[4], job[5], job[6])   
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

def execute_qaoa_job1(n_vertices, n_layers, n_steps, n_samples, n_isomorph_max, max_graph= None, max_job = None, parallel_task= True):
    
    start_time = time.time()
    write_to_progress_file(f"start of QAOA - job1")

    def generate_job_list_job1(isomorphic_graph_lists, n_layers, n_steps, n_samples):
        job_lists_QAOA = [[graph,n_vertices, n_layers,"QAOA", f"unlabeledGraph_{graph_to_string(graph)}", n_steps, n_samples] for graph in isomorphic_graph_lists]
        job_lists_fQAOA = [[graph,n_vertices,n_layers,"fQAOA", f"unlabeledGraph_{graph_to_string(graph)}", n_steps, n_samples] for graph in isomorphic_graph_lists]
        job_lists_iso = []
        for ii, graph in enumerate(isomorphic_graph_lists):
            isomorphic_graphs = generate_isomorphics_from_combination(graph,max_isomorphism_number=n_isomorph_max)
            for ij, isomorph_graph in enumerate(isomorphic_graphs):                   
                job_lists_iso.append([isomorph_graph,n_vertices, n_layers, "QAOA", f"isomorphGraph{ii}_{graph_to_string(graph)}", n_steps, n_samples])
                job_lists_iso.append([isomorph_graph,n_vertices, n_layers, "fQAOA", f"isomorphGraph{ii}_{graph_to_string(graph)}", n_steps, n_samples])

        all_jobs = job_lists_QAOA + job_lists_fQAOA + job_lists_iso
        return all_jobs

    np.random.seed(42)

    isomorphic_graphs = generate_all_connected_graphs(n_vertices, True)
    write_to_progress_file(f"graphs generated")

    isomorphic_graphs_graphs = [graph[0] for graph in isomorphic_graphs]
    isomorphic_graph_number = len(isomorphic_graphs_graphs)      

    if max_graph != None: #limit to amount of graph number if needed. TB Implemented: sampling according to weight
        isomorphic_graphs_graphs = isomorphic_graphs_graphs[:max_graph]

    all_jobs = generate_job_list_job1(isomorphic_graphs_graphs, n_layers, n_steps, n_samples)
    job_count = 0

    if max_job != None: #limit to amount of graph number if needed. TB Implemented: sampling according to weight
        all_jobs = all_jobs[:max_graph]

    if not parallel_task:
        results_list = run_jobs(all_jobs)
    else:
        results_list = run_jobs_parallel(all_jobs)

    formatted_datatime = datetime.now().strftime("%m%d%H%M")
    #np.savetxt(f"/home/jcjcp/scratch/jcjcp/QAOA/MaxCut/src/pkg/qaoa_job1_{formatted_datatime}.txt", results_list, fmt='%s', delimiter='\t')
    np.savetxt(f"qaoa_job1_{formatted_datatime}.txt", results_list, fmt='%s', delimiter='\t')

