from QAOA_utils import *
import QAOA, graph_methods
import time, json
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial

def execute_job_parallel(job):
    graph, n_vertices, n_layers, method, identifier, n_steps, n_samples = job
    #start_time = time.time()
    #write_to_progress_file(f"starting job:{method}_{identifier}_{n_vertices}-vertices_{n_layers}-layers")
    print(f"    into job execution of {identifier} ")  
    result = execute_qaoa_subjob1(graph, n_vertices, n_layers, method, identifier, n_steps, n_samples)
    print(f"    end of job execution of {identifier} ")  
    #write_to_progress_file(f"done job: {method}_{identifier}_{n_vertices}-vertices_{n_layers}-layers. time taken: {time.time() - start_time}")
    return result

def run_jobs(all_jobs):
    loop_count = 0
    results_list = []
    start_time = time.time()
    job_count = len(all_jobs)
    for job in all_jobs:
        loop_count += 1            
        result = execute_qaoa_subjob1(job[0],job[1],job[2], job[3], job[4], job[5], job[6])   
        results_list.append(result)                   
    return results_list

def run_jobs_parallel(all_jobs):

    # Determine the number of processes to use
    num_processes = min(cpu_count(), len(all_jobs))  # Use all available cores or limit to number of jobs

    # Create a multiprocessing Pool with the determined number of processes
    with Pool(processes=num_processes) as pool:
        # Map jobs to the pool for parallel execution
        results = pool.map(execute_job_parallel, all_jobs)

    return results

def calculate_add_ratios_to_results_list(results_list):
    fQAOA_list = [entry for entry in results_list if entry[0] == 'fQAOA']
    QAOA_list = [entry for entry in results_list if entry[0] == 'QAOA']

    ratios = []
    for fQAOA_entry in fQAOA_list:

        QAOA_entry = [QAOA_entry for QAOA_entry in QAOA_list if QAOA_entry[2] == fQAOA_entry[2]]
       
        if not QAOA_entry:
            continue

        QAOA_mean = float(QAOA_entry[0][5])

        if not QAOA_mean > 0:
            continue

        ratio =  float(fQAOA_entry[5]) / QAOA_mean
        # [cost_layer,label, graph_to_string(graph), most_common_element, most_common_element_count_ratio, mean, maximum, stdev, str(graph_results_parameters)]
        ratios.append(["fQAOA/QAOA", "mean_ratio-"+str(fQAOA_entry[1]), str(fQAOA_entry[2]), "-", "-",str(ratio),"-","-","-"  ])

    return results_list + ratios

def execute_qaoa_subjob1(graph,n_vertices, n_layers, cost_layer, label, n_steps = 30, n_samples = 200): 
    #[isomorph_graph,n_vertices, n_layers, "QAOA", f"isomorphGraph{ii}_{graph_to_string(graph)}", n_steps, n_samples]
    print(f"    start of job execution of {label} (inner method)")  
    np.random.seed(42)
    start_time = time.time() #(graph, n_wires, n_layers, cost_layer = "QAOA", n_steps = 30, n_samples = 200, lightning_device = True, mixer_layer = "fermionic_Ryy"):
    graph_results = QAOA.qaoa_maxcut(graph, n_vertices,n_layers, cost_layer=cost_layer , n_steps = n_steps, n_samples = n_samples, label=label) #n_layer = n_vertices
    graph_results_distribution, graph_results_parameters  = graph_results.bit_strings_objectives_distribution, graph_results.parameters
    most_common_element, most_common_element_count_ratio, mean, maximum, stdev = compute_stats(graph_results_distribution)

    elapsed_time_seconds = time.time() - start_time
    elapsed_time_formatted = f"{int(elapsed_time_seconds // 60)} mins {int(elapsed_time_seconds % 60)} secs"
    print(f"    end of job execution of {label}, time elapsed: {elapsed_time_formatted} ")  

    graph_from_label = graph_from_label_string(label)

    return [cost_layer,label, graph_from_label, graph_to_string(graph), mean, maximum, most_common_element, most_common_element_count_ratio, elapsed_time_formatted]
    #return [cost_layer,label, graph_to_string(graph), most_common_element, most_common_element_count_ratio, mean, maximum, stdev, str(graph_results_parameters),elapsed_time_formatted]

def store_jobs(jobs, job_names):

    #create the corresponding directory
    subdirectory = "stored_jobs"

    # Check if the subdirectory exists and create it if it doesn't
    if not os.path.exists(subdirectory):
        os.makedirs(subdirectory)

    json_file_path = os.path.join(subdirectory, f"{job_names}.json")
    with open(json_file_path, 'w') as f:
        json.dump(jobs, f)

def retrieve_stored_jobs(job_names):

    subdirectory  = "stored_jobs"
    json_file_path = os.path.join(subdirectory, f"{job_names}.json")

    #Open the JSON file and load the data
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    return data

def get_job1_names_from_parameters(n_vertices, n_layers, n_steps, n_samples, n_isomorph_max, max_unlabeled_graph= None, max_job = None):
    #[[[[0, 1], [2, 3]], 4, 4, "QAOA", "unlabeledGraph_0123", 20, 100]
    job_names = f"job1-vertices_{n_vertices}_layers_{n_layers}_steps_{n_steps}_samples_{n_samples}_isomorphmax_{str(n_isomorph_max)}_maxunlabeledgraph_{str(max_unlabeled_graph)}_maxjob_{max_job}"
    return job_names

def get_result_name_from_job(job):
    #n_vertices, n_layers, cost_layer, label, n_steps = 30, n_samples = 200): 
    return f"vertices_{job[1]}_layers_{job[2]}_costlayer_{job[3]}_label_{job[4]}_steps_{job[5]}_samples_{job[6]}"

def generate_job_list_job1(isomorphic_graph_lists, n_layers, n_steps, n_samples, n_vertices, n_isomorph_max):
    job_lists_QAOA = [[graph,n_vertices, n_layers,"QAOA", f"unlabeledGraph_{graph_to_string(graph)}", n_steps, n_samples] for graph in isomorphic_graph_lists]
    job_lists_fQAOA = [[graph,n_vertices,n_layers,"fQAOA", f"unlabeledGraph_{graph_to_string(graph)}", n_steps, n_samples] for graph in isomorphic_graph_lists]
    job_lists_iso = []
    for ii, graph in enumerate(isomorphic_graph_lists):
        isomorphic_graphs = graph_methods.generate_isomorphics_from_combination(graph,max_isomorphism_number=n_isomorph_max)
        isomorphic_graphs = isomorphic_graphs[1:] #the first generated isomorphic graph is identity
        for ij, isomorph_graph in enumerate(isomorphic_graphs):                   
            job_lists_iso.append([isomorph_graph,n_vertices, n_layers, "QAOA", f"isomorphGraph{ij}_{graph_to_string(graph)}", n_steps, n_samples])
            job_lists_iso.append([isomorph_graph,n_vertices, n_layers, "fQAOA", f"isomorphGraph{ij}_{graph_to_string(graph)}", n_steps, n_samples])

    all_jobs = job_lists_QAOA + job_lists_fQAOA + job_lists_iso
    
    return all_jobs

def generate_jobs1(n_vertices, n_layers, n_steps, n_samples, n_isomorph_max, max_unlabeled_graph= None, max_job = None):

    start_time = time.time()
    np.random.seed(42)
    unlabeled_graphs = graph_methods.generate_all_connected_graphs(n_vertices, True)
    print(f"graphs generated")

    unlabeled_graphs_graphs = [graph[0] for graph in unlabeled_graphs]
    unlabeled_graph_number = len(unlabeled_graphs_graphs)
    print(f"number of unlabeled graphs:  {unlabeled_graph_number}")      

    if max_unlabeled_graph != None: #limit to amount of unlabeled graph number if needed. TB Implemented: sampling according to weight
        unlabeled_graphs_graphs = unlabeled_graphs_graphs[:max_unlabeled_graph]

    all_jobs = generate_job_list_job1(unlabeled_graphs_graphs, n_layers, n_steps, n_samples, n_vertices, n_isomorph_max)
    print(f"jobs created, number of jobs:  {len(all_jobs)}. Max Jobs: {max_job} (no lim: -1)")
    job_count = 0

    if max_job != -1: #limit to amount of graph number if needed. TB Implemented: sampling according to weight
        all_jobs = all_jobs[:max_job]

    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    return all_jobs

def job1(n_vertices, n_layers, n_steps, n_samples, n_isomorph_max, max_unlabeled_graph= None, max_job = None, parallel_task= True):        
    print("start of QAOA - job1")
    all_jobs = generate_jobs1(n_vertices, n_layers, n_steps, n_samples, n_isomorph_max, max_unlabeled_graph= None, max_job = None)
    jobnames = get_job1_names_from_parameters(n_vertices, n_layers, n_steps, n_samples, n_isomorph_max, max_unlabeled_graph= None, max_job = None)
    results_list = execute_mp_jobs(all_jobs)
    process_results_save(results_list, jobnames)

def job1_generate_save_jobs(n_vertices, n_layers, n_steps, n_samples, n_isomorph_max, max_unlabeled_graph= None, max_job = None):
    all_jobs = generate_jobs1(n_vertices, n_layers, n_steps, n_samples, n_isomorph_max, max_unlabeled_graph, max_job)
    job_names = get_job1_names_from_parameters(n_vertices, n_layers, n_steps, n_samples, n_isomorph_max, max_unlabeled_graph, max_job)
    store_jobs(all_jobs,job_names)

def job1_retrieve_jobs(n_vertices, n_layers, n_steps, n_samples, n_isomorph_max, max_unlabeled_graph= None, max_job = None):
    job_names = get_job1_names_from_parameters(n_vertices, n_layers, n_steps, n_samples, n_isomorph_max, max_unlabeled_graph, max_job)
    all_jobs = retrieve_stored_jobs(job_names)
    return all_jobs

def execute_mp_jobs(jobs, parallel_task = True):
    results_list = []
    if not parallel_task:
        print(f"jobs start not in parallel")   
        results_list = run_jobs(jobs)
    else:
        print(f"jobs start in parallel")   
        results_list = run_jobs_parallel(jobs)
    return results_list

def execute_single_job(job):
    #[[[0, 1], [2, 3]], 4, 4, "QAOA", "unlabeledGraph_0123", 20, 100]
    #(graph,n_vertices, n_layers, cost_layer, label, n_steps = 30, n_samples = 200): 
    result = execute_qaoa_subjob1(job[0],job[1],job[2], job[3], job[4], job[5], job[6])  
    jobname = get_result_name_from_job(job)
    save_single_job_result(result,jobname)

def save_single_job_result(result,jobname):
    
    subdirectory = "stored_job_results"

    # Convert the list to a NumPy array
    result_np = np.array(result)

    # Save the array to a .npy file
    full_path = os.path.join(subdirectory, f"{jobname}.npy")

    # Create the subdirectory if it doesn't exist
    os.makedirs(subdirectory, exist_ok=True)

    # Save the array to the specified subdirectory
    
    if not os.path.exists(subdirectory):
        os.makedirs(subdirectory)

    np.save(full_path, result_np)
   
def store_jobs(jobs, job_names):

    #create the corresponding directory
    subdirectory = "stored_jobs"

    # Check if the subdirectory exists and create it if it doesn't
    if not os.path.exists(subdirectory):
        os.makedirs(subdirectory)

    json_file_path = os.path.join(subdirectory, f"{job_names}.json")
    with open(json_file_path, 'w') as f:
        json.dump(jobs, f)
    
def retrieve_single_job_result(resultname):

    subdirectory = "stored_job_results"

    # Save the array to a .npy file
    full_path = os.path.join(subdirectory, f"{resultname}.npy")

    if not os.path.exists(full_path):
        return

    # Load the NumPy array from the file
    loaded_array = np.load(full_path)

    # Convert the NumPy array back to a list
    loaded_list = loaded_array.tolist()

    return loaded_list

def retrieve_job_result_names_list(result_names):
    results_list = []
    for result_name in result_names:
        result = retrieve_single_job_result(result_name)
        results_list.append(result)
    return results_list

def job1_execute_slurmarray(n_vertices, n_layers, n_steps, n_samples, n_isomorph_max, max_unlabeled_graph= None, max_job = None, task_id = None, mock = False):

    if task_id is None or task_id == -1:
        return
    
    if mock:
        task_ids = range(max_job)
    else:
        task_ids = task_id
    
    for task_id in task_ids:   
    #load job list of job1
        jobnames = get_job1_names_from_parameters(n_vertices, n_layers, n_steps, n_samples, n_isomorph_max, max_unlabeled_graph, max_job)
        all_jobs = retrieve_stored_jobs(jobnames)

        # the task id array is bigger than the number of jobs
        if task_id >= len(all_jobs):
            return
        
        execute_single_job(all_jobs[task_id])

def job1_retrieve_merge_results(n_vertices, n_layers, n_steps, n_samples, n_isomorph_max, max_unlabeled_graph= None, max_job = None):

    jobnames = get_job1_names_from_parameters(n_vertices, n_layers, n_steps, n_samples, n_isomorph_max, max_unlabeled_graph, max_job)
    all_jobs = retrieve_stored_jobs(jobnames)

    results_list = []
    for job in all_jobs:
        resultname = get_result_name_from_job(job)
        result = retrieve_single_job_result(resultname)

        if result is None:
            continue

        results_list.append(result)
    
    return results_list

def job1_process_results(n_vertices, n_layers, n_steps, n_samples, n_isomorph_max, max_unlabeled_graph= None, max_job = None):
    print("start of result merge")
    results_list = job1_retrieve_merge_results(n_vertices, n_layers, n_steps, n_samples, n_isomorph_max, max_unlabeled_graph, max_job)
    print("result list obtained")
    jobnames = get_job1_names_from_parameters(n_vertices, n_layers, n_steps, n_samples, n_isomorph_max, max_unlabeled_graph, max_job)
    print("jobname obtained")
    process_results_save(results_list,jobnames)

def process_results_save(results_list, jobnames):
    #results_list = calculate_ratios_from_results_list(results_list)
    #run_times = [float(element[-1]) for element in results_list]
    results_list = calculate_add_ratios_to_results_list(results_list)
    print("ratio calculated")
    
    # Calculate the mean of the "run time" values
    # mean_run_time = sum(run_times) / (len(run_times) * len(run_times))
    #results_list = [["cost_layer","label", "graph", "most_common_element", "most_common_element_count_ratio", "mean", "maximum", "stdev", "layer parameters", "run time"]] + results_list
    results_list = ["cost_layer","label", "simulated graph", "isomorphic graph", "mean", "maximum", "most_common_element", "most_common_element_count_ratio", "elapsed_time_formatted"] + results_list
    
    # Define the subdirectory name
    subdirectory = "merged_processed_results"

    # Ensure the subdirectory exists
    if not os.path.exists(subdirectory):
        os.makedirs(subdirectory)

    # Construct the file path
    file_path = os.path.join(subdirectory, jobnames + ".txt")
   
    # Save the results list to the specified JSON file
    with open(file_path, 'w') as f:
        json.dump(results_list, f)
    print("saved")

