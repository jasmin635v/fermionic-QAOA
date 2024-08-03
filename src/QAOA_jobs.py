from QAOA_utils import *
import QAOA, graph_methods, time, json, random, os
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial
import subprocess


def execute_job_parallel(job):
    graph, n_vertices, n_layers, method, identifier, n_steps, n_samples = job
    print(f"    into job execution of {identifier} ")
    result = execute_qaoa_subjob(
        graph, n_vertices, n_layers, method, identifier, n_steps, n_samples)
    print(f"    end of job execution of {identifier} ")
    return result

def run_jobs(all_jobs):
    results_list = []
    start_time = time.time()
    job_count = len(all_jobs)
    for job in all_jobs:
        result = execute_qaoa_subjob(
            job[0], job[1], job[2], job[3], job[4], job[5], job[6])
        results_list.append(result)
    return results_list

def run_jobs_parallel(all_jobs):

    # Determine the number of processes to use
    # Use all available cores or limit to number of jobs
    num_processes = min(cpu_count(), len(all_jobs))
    # Create a multiprocessing Pool with the determined number of processes
    with Pool(processes=num_processes) as pool:
        results = pool.map(execute_job_parallel, all_jobs)

    return results

def calculate_add_ratios_to_results_list(results_list):  # UNUSED
    fQAOA_list = [entry for entry in results_list if entry[0] == 'fQAOA']
    QAOA_list = [entry for entry in results_list if entry[0] == 'QAOA']

    ratios = []
    for fQAOA_entry in fQAOA_list:

        QAOA_entry = [
            QAOA_entry for QAOA_entry in QAOA_list if QAOA_entry[2] == fQAOA_entry[2]]

        if not QAOA_entry:
            continue

        QAOA_mean = float(QAOA_entry[0][4])

        if not QAOA_mean > 0:
            continue

        ratio = float(fQAOA_entry[4]) / QAOA_mean
        # [cost_layer,label, graph_to_string(graph), most_common_element, most_common_element_count_ratio, mean, maximum, stdev, str(graph_results_parameters)]
        ratios.append(["fQAOA/QAOA", str(fQAOA_entry[1]), str(fQAOA_entry[2]), str(
            fQAOA_entry[3]), str(ratio), "-", "-", "-", fQAOA_entry[8], fQAOA_entry[9]])

    return results_list + ratios

def execute_qaoa_subjob(graph, n_vertices, n_layers, cost_layer, label, n_steps=30, n_samples=200):
    # [isomorph_graph,n_vertices, n_layers, "QAOA", f"isomorphGraph{ii}_{graph_to_string(graph)}", n_steps, n_samples]
    print(f"    start of job execution of {label} - {n_layers} layers")

    np.random.seed(42)
    # (graph, n_wires, n_layers, cost_layer = "QAOA", n_steps = 30, n_samples = 200, lightning_device = True, mixer_layer = "fermionic_Ryy"):
    start_time = time.time()



    graph_results = QAOA.qaoa_maxcut(graph, n_vertices, n_layers, cost_layer=cost_layer,
                                     n_steps=n_steps, n_samples=n_samples, label=label)  # n_layer = n_vertices
    graph_results_distribution, graph_results_parameters = graph_results.bit_strings_objectives_distribution, graph_results.parameters
    most_common_element, most_common_element_count_ratio, mean, maximum, stdev = compute_stats(
        graph_results_distribution)
    graph_from_label = graph_from_label_string(label)

    elapsed_time_formatted = f"{int(
        (time.time() - start_time) // 60)} mins {(int(time.time() - start_time) % 60)} secs"
    print(f"    end of job execution of {
          label}, time elapsed: {elapsed_time_formatted} ")
    return [cost_layer, label, graph_from_label, graph_to_string(graph), mean, maximum, most_common_element, most_common_element_count_ratio, n_layers, n_steps, elapsed_time_formatted]

def retrieve_stored_jobs(job_names):

    script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create the corresponding directory
    subdirectory = os.path.join(script_dir, "stored_jobs")

    json_file_path = os.path.join(subdirectory, f"{job_names}.json")

    # Open the JSON file and load the data
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    return data

def get_job_names_from_parameters(n_vertices, n_layers, n_samples, n_isomorph_max, max_unlabeled_graph=None, max_job=None):
    job_names = f"job-vertices_{n_vertices}_layers_{str(n_layers)}_samples_{n_samples}_isomorphmax_{str(n_isomorph_max)}_maxunlabeledgraph_{str(max_unlabeled_graph)}_maxjob_{str(max_job)}"
    return job_names

def get_job_names_from_parameters_graphs(n_vertices, n_isomorph_max, max_unlabeled_graph=None, max_job=None):
    if max_job == -1:
        max_job = "None"

    job_names = f"job-vertices_{n_vertices}_isomorphmax_{str(
        n_isomorph_max)}_maxunlabeledgraph_{str(max_unlabeled_graph)}_maxjob_{max_job}"
    return job_names

def get_result_name_from_job(job):
    # n_vertices, n_layers, cost_layer, label, n_steps = 30, n_samples = 200):
    return f"vertices_{job[1]}_layers_{job[2]}_costlayer_{job[3]}_label_{job[4]}_steps_{job[5]}_samples_{job[6]}"

def generate_job_list(isomorphic_graph_lists, n_layers, n_steps, n_samples, n_vertices, n_isomorph_max):
    job_lists_QAOA = [[graph, n_vertices, n_layers, "QAOA", f"unlabeledGraph_{
        graph_to_string(graph)}", n_steps, n_samples] for graph in isomorphic_graph_lists]
    job_lists_fQAOA = [[graph, n_vertices, n_layers, "fQAOA", f"unlabeledGraph_{
        graph_to_string(graph)}", n_steps, n_samples] for graph in isomorphic_graph_lists]
    job_lists_iso = []

    if n_isomorph_max > 0:

        for ii, graph in enumerate(isomorphic_graph_lists):
            isomorphic_graphs = graph_methods.generate_isomorphics_from_combination(
                graph, max_isomorphism_number=n_isomorph_max)
            # the first generated isomorphic graph is identity
            isomorphic_graphs = isomorphic_graphs[1:]
            for ij, isomorph_graph in enumerate(isomorphic_graphs):
                job_lists_iso.append([isomorph_graph, n_vertices, n_layers, "QAOA", f"isomorphGraph{
                                    ij}_{graph_to_string(graph)}", n_steps, n_samples])
                job_lists_iso.append([isomorph_graph, n_vertices, n_layers, "fQAOA", f"isomorphGraph{
                                    ij}_{graph_to_string(graph)}", n_steps, n_samples])

    all_jobs = job_lists_QAOA + job_lists_fQAOA + job_lists_iso

    return all_jobs

def generate_job_graphslist(isomorphic_graph_lists, n_vertices, n_isomorph_max):
    job_lists_QAOA = [[graph, n_vertices, f"unlabeledGraph_{graph_to_string(graph)}"] for graph in isomorphic_graph_lists]
    job_lists_iso = []
    if n_isomorph_max != 0 and n_isomorph_max is not None:
        for ii, graph in enumerate(isomorphic_graph_lists):
            isomorphic_graphs = graph_methods.generate_isomorphics_from_combination(graph, max_isomorphism_number=n_isomorph_max)
            # the first generated isomorphic graph is identity
            isomorphic_graphs = isomorphic_graphs[1:]
            for ij, isomorph_graph in enumerate(isomorphic_graphs):
                job_lists_iso.append([isomorph_graph, n_vertices, f"isomorphGraph{ij}_{graph_to_string(graph)}"])

    all_jobs = job_lists_QAOA + job_lists_iso

    return all_jobs
def generate_jobs1(n_vertices, n_layers, n_steps, n_samples, n_isomorph_max, max_unlabeled_graph=None, max_job=None, graph_only=False):

    start_time = time.time()
    np.random.seed(42)

    if max_job == None or max_job == "None":
        max_job = -1


    unlabeled_graphs = graph_methods.generate_all_connected_graphs(
        n_vertices, True)
    print(f"graphs generated")

    unlabeled_graphs_graphs = [graph[0] for graph in unlabeled_graphs]
    unlabeled_graph_number = len(unlabeled_graphs_graphs)
    print(f"number of unlabeled graphs:  {unlabeled_graph_number}")

    if max_unlabeled_graph != None:  # limit to amount of unlabeled graph number if needed. TB Implemented: sampling according to weight
        unlabeled_graphs_graphs = unlabeled_graphs_graphs[:max_unlabeled_graph]

    if graph_only:
        all_jobs = generate_job_graphslist( unlabeled_graphs_graphs, n_vertices, n_isomorph_max)
    else:
        all_jobs = generate_job_list(unlabeled_graphs_graphs, n_layers, n_steps, n_samples, n_vertices, n_isomorph_max)
    
    print(f"jobs created, number of jobs:  {len(all_jobs)}. Max Jobs: {max_job} (no lim: -1)")
    job_count = 0

    if max_job != -1:  # limit to amount of graph number if needed. TB Implemented: sampling according to weight
        all_jobs = all_jobs[:max_job]

    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    return all_jobs

def generate_store_n1_jobs_from_graph(jobname, graph, n_jobs, n_layers = 3, n_samples=400):
    n1_jobs = generate_n1_jobs_from_graph(jobname, graph, n_jobs, n_layers = 3, n_samples=400)
    store_jobs(n1_jobs,jobname)

def generate_n1_jobs_from_graph(graph, n_jobs, n_layers = 3, n_samples=400):
    graph_str= graph_to_string(graph)
    graph = string_graph_to_graph(graph_str)
    max_vertice = max(max(edge) for edge in graph)
    new_vertice = max_vertice + 1
    n1_graphs = graph_methods.generate_all_n1_graphs_from_n_graph(graph)  
    n1_graphs_unlbl = [unlbl_graph for unlbl_graph in n1_graphs if unlbl_graph[0] == unlbl_graph[2]]
    n1_graphs_sorted = sorted(n1_graphs_unlbl, key=lambda x: x[1],reverse=True)
    n1_graphs_sorted_slice = n1_graphs_sorted[:n_jobs] 

    n1_graphs_sorted_slice_graphs = [list(item[0].edges()) for item in n1_graphs_sorted_slice]

    n1_jobs = generate_job_list(n1_graphs_sorted_slice_graphs,n_layers,None,n_samples,new_vertice,0)
    return n1_jobs

def job_multiprocess(n_vertices, n_layers, n_steps, n_samples, n_isomorph_max, max_unlabeled_graph=None, max_job=None, parallel_task=True):
        #multiprocessing doesnt work properly on slurm, do not use
    print("start of QAOA - job1 - multiprocessing")

    all_jobs = generate_jobs1(n_vertices, n_layers, n_steps, n_samples,
                              n_isomorph_max, max_unlabeled_graph=None, max_job=None)
    jobnames = get_job_names_from_parameters(n_vertices, n_layers, n_samples, n_isomorph_max, max_unlabeled_graph=None, max_job=None)
    results_list = execute_mp_jobs(all_jobs)
    process_results_save(results_list, jobnames)

def execute_mp_jobs(jobs, parallel_task=True):
    results_list = []
    if not parallel_task:
        print(f"jobs start not in parallel")
        results_list = run_jobs(jobs)
    else:
        print(f"jobs start in parallel")
        results_list = run_jobs_parallel(jobs)
    return results_list

def execute_single_job(job, mock = False):

    # [[[0, 1], [2, 3]], 4, 4, "QAOA", "unlabeledGraph_0123", 20, 100]
    # (graph,n_vertices, n_layers, cost_layer, label, n_steps = 30, n_samples = 200):
    if mock == False:
        result = execute_qaoa_subjob(
            job[0], job[1], job[2], job[3], job[4], job[5], job[6])
    else:
        result = ["mock0", "mock1", "mock2", "mock3", "mock4", "mock5", "mock6", "mock7", "mock8", "mock9", "mock10"]
    
    return result
    #jobname = get_result_name_from_job(job)
    #save_single_job_result(result, jobname)

def save_single_job_result(result, jobname):

    script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create the corresponding directory
    subdirectory = os.path.join(script_dir, "stored_job_results")

    # Create the subdirectory if it doesn't exist
    os.makedirs(subdirectory, exist_ok=True)

    json_file_path = os.path.join(subdirectory, f"{jobname}.json")
    
    with open(json_file_path, 'w') as f:
        json.dump(result, f)

def store_jobs(jobs, job_names):

    script_dir = os.path.dirname(os.path.abspath(__file__))
        
    # Create the corresponding directory
    subdirectory = os.path.join(script_dir, "stored_jobs")

    # Check if the subdirectory exists and create it if it doesn't
    if not os.path.exists(subdirectory):
        os.makedirs(subdirectory)

    json_file_path = os.path.join(subdirectory, f"{job_names}.json")
    with open(json_file_path, 'w') as f:
        json.dump(jobs, f)

def retrieve_single_job_result(resultname):

    script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create the corresponding directory
    subdirectory = os.path.join(script_dir, "stored_job_results")

    if not resultname.endswith('.json'):
        resultname = resultname + '.json'

    # Save the array to a .npy file
    full_path = os.path.join(subdirectory, resultname)

    if not os.path.exists(full_path):
        return

     # Open the JSON file and load the data
    with open(full_path, 'r') as f:
        data = json.load(f)

    return data

def retrieve_job_result_names_list(result_names):
    results_list = []
    for result_name in result_names:
        result = retrieve_single_job_result(result_name)
        results_list.append(result)
    return results_list

def create_joblist_from_jobgraphlist(all_jobs_graphs, n_layers, n_steps, n_samples):
    all_jobs = []
    for sublist in all_jobs_graphs:
        graph, n_vertices, label = sublist
        all_jobs.append([graph, n_vertices, n_layers,"QAOA", label, str(n_steps), n_samples])
        all_jobs.append([graph, n_vertices, n_layers,"fQAOA", label, str(n_steps), n_samples])
    return all_jobs

def job_execute_slurmarray(n_vertices, n_layers_array, n_steps=None, n_samples=400, n_isomorph_max=None, max_unlabeled_graph=None, max_job=None, task_id=None):
    #run for many layers sequentially for each graph
    
    job_graph_names = get_job_names_from_parameters_graphs(n_vertices, n_isomorph_max, max_unlabeled_graph, max_job)
    all_jobs_graphs = retrieve_stored_jobs(job_graph_names)

    for n_layers in n_layers_array:
    
        if task_id is None or task_id == -1:
            return

        all_jobs = create_joblist_from_jobgraphlist(all_jobs_graphs, n_layers, n_steps, n_samples)
        result = execute_slurmarray(all_jobs, task_id=task_id)

        jobname = get_result_name_from_job(all_jobs[task_id])
        save_single_job_result(result, jobname)

def job_slurm_execute_slurmarray_from_stored_job_graph_name(jobname, task_id, n_layers_array = [3], n_sample = 400):

    all_jobs_graphs = retrieve_stored_jobs(jobname)
    for n_layers in n_layers_array:

        if task_id is None or task_id == -1:
            return

        all_jobs = create_joblist_from_jobgraphlist(all_jobs_graphs, n_layers, n_steps = None, n_samples = n_sample)
        result = execute_slurmarray(all_jobs, task_id=task_id)
        save_single_job_result(result, jobname+"task"+str(task_id))

def job_execute_slurmarray_from_stored_job_name(jobname, task_id, add_last_job_column_to_result = True):

    all_jobs = retrieve_stored_jobs(jobname)
    if task_id is None or task_id == -1:
        return

    result = execute_slurmarray(all_jobs, task_id=task_id)

    if add_last_job_column_to_result:
        result = [result, all_jobs[task_id][-1]]

    save_single_job_result(result, jobname+"task"+str(task_id))

def job_execute_vertice_converge_job(n_layer = 3, n_sample = 400, n_graphs = 3):

    # start with this given graph 01_02_13_23 (the worst for fQAOA in previously obtained results)
    graph_4_vertices = [(0,1),(0,2),(1,3),(2,3)]

    current_time = datetime.now()
    formatted_time = current_time.strftime("%H%M%S")
    all_results = []
    continue_flag = True
    base_graph = graph_4_vertices.copy()
    vertices = 5
    while continue_flag:

        generate_store_n1_jobs_from_graph(f"job_vertices_{vertices}_layer_{n_layer}_reg_{formatted_time}",base_graph,n_graphs,n_layer,n_sample)

        job_ids = []
        job_name = f"job_vertices_{vertices}_layer_{n_layer}_reg_{formatted_time}"
        job_script = return_slurm_array_test_script_job_execute_slurmarray_from_job_name_string(job_name, n_layer, n_graphs, n_sample)
        job_id = submit_slurm_job(job_script)
        job_ids.append(job_id)

        while not check_job_id_state_completed_or_failed(job_ids): # continue check job state until it is completed
            time.sleep(60) #wait a min
        
        if check_job_id_state_failed(job_ids):
            break

        #retrieve job results
        result_list = []
        for task_number in range(n_graphs):
            result = retrieve_single_job_result(f"job_vertices_{vertices}_layer_{n_layer}_reg_+{formatted_time}task{str(task_number)}")
            result_list.append(result)
        
        all_results.append(result_list)
        
        #get the minimum mean for fQAOA 
        fQAOA_entries = [entry for entry in result_list if entry[0] == "fQAOA"]
        min_mean_entry = min(fQAOA_entries, key=lambda x: x[4], default=None)  # Using 'mean' at index 4
        result_graph_string = min_mean_entry[3] if min_mean_entry else "No entries found with cost_layer equal to 'fQAOA'."
        
        # get new graph
        next_graph = string_graph_to_graph(result_graph_string)
        base_graph = next_graph

        vertices += 1 #next vertice number

def execute_slurmarray(all_jobs, task_id=None):
    
    mock = False #set true in debug
    #mock = True

    if mock:
        task_ids = range(len(all_jobs))
    else:
        task_ids = [task_id]

    for task_id in task_ids:

        # the task id array is bigger than the number of jobs
        if task_id >= len(all_jobs):
            return

        result = execute_single_job(all_jobs[task_id], mock) #Remove
        return result
        #jobname = get_result_name_from_job(all_jobs[task_id])
        #save_single_job_result(result, jobname)

def job_retrieve_merge_results(n_vertices, n_layers, n_samples):
    result_name = []
    result_names= get_possible_jobnames_from_params(n_vertices, n_layers, n_samples)
    results_list = retrieve_result_list(result_names)
    return results_list

def retrieve_result_list(result_names):

    results_list = []
    for result_name in result_names:
        result = retrieve_single_job_result(result_name)

        if result is None:
            continue

        results_list.append(result)

    return results_list

def job_retrieve_merge_results_from_jobname(jobname):
    result_names = get_results_with_jobname([jobname])
    results_list = retrieve_result_list(result_names)
    process_results_save(results_list,jobname)

def job_process_results(n_vertices, n_layers, n_steps= None, n_samples = None, n_isomorph_max = None, max_unlabeled_graph = None, max_job = None):
    print(f"start of result merge vertice {n_vertices}, layers {str(n_layers)}, n_samples {n_samples}")
    # results_list = job1_retrieve_merge_results(n_vertices, n_layers, n_steps, n_samples, n_isomorph_max, max_unlabeled_graph, max_job)
    results = job_retrieve_merge_results(n_vertices = n_vertices, n_layers = n_layers, n_samples = n_samples)
    jobnames = get_job_names_from_parameters(n_vertices, n_layers, n_samples = n_samples, n_isomorph_max = n_isomorph_max, max_unlabeled_graph = max_unlabeled_graph, max_job = max_job)
    process_results_save(results, jobnames)
    print("result merged and saved")

def job_process_results_layers(n_vertices, layer_list, n_samples = 400, n_isomorph_max = None, max_unlabeled_graph = None, max_job = None):
    layers_string = "".join(str(x) for x in layer_list)
    print(f"start of result merge vertice {n_vertices}, layers {layers_string}, n_samples {n_samples}")

    results_all_layers = []
    for n_layer in layer_list:
        results = job_retrieve_merge_results(n_vertices = n_vertices, n_layers = n_layer, n_samples = n_samples)    
        results = [inner_list + [n_layer] for inner_list in results]
        results_all_layers.extend(results)
        

    results = merge_result_for_layers(results_all_layers)

    #["fQAOA", "unlabeledGraph__0123", "_0123", "_0123", "1.3675", "2", "2", "0.23", "3", "None", "1 mins 57 secs", "n_layer"]
    jobnames = get_job_names_from_parameters(n_vertices, layers_string, n_samples = n_samples, n_isomorph_max = n_isomorph_max, max_unlabeled_graph = max_unlabeled_graph, max_job = max_job)
    
    process_results_save(results, jobnames)

def merge_result_for_layers(data):
    grouped_data = {}

    # Iterate through the input data
    for entry in data:
        key = (entry[0], entry[1])  # Use the first and second elements as the key
        n_layer_value = str(entry[-1])  # Convert the last element (n_layer) to an integer
        fifth_element_value = str(entry[4])  # Convert the fifth element to a float

        # If the key does not exist in the dictionary, create it
        if key not in grouped_data:
            grouped_data[key] = {}

        # Store the fifth element value indexed by the n_layer value
        grouped_data[key][n_layer_value] = fifth_element_value

    # Construct the final list
    final_list = []
    for key, values in grouped_data.items():
        # Extract the first two elements
        result_entry = [key[0], key[1]]

        # Collect the fifth element values sorted by n_layer
        for n_layer in sorted(values.keys()):
            result_entry.append(values[n_layer])

        # Append the result entry to the final list
        final_list.append(result_entry)

    return final_list

def process_results_save(results_list, jobnames):

    script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create the corresponding directory
    subdirectory = os.path.join(script_dir, "merged_processed_results")

    # Ensure the subdirectory exists
    if not os.path.exists(subdirectory):
        os.makedirs(subdirectory)

    # Construct the file path
    file_path = os.path.join(subdirectory, jobnames + ".txt")

    with open(file_path, 'w') as f:
        json.dump(results_list, f)
    print("saved")

def job_generate_save_graphs(n_vertices, n_isomorph_max, max_unlabeled_graph=None, max_job=None):
    all_jobs_graphs = generate_jobs1(n_vertices, n_layers=None, n_steps=None, n_samples=None,n_isomorph_max=n_isomorph_max, max_unlabeled_graph=max_unlabeled_graph, max_job=max_job, graph_only=True)
    job_names_graph = get_job_names_from_parameters_graphs( n_vertices, n_isomorph_max, max_unlabeled_graph, max_job)
    store_jobs(all_jobs_graphs, job_names_graph)

def job_generate_save_graphs_vertice_sequence(initial_graph, jobname = "job-sequence", n_graphs =3, depth =3, n_layer = 3, n_samples = 400 ):
    n1_graphs_jobs = generate_n1_jobs_from_graph(initial_graph,n_graphs,n_layer,n_samples)
    [n1_graphs_job.append(graph_to_string(initial_graph)) for n1_graphs_job in n1_graphs_jobs]
    all_jobs = []
    all_jobs.extend(n1_graphs_jobs)
    for n1_graph_job in n1_graphs_jobs:
        graph = n1_graph_job[0]
        n2_graph_jobs = generate_n1_jobs_from_graph(graph,n_graphs,n_layer,n_samples)
        [n2_graph_job.append(graph_to_string(n1_graph_job[0])) for n2_graph_job in n2_graph_jobs]
        all_jobs.extend(n2_graph_jobs)

    store_jobs(all_jobs, jobname)

def get_possible_jobnames_from_params(n_vertices, n_layers, n_samples=400, n_steps=None):
    parameters = [f"vertices_{n_vertices}", f"layers_{n_layers}",f"samples_{n_samples}"]    
    return  get_results_with_jobname(parameters)

def get_results_with_jobname(parameters):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    subdirectory = os.path.join(script_dir, "stored_job_results")
    all_files = os.listdir(subdirectory)
    json_files = [f for f in all_files if f.endswith('.json')]
    filtered_files = [f for f in json_files if all(
        sub in f for sub in parameters)]
    return filtered_files

def test_slurm_state():
    job_script = return_slurm_array_test_script_string()
    print("job script obtained")
    job_id = submit_slurm_job(job_script)
    print("slurm job submitted")
    print("job_ids: " + job_id)
    job_states = check_job_id_state(job_id, True)
    print(job_states)
    while not check_job_id_state_completed_or_failed(job_id, True):
        print("waiting 30 sec")
        time.sleep(30)
