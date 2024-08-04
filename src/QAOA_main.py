import QAOA_jobs
from QAOA_utils import parse_args
from multiprocessing import freeze_support

# needed for multiprocessing
if __name__ == '__main__':
    freeze_support() 
    args = parse_args()
    print(args)

    n_samples = args.n_samples
    n_layers = args.n_layers
    n_qubits = args.n_qubits
    n_steps = args.n_steps
    job_name = args.job_name
    task_id = args.task_id
    stored_job_name = args.stored_job_name
    start_graph = args.start_graph

    max_isomorph_number = args.max_isomorph_number
    if max_isomorph_number == -1 or max_isomorph_number == 0:
        max_isomorph_number = None

    max_job = args.max_job
    if max_job == -1 or max_job ==0:
        max_job = None

    maxunlblgraph = args.max_unlbl_graph
    if maxunlblgraph == -1 or maxunlblgraph == 0:
        maxunlblgraph = None


    #STEP 1 use this to create a list of graphs with n vertices / qbits that slurm will use
    #each graph will have n isomorphic graphs to unlabeled graphs (a graph that can be obtained by shuffling labels)
    #there is n max unlabeled graph and m max jobs (the generated array is sliced to m entries or max jobs)
    if job_name == "job_generate_graphs":
        QAOA_jobs.job_generate_save_graphs(n_vertices=n_qubits, n_isomorph_max=max_isomorph_number, max_unlabeled_graph=maxunlblgraph, max_job=max_job)

    if job_name == "job_generate_graphs_vertice_sequence":
        QAOA_jobs.job_generate_save_graphs_vertice_sequence(start_graph,stored_job_name)

    #STEP 2: run this from slurm with a slurm array. max_unlabeled_graph, n_isomorph_max and max_job parameters will fetch the previously created graph list
    #use the size of slurm array corresponding to the number of jobs
    elif job_name == "job_execute_slurmarray":
        QAOA_jobs.job_execute_slurmarray(n_vertices=n_qubits, n_layers_array=n_layers, n_samples=400, n_steps="None", max_unlabeled_graph=maxunlblgraph, n_isomorph_max=max_isomorph_number, max_job=max_job, task_id=task_id)

    elif job_name == "job_slurm_execute_slurmarray_from_job_graph_name": #used internally to spawn jobs
        QAOA_jobs.job_slurm_execute_slurmarray_from_stored_job_graph_name(stored_job_name, task_id, n_layers_array = [3], n_sample = 400)

    elif job_name == "job_execute_slurm_array_from_jobname":
        QAOA_jobs.job_execute_slurmarray_from_stored_job_name(stored_job_name, task_id)
    
    elif job_name == "job_vertices_converge_job": #Work in progress
        QAOA_jobs.job_execute_vertice_converge_job(n_layer = [3], n_sample = 400)
    
    elif job_name == "job_process_results":
        QAOA_jobs.job_process_results(n_vertices=n_qubits, n_layers=n_layers, n_samples=400)

    elif job_name == "job_process_results_from_jobname":
        QAOA_jobs.job_retrieve_merge_results_from_jobname(job_name)

    elif job_name == "job_process_results_layers":
        QAOA_jobs.job_process_results_layers(n_vertices=n_qubits, layer_list=n_layers, n_samples=400)
   
    elif job_name == "job_multiprocess":  # with multiprocessing (too slow on slurm)
        QAOA_jobs.job_multiprocess(n_vertices=n_qubits, n_layers=n_layers, n_samples=n_samples, n_steps=n_steps, n_isomorph_max=max_isomorph_number, max_job=max_job)

    elif job_name == "test_slurm_state":
        QAOA_jobs.test_slurm_state()
