import QAOA_jobs
from QAOA_utils import parse_args
from multiprocessing import freeze_support

if __name__ == '__main__':

    freeze_support() # needed for multiprocessing
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

    print(job_name)

    #STEP 1: generate the jobs
    # Generate jobs with for given parameters
    if job_name == "job_generate_graphs": 
        QAOA_jobs.job_generate_save_graphs(n_vertices=n_qubits, n_isomorph_max=max_isomorph_number, max_unlabeled_graph=maxunlblgraph, max_job=max_job)

    # Generate sequence of jobs for vertice sequence from a given start_graph
    if job_name == "job_generate_graphs_sequence":
        QAOA_jobs.job_generate_save_graphs_vertice_sequence(start_graph,stored_job_name)

    # Generate for all isomorphic graph of 5 and 6 vertices
    if job_name == "job_generate_graphs_vertice56":
        QAOA_jobs.job_generate_save_graphs_vertice56(stored_job_name)

    #STEP 2: run this from slurm with a slurm array. max_unlabeled_graph, n_isomorph_max and max_job parameters will fetch the previously created graph list
    #use the size of slurm array corresponding to the number of jobs

    #run the job corresponding to the parameters
    elif job_name == "job_execute_slurmarray":
        QAOA_jobs.job_execute_slurmarray(n_vertices=n_qubits, n_layers_array=n_layers, n_samples=400, n_steps="None", max_unlabeled_graph=maxunlblgraph, n_isomorph_max=max_isomorph_number, max_job=max_job, task_id=task_id)

    #run the job from the given name
    elif job_name == "job_slurm_execute_slurmarray_from_jobname":
        QAOA_jobs.job_slurm_execute_slurmarray_from_stored_job_graph_name(stored_job_name, task_id, n_layers, n_samples)

    #STEP 3: merge individual results in stored_job_results

    #merge individual results with given parameters
    elif job_name == "job_merge_results":
        QAOA_jobs.job_process_results(n_vertices=n_qubits, n_layers=n_layers, n_samples=400)

    #merge individual results having job name in individual results
    elif job_name == "job_merge_results_from_jobname":
        QAOA_jobs.job_retrieve_merge_results_from_jobname(stored_job_name)
    
    #STEP 4 (facultative): further process merged results

    #Get fQAOA and QAOA mean on the same entry and root graph (need additional column from job for root graph)
    elif job_name == "job_process_merged_sequence_results_from_jobname":
        QAOA_jobs.job_process_merged_sequence_results_from_jobname(stored_job_name)

    #Merge same results for multiple layers
    elif job_name == "job_process_results_layers":
        QAOA_jobs.job_process_results_layers(n_vertices=n_qubits, layer_list=n_layers, n_samples=400)
   
    #Old code (generate and run graph with multiprocessing)
    elif job_name == "job_multiprocess":  # with multiprocessing (too slow on slurm)
        QAOA_jobs.job_multiprocess(n_vertices=n_qubits, n_layers=n_layers, n_samples=n_samples, n_steps=n_steps, n_isomorph_max=max_isomorph_number, max_job=max_job)

