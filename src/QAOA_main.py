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

    max_isomorph_number = args.max_isomorph_number
    if max_isomorph_number == -1:
        max_isomorph_number = None

    max_job = args.max_job
    if max_job == -1:
        max_job = None

    maxunlblgraph = args.max_unlbl_graph
    if maxunlblgraph == -1:
        maxunlblgraph = None

    if job_name == "job_generate_graphs":
        QAOA_jobs.job_generate_save_graphs(n_vertices=n_qubits, n_isomorph_max=max_isomorph_number, max_unlabeled_graph=maxunlblgraph, max_job=max_job)

    elif job_name == "job_execute_slurmarray":
        QAOA_jobs.job_execute_slurmarray(n_vertices=n_qubits, n_layers_array=n_layers, n_samples=n_samples, n_steps="None", max_unlabeled_graph=maxunlblgraph, n_isomorph_max=None, max_job=None, task_id=task_id)

    elif job_name == "job1_process_results":
        QAOA_jobs.job1_process_results(n_vertices=n_qubits, n_layers=n_layers, n_samples=n_samples, n_steps=n_steps, n_isomorph_max=max_isomorph_number, max_job=max_job)
    
    elif job_name == "job2_process_results":
        QAOA_jobs.job2_process_results(n_vertices=n_qubits, n_layers=n_layers, n_samples=n_samples)
    
    elif job_name == "job_multiprocess":  # with multiprocessing (too slow on slurm)
        QAOA_jobs.job_multiprocess(n_vertices=n_qubits, n_layers=n_layers, n_samples=n_samples, n_steps=n_steps, n_isomorph_max=max_isomorph_number, max_job=max_job)


