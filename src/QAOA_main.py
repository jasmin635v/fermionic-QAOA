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
    max_isomorph_number = args.max_isomorph_number
    max_job = args.max_job
    job_name = args.job_name
    task_id = args.task_id
    mock = args.mock
    maxunlblgraph = args.max_unlbl_graph

    if job_name == "printargs":
        print("parsed max_unlabeled_graphs " + str(maxunlblgraph))

    if job_name == "job1":  # with multiprocessing (too slow on slurm)
        QAOA_jobs.job1(n_vertices=n_qubits, n_layers=n_layers, n_samples=n_samples, n_steps=n_steps, n_isomorph_max=max_isomorph_number, max_job=max_job)

    elif job_name == "job1_generate_graphs":
        QAOA_jobs.job1_generate_save_graphs(n_vertices=n_qubits, n_isomorph_max=max_isomorph_number, max_job=max_job)

    elif job_name == "job2_generate_graphs":
        QAOA_jobs.job2_generate_save_graphs(n_vertices=n_qubits, max_unlabeled_graph=maxunlblgraph)

    elif job_name == "job1_generate":
        QAOA_jobs.job1_generate_save_jobs(n_vertices=n_qubits, n_layers=n_layers, n_samples=n_samples,n_steps=n_steps, n_isomorph_max=max_isomorph_number, max_job=max_job)

    elif job_name == "job1_execute_slurmarray":
        QAOA_jobs.job1_execute_slurmarray(n_vertices=n_qubits, n_layers=n_layers, n_samples=n_samples,n_steps=n_steps, n_isomorph_max=max_isomorph_number, max_job=max_job, task_id=task_id, mock=mock)
    
    elif job_name == "job2_execute_slurmarray":
        QAOA_jobs.job1_execute_slurmarray(n_vertices=n_qubits, n_layers=n_layers, n_samples=n_samples, n_steps="None", max_unlabeled_graph=maxunlblgraph, n_isomorph_max=None, max_job=None, task_id=task_id, mock=mock)

    elif job_name == "job1_process_results":
        QAOA_jobs.job1_process_results(n_vertices=n_qubits, n_layers=n_layers, n_samples=n_samples, n_steps=n_steps, n_isomorph_max=max_isomorph_number, max_job=max_job)
    
    else:
        QAOA_jobs.job1(n_vertices=n_qubits, n_layers=n_layers, n_samples=n_samples,n_steps=n_steps, n_isomorph_max=max_isomorph_number, max_job=max_job)
