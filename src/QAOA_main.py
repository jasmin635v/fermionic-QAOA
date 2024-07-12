
import QAOA_jobs
from QAOA_utils import parse_args
from multiprocessing import freeze_support

#needed for multiprocessing
if __name__ == '__main__':
    freeze_support() 
    args = parse_args()

    n_samples = args.n_samples  
    n_layers = args.n_layers   
    n_qubits = args.n_qubits
    n_steps = args.n_steps
    max_isomorph_number = args.max_isomorph_number
    max_job = args.max_job
    job_name = args.job_name
    task_id = args.task_id

    if job_name == "job1": #run it without slurm array but multiprocessing (will not work on slurm, too slow?)
        QAOA_jobs.job1(n_vertices = n_qubits, n_layers = n_layers, n_samples = n_samples, n_steps = n_steps, n_isomorph_max = max_isomorph_number, max_job=max_job)
    elif job_name == "job1_generate":
        QAOA_jobs.job1_generate_save_jobs(n_vertices = n_qubits, n_layers = n_layers, n_samples = n_samples, n_steps = n_steps, n_isomorph_max = max_isomorph_number, max_job=max_job)
    elif job_name == "job1_execute_slurmarray":
        QAOA_jobs.job1_execute_slurmarray(n_vertices = n_qubits, n_layers = n_layers, n_samples = n_samples, n_steps = n_steps, n_isomorph_max = max_isomorph_number, max_job=max_job, task_id=task_id)
    elif job_name == "job1_process_results":
        QAOA_jobs.job1_process_results(n_vertices = n_qubits, n_layers = n_layers, n_samples = n_samples, n_steps = n_steps, n_isomorph_max = max_isomorph_number, max_job=max_job)
    else:
        QAOA_jobs.job1(n_vertices = n_qubits, n_layers = n_layers, n_samples = n_samples, n_steps = n_steps, n_isomorph_max = max_isomorph_number, max_job=max_job)
    
