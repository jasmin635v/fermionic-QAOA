
import QAOA_jobs
from QAOA_utils import parse_args
from multiprocessing import freeze_support

#needed for multiprocessing
if __name__ == '__main__':
    freeze_support() 
    args = parse_args()

    n_samples = args.n_samples  # Assuming you want the first value if nargs="+"
    n_layers = args.n_layers    # Assuming you want the first value if nargs="+"
    n_qubits = args.n_qubits
    n_steps = args.n_steps
    max_isomorph_number = args.max_isomorph_number
    max_job = args.max_job


    #QAOA_jobs.execute_qaoa_job1(n_vertices = 4, n_layers = 4, n_samples = 1, n_steps = 1, n_isomorph_max = 0, max_unlabeled_graph=1, parallel_task=True)
    #QAOA_jobs.execute_qaoa_job1(n_vertices = 4, n_layers = 4, n_samples = 100, n_steps = 20, n_isomorph_max = 3, parallel_task=True)
    QAOA_jobs.execute_qaoa_job1(n_vertices = n_qubits, n_layers = n_layers, n_samples = n_samples, n_steps = n_steps, n_isomorph_max = max_isomorph_number, max_job=max_job, parallel_task=True)