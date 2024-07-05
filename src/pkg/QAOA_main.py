import QAOA_jobs
from multiprocessing import freeze_support

#needed for multiprocessing
if __name__ == '__main__':
    freeze_support() 

    #QAOA_jobs.execute_qaoa_job1(n_vertices = 4, n_layers = 4, n_samples = 1, n_steps = 1, n_isomorph_max = 0, max_job=2, parallel_task=True)
    QAOA_jobs.execute_qaoa_job1(n_vertices = 4, n_layers = 4, n_samples = 100, n_steps = 20, n_isomorph_max = 3, parallel_task=True)