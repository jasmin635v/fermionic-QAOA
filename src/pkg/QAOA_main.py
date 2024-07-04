import QAOA_jobs

#QAOA_jobs.execute_qaoa_job1(n_vertices = 4, n_layers = 4, n_samples = 1, n_steps = 1, n_isomorph_max = 0, max_graph=1, parallel_task=True)
QAOA_jobs.execute_qaoa_job1(n_vertices = 4, n_layers = 4, n_samples = 100, n_steps = 20, n_isomorph_max = 3, parallel_task=True)