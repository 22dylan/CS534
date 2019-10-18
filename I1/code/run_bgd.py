from BatchGradientDescent import bgd

def run_bgd(data, step_size_vals, lambda_vals):
	w = {}
	iterations = {}
	for step_size in step_size_vals:
		w[step_size] = {}
		iterations[step_size] = {}
		for lambda_reg in lambda_vals:
			w[step_size][lambda_reg], iterations[step_size][lambda_reg] =  bgd(lambda_reg, step_size,data)
	return w, iterations