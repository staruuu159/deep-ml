import numpy as np
def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	a1 = np.array(matrix)
	if mode == 'row':
		means = a1.mean(axis = 1).tolist()
	if mode == 'column':
		means = a1.mean(axis = 0).tolist()
	return means