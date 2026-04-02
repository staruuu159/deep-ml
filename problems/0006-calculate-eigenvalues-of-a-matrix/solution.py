import numpy as np
def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	a1 = np.array(matrix)
	eigenvalues = np.linalg.eigvals(a1)
	return eigenvalues.tolist()