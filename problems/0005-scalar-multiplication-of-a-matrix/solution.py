import numpy as np
def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
	# Your code here
	a1 = np.array(matrix)
	result =  a1 * scalar
	return result.tolist()
	pass