import numpy as np
def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	np_a = np.array(a)
	np_b = np.array(b)
	if np_a.shape[1] == np_b.shape[0]:
		a1 = np_a @ np_b
		return a1.tolist()
	else:
		return -1
	pass