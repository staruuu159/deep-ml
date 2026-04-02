import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
	A, T, S = np.array(A), np.array(T), np.array(S)
	if T.shape[0] != T.shape[1] or S.shape[0] != S.shape[1]:
		return -1
	rank1 = np.linalg.matrix_rank(T)
	rank2 = np.linalg.matrix_rank(S)
	if rank1 != T.shape[0] or rank2 != S.shape[0]:
		return -1
	transformed_matrix = np.linalg.inv(T) @ A @ S
	return transformed_matrix.tolist()