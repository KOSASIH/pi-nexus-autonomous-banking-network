# math_utils.py

import math

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_angle(x1, y1, x2, y2):
    return math.atan2(y2 - y1, x2 - x1)

def calculate_matrix_inverse(matrix):
    return np.linalg.inv(matrix)

def calculate_eigenvalues(matrix):
    return np.linalg.eigvals(matrix)
