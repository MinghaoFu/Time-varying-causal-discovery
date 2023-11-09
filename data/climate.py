import pickle
import numpy as np

def generate_data(args):
    with open(args.data_path, 'rb') as file:
        data = pickle.load(file)
    d = 10#data.shape[1]
    
    data = data[:, :d]
    args.num = data.shape[0]
    m_true = generate_band_matrix(args.num, d, args.distance)
    args.d = d
    
    return data, m_true

def generate_band_matrix(n, d, bandwidth):
    matrix = np.zeros((d, d))
    for i in range(d):
        for j in range(max(0, i - bandwidth), min(d, i + bandwidth + 1)):
            if i != j:
                matrix[i, j] = 1
    return np.tile(matrix, (n, 1, 1))