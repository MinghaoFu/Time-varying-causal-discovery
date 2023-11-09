import torch
import torch.nn as nn
import networkx as nx
import numpy as np
import itertools

from .base import check_tensor

def sample_n_different_integers(n, low, high, random_seed=None):
    # Create a random number generator with a specified random seed (or without)
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng()

    # Check if the interval contains enough unique integers
    if high - low < n:
        raise ValueError("Interval does not contain enough unique integers.")

    # Create an array of all integers in the interval
    all_integers = np.arange(low, high)

    # Shuffle the integers and take the first 'n' as the sample
    rng.shuffle(all_integers)
    sampled_integers = all_integers[:n]
    
    return sampled_integers

def top_k_abs_tensor(tensor, k):
    d = tensor.shape[0]
    abs_tensor = torch.abs(tensor)
    _, indices = torch.topk(abs_tensor.view(-1), k)
    
    flat_tensor = tensor.view(-1)
    flat_zero_tensor = torch.zeros_like(flat_tensor)
    flat_zero_tensor[indices] = flat_tensor[indices]
    
    zero_tensor = check_tensor(flat_zero_tensor.view(d, d))
    
    
    # batch_size, d, _ = tensor.shape
    # values, indices = torch.topk(tensor.view(batch_size, -1), k=k, dim=-1)
    # result = torch.zeros_like(tensor).view(batch_size, -1)
    # result.scatter_(1, indices, values)
    # result = result.view(batch_size, d, d)
    return zero_tensor

def is_dag(B):
    """Check whether B corresponds to a DAG.

    Args:
        B (numpy.ndarray): [d, d] binary or weighted matrix.
    """
    return nx.is_directed_acyclic_graph(nx.DiGraph(B))

def threshold_till_dag(B):
    """Remove the edges with smallest absolute weight until a DAG is obtained.

    Args:
        B (numpy.ndarray): [d, d] weighted matrix.

    Returns:
        numpy.ndarray: [d, d] weighted matrix of DAG.
        float: Minimum threshold to obtain DAG.
    """
    if is_dag(B):
        return B, 0

    B = np.copy(B)
    # Get the indices with non-zero weight
    nonzero_indices = np.where(B != 0)
    # Each element in the list is a tuple (weight, j, i)
    weight_indices_ls = list(zip(B[nonzero_indices],
                                 nonzero_indices[0],
                                 nonzero_indices[1]))
    # Sort based on absolute weight
    sorted_weight_indices_ls = sorted(weight_indices_ls, key=lambda tup: abs(tup[0]))

    for weight, j, i in sorted_weight_indices_ls:
        if is_dag(B):
            # A DAG is found
            break

        # Remove edge with smallest absolute weight
        B[j, i] = 0
        dag_thres = abs(weight)

    return B, dag_thres

def postprocess(B, graph_thres=0.3):
    """Post-process estimated solution:
        (1) Thresholding.
        (2) Remove the edges with smallest absolute weight until a DAG
            is obtained.

    Args:
        B (numpy.ndarray): [d, d] weighted matrix.
        graph_thres (float): Threshold for weighted matrix. Default: 0.3.

    Returns:
        numpy.ndarray: [d, d] weighted matrix of DAG.
    """
    B = np.copy(B)

    B[np.abs(B) <= graph_thres] = 0    # Thresholding
    B, _ = threshold_till_dag(B)

    return B

def is_dag(B):
    """Check whether B corresponds to a DAG.

    Args:
        B (numpy.ndarray): [d, d] binary or weighted matrix.
    """
    return nx.is_directed_acyclic_graph(nx.DiGraph(B))

def bin_mat(B):
    return np.where(B != 0, 1, 0)

def is_markov_equivalent(B1, B2):
    '''
        Judge whether two matrices have the same v-structures. a->b<-c
        row -> col
    '''
    def find_v_structures(adj_matrix):
        G = nx.DiGraph(adj_matrix)
        v_structures = []

        for node in G.nodes():
            parents = list(G.predecessors(node))
            for pair in itertools.combinations(parents, 2):
                if not G.has_edge(pair[0], pair[1]) and not G.has_edge(pair[1], pair[0]):
                    v_structures.append((pair[0], node, pair[1]))

        return v_structures
    
    def find_skeleton(B, v_structures):
        inds = []
        for v in v_structures:
            inds.extend([(v[0], v[1]), (v[1], v[0]), (v[2], v[1]), (v[1], v[2])])
        inds = np.array(inds)
        B[inds[:, 0], inds[:, 1]] = 0
        
        return np.logical_or(B, B.T).astype(int)
    
    B1 = bin_mat(B1)
    B2 = bin_mat(B2)
    v_structures1 = find_v_structures(B1)
    v_structures2 = find_v_structures(B2)
    sk1 = find_skeleton(B1, v_structures1)
    sk2 = find_skeleton(B2, v_structures1)

    return set(v_structures1) == set(v_structures2) and sk1 == sk2

def random_zero_array(arr, zero_ratio, constraint=None):
    '''
        Randomly set some elements in an array to 0
    '''
    if constraint is None:
        original_shape = arr.shape
        arr = arr.flatten()
        inds = np.random.choice(np.arange(len(arr)), size=int(len(arr) * zero_ratio), replace=False)
        arr[inds] = 0
        result = arr.reshape(original_shape)
    return result
