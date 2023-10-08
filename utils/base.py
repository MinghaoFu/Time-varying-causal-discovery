import numpy as np
import pandas as pd
import os
import warnings
import torch
import itertools
import pytest
import shutil

from torchsummary import summary
from sklearn.linear_model import LinearRegression, Lasso
from torch.utils.data import Dataset
from sklearn.decomposition import PCA, FastICA, fastica
from sklearn.decomposition._fastica import _gs_decorrelation
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import assert_allclose
from scipy import linalg, stats
from tqdm import tqdm
from time import time

def makedir(path, remove_exist=False):
    if remove_exist and os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def check_tensor(data):
    if not torch.is_tensor(data):
        if isinstance(data, np.ndarray):
            data = torch.tensor(data)
        elif isinstance(data, (list, tuple)):
            data = torch.tensor(np.array(data))
        else:
            raise ValueError("Unsupported data type. Please provide a list, NumPy array, or PyTorch tensor.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return data.to(device, dtype=torch.float)

def covariance(x):
    mean = np.mean(x, axis=-1, keepdims=True)
    n = np.shape(x)[1] - 1
    m = x - mean

    return (m.dot(m.T))/n

def center_and_norm(x, axis=-1):
    """Centers and norms x **in place**

    Parameters
    -----------
    x: ndarray
        Array with an axis of observations (statistical units) measured on
        random variables.
    axis: int, optional
        Axis along which the mean and variance are calculated.
    """
    x = np.rollaxis(x, axis)
    x -= x.mean(axis=0)
    x /= x.std(axis=0)

def generate_six_nodes_DAG(random_state, x_size=1000):
    x3 = random_state.uniform(size=x_size)
    x0 = 3.0 * x3 + random_state.uniform(size=x_size)
    x2 = 6.0 * x3 + random_state.uniform(size=x_size)
    x1 = 3.0 * x0 + 2.0 * x2 + random_state.uniform(size=x_size)
    x5 = 4.0 * x0 + random_state.uniform(size=x_size)
    x4 = 8.0 * x0 - 1.0 * x2 + random_state.uniform(size=x_size)
    X = np.array([x0, x1, x2, x3, x4, x5]).T
    return X

def dict_to_class(**dict):
    class _dict_to_class:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    return _dict_to_class(**dict)
        
def test_fastica_simple(add_noise, global_random_state, global_dtype, mixing):
    # Test the FastICA algorithm on very simple data.
    rng = global_random_state
    n_samples = 1000
    # Generate two sources:
    s1 = (2 * np.sin(np.linspace(0, 100, n_samples)) > 0) - 1
    s2 = stats.t.rvs(1, size=n_samples, random_state=global_random_state)
    s = np.c_[s1, s2].T
    center_and_norm(s)
    s = s.astype(global_dtype)
    s1, s2 = s

    # Mixing angle
    phi = 0.6
    mixing = np.array([[np.cos(phi), np.sin(phi)], [np.sin(phi), -np.cos(phi)]])
    mixing = mixing.astype(global_dtype)
    m = np.dot(mixing, s)

    if add_noise:
        m += 0.1 * rng.randn(2, 1000)

    center_and_norm(m)

    # function as fun arg
    def g_test(x):
        return x**3, (3 * x**2).mean(axis=-1)

    algos = ["parallel", "deflation"]
    nls = ["logcosh", "exp", "cube", g_test]
    whitening = ["arbitrary-variance", "unit-variance", False]
    for algo, nl, whiten in itertools.product(algos, nls, whitening):
        if whiten:
            k_, mixing_, s_ = fastica(
                m.T, fun=nl, whiten=whiten, algorithm=algo, random_state=rng
            )
            with pytest.raises(ValueError):
                fastica(m.T, fun=np.tanh, whiten=whiten, algorithm=algo)
        else:
            pca = PCA(n_components=2, whiten=True, random_state=rng)
            X = pca.fit_transform(m.T)
            k_, mixing_, s_ = fastica(
                X, fun=nl, algorithm=algo, whiten=False, random_state=rng
            )
            with pytest.raises(ValueError):
                fastica(X, fun=np.tanh, algorithm=algo)
        s_ = s_.T
        # Check that the mixing model described in the docstring holds:
        if whiten:
            # XXX: exact reconstruction to standard relative tolerance is not
            # possible. This is probably expected when add_noise is True but we
            # also need a non-trivial atol in float32 when add_noise is False.
            #
            # Note that the 2 sources are non-Gaussian in this test.
            atol = 1e-5 if global_dtype == np.float32 else 0
            assert_allclose(np.dot(np.dot(mixing_, k_), m), s_, atol=atol)

        center_and_norm(s_)
        s1_, s2_ = s_
        # Check to see if the sources have been estimated
        # in the wrong order
        if abs(np.dot(s1_, s2)) > abs(np.dot(s1_, s1)):
            s2_, s1_ = s_
        s1_ *= np.sign(np.dot(s1_, s1))
        s2_ *= np.sign(np.dot(s2_, s2))

        # Check that we have estimated the original sources
        if not add_noise:
            assert_allclose(np.dot(s1_, s1) / n_samples, 1, atol=1e-2)
            assert_allclose(np.dot(s2_, s2) / n_samples, 1, atol=1e-2)
        else:
            assert_allclose(np.dot(s1_, s1) / n_samples, 1, atol=1e-1)
            assert_allclose(np.dot(s2_, s2) / n_samples, 1, atol=1e-1)

    # Test FastICA class
    _, _, sources_fun = fastica(
        m.T, fun=nl, algorithm=algo, random_state=global_random_seed
    )
    ica = FastICA(fun=nl, algorithm=algo, random_state=global_random_seed)
    sources = ica.fit_transform(m.T)
    assert ica.components_.shape == (2, 2)
    assert sources.shape == (1000, 2)

    assert_allclose(sources_fun, sources)
    # Set atol to account for the different magnitudes of the elements in sources
    # (from 1e-4 to 1e1).
    atol = np.max(np.abs(sources)) * (1e-5 if global_dtype == np.float32 else 1e-7)
    assert_allclose(sources, ica.transform(m.T), atol=atol)

    assert ica.mixing_.shape == (2, 2)

    ica = FastICA(fun=np.tanh, algorithm=algo)
    with pytest.raises(ValueError):
        ica.fit(m.T)

def linear_regression_initialize(x: np.array, distance, ):
    model = LinearRegression()
    n, d = x.shape
    B_init = np.zeros((d, d))
    for i in range(d):
        start = max(i - distance, 0)
        end = min(i + distance + 1, d)

        model.fit(np.concatenate((x[:, start : i], x[:, i + 1 : end]), axis=1), x[:, i])
        B_init[i][start : i] = model.coef_[ : min(i, distance)]#np.pad(np.insert(model.coef_, min(i, distance), 0.), (start, d - end), 'constant')
        B_init[i][i + 1 : end] = model.coef_[min(i, distance) : ]
    
    return B_init

def model_info(model, input_shape):
    summary(model, input_shape)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"--- Total Parameters: {total_params}, Trainable Parameters: {trainable_params}")
    
    input_data = check_tensor(torch.randn([1] + list(input_shape)))
    start_time = time.time()
    with torch.no_grad():
        _ = model(input_data)
    inference_time = time.time() - start_time
    
    print(f"Inference Time: {inference_time * 1e3:.2f} ms")
    
def save_epoch_log(args, model, m_true, X, T_tensor, epoch):
    model.eval()

    row_indices, col_indices = np.nonzero(m_true[0])
    edge_values = m_true[:, row_indices, col_indices]
    values = []
    values_true =[]
    for _ in range(len(edge_values)):
        values.append([])
        values_true.append([])
    for k in range(args.num):
        B = model(T_tensor[k:k+1])
        B = B.view(X.shape[1], X.shape[1]).cpu().detach().numpy()
        
        for idx, (i, j) in enumerate(zip(row_indices, col_indices)):
            values[idx].append(B[i, j])
            values_true[idx].append(m_true[k][i, j])

    save_epoch_dir = os.path.join(args.save_dir, f'epoch_{epoch}')
    makedir(save_epoch_dir)
    print(f'--- Save figures...')
    for idx, (i, j) in enumerate(zip(row_indices, col_indices)):
        plt.plot(values[idx], label='Pred' + str(idx))
        if args.synthetic:
            plt.plot(values_true[idx], label = 'True' + str(idx))
        plt.legend() 
        plt.savefig(os.path.join(save_epoch_dir, f'({i}, {j})_trend.png'), format='png')
        plt.show()
        plt.clf()

    print('-- Save results...')
    np.save(os.path.join(save_epoch_dir, 'prediction.npy'), np.round(model.Bs, 4))
    np.save(os.path.join(save_epoch_dir, 'ground_truth.npy'), np.round(m_true, 4))
    
    print('-- Save gradient changes...')
    plt.plot(model.gradient)
    plt.title('Gradient')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()
    plt.savefig(os.path.join(save_epoch_dir, 'gradient_change.png'), format='png')