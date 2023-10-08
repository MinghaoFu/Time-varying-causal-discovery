import numpy as np
from regain.covariance import LatentTimeGraphicalLasso
from regain.datasets import make_dataset
from regain.utils import error_norm_time

np.random.seed(42)
data = make_dataset(n_dim_lat=1, n_dim_obs=3)
X = data.X
y = data.y
theta = data.thetas

mdl = LatentTimeGraphicalLasso(max_iter=50).fit(X, y)
print("Error: %.2f" % error_norm_time(theta, mdl.precision_))