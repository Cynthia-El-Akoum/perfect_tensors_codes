import numpy as np

def is_rank1_tensor(T, tol=1e-8):
    """Checks if a tensor is rank-1 (decomposable into a single tensor product)."""
    dims = T.shape # Gives the shape of the tensor as a tuple where each element corresponds to the dimension of a subspace, e.g. (3,3,3,3)
    ndim = len(dims) # Defines the order of the tensor i.e. the number of its indices

    # Generates all possible flattenings
    for i in range(ndim):
        # Flatten all dimensions except the i-th
        remaining_dims = [j for j in range(ndim) if j != i]# Clarifies what dimensions are being grouped
        flattened = np.reshape(T, (dims[i], -1))  # Reshapes into a matrix, -1 automatically computes the size of the dimensions in remaining_dims

        # Computes matrix rank (number of singular values above tolerance)
        s = np.linalg.svd(flattened, compute_uv=False)
        rank = np.sum(s > tol)

        if rank != 1:
            return False
    return True
