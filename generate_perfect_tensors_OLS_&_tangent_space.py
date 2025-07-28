import numpy as np
from numpy.linalg import pinv, norm
from scipy.linalg import expm  # Imports the matrix exponential
from scipy.linalg import null_space # Computes the kernel of a map

# Kronecker delta
def kronecker_delta(i, j):
    return int(i == j)

# Converts an OLS to a tensor
def OLS_to_tensor(OLS, d):
    """
    Converts an OLS into a tensor.
    Parameters:
        - OLS: the orthogonal latin square defined as a dxd table;
        - d: the order of the OLS.
    Returns:
        Φ: tensor with the shape (d,d,d,d)
    """
    Phi = np.zeros((d, d, d, d), dtype=complex)
    for i in range(d):
        for j in range(d):
            for k in range(d):
                for l in range(d):
                    Phi[i, j, k, l] = kronecker_delta(OLS[k, l][0], i) * kronecker_delta(OLS[k, l][1], j)
    return Phi

# Checks if a tensor corresponds to an OLS and computes it in the latter case
def tensor_to_OLS(Phi, d):
    """
    Attempt to extract OLS from a 4D tensor Φ.
    Parametrs:
        -Φ: a (d,d,d,d) tensor;
        -d: the dimension of each entry of Φ.
    Returns:
        - OLS if Φ corresponds to valid OLS;
        - None otherwise.
    """
    # Verifies tensor has binary 0/1 values
    if not np.all((np.abs(Phi) == 0) or (np.abs(Phi) == 1)):
        return None

    # Initializes OLS structure (d×d array of tuples)
    OLS = np.empty((d, d), dtype=object)

    # For each (k,l) position, finds the unique (i,j) where Phi[i,j,k,l] = 1
    for k in range(d):
        for l in range(d):
            matches = np.where(np.abs(Phi[:, :, k, l]) > 0.5) # Uses mid-point 0.5 to distingues efficiently between 0 and 1 with floating-points
            if len(matches[0]) != 1 or len(matches[1]) != 1:
                return None  # Not exactly one 1 in the slice
            i, j = matches[0][0], matches[1][0]
            OLS[k, l] = (i, j)

    # Verifies orthogonality
    # Converts OLS to two separate squares L1 and L2
    L1 = np.array([[OLS[k,l][0] for l in range(d)] for k in range(d)])
    L2 = np.array([[OLS[k,l][1] for l in range(d)] for k in range(d)])

    # Checks all pairs (L1,L2) are orthogonal
    pairs = set() # Uses a set and not a list because it doesn't automatically remove duplicates and is hence useful for uniqueness check
    for k in range(d):
        for l in range(d):
            pair = (L1[k,l], L2[k,l])
            if pair in pairs: # Checks for repeated pairs
                return None  # Repeated pair i.e not orthogonal
            pairs.add(pair)

    return OLS


# Flattening map
def flatten(X, input_dims, output_dims, d):
    """
    Flattens the tensor given a bipartition.
    Parametrs:
        - X: tensor of shape (d,d,d,d);
        - input_dims: tuple or list with the input dimensions;
        - output_dims: tuple or list with the output dimensions;
        - d: dimension of the subsytems.
    Returns:
        The flattened matrix X
    """
    return X.transpose(*input_dims, *output_dims).reshape(d*d, d*d)

# Inverse flattening map
def flatten_inv(flattened, input_dims, output_dims, d):
    """
    Reshapes a matrix into a tensor given a bipartition.
    Parametrs:
        - flattened: a d**2 x d**2 matrix;
        - input_dims: tuple or list with the input dimensions;
        - output_dims: tuple or list with the output dimensions;
        - d: dimension of the subsytems.
    Returns:
        The restored tensor
    """
    # Step 1: Reshape back to (d, d, d, d)
    intermediate = flattened.reshape(d, d, d, d)

    # Step 2: Reverse the transpose
    dims_permutation = list(input_dims) + list(output_dims)
    original_order = np.argsort(dims_permutation)
    restored = intermediate.transpose(*original_order)

    return restored

# Computes the exponential map at F_i(Phi)
def exp_map(Phi, input_dims, output_dims, d, X):
    """
    Computes the exponential map at F_i(Phi) using: exp_F_i(Φ)(X) = F_i(Φ).exp_Id(F^{−1}_i(Φ) X).
    Treats both cases of symbolic and numeric argument X.
    Parameters:
        - Φ: tensor of shape (d,d,d,d);
        - input_dims: tuple or list with the input dimensions;
        - output_dims: tuple or list with the output dimensions;
        - d: dimension of the subsytems;
        - X: the mtrix argument of the exponenetial.
    Returns:
        The matrix image of X via the exponential map.
    """
    # Computes the falttened matrix of the tensor Phi
    F_Phi = flatten(Phi, input_dims, output_dims, d)
    F_Phi_inv = pinv(F_Phi) # Computes the inverse matrix
    if isinstance(X, (sp.Matrix, np.matrix)): # Checks if X is symbolic
        X_np = np.array(X.tolist(), dtype=complex) # Transforms it back into a numpy array
    else:
        X_np = X # Does nothing for a numerical X
    exp_part = expm(F_Phi_inv @ X_np) # Computes the exponenetial part in the expression seperately
    return F_Phi @ exp_part # Computes the image of X via the exponenetial map


# Verifies the condition C (numerical and symbolic)
def verify_condition_C(Phi, X, d, tol=1e-8):
    """
    Verifies the condition C for both numerical and symbolic arguments X.
    Condition C:
        F₁⁻¹(exp_{F₁(Φ)}(F₁(X))) = F₂⁻¹(exp_{F₂(Φ)}(F₂(X))) = F₃⁻¹(exp_{F₃(Φ)}(F₃(X)))
    Where:
       - Φ: tensor of shape (d,d,d,d);
       - X: is the tensor to verify if it verifies condition C.
    Returns:
       - condition_holds: bool;
       - terms: the terms of the condition C.
    """

    flattenings = [([0, 1], [2, 3]), ([0, 2], [1, 3]), ([0, 3], [1, 2])] # Defines all the possible bipartitions
    terms = []
    for i in range(3):# Generates all the terms of the condition for both symbolic and numerical X
        in_dims, out_dims= flattenings[i]
        if isinstance(X, sp.Matrix):# Computes the exponential terms for symbolic X
            X_np = np.array(X.tolist(), dtype=complex) # Transforms symbolic X into a numpy array
            F_X= flatten(X_np, in_dims, out_dims, d)
            exp_result = exp_map(Phi, in_dims, out_dims, d, F_X)
        else:# Computes the exponential terms for numerical X
            F_X= flatten(X, in_dims, out_dims, d)
            exp_result = exp_map(Phi, in_dims, out_dims, d, F_X)

        term = flatten_inv(np.array(exp_result).astype(complex), in_dims, out_dims, d)
        terms.append(term)
    diffs = [ # Computes the norms of the differences between each pair of terms
        np.linalg.norm(terms[0] - terms[1]),
        np.linalg.norm(terms[0] - terms[2]),
        np.linalg.norm(terms[1] - terms[2])
    ]
    condition_holds = np.all(np.array(diffs) < tol) # Compares the norms in diffs to the tolerence in order to verify the equality
    return condition_holds, terms


def compute_tangent_space(Phi, d, i):
    """
    Computes the tangent space T at point Phi for a single flattening i:
    F_i(X)F_i(Φ)† + F_i(Φ)F_i(X)† = 0
    It vectorizes F_i(X), and solve for the tangent space using a lyapunov like equations approach
    Parameters:
        Φ: 4D tensor of shape (d,d,d,d)
        d: local dimension
        i: index of the flattening (0, 1, or 2)

    Returns:
        basis: list of X tensors (d,d,d,d) forming a basis of the solution space
    """

    flattenings = [
        ([0, 1], [2, 3]),  # F1
        ([0, 2], [1, 3]),  # F2
        ([0, 3], [1, 2])   # F3
    ]
    in_dims, out_dims = flattenings[i]

    F_Phi = flatten(Phi, in_dims, out_dims, d)
    I_d2 = np.eye(d*d)

    term1 = np.kron(I_d2, F_Phi)
    term2 = np.kron(F_Phi.conjugate(), I_d2)

    Pi = np.zeros((d**4, d**4))  # Calculates Pi, the commutation matrix such that, vec(X†)=Pi vec(X)
    for idx in range(d**4):
        multi_idx = np.unravel_index(idx, (d, d, d, d))
        transposed_idx = (multi_idx[0], multi_idx[1], multi_idx[3], multi_idx[2])
        idx_t = np.ravel_multi_index(transposed_idx, (d, d, d, d))
        Pi[idx_t, idx] = 1

    constraint = term1 @ Pi + term2
    A = constraint

    # Finds the F_i(X) as vector elements in the kermel of A
    null_basis = null_space(A)

    # Constructs the basis by applying the inverse flattening functions on F_i(X)
    basis = []
    for k in range(null_basis.shape[1]):
        F_X = null_basis[:, k].reshape(d*d, d*d)
        X = flatten_inv(F_X, in_dims, out_dims, d)
        basis.append(X)

    return basis
