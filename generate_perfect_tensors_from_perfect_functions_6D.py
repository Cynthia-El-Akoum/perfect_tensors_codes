import numpy as np
from numpy.linalg import matrix_power # Computes matrix power
from itertools import product
from numpy.linalg import eig
import Perfect_tensor_verification # verifies if a given tensor is perfect

# Initializes basis and parameters
d = 6 #dimension of each of the 4 subspaces
I6 = np.eye(d)
basis = [I6[:, i] for i in range(d)] # Generates |i> basis of dimension 6
omega_6 = np.exp(2j * np.pi / d) # The 6-th root of unity
omega_3 = np.exp(2j * np.pi / 3) # The 3rd root of unity

# Generalized Pauli operators (shift X and phase Z)
X = np.zeros((d, d), dtype=complex)
Z = np.zeros((d, d), dtype=complex)
for i in range(d):
    X += np.outer(basis[(i + 1) % d], basis[i])  # X|i⟩ = |(i+1) mod 6⟩
    Z += omega_6**i * np.outer(basis[i], basis[i])  # Z|i⟩ = omega_6**i |i⟩

# Precomputes matrix powers for efficiency
X_powers = [matrix_power(X, i) for i in range(d)]
Z_powers = [matrix_power(Z, j) for j in range(d)]

# Constructs Weyl operators W_{i,j} = ω^{ij/2} X^i Z^j
Weyl_operators = np.zeros((d, d, d, d), dtype=complex)
for i, j in product(range(d), repeat=2):
    phase = omega_6 ** ((i * j) / 2)
    Weyl_operators[i, j] = phase * (X_powers[i] @ Z_powers[j])

# Verifies unitarity
for i, j in product(range(d), repeat=2):
    op = Weyl_operators[i, j]
    assert np.allclose(op @ op.conj().T, np.eye(d)), f"Unitarity failed for (i,j)=({i},{j})"

# Computes maximally entangled state |ϕ⟩ = (1/√6) ∑_{k=0}^5 |k⟩ ⊗ |k⟩
phi = np.sum([np.kron(basis[k], basis[k]).reshape(-1, 1) for k in range(d)], axis=0) / np.sqrt(d)

# States |phi_a⟩ = (W_{i,j} tensor I) |phi⟩ for a = (i,j)
phi_a = []
for i, j in product(range(d), repeat=2):
    phi_a.append(np.kron(Weyl_operators[i, j], I6) @ phi)

#Convert phi_a to a numpy array for easier manipulation
phi_a = np.array(phi_a).reshape(d, d, -1)  # Shape: (6, 6, 36) , where reshape(-1) -> (n, )


# Defines the sparse and symmetric solutions
def lambda_sparse_symmetric(a, sparse= True): # a=(i, j) an element in Z_6 x Z_6
    k= a[0]%3
    x= a[0]%2
    l= a[1]%3
    y= a[1]%2
    m= (x-y)%3
    P= k**2 + l**2
    if sparse:
        Q= (l+m)**2
    else: #symmetric solution
        Q= - (k+l+m)**2
    if (x, y)==(1,1):
        return omega_3**(P)
    else:
        return omega_3**(P+Q)

# Constructs the perfect tensor from the perfect function solution
U_lambda = np.zeros((36, 36), dtype=complex) # Initializes the tensor
for i in range(6):
    for j in range(6):
        idx = 6 * i + j
        U_lambda += lambda_sparse_symmetric([i, j]) * np.outer(phi_a[i, j], phi_a[i, j])

# Eliminates the numerical noise
def clean_tensor(tensor, threshold=1e-14):
    cleaned = np.copy(tensor)
    cleaned.real[np.abs(cleaned.real) < threshold] = 0.0
    cleaned.imag[np.abs(cleaned.imag) < threshold] = 0.0
    return cleaned

U_lambda= clean_tensor(U_lambda)

# Defines the corresponding partial transpose and realignement matrices using the Perfect_tensor_verification functions
U_lambda_Gamma= Perfect_tensor_verification.partial_transpose(U_lambda, 6, 6)
U_lambda_R= Perfect_tensor_verification.realignment(U_lambda, 6, 6)

# Checks for 2-unitarity
Perfect_tensor_verification.check_perfect_tensor(U_lambda, U_lambda_R, U_lambda_Gamma, 36)
