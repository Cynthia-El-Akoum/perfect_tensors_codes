import numpy as np
from numpy.linalg import eig
from itertools import product

# Define constants
d = 2  # Qubit dimension
n = 5  # Number of qubits for the code

# Computational basis states
I2 = np.eye(d)
basis = [I2[:, i] for i in range(d)]

# Pauli matrices
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Stabilizer generators
St1 = np.kron(X, np.kron(Z, np.kron(Z, np.kron(X, I2))))
St2 = np.kron(I2, np.kron(X, np.kron(Z, np.kron(Z, X))))
St3 = np.kron(X, np.kron(I2, np.kron(X, np.kron(Z, Z))))
St4 = np.kron(Z, np.kron(X, np.kron(I2, np.kron(X, Z))))

# Generates full stabilizer group
generators = [St1, St2, St3, St4] # Stabilizer group basis
St = []
for coeffs in product([0, 1], repeat=4): # Creates all possible 4 elements tuples of 0 and 1, there are 16 elements
    g = np.eye(2**n, dtype=complex)
    for i, bit in enumerate(coeffs): #  bit is an element in the tuple and i is its index
        if bit: # If bit is 1 then the genrators[i] is included in the product
            g = g @ generators[i] # Performs the matrix multiplication of all the genrators
    St.append(g)
print(len(np.array(St)))
order_ST = 16  # Size of stabilizer group
dim_St=4

# Logical operators
X_bar = np.kron(X, np.kron(X, np.kron(X, np.kron(X, X)))) # X_bar= X ⊗ X ⊗ X ⊗ X ⊗ X
Z_bar = np.kron(Z, np.kron(Z, np.kron(Z, np.kron(Z, Z)))) # Z_bar= Z ⊗ Z ⊗ Z ⊗ Z ⊗ Z

# Generates logical states
ket_05 = np.kron(basis[0], np.kron(basis[0], np.kron(basis[0], np.kron(basis[0], basis[0])))) # the ket |00000>
ket_0tild = np.zeros((32,), dtype=complex) #Initializes |0^tild>
for st in St: # Computes |0^tild>= 1/order_St * sum_{s in St} s*|00000>
    ket_0tild += (st @ ket_05)
ket_0tild= ket_0tild/ np.sqrt(dim_St) # Normalizes with the dimension of St
ket_1tild = X_bar @ ket_0tild # Computes |1^tild> = X_bar |0^tild>

# Constructs 6-qubit state using the 5-qubits state using tensor product
psi = np.kron(ket_0tild, basis[0]) + np.kron(ket_1tild, basis[1])

# Reshapes to 8×8 matrix
T_matrix = psi.reshape((8,8))

# Prints the generated matrix
print(T_matrix)

# Verifies unitarity
unitary_check = T_matrix.conj().T @ T_matrix
print("\nIs unitary?", np.allclose(unitary_check, np.eye(8), atol=1e-10))


# Alternative construction of 6-index tensor from entries of psi
T = np.zeros((2, 2, 2, 2, 2, 2), dtype=complex)
for indices in product([0, 1], repeat=6): #Computes all possible 6 elements tuples of 0 and 1
    i, j, k, ip, jp, kp = indices
    index = 32*i + 16*j + 8*k + 4*ip + 2*jp + kp
    T[i, j, k, ip, jp, kp] = psi[index]


# Verifies both methods give same result
assert np.allclose(T.reshape((8, 8)), T_matrix)

# Check eigenvalues of the matrix associated with T
# Since T is generated from a perfect state, its eigenvalues correspond as well to the order 6 roots of the unity 
eigenvalues, eigenvectors = eig(T_matrix)
print("\nEigenvalues:")
print(eigenvalues)
