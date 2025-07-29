import numpy as np
from numpy.linalg import eig
from numpy.linalg import inv
from rank1_verification import is_rank1_tensor #This code checks if a tensor is of rank-1
from compute_example9_parametrized_families import psi_mat

#Defines dimensions
d=3 # Dimension of each subsystem
Id= np.eye(d)
Id2= np.eye(d**2)
basis= [Id[:, i] for i in range(d)] # Generates the {|i>, i=0, ..., d-1}  basis of  d-dimentional A
basis_w= [Id2[:, i] for i in range(d**2)] # Generates the {|i>, i=0, ..., d**2-1} basis of d**2-dimentional Ω



# Creates A_ij= |i><j| basis
A= np.zeros((d,d,d,d), dtype=complex) # Creates a dxd table where each element is a dxd matrix
for i in range(d):
    for j in range(d):
        A[i,j]= np.outer(basis[i], basis[j]) # Performs the outer product

# Creates A_tild_ij = A_ij ⊗ Id = A |i><j| basis
A_tild= np.zeros((d, d, d**2, d**2), dtype=complex) # Creates a dxd table where each element is a d**2xd**2 matrix
for i in range(d):
    for j in range(d):
        A_tild[i,j]= np.kron(A[i, j], Id)



# Creates U d**2xd**2 the 2-unitary matrix
U = np.array([
    """ Enter the 2-unitary matrix """
], dtype=complex)

# The adjoint matrix of U
U_dagger= U.conjugate().T

# Creates B_ij = U†(A_ij ⊗ I)U basis
B_tild=np.zeros((d, d, d**2, d**2), dtype= complex) # Creates a dxd table where each element is a d**2xd**2 matrix
for i in range(d):
    for j in range(d):
        B_tild[i,j]= U_dagger @ A_tild[i,j] @ U


# Generates the S tensor associated with the S map used in 2-unitary analysis
S= np.zeros((d,d,d,d,d,d,d,d), dtype= complex)# Initializes the S tensor
for i in range(d):
    for j in range(d):
        for k in range(d):
            for l in range(d):
                for ip in range(d):
                    for jp in range(d):
                        for kp in range(d):
                            for lp in range(d):
                                outer_ijkl= A_tild[i,j] @ B_tild[k,l] # Computes the action of prod_(A,B) on the tensor product basis of (A ⊗ B)
                                inner_ipjpkplp= B_tild[kp,lp] @ A_tild[ip, jp] #Computes the representatives of the tensor product basis of (B ⊗ A) in Ω
                                c= np.trace(inner_ipjpkplp.conjugate().T @ outer_ijkl) # Performs the Hilbert-Schmidt inner product
                                S[i,j,k,l,ip,jp,kp,lp]= c




S_reshaped= S.reshape(d**4,d**4) # Reshapes the tensor into a matrix in order to study its eigenvectors and eigenvalues

eigenvalues, eigenvectors = eig(S_reshaped) # Computes eigenvalues and eigenvectors with python


#to avoid the prints when importing
if __name__ == "__main__":
    # Checks for the unitarity of the S map
    is_unitary = np.allclose(S_reshaped.conj().T @ S_reshaped, np.eye(d**4), atol=1e-10)
    print(f"Is the matrix unitary? {is_unitary}")

    # Checks for the normality of the S map
    is_normal = np.allclose(S_reshaped.conj().T @ S_reshaped, S_reshaped @ S_reshaped.conj().T, atol=1e-10)
    print(f"Is the matrix normal? {is_normal}")

    # Checks if all the eigenvalues are roots of unity
    tolerance = 1e-10
    for i, l in enumerate(eigenvalues):
        # Check if |λ^d - 1| is within numerical tolerance
        if not np.isclose(l**d, 1, atol=tolerance):
            print(f"Eigenvalue {i+1}: {l:.6f}, does not satisfy λ^{d} = 1 (value = {l**d:.6f})")

    # Displays the eigenvalues and the corresponding eigenvectors
    for i in range(d**4):
        if np.abs(eigenvalues[i]) > 1e-8:
            print(f"Eigenvalue {i}: {eigenvalues[i]:.2f}")
            print(f"Eigenvector {i}: {eigenvectors[:, i].reshape(d**2,d**2)}")

    # Checks which eigenvectors correspond to rank-1 tensors under reshaping
    counter=0 #Counts the number of rank-1 eigenvectos
    for i in range(d**4):
        rank1= is_rank1_tensor(eigenvectors[:, i].reshape((d,d,d,d)))
        if rank1==True:
            counter+=1
            print(f"Eigenvalue {i}: {eigenvalues[i]:.2f}")
            print(f"Eigenvector {i}: {eigenvectors[:, i].reshape(d**2,d**2)} is rank1")

    print(f"counter= {counter}")
