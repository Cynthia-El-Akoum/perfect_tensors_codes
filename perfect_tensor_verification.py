import numpy as np

# Computes the partial transpose matrix of a tensor
def partial_transpose(U, dimA, dimB):
    """Computes the partial transpose of U"""
    # Reshapes U to a tensor with indices (i,k,j,l)
    U_tensor = U.reshape(dimA, dimB, dimA, dimB)
    # Swaps j <-> k to get (i,j,k,l) then reshapes back
    U_Gamma = U_tensor.transpose(0, 3, 2, 1).reshape(dimA*dimB, dimA*dimB) #  In transpose(arguments): "arguments" indicate the new indices transposition
    return U_Gamma                                                         #i.e. 0 <-> 0, 1 <-> 3, 2 <-> 2, 3 <-> 1 and reshapes into a matrix

# Computes the realignment matrix of a tensor
def realignment(U, dimA, dimB):
    """Computes the realignment of U"""
    # Reshapes U to a tensor with indices (i,j,k,l)
    U_tensor = U.reshape(dimA, dimB, dimA, dimB)
    # Permutes indices to (i,k,j,l) and reshapes
    U_R = U_tensor.transpose(0, 2, 1, 3).reshape(dimA*dimB, dimA*dimB)
    return U_R

def check_perfect_tensor(U, U_R, U_Gamma, dimI):
    """Checks the unitarity of U, U_R, U_Gamma
       dimI is the dimension of the identity matrix to consider"""
    Id= np.eye(dimI) #if U is dimAxdimB then dimI=dimA
    # Defines the adjoint maps
    U_dagger= U.conjugate().T
    U_R_dagger= U_R.conjugate().T
    U_Gamma_dagger= U_Gamma.conjugate().T

    # Checks the unitarity of the matrices
    unitarity_U= np.allclose( U_dagger @ U, Id, atol=1e-8)
    unitarity_U_R= np.allclose( U_R_dagger @ U_R, Id, atol=1e-8)
    unitarity_U_Gamma= np.allclose( U_Gamma_dagger @ U_Gamma, Id, atol=1e-8)

    # Defines conditions of 2-unitarity
    if unitarity_U and (unitarity_U_R and unitarity_U_Gamma) :
        return print("U corresponds to a perfect tensor")
    else:
        return print(f"U doesn't correspond to a perfect tensor : unitarity_U: {unitarity_U} \nunitarity_U_R: {unitarity_U_R} \nunitarity_U_Gamma: {unitarity_U_Gamma}")
