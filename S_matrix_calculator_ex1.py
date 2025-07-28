import numpy as np
from numpy.linalg import eig
from numpy.linalg import inv
from rank1_verification import is_rank1_tensor

#This function is different because it acts on S maps that correspond to two-unitaries in P(6,d)
#define dimensions
dA1=4
dA2=2
dB=8
I2= np.eye(2)
I4= np.eye(4)
basisA1= [I4[:, i] for i in range(dA1)]
basisA2= [I2[:, i] for i in range(dA2)]


#create A_ij= |i><j| basis for A -> 4x4
A1= np.zeros((4,4,4,4), dtype=complex) #creates a dxd table where each element is a dxd matrix
for i in range(4):
    for j in range(4):
        A1[i,j]= np.outer(basisA1[i], basisA1[j])

A1_tild= np.zeros((4, 4, 8, 8), dtype=complex)
for i in range(4):
    for j in range(4):
        A1_tild[i,j]= np.kron(A1[i, j], I2)
        norm1A=np.linalg.norm(A1_tild[i, j], 'fro')
        A1_tild[i,j]*=norm1A

#create A_ij= |i><j| basis for A -> 2x2
A2= np.zeros((2,2,2,2), dtype=complex) #creates a dxd table where each element is a dxd matrix
for i in range(2):
    for j in range(2):
        A2[i,j]= np.outer(basisA2[i], basisA2[j])

A2_tild= np.zeros((2, 2, 8, 8), dtype=complex) #creates a dxd table where each element is a d**2xd**2 matrix
for i in range(2):
    for j in range(2):
        A2_tild[i,j]= np.kron(A2[i, j], np.kron(I2, I2))

#taking dimA=4 but A ~ C^2 ⊗ C^2
A3_tild= np.zeros((4, 4, 8, 8), dtype=complex) #creates a dxd table where each element is a d**2xd**2 matrix
for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                row_idx = 2 * i + k
                col_idx = 2 * j + l
                A3_tild[row_idx, col_idx] = np.kron(A2[i, j], np.kron(A2[k, l], I2))
                norm3A=np.linalg.norm(A3_tild[row_idx, col_idx], 'fro')
                A3_tild[row_idx, col_idx]*=norm3A


#create T 8x8 matrix
T= np.array([[ 0.5,  0. ,  0. , -0.5,  0. , -0.5, -0.5,  0. ],
             [ 0. , -0.5,  0.5,  0. , -0.5,  0. ,  0. , -0.5],
             [ 0. , -0.5,  0.5,  0. ,  0.5,  0. ,  0. ,  0.5],
             [-0.5,  0. ,  0. ,  0.5,  0. , -0.5, -0.5,  0. ],
             [ 0. , -0.5, -0.5,  0. ,  0.5,  0. ,  0. , -0.5],
             [ 0.5,  0. ,  0. ,  0.5,  0. ,  0.5, -0.5,  0. ],
             [-0.5,  0. ,  0. , -0.5,  0. ,  0.5, -0.5,  0. ],
             [ 0. , -0.5, -0.5,  0. , -0.5,  0. ,  0. ,  0.5]],
            dtype= complex)

T_dagger= T.conjugate().transpose()


B1_tild=np.zeros((dA1, dA1, dA1*2, dA1*2), dtype= complex)
for i in range(dA1):
    for j in range(dA1):
        B1_tild[i,j]= T_dagger @ A1_tild[i,j] @ T
        norm1B=np.linalg.norm(B1_tild[i, j], 'fro')
        B1_tild[i,j]*=norm1B

B2_tild=np.zeros((dA2, dA2, dA2**3, dA2**3), dtype= complex)
for i in range(dA2):
    for j in range(dA2):
        B2_tild[i,j]= T_dagger @ A2_tild[i,j] @ T

B3_tild=np.zeros((4, 4, 8, 8), dtype= complex)
for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                index=[2*i+k,2*j+l]
                B3_tild[index[0], index[1]]= T_dagger @ A3_tild[index[0], index[1]] @ T
                norm3B=np.linalg.norm(B3_tild[row_idx, col_idx], 'fro')
                B3_tild[row_idx, col_idx]*=norm3B #why multiply16
d=2
counter= 0
#S= np.zeros((81,81), dtype= complex)
S2= np.zeros((2,2,2,2,2,2,2,2), dtype= complex)
for i in range(2):
    for j in range(2):
        for k in range(d):
            for l in range(d):
                for ip in range(2):
                    for jp in range(2):
                        for kp in range(d):
                            for lp in range(d):
                                    outer_ijkl= A2_tild[i,j] @ B2_tild[k,l]
                                    inner_ipjpkplp= B2_tild[kp,lp] @ A2_tild[ip, jp]
                                    c= np.trace(inner_ipjpkplp.conjugate().T @ outer_ijkl)/2 #normalized
                                    #if c >1e-8:
                                        #counter+=1
                                        #print(f"c[{i},{j},{k},{l},{ip},{jp},{kp},{lp}]= {c}")
                                    S2[i,j,k,l,ip,jp,kp,lp]= c


#print(counter)
S1= np.zeros((2,2,2,2,2,2,2,2), dtype= complex)
S3= np.zeros((2,2,2,2,2,2,2,2), dtype= complex)
for i in range(2):
    for j in range(2):
        for k in range(d):
            for l in range(d):
                for ip in range(2):
                    for jp in range(2):
                        for kp in range(d):
                            for lp in range(d):
                                index=[2*i+k,2*j+l]
                                indexp=[2*ip+kp,2*jp+lp]
                                outer_ijkl= A1_tild[index[0], index[1]] @ B1_tild[k,l]
                                inner_ipjpkplp= B1_tild[kp,lp] @ A1_tild[indexp[0], indexp[1]]
                                c1= np.trace(inner_ipjpkplp.conjugate().T @ outer_ijkl)/4 #normalized
                                S1[i,j,k,l,ip,jp,kp,lp]= c1
                                outer_ijkl3= A3_tild[index[0], index[1]] @ B3_tild[k,l]
                                inner_ipjpkplp3= B3_tild[kp,lp] @ A3_tild[indexp[0], indexp[1]]
                                #print(np.shape(outer_ijkl3), np.shape(inner_ipjpkplp3))
                                c3= np.trace(inner_ipjpkplp3.conjugate().T @ outer_ijkl3)#normalized
                                S3[i,j,k,l,ip,jp,kp,lp]= c3

"""
S2_reshaped= S2.reshape(16,16)

eigenvalues, eigenvectors = eig(S2_reshaped)

is_unitary = np.allclose(S2_reshaped.conj().T @ S2_reshaped, np.eye(16), atol=1e-10)
print(f"Is the matrix unitary? {is_unitary}")

# Normality check
is_normal = np.allclose(S2_reshaped.conj().T @ S2_reshaped, S2_reshaped @ S2_reshaped.conj().T, atol=1e-10)
print(f"Is the matrix normal? {is_normal}")

for i in range(16):
    if np.abs(eigenvalues[i]) > 1e-8:
        print(f"Eigenvalue {i}: {eigenvalues[i]:.2f}")
        print(f"Eigenvector {i}: {eigenvectors[:, i]}")
"""
"""
S3_reshaped= S3.reshape(16,16)

eigenvalues, eigenvectors = eig(S3_reshaped)

is_unitary = np.allclose(S3_reshaped.conj().T @ S3_reshaped, np.eye(16), atol=1e-10)
print(f"Is the matrix unitary? {is_unitary}")

# Normality check
is_normal = np.allclose(S3_reshaped.conj().T @ S3_reshaped, S3_reshaped @ S3_reshaped.conj().T, atol=1e-10)
print(f"Is the matrix normal? {is_normal}")

for i in range(16):
    if np.abs(eigenvalues[i]) > 1e-8:
        print(f"Eigenvalue {i}: {eigenvalues[i]:.2f}")
        print(f"Eigenvector {i}: {eigenvectors[:, i]}")

"""
S1_reshaped= S1.reshape(16,16)

eigenvalues, eigenvectors = eig(S1_reshaped)

is_unitary = np.allclose(S1_reshaped.conj().T @ S1_reshaped, np.eye(16), atol=1e-10)
print(f"Is the matrix unitary? {is_unitary}")

# Normality check
is_normal = np.allclose(S1_reshaped.conj().T @ S1_reshaped, S1_reshaped @ S1_reshaped.conj().T, atol=1e-10)
print(f"Is the matrix normal? {is_normal}")

for i in range(16):
    if np.abs(eigenvalues[i]) > 1e-8:
        print(f"Eigenvalue {i}: {eigenvalues[i]:.2f}")
        print(f"Eigenvector {i}: {eigenvectors[:, i]}")

counter=0
for i in range(16):
    rank1= is_rank1_tensor(eigenvectors[:, i].reshape((2,2,2,2)))
    if rank1==True:
        counter+=1
        print(f"Eigenvector {i}: {eigenvectors[:, i]} is rank1")
        print(f"Eigenvalue {i}: {eigenvalues[i]:.2f}")

print(f"counter= {counter}")
