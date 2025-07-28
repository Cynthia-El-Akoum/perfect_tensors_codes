import numpy as np
from math import cos, sinh, cosh
import Perfect_tensor_verification as ptv

def compute_psi(t):
    """
    Compute the Ψ tensor for given parameters t = [t1, t2, t3, t4]

    Returns:
        A 3x3x3x3 numpy array (with complex dtype) representing Ψ_{ijkl}
    """
    t1, t2, t3, t4 = t
    t_norm = np.sqrt(t1**2 + t2**2 + t3**2 + t4**2)
    psi = np.zeros((3, 3, 3, 3), dtype=complex)

    # Handle special case when t_norm is 0 to avoid division by zero
    if t_norm == 0:
        return psi  # All components remain zero

    # Group 1
    val1 = -np.sinh(t_norm * 1j) * (t3 - 1j*t4) / (t_norm * 1j)
    psi[0,0,0,0] = psi[1,0,1,1] = psi[2,0,2,2] = val1

    # Group 2
    val2 = (cos(t_norm) - 1) * (t1 + t2*1j) * (t3 - t4*1j) / t_norm**2
    psi[0,1,0,0] = psi[1,1,1,1] = psi[2,1,2,2] = val2

    # Group 3
    val3 = ((t3**2 + t4**2)*cos(t_norm) + t1**2 + t2**2) / t_norm**2
    psi[0,2,0,0] = psi[1,2,1,1] = psi[2,2,2,2] = val3

    # Group 4
    val4 = np.cosh(t_norm * 1j)  # Using numpy's cosh for complex support
    psi[2,0,0,1] = psi[0,0,1,2] = psi[1,0,2,0] = val4

    # Group 5
    val5 = np.sinh(t_norm * 1j) * (t1 + 1j*t2) / (t_norm * 1j)
    psi[2,1,0,1] = psi[0,1,1,2] = psi[1,1,2,0] = val5

    # Group 6
    val6 = np.sinh(t_norm * 1j) * (t3 + 1j*t4) / (t_norm * 1j)
    psi[2,2,0,1] = psi[0,2,1,2] = psi[1,2,2,0] = val6

    # Group 7
    val7 = -np.sinh(t_norm * 1j) * (t1 - 1j*t2) / (t_norm * 1j)
    psi[1,0,0,2] = psi[2,0,1,0] = psi[0,0,2,1] = val7

    # Group 8
    val8 = ((t1**2 + t2**2)*cos(t_norm) + t3**2 + t4**2) / t_norm**2
    psi[1,1,0,2] = psi[2,1,1,0] = psi[0,1,2,1] = val8

    # Group 9
    val9 = (cos(t_norm) - 1) * (t1 - t2*1j) * (t3 + t4*1j) / t_norm**2
    psi[1,2,0,2] = psi[2,2,1,0] = psi[0,2,2,1] = val9

    return psi

def visualize_psi(psi):
    """Visualize non-zero components of the Ψ tensor with better formatting"""
    print("Non-zero components of Ψ:")
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    val = psi[i,j,k,l]
                    if abs(val) > 1e-10:
                        # Format real and imaginary parts separately
                        real_part = np.real(val)
                        imag_part = np.imag(val)

                        if abs(imag_part) < 1e-10:
                            print(f"Ψ_{i+1}{j+1}{k+1}{l+1} = {real_part:.6f}")
                        elif abs(real_part) < 1e-10:
                            print(f"Ψ_{i+1}{j+1}{k+1}{l+1} = {imag_part:.6f}j")
                        else:
                            sign = '+' if imag_part >= 0 else '-'
                            print(f"Ψ_{i+1}{j+1}{k+1}{l+1} = {real_part:.6f} {sign} {abs(imag_part):.6f}j")

# Example usage
t_params = np.random.rand(10, 4)*8

for t in t_params:
    psi_tensor = compute_psi(t)
    psi_mat= psi_tensor.reshape(9,9)
    psi_Gamma= ptv.partial_transpose(psi_tensor, 3, 3)
    psi_R= ptv.realignment(psi_tensor, 3, 3)
    ptv.check_perfect_tensor(psi_mat, psi_R, psi_Gamma, 3**2)
    #visualize_psi(psi_tensor)
