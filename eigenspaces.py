import numpy as np
from collections import defaultdict
from compute_S_map import  S_reshaped
from rank1_verification import is_rank1_tensor #This code checks if a tensor is of rank-1


def compute_eigensystem(matrix, tol=1e-8):
    """
    Computes eigenvalues, degeneracies, eigenspaces, and checks rank-1 tensor status for d**4×d**4 matrices.

    Parameters:
        matrix (np.array): Input square matrix of size d**4×d**4
        tol (float): Tolerance for numerical comparisons

    Returns:
        dict: {
            'eigenvalues': array of unique eigenvalues,
            'degeneracies': array of multiplicities,
            'eigenspaces': list of eigenspace matrices,
            'rank1_vectors': list of lists showing which vectors are rank-1 tensors,
            'rank1_eigenvectors': dictionary of rank-1 vectors.
        }
    """
    # Verify matrix is d⁴×d⁴
    n = matrix.shape[0]
    d = int(round(n ** 0.25))
    if d**4 != n:
        print(f"Expected d**4×d**4 matrix but got {n}×{n}. Using (d**2)×(d**2) reshape instead.")
        reshape_dims = (d*d, d*d)
    else:
        reshape_dims = (d,d,d,d)

    # Computes eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Rounds eigenvalues to handle numerical noise
    rounded_eigenvalues = np.round(eigenvalues, int(-np.log10(tol)))

    # Finds unique eigenvalues and degeneracies
    unique_eigenvalues, degeneracies = np.unique(rounded_eigenvalues, return_counts=True)

    # Groups eigenvectors by eigenvalue
    eigen_dict = defaultdict(list)
    for val, vec in zip(rounded_eigenvalues, eigenvectors.T):
        eigen_dict[val].append(vec)

    # Analyzes eigenspaces and tracks rank-1 vectors
    eigenspaces = []
    rank1_info = []
    rank1_eigenvectors = defaultdict(list)

    for val in unique_eigenvalues:
        eigenspace = np.array(eigen_dict[val]).T
        eigenspaces.append(eigenspace)

        # Checks rank-1 status for each eigenvector
        current_rank1 = []
        for i in range(eigenspace.shape[1]):
            vector = eigenspace[:, i]

            # Reshapes and checks rank-1
            try:
                tensor = vector.reshape(reshape_dims)
                is_rank1 = is_rank1_tensor(tensor, tol)
                current_rank1.append(is_rank1)
                if is_rank1:
                    rank1_eigenvectors[val].append(vector)  # Stores rank-1 vectors
            except:
                current_rank1.append(False)

        rank1_info.append(current_rank1)

    return {
        'eigenvalues': unique_eigenvalues,
        'degeneracies': degeneracies,
        'eigenspaces': eigenspaces,
        'rank1_vectors': rank1_info,
        'rank1_eigenvectors': dict(rank1_eigenvectors)  #dictionary of rank-1 vectors
    }

def visualize_eigenspaces(matrix):
    """
    Displays eigenspaces and identifies rank-1 eigenvectors

    Parameters:
        matrix (np.array): Input square matrix to analyze
    """
    result = compute_eigensystem(matrix)
    counter = 0  # Count of rank-1 tensors

    print("\nEigenspace Analysis:")

    for i, (val, eigenspace, rank1_flags) in enumerate(zip(
        result['eigenvalues'],
        result['eigenspaces'],
        result['rank1_vectors']
    )):
        print(f"\nEigenspace {i+1}: λ = {val:.4f} (Degeneracy: {eigenspace.shape[1]})")

        for j in range(eigenspace.shape[1]):
            vector = eigenspace[:, j]
            if rank1_flags[j]:
                status = "★ Rank-1 tensor ★"
                counter += 1
            else:
                status = "Not rank-1 tensor"

            print(f"\nEigenvector {j+1} ({status}):")
            print(np.array2string(vector, precision=4, suppress_small=True))

    print(f"\nSummary: Found {counter} rank-1 eigenvectors out of {sum(result['degeneracies'])} total eigenvectors")

def summarize_rank1_eigenvectors(matrix):
    """
    Computes and summarizes rank-1 eigenvectors for a given matrix.

    Parameters:
        matrix (np.array): Input square matrix to analyze
    """
    result = compute_eigensystem(matrix)

    print("\nComputed Eigensystem:")
    print(f"Eigenvalues: {result['eigenvalues']}")
    print(f"Degeneracies: {result['degeneracies']}")
    print("\nRank-1 Eigenvectors Summary:")

    counter = 0
    for eigenvalue, vectors in result['rank1_eigenvectors'].items():
        print(f"\nλ = {eigenvalue}: Found {len(vectors)} rank-1 eigenvector(s)")
        counter += len(vectors)
        for i, vec in enumerate(vectors):
            print(f"  Rank-1 eigenvector {i+1}:")
            print(np.array2string(vec, precision=4, suppress_small=True))

    print(f"\nTotal rank-1 eigenvectors found: {counter}")
    return result

# Test
if __name__ == "__main__":
    result = summarize_rank1_eigenvectors(S_reshaped)
    print('-'*100)
    visualize_eigenspaces(S_reshaped)
