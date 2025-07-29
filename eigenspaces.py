import numpy as np
from collections import defaultdict
from compute_S_map import S_reshaped
from rank1_verification import is_rank1_tensor

def compute_eigensystem(matrix, tol=1e-8):
    """
    Computes eigenvalues, degeneracies, eigenspaces, and checks rank-1 tensor status
    for both (d,d,d,d) and (d²,d²) reshapes of eigenvectors.

    Parameters:
        matrix (np.array): Input square matrix of size d⁴×d⁴
        tol (float): Tolerance for numerical comparisons

    Returns:
        dict: {
            'eigenvalues': array of unique eigenvalues,
            'degeneracies': array of multiplicities,
            'eigenspaces': list of eigenspace matrices,
            'rank1_vectors_4d': list of lists showing rank-1 in (d,d,d,d) reshape,
            'rank1_vectors_2d': list of lists showing rank-1 in (d²,d²) reshape,
            'rank1_eigenvectors_4d': dict of rank-1 vectors in (d,d,d,d),
            'rank1_eigenvectors_2d': dict of rank-1 vectors in (d²,d²)
        }
    """
    n = matrix.shape[0]
    d = int(round(n ** 0.25))
    if d**4 != n:
        print(f"Expected d⁴×d⁴ matrix but got {n}×{n}. Using (d²)×(d²) reshape only.")
        reshape_dims_4d = None
        reshape_dims_2d = (d*d, d*d)
    else:
        reshape_dims_4d = (d,d,d,d)
        reshape_dims_2d = (d*d, d*d)

    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    rounded_eigenvalues = np.round(eigenvalues, int(-np.log10(tol)))
    unique_eigenvalues, degeneracies = np.unique(rounded_eigenvalues, return_counts=True)

    eigen_dict = defaultdict(list)
    for val, vec in zip(rounded_eigenvalues, eigenvectors.T):
        eigen_dict[val].append(vec)

    eigenspaces = []
    rank1_info_4d = []
    rank1_info_2d = []
    rank1_eigenvectors_4d = defaultdict(list)
    rank1_eigenvectors_2d = defaultdict(list)

    for val in unique_eigenvalues:
        eigenspace = np.array(eigen_dict[val]).T
        eigenspaces.append(eigenspace)

        current_rank1_4d = []
        current_rank1_2d = []

        for i in range(eigenspace.shape[1]):
            vector = eigenspace[:, i]

            # Check rank-1 in (d²,d²) reshape
            try:
                tensor_2d = vector.reshape(reshape_dims_2d)
                is_rank1_2d = is_rank1_tensor(tensor_2d, tol)
                current_rank1_2d.append(is_rank1_2d)
                if is_rank1_2d:
                    rank1_eigenvectors_2d[val].append(vector)
            except:
                current_rank1_2d.append(False)

            # Check rank-1 in (d,d,d,d) reshape if possible
            if reshape_dims_4d is not None:
                try:
                    tensor_4d = vector.reshape(reshape_dims_4d)
                    is_rank1_4d = is_rank1_tensor(tensor_4d, tol)
                    current_rank1_4d.append(is_rank1_4d)
                    if is_rank1_4d:
                        rank1_eigenvectors_4d[val].append(vector)
                except:
                    current_rank1_4d.append(False)
            else:
                current_rank1_4d.append(False)  # Mark as False when 4d reshape isn't possible

        rank1_info_4d.append(current_rank1_4d)
        rank1_info_2d.append(current_rank1_2d)

    return {
        'eigenvalues': unique_eigenvalues,
        'degeneracies': degeneracies,
        'eigenspaces': eigenspaces,
        'rank1_vectors_4d': rank1_info_4d,
        'rank1_vectors_2d': rank1_info_2d,
        'rank1_eigenvectors_4d': dict(rank1_eigenvectors_4d),
        'rank1_eigenvectors_2d': dict(rank1_eigenvectors_2d)
    }

def visualize_eigenspaces_4d(matrix):
    """Displays rank-1 results for (d,d,d,d) reshaping"""
    result = compute_eigensystem(matrix)
    counter = 0

    print("\nEigenspace Analysis (4D Reshape):")
    for i, (val, eigenspace, rank1_states) in enumerate(zip(result['eigenvalues'],result['eigenspaces'],result['rank1_vectors_4d'])):
        print(f"\nEigenspace {i+1}: λ = {val:.4f} (Degeneracy: {eigenspace.shape[1]})")

        for j in range(eigenspace.shape[1]):
            vector = eigenspace[:, j]
            if rank1_states[j]:
                print(f"  ★ Rank-1 eigenvector {j+1} (4D)")
                counter += 1
                print(np.array2string(vector, precision=4, suppress_small=True))# Returns a string representation of an array # suppress_small displays 0 instead of really small values like 1e-10

    print(f"\nFound {counter} rank-1 eigenvectors in (d,d,d,d) reshape")

def visualize_eigenspaces_2d(matrix):
    """Displays rank-1 results for (d²,d²) reshaping"""
    result = compute_eigensystem(matrix)
    counter = 0

    print("\nEigenspace Analysis (2D Reshape):")
    for i, (val, eigenspace, rank1_states) in enumerate(zip(result['eigenvalues'],result['eigenspaces'],result['rank1_vectors_2d'])):
        print(f"\nEigenspace {i+1}: λ = {val:.4f} (Degeneracy: {eigenspace.shape[1]})")

        for j in range(eigenspace.shape[1]):
            vector = eigenspace[:, j]
            if rank1_states[j]:
                print(f"  ★ Rank-1 eigenvector {j+1} (2D)")
                counter += 1
                print(np.array2string(vector, precision=4, suppress_small=True))

    print(f"\nFound {counter} rank-1 eigenvectors in (d²,d²) reshape")

def summarize_rank1_eigenvectors(matrix):
    """Summarizes rank-1 eigenvectors for both reshape cases"""
    result = compute_eigensystem(matrix)

    print("\nComputed Eigensystem:")
    print(f"Eigenvalues: {result['eigenvalues']}")
    print(f"Degeneracies: {result['degeneracies']}")

    # 4D reshape results
    print("\nRank-1 Eigenvectors (4D Reshape):")
    counter_4d = 0
    for eigenvalue, vectors in result['rank1_eigenvectors_4d'].items():
        print(f"\nλ = {eigenvalue}: Found {len(vectors)} rank-1 eigenvector(s)")
        counter_4d += len(vectors)
    print(f"Total 4D rank-1: {counter_4d}")

    # 2D reshape results
    print("\nRank-1 Eigenvectors (2D Reshape):")
    counter_2d = 0
    for eigenvalue, vectors in result['rank1_eigenvectors_2d'].items():
        print(f"\nλ = {eigenvalue}: Found {len(vectors)} rank-1 eigenvector(s)")
        counter_2d += len(vectors)
    print(f"Total 2D rank-1: {counter_2d}")

    return result

if __name__ == "__main__":
    result = summarize_rank1_eigenvectors(S_reshaped)

    print("\n" + "="*100)
    print("Detailed 4D Reshape Analysis")
    visualize_eigenspaces_4d(S_reshaped)

    print("\n" + "="*100)
    print("Detailed 2D Reshape Analysis")
    visualize_eigenspaces_2d(S_reshaped)
