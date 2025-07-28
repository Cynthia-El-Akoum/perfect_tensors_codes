# Perfect Tensors Codebase

## Overview

In this internship, we aimed to construct perfect tensors using three primary methods provided in the literature, and to further study the eigenvalues and eigenvectors of their associated **S map**, defined in detail in the report (%insert sharing link)

To achieve this, we primarily used the Python library `numpy` and developed code for both generating perfect tensors and computing the associated **S** tensor and its matrix representation to evaluate its spectrum.

## Codebase Structure

The project is structured across seven Python files, each responsible for a specific part of the pipeline for generating perfect tensors, computing the associated **S** map, and analyzing its spectral properties.

- `generate_perfect_tensors_OLS_and_tangent_space.py`: Constructs perfect tensors using OLS methods and computes the tangent space to generate parametrized families of perfect tensors.

- `generate_perfect_tensors_from_perfect_functions_6D.py`: Computes explicit hand-made order-6 perfect tensors for testing and analysis.

- `generate_perfect_tensors_from_AME_6qubits.py`: Computes a perfect tensor acting on 6 qubits, included for comparison. This example is not used for the spectral characterization of the **S** map.

- `perfect_tensor_verification.py`: Provides functions to verify whether a given tensor satisfies perfect tensor conditions.

- `compute_S_map.py`: Main script to compute the **S** map for a given dimension `d`, evaluate its spectrum, and call other modules to analyze eigenvectors.

- `rank1_verification.py`: Contains the function `is_rank1_tensor()` to check whether a given tensor is rank-1. Used across files for eigenvector validation.

- `compute_example9_parametrized_families.py`: Generates a concrete example of parametrized families of perfect tensors of dimension 3, derived from the perfect tensor from OLS(3). Based on a construction from [REF].

