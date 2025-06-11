#
# Bray's Loops 4.0 - Milestone 1: Final Functional Implementation
# This script contains a mathematically correct, functional implementation of the
# finite-difference topological charge calculator. It replaces all previous placeholders.
#

import numpy as np

# --- Configuration ---
GRID_SIZE = 32 # Reduced for faster execution of the complex calculation
TOLERANCE = 1e-15
VALIDATION_TOLERANCE = 5e-2 # Looser tolerance for lattice calculations

# --- Component 1: Field Generation (Outputting SU(2) Matrices) ---

def _get_su2_matrix(s0, s_vector):
    """Converts a 4-vector to an SU(2) matrix."""
    s1, s2, s3 = s_vector
    return np.array([[s0 + 1j * s3, s2 + 1j * s1],
                     [-s2 + 1j * s1, s0 - 1j * s3]])

def generate_skyrmion_field_su2(grid_size):
    """Generates a 3D skyrmion field (Q=1) as a grid of SU(2) matrices."""
    field = np.zeros((grid_size, grid_size, grid_size, 2, 2), dtype=np.complex128)
    center_offset = (grid_size - 1) / 2.0
    width = grid_size / 10.0

    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                pos = np.array([i - center_offset, j - center_offset, k - center_offset])
                radius = np.linalg.norm(pos)
                F_r = np.pi * np.exp(-radius / width)
                s0 = np.cos(F_r)
                s_vector_magnitude = np.sin(F_r)
                
                if radius < TOLERANCE:
                    s_vector = np.array([0,0,0])
                else:
                    unit_vector = pos / radius
                    s_vector = s_vector_magnitude * unit_vector
                
                field[i, j, k] = _get_su2_matrix(s0, s_vector)
    return field

# --- Component 2: Functional Diagnostic Tool ---

def calculate_topological_charge(phi):
    """
    Calculates topological charge using the correct lattice formula.
    This is a direct implementation of q = (1/24pi^2) * Tr(L_x L_y L_z + permutations).
    """
    # Get the conjugate transpose field, phi_dagger
    phi_dagger = np.conj(phi).transpose(0, 1, 2, 4, 3)

    # Calculate Link variables U_mu = phi_dagger(x) * phi(x+mu)
    U = []
    for axis in range(3): # 0=x, 1=y, 2=z
        U.append(phi_dagger @ np.roll(phi, -1, axis=axis))

    # Calculate derivative operators L_mu = U_mu - U_mu_dagger
    L = []
    for mu in range(3):
        L.append(U[mu] - np.conj(U[mu]).transpose(0, 1, 2, 4, 3))

    # Calculate the full anti-symmetric trace
    # Tr( L_x L_y L_z - L_x L_z L_y + L_y L_z L_x - ... )
    # This corresponds to 3! = 6 permutations
    term1 = L[0] @ L[1] @ L[2]
    term2 = L[0] @ L[2] @ L[1]
    term3 = L[1] @ L[2] @ L[0]
    term4 = L[1] @ L[0] @ L[2]
    term5 = L[2] @ L[0] @ L[1]
    term6 = L[2] @ L[1] @ L[0]
    
    density_matrix = term1 - term2 + term3 - term4 + term5 - term6
    
    # Take the trace and sum over the volume
    # The result is imaginary, so we take the imaginary part.
    total_charge_density = np.imag(np.trace(density_matrix, axis1=3, axis2=4))
    
    # Apply normalization constant
    total_charge = np.sum(total_charge_density) / (24 * np.pi**2)
    
    return total_charge

# --- Main Execution and Validation Block ---

if __name__ == "__main__":
    print("--- Running Functional Diagnostic Implementation ---")
    
    # 1. Generate the Q=1 Skyrmion field
    print("Generating SU(2) skyrmion field (Q=1)...")
    skyrmion_su2 = generate_skyrmion_field_su2(GRID_SIZE)
    print("Generation complete.")

    # 2. Run the functional diagnostic tool
    print("\nValidating functional diagnostic tool...")
    q_skyrmion = calculate_topological_charge(skyrmion_su2)

    print(f"  Measured charge for Skyrmion (Q=1): {q_skyrmion:.4f}")

    # 3. Assert correctness
    assert abs(q_skyrmion - 1.0) < VALIDATION_TOLERANCE, "Skyrmion charge validation FAILED"

    print("\n✅ SUCCESS: The functional finite-difference diagnostic tool is validated.")
    print("   This component now serves as our 'Reference Implementation'.")
    print("\nWith this validated component, we will now proceed to build the second, independent geometric calculator for cross-validation.")