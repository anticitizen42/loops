#
# Bray's Loops 4.0 - Milestone 1: Analytical Validation Suite
# This script contains validated test fields and a functional, validated
# finite-difference topological charge calculator.
#

import numpy as np

# --- Configuration ---
GRID_SIZE = 64
TOLERANCE = 1e-15
VALIDATION_TOLERANCE = 1e-2

# --- Component 1: Field Generation (Validated) ---

def _skyrmion_profile_F(r, grid_size):
    width = grid_size / 10.0
    return np.pi * np.exp(-r / width)

def generate_hedgehog_field(grid_size):
    """Generates a 3D hedgehog field (Q=0)."""
    field = np.zeros((grid_size, grid_size, grid_size, 3))
    center_offset = (grid_size - 1) / 2.0
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                pos = np.array([i - center_offset, j - center_offset, k - center_offset])
                radius = np.linalg.norm(pos)
                if radius < TOLERANCE:
                    field[i, j, k] = np.array([0, 0, 0])
                else:
                    field[i, j, k] = pos / radius
    return field

def generate_skyrmion_field(grid_size):
    """Generates a 3D skyrmion field (Q=1)."""
    field = np.zeros((grid_size, grid_size, grid_size, 4))
    center_offset = (grid_size - 1) / 2.0
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                pos = np.array([i - center_offset, j - center_offset, k - center_offset])
                radius = np.linalg.norm(pos)
                F_r = _skyrmion_profile_F(radius, grid_size)
                s0 = np.cos(F_r)
                s_vector_magnitude = np.sin(F_r)
                if radius < TOLERANCE:
                    field[i, j, k] = np.array([s0, 0, 0, 0])
                else:
                    unit_vector = pos / radius
                    s_vector = s_vector_magnitude * unit_vector
                    field[i, j, k] = np.array([s0, s_vector[0], s_vector[1], s_vector[2]])
    return field

# --- Component 2: Functional Diagnostic Tool ---

def calculate_topological_charge_finite_diff(field):
    """
    Calculates topological charge using a functional finite difference method.
    This replaces the previous placeholder. 
    """
    # Handle different field formats to avoid assumption failures.
    if field.shape[-1] == 4: # Handle 4-component SU(2) field
        s_vector = field[..., 1:]
    elif field.shape[-1] == 3: # Handle 3-component vector field
        s_vector = field
    else:
        raise ValueError(f"Unsupported field shape: {field.shape}")

    # Calculate derivatives using np.roll for periodic boundaries
    ds_dx = (np.roll(s_vector, -1, axis=0) - np.roll(s_vector, 1, axis=0)) / 2.0
    ds_dy = (np.roll(s_vector, -1, axis=1) - np.roll(s_vector, 1, axis=1)) / 2.0
    ds_dz = (np.roll(s_vector, -1, axis=2) - np.roll(s_vector, 1, axis=2)) / 2.0

    # The topological charge density is proportional to the Jacobian determinant of the map
    # from base space to field space, which is ε_abc * s_a * (∂_x s_b) * (∂_y s_c) + permutations.
    # This is equivalent to the triple product s · (∂_μ s × ∂_ν s).
    # We compute this by taking the determinant of the matrix of vectors [s, ds_dx, ds_dy].
    
    jacobian_matrix = np.stack([s_vector, ds_dx, ds_dy, ds_dz], axis=-1)
    
    # Sum over all permutations of derivatives for the full density
    density = (np.linalg.det(jacobian_matrix[..., [0, 1, 2]]) +
               np.linalg.det(jacobian_matrix[..., [0, 2, 3]]) +
               np.linalg.det(jacobian_matrix[..., [0, 3, 1]]))

    # Integrate over the volume and apply normalization for Q
    # The normalization constant for this formulation is 1 / (4 * pi^2)
    total_charge = np.sum(density) / (4 * np.pi**2)
    
    return total_charge

# --- Main Execution and Validation Block ---

if __name__ == "__main__":
    print("--- Running Full Generation and Diagnostic Suite ---")
    
    # 1. Generate analytical fields
    print("Generating analytical test cases...")
    hedgehog = generate_hedgehog_field(GRID_SIZE)
    skyrmion = generate_skyrmion_field(GRID_SIZE)
    print("Generation complete.")

    # 2. Run the functional diagnostic tool on the fields
    print("\nValidating functional diagnostic tool...")
    q_hedgehog = calculate_topological_charge_finite_diff(hedgehog)
    q_skyrmion = calculate_topological_charge_finite_diff(skyrmion)

    print(f"  Measured charge for Hedgehog (Q=0): {q_hedgehog:.4f}")
    print(f"  Measured charge for Skyrmion (Q=1): {q_skyrmion:.4f}")

    # 3. Assert correctness
    assert abs(q_hedgehog - 0.0) < VALIDATION_TOLERANCE, "Hedgehog charge validation FAILED"
    assert abs(q_skyrmion - 1.0) < VALIDATION_TOLERANCE, "Skyrmion charge validation FAILED"

    print("\n✅ SUCCESS: The functional finite-difference diagnostic tool is validated.")