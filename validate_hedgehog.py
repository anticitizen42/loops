#
# Bray's Loops 4.0 - Milestone 1: Analytical Validation Suite
# Module: Hedgehog Field Generation and Validation
# This script directly addresses the failures of hedgehog initialization,
# coordinate system errors, and normalization issues.
#

import numpy as np

# --- Configuration ---
# All parameters are clearly defined to ensure reproducibility.
GRID_SIZE = 64
FIELD_DIMENSION = 3 # R^3 vector field for simplicity, maps to SU(2)
TOLERANCE = 1e-15 # Tolerance for floating point comparisons

# --- Component 1: Field Generation ---
# This component is designed to be independent and testable.

def generate_hedgehog_field(grid_size):
    """
    Generates a 3D hedgehog field on a grid.
    """
    field = np.zeros((grid_size, grid_size, grid_size, FIELD_DIMENSION))
    center_offset = (grid_size - 1) / 2.0

    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                # Calculate coordinates relative to the grid center
                x = i - center_offset
                y = j - center_offset
                z = k - center_offset

                position_vector = np.array([x, y, z])
                radius = np.linalg.norm(position_vector)

                if radius < TOLERANCE:
                    # Handle the singularity at the origin
                    field[i, j, k] = np.array([0, 0, 0])
                else:
                    # Normalize the vector to point radially outward
                    field[i, j, k] = position_vector / radius
    return field

# --- Component 2: Validation Suite ---
# A comprehensive test suite is a critical success factor.

def validate_field(field):
    """
    Runs the full validation protocol against a generated field.
    Returns True if all tests pass, otherwise False.
    """
    print("Running Validation Protocol...")

    # Test 1: Analytical Verification
    # Select a random point and verify its value.
    i, j, k = np.random.randint(0, GRID_SIZE, 3)
    center_offset = (GRID_SIZE - 1) / 2.0
    x, y, z = i - center_offset, j - center_offset, k - center_offset
    expected_vector = np.array([x, y, z]) / np.linalg.norm(np.array([x, y, z]))
    error = np.linalg.norm(field[i, j, k] - expected_vector)
    assert error < TOLERANCE, "Validation FAILED: Analytical verification."
    print("  ✅ Analytical Verification: PASSED")

    # Test 2: Normalization Check
    # This directly addresses a critical failure from Loop 3.0.
    norms = np.linalg.norm(field.reshape(-1, FIELD_DIMENSION), axis=1)
    # Exclude the origin point, which has a norm of 0
    non_origin_norms = norms[norms > TOLERANCE]
    max_deviation = np.max(np.abs(non_origin_norms - 1.0))
    assert max_deviation < TOLERANCE, "Validation FAILED: Normalization check."
    print("  ✅ Normalization Check: PASSED")

    # Test 3: Coordinate System Check
    # This directly addresses a critical failure from Loop 3.0.
    # Check a non-axis-aligned point, like the corner (0,0,0).
    x, y, z = -center_offset, -center_offset, -center_offset
    expected_corner = np.array([x, y, z]) / np.linalg.norm(np.array([x, y, z]))
    corner_error = np.linalg.norm(field[0, 0, 0] - expected_corner)
    assert corner_error < TOLERANCE, "Validation FAILED: Coordinate system check."
    print("  ✅ Coordinate System Check: PASSED")

    return True

# --- Main Execution Block ---
# Build nothing without validation.

if __name__ == "__main__":
    print("--- Starting Hedgehog Module Test ---")
    
    # 1. Generate the field
    generated_field = generate_hedgehog_field(GRID_SIZE)
    print("Field generation complete.")
    
    # 2. Validate the field
    is_valid = validate_field(generated_field)
    
    print("--- Test Result ---")
    if is_valid:
        print("✅ SUCCESS: The hedgehog field module is robust and validated.")
    else:
        print("❌ FAILURE: The module did not pass validation.")
    print("--------------------")