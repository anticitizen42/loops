#
# Bray's Loops 4.0 - Foundational 2D Validation
# Final corrected version. The integration path is now correctly defined
# as counter-clockwise, resolving the sign error.
#

import numpy as np

# --- Configuration ---
GRID_SIZE_2D = 64
VALIDATION_TOLERANCE = 1e-9

# --- Component 1: 2D Test Case Generation ---

def generate_vortex_2d(grid_size):
    """
    Generates a 2D vector field with a single vortex at the center.
    This field has a known winding number of +1.
    """
    field = np.zeros((grid_size, grid_size, 2)) # Field of 2D vectors
    center = (grid_size - 1) / 2.0

    for i in range(grid_size):
        for j in range(grid_size):
            x = i - center
            y = j - center
            r_squared = x**2 + y**2
            
            if r_squared < 1e-6:
                field[i, j] = np.array([0.0, 0.0])
            else:
                # Standard vortex formula for W=+1
                field[i, j] = np.array([-y, x]) / r_squared
    return field

# --- Component 2: Functional 2D Winding Number Calculator ---

def calculate_winding_number_2d(field):
    """
    Calculates the winding number of a 2D vector field by integrating the
    angle change around a closed, counter-clockwise loop.
    """
    grid_size = field.shape[0]
    path_radius = grid_size // 4
    center_idx = int(grid_size / 2) # Use int for indexing
    
    path = []
    # CORRECTED to trace a counter-clockwise path
    # Bottom edge, left to right
    for i in range(-path_radius, path_radius + 1):
        path.append((center_idx + i, center_idx - path_radius))
    # Right edge, bottom to top
    for j in range(-path_radius + 1, path_radius + 1):
        path.append((center_idx + path_radius, center_idx + j))
    # Top edge, right to left
    for i in range(path_radius - 1, -path_radius - 1, -1):
        path.append((center_idx + i, center_idx + path_radius))
    # Left edge, top to bottom
    for j in range(path_radius - 1, -path_radius - 1, -1):
        path.append((center_idx - path_radius, center_idx + j))

    total_angle_change = 0.0
    
    vx_start, vy_start = field[path[0]]
    last_angle = np.arctan2(vy_start, vx_start)

    for i in range(1, len(path)):
        vx, vy = field[path[i]]
        current_angle = np.arctan2(vy, vx)
        
        delta_angle = current_angle - last_angle
        if delta_angle > np.pi:
            delta_angle -= 2 * np.pi
        elif delta_angle < -np.pi:
            delta_angle += 2 * np.pi
            
        total_angle_change += delta_angle
        last_angle = current_angle
        
    winding_number = total_angle_change / (2 * np.pi)
    return winding_number

# --- Main Execution and Validation Block ---

if __name__ == "__main__":
    print("--- Running Foundational 2D Validation (Corrected Path) ---")
    
    # 1. Generate the 2D vortex test case
    print("Generating 2D vortex field (W=+1)...")
    vortex_field = generate_vortex_2d(GRID_SIZE_2D)
    print("Generation complete.")

    # 2. Run the functional 2D winding number calculator
    print("\nValidating 2D calculator...")
    winding_number = calculate_winding_number_2d(vortex_field)

    print(f"  Measured Winding Number: {winding_number}")

    # 3. Assert correctness
    assert abs(winding_number - 1.0) < VALIDATION_TOLERANCE, "Winding number validation FAILED"

    print("\n✅ SUCCESS: The 2D winding number calculator is fully validated.")
    print("   This provides the 'bulletproof' foundation required to proceed.")