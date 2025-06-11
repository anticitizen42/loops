#
# Bray's Loops - Incremental Assembly, Step 1
# Objective: Prove we can successfully isolate the k=0 (zero-frequency)
# component of a field in Fourier space.
#

import numpy as np
import cupy as cp

# --- Configuration ---
GRID_SIZE_3D = 32

if __name__ == "__main__":
    print("--- Step 1: Isolating the k=0 Mode ---")

    # 1. Create a simple field on the CPU. A field of all '1s'.
    field_cpu = np.ones((GRID_SIZE_3D, GRID_SIZE_3D, GRID_SIZE_3D), dtype=np.float64)
    # Add a small variation to make it non-trivial
    field_cpu[0,0,1] = 5.0
    
    # 2. Move to GPU
    field_gpu = cp.asarray(field_cpu)
    print("Field created on GPU.")

    # 3. Fourier Transform the field
    field_k = cp.fft.fftn(field_gpu)
    print("Field transformed to Fourier space.")

    # 4. Isolate the k=0 component (the value at index 0,0,0)
    k0_component = field_k[0, 0, 0]
    
    # The value of the k=0 component should be the sum of all elements in the real-space field.
    # Sum = (32*32*32 - 1) * 1.0 + 5.0 = 32768 - 1 + 5 = 32772
    expected_value = np.sum(field_cpu)
    print(f"\nExpected value at k=0: {expected_value}")
    print(f"Measured value at k=0: {k0_component.get()}")

    # 5. Assert that the measured value is correct
    assert abs(k0_component.get() - expected_value) < 1e-9, "Failed to isolate k=0 component."

    print("\n✅ SUCCESS: The k=0 mode was successfully isolated and its value is correct.")
    print("   This confirms the first piece of our logic is sound.")