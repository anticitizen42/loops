#
# Bray's Loops 4.0 - Phase I, Milestone 3: GPU-Accelerated 3D Engine
# This script upgrades the validated 3D engine to CUDA using the CuPy library.
# This aligns with the "GPU acceleration" requirement for performance. 
#

import numpy as np
import cupy as cp # Import CuPy, the CUDA-accelerated NumPy equivalent

# --- Configuration ---
GRID_SIZE_3D = 64 # Increased grid size to demonstrate GPU advantage
TIME_STEPS = 10000
DT = 0.005 
C = 1.0

# --- Yoshida 4th Order Symplectic Integrator Coefficients ---
# (These remain unchanged)
C1 = 0.6756035959798288
C2 = -0.1756035959798288
D1 = 1.3512071919596576
D2 = -1.7024143839193152

# --- Component 1: 3D Physics Engine (on GPU) ---

def calculate_laplacian_spectral_3d(field_gpu):
    """Calculates the 3D Laplacian using CuPy's accelerated FFT library."""
    # The logic is identical to the NumPy version, but operations run on the GPU.
    # Use cupy.fft for GPU-based Fourier transforms.
    field_k = cp.fft.fftn(field_gpu)
    
    # Create wave numbers directly on the GPU.
    k = cp.fft.fftfreq(field_gpu.shape[0]) * 2 * np.pi
    kx, ky, kz = cp.meshgrid(k, k, k, indexing='ij')
    ksquared = kx**2 + ky**2 + kz**2
    
    laplacian_k = -ksquared * field_k
    return cp.real(cp.fft.ifftn(laplacian_k))

def yoshida_step_3d(phi_gpu, pi_gpu):
    """Performs a single GPU-accelerated time step."""
    # All array arithmetic now happens on the GPU via CuPy.
    phi_new = phi_gpu + C1 * DT * pi_gpu
    pi_new = pi_gpu + D1 * DT * (C**2 * calculate_laplacian_spectral_3d(phi_new))

    phi_new = phi_new + C2 * DT * pi_new
    pi_new = pi_new + D2 * DT * (C**2 * calculate_laplacian_spectral_3d(phi_new))
    
    phi_new = phi_new + C2 * DT * pi_new
    pi_new = pi_new + D1 * DT * (C**2 * calculate_laplacian_spectral_3d(phi_new))
    
    phi_new = phi_new + C1 * DT * pi_new
    
    return phi_new, pi_new

# --- Component 2: 3D Diagnostics (on GPU) ---

def calculate_energy_3d(phi_gpu, pi_gpu):
    """Calculates the total energy of the system on the GPU."""
    # Use CuPy for all calculations.
    k = cp.fft.fftfreq(phi_gpu.shape[0]) * 2 * np.pi
    kx, ky, kz = cp.meshgrid(k, k, k, indexing='ij')
    phi_k = cp.fft.fftn(phi_gpu)
    
    grad_phi_x = cp.real(cp.fft.ifftn(1j * kx * phi_k))
    grad_phi_y = cp.real(cp.fft.ifftn(1j * ky * phi_k))
    grad_phi_z = cp.real(cp.fft.ifftn(1j * kz * phi_k))
    
    potential_energy = 0.5 * C**2 * (grad_phi_x**2 + grad_phi_y**2 + grad_phi_z**2)
    kinetic_energy = 0.5 * pi_gpu**2
    
    # cp.sum performs the reduction on the GPU.
    return cp.sum(potential_energy + kinetic_energy)

# --- Main Execution and Validation Block ---

if __name__ == "__main__":
    print("--- Running GPU-Accelerated 3D Scalar Field Evolution ---")
    
    # 1. Setup initial conditions on the CPU (NumPy)
    x = np.arange(GRID_SIZE_3D)
    xx, yy, zz = np.meshgrid(x, x, x, indexing='ij')
    
    k_space = 2 * np.pi / GRID_SIZE_3D
    phi_initial_cpu = np.cos(k_space * xx) * np.cos(k_space * yy) * np.cos(k_space * zz)
    pi_initial_cpu = np.zeros(phi_initial_cpu.shape)
    
    # 2. Move data from CPU to GPU memory
    print("Moving data to GPU...")
    phi = cp.asarray(phi_initial_cpu)
    pi = cp.asarray(pi_initial_cpu)
    
    # 3. Calculate initial energy and run simulation on the GPU
    initial_energy = calculate_energy_3d(phi, pi)
    # To print the value, we must move it back to the CPU via .get()
    print(f"Initial Energy (on GPU): {initial_energy.get():.14f}")

    print(f"Running {TIME_STEPS} steps on the GPU...")
    for step in range(TIME_STEPS):
        phi, pi = yoshida_step_3d(phi, pi)

    # Synchronize the GPU to ensure all calculations are complete before timing
    cp.cuda.Stream.null.synchronize()
    print("Simulation complete.")

    final_energy = calculate_energy_3d(phi, pi)
    print(f"Final Energy after {TIME_STEPS} steps: {final_energy.get():.14f}")
    
    # 4. Assert energy conservation
    energy_drift = final_energy - initial_energy
    print(f"Total Energy Drift: {energy_drift.get():.4e}")
    
    assert abs(energy_drift.get()) < 1e-10, "Energy conservation FAILED on GPU"

    print("\n✅ SUCCESS: The GPU-accelerated 3D evolution engine is validated.")
    print("   Phase I: Foundation Reconstruction is complete.")