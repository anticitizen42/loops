#
# Bray's Loops 4.0 - Foundational Validation of Nonlinear Dynamics
# This script adds a nonlinear phi-four term to the 3D engine to validate
# the handling of nonlinear potentials via energy conservation.
#

import numpy as np
import cupy as cp

# --- Configuration ---
GRID_SIZE_3D = 32
TIME_STEPS = 10000
DT = 0.002
C = 1.0
LAMBDA = 0.1 # Strength of the nonlinear term

# --- Yoshida 4th Order Symplectic Integrator Coefficients ---
C1 = 0.6756035959798288
C2 = -0.1756035959798288
D1 = 1.3512071919596576
D2 = -1.7024143839193152

# --- Component 1: 3D Physics Engine (with Nonlinearity) ---

def calculate_laplacian_spectral_3d(field_gpu):
    """Calculates the 3D Laplacian using spectral methods."""
    field_k = cp.fft.fftn(field_gpu)
    k = cp.fft.fftfreq(field_gpu.shape[0]) * 2 * np.pi
    kx, ky, kz = cp.meshgrid(k, k, k, indexing='ij')
    ksquared = kx**2 + ky**2 + kz**2
    laplacian_k = -ksquared * field_k
    return cp.real(cp.fft.ifftn(laplacian_k))

def get_force(phi_gpu):
    """Calculates the force term, now including nonlinearity."""
    # Force = nabla^2(phi) - lambda*phi^3
    linear_force = C**2 * calculate_laplacian_spectral_3d(phi_gpu)
    nonlinear_force = -LAMBDA * phi_gpu**3
    return linear_force + nonlinear_force

def yoshida_step_3d(phi_gpu, pi_gpu):
    """Performs a single GPU-accelerated time step for the nonlinear system."""
    phi_new = phi_gpu + C1 * DT * pi_gpu
    pi_new = pi_gpu + D1 * DT * get_force(phi_new)

    phi_new = phi_new + C2 * DT * pi_new
    pi_new = pi_new + D2 * DT * get_force(phi_new)
    
    phi_new = phi_new + C2 * DT * pi_new
    pi_new = pi_new + D1 * DT * get_force(phi_new)
    
    phi_new = phi_new + C1 * DT * pi_new
    
    return phi_new, pi_new

# --- Component 2: Nonlinear Diagnostics ---

def calculate_energy_phi4_3d(phi_gpu, pi_gpu):
    """Calculates the total energy for the phi-four system."""
    # Energy = integral of (0.5*pi^2 + 0.5*(grad phi)^2 + 0.25*lambda*phi^4)
    k = cp.fft.fftfreq(phi_gpu.shape[0]) * 2 * np.pi
    kx, ky, kz = cp.meshgrid(k, k, k, indexing='ij')
    phi_k = cp.fft.fftn(phi_gpu)
    
    grad_phi_x = cp.real(cp.fft.ifftn(1j * kx * phi_k))
    grad_phi_y = cp.real(cp.fft.ifftn(1j * ky * phi_k))
    grad_phi_z = cp.real(cp.fft.ifftn(1j * kz * phi_k))
    
    gradient_energy = 0.5 * C**2 * (grad_phi_x**2 + grad_phi_y**2 + grad_phi_z**2)
    kinetic_energy = 0.5 * pi_gpu**2
    potential_energy = 0.25 * LAMBDA * phi_gpu**4
    
    return cp.sum(gradient_energy + kinetic_energy + potential_energy)

# --- Main Execution and Validation Block ---

if __name__ == "__main__":
    print("--- Running 3D Nonlinear Scalar Field Validation ---")
    
    # 1. Setup initial conditions
    x = np.arange(GRID_SIZE_3D)
    xx, yy, zz = np.meshgrid(x, x, x, indexing='ij')
    k_space = 2 * np.pi / GRID_SIZE_3D
    phi_initial_cpu = np.sin(k_space * xx) * np.sin(k_space * yy) * np.sin(k_space * zz)
    pi_initial_cpu = np.zeros(phi_initial_cpu.shape)
    
    print("Moving data to GPU...")
    phi = cp.asarray(phi_initial_cpu)
    pi = cp.asarray(pi_initial_cpu)
    
    # 2. Calculate initial energy and run simulation
    initial_energy = calculate_energy_phi4_3d(phi, pi)
    print(f"Initial Energy of Nonlinear System: {initial_energy.get():.14f}")

    print(f"Running {TIME_STEPS} steps on the GPU with nonlinearity (lambda={LAMBDA})...")
    for step in range(TIME_STEPS):
        phi, pi = yoshida_step_3d(phi, pi)

    cp.cuda.Stream.null.synchronize()
    print("Simulation complete.")

    final_energy = calculate_energy_phi4_3d(phi, pi)
    print(f"Final Energy after {TIME_STEPS} steps: {final_energy.get():.14f}")
    
    energy_drift = final_energy - initial_energy
    print(f"Total Energy Drift: {energy_drift.get():.4e}")
    
    assert abs(energy_drift.get()) < 1e-10, "Energy conservation FAILED for nonlinear system"

    print("\n✅ SUCCESS: The GPU-accelerated 3D engine is validated for nonlinear dynamics.")
    print("   Phase I: Foundation Reconstruction is now truly complete.")