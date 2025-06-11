#
# Bray's Loops 4.0 - Phase I, Milestone 3: 3D Field Infrastructure
# This script implements a 3D scalar field evolution, extending the
# validated 2D engine. It confirms energy conservation in 3D.
#

import numpy as np

# --- Configuration ---
GRID_SIZE_3D = 32 # Reduced for faster 3D execution
TIME_STEPS = 10000
DT = 0.001 # Time step
C = 1.0 # Wave speed

# --- Yoshida 4th Order Symplectic Integrator Coefficients ---
C1 = 0.6756035959798288
C2 = -0.1756035959798288
D1 = 1.3512071919596576
D2 = -1.7024143839193152

# --- Component 1: 3D Physics Engine ---

def calculate_laplacian_spectral_3d(field):
    """Calculates the 3D Laplacian using spectral methods."""
    # Fourier transform the field in 3D
    field_k = np.fft.fftn(field)
    # Get wave numbers for 3D
    k = np.fft.fftfreq(field.shape[0]) * 2 * np.pi
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    ksquared = kx**2 + ky**2 + kz**2
    # The Laplacian in Fourier space is -k^2 * field_k
    laplacian_k = -ksquared * field_k
    # Inverse transform to get the result
    return np.real(np.fft.ifftn(laplacian_k))

def yoshida_step_3d(phi, pi):
    """Performs a single time step using the Yoshida 4th-order integrator in 3D."""
    # Using the same "gold standard" integrator 
    phi_new = phi + C1 * DT * pi
    pi_new = pi + D1 * DT * (C**2 * calculate_laplacian_spectral_3d(phi_new))

    phi_new = phi_new + C2 * DT * pi_new
    pi_new = pi_new + D2 * DT * (C**2 * calculate_laplacian_spectral_3d(phi_new))
    
    phi_new = phi_new + C2 * DT * pi_new
    pi_new = pi_new + D1 * DT * (C**2 * calculate_laplacian_spectral_3d(phi_new))
    
    phi_new = phi_new + C1 * DT * pi_new
    
    return phi_new, pi_new

# --- Component 2: 3D Diagnostics ---

def calculate_energy_3d(phi, pi):
    """Calculates the total energy of the 3D system."""
    # Energy = integral of (0.5 * pi^2 + 0.5 * c^2 * (grad phi)^2)
    k = np.fft.fftfreq(phi.shape[0]) * 2 * np.pi
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    phi_k = np.fft.fftn(phi)
    
    grad_phi_x = np.real(np.fft.ifftn(1j * kx * phi_k))
    grad_phi_y = np.real(np.fft.ifftn(1j * ky * phi_k))
    grad_phi_z = np.real(np.fft.ifftn(1j * kz * phi_k))
    
    potential_energy = 0.5 * C**2 * (grad_phi_x**2 + grad_phi_y**2 + grad_phi_z**2)
    kinetic_energy = 0.5 * pi**2
    
    return np.sum(potential_energy + kinetic_energy)

# --- Main Execution and Validation Block ---

if __name__ == "__main__":
    print("--- Running 3D Scalar Field Evolution Validation ---")
    
    # 1. Setup initial conditions for a 3D standing wave
    x = np.arange(GRID_SIZE_3D)
    xx, yy, zz = np.meshgrid(x, x, x, indexing='ij')
    
    k_space = 2 * np.pi / GRID_SIZE_3D
    phi_initial = np.cos(k_space * xx) * np.cos(k_space * yy) * np.cos(k_space * zz)
    pi_initial = np.zeros(phi_initial.shape)
    
    phi, pi = phi_initial, pi_initial
    
    # 2. Calculate initial energy and run simulation
    initial_energy = calculate_energy_3d(phi, pi)
    print(f"Initial Energy: {initial_energy:.14f}")

    for step in range(TIME_STEPS):
        phi, pi = yoshida_step_3d(phi, pi)

    final_energy = calculate_energy_3d(phi, pi)
    print(f"Final Energy after {TIME_STEPS} steps: {final_energy:.14f}")
    
    # 3. Assert energy conservation
    energy_drift = final_energy - initial_energy
    print(f"Total Energy Drift: {energy_drift:.4e}")
    
    assert abs(energy_drift) < 1e-10, "Energy conservation FAILED in 3D"

    print("\n✅ SUCCESS: The 3D evolution engine is validated.")
    print("   Phase I: Foundation Reconstruction is complete.")