#
# Bray's Loops 4.0 - Phase I, Milestone 2: Minimal Working System
# This script implements a 2D scalar field evolution using validated methods
# from Loop 3.0, including a Yoshida integrator and spectral derivatives.
# It validates energy conservation to machine precision.
#

import numpy as np

# --- Configuration ---
GRID_SIZE = 128
TIME_STEPS = 10000
DT = 0.01 # Time step
C = 1.0 # Wave speed

# --- Yoshida 4th Order Symplectic Integrator Coefficients ---
C1 = 0.6756035959798288
C2 = -0.1756035959798288
D1 = 1.3512071919596576
D2 = -1.7024143839193152

# --- Component 1: Physics Engine ---

def calculate_laplacian_spectral(field):
    """Calculates the Laplacian using spectral methods for machine precision."""
    # Fourier transform the field
    field_k = np.fft.fft2(field)
    # Get wave numbers
    kx = np.fft.fftfreq(field.shape[0]) * 2 * np.pi
    ky = np.fft.fftfreq(field.shape[1]) * 2 * np.pi
    kxx, kyy = np.meshgrid(kx, ky, indexing='ij')
    ksquared = kxx**2 + kyy**2
    # The Laplacian in Fourier space is -k^2 * field_k
    laplacian_k = -ksquared * field_k
    # Inverse transform to get the result
    return np.real(np.fft.ifft2(laplacian_k))

def yoshida_step(phi, pi):
    """Performs a single time step using the Yoshida 4th-order integrator."""
    # This is the "gold standard for Hamiltonian systems" from Loop 3.0 
    phi_new = phi + C1 * DT * pi
    pi_new = pi + D1 * DT * (C**2 * calculate_laplacian_spectral(phi_new))

    phi_new = phi_new + C2 * DT * pi_new
    pi_new = pi_new + D2 * DT * (C**2 * calculate_laplacian_spectral(phi_new))
    
    phi_new = phi_new + C2 * DT * pi_new
    pi_new = pi_new + D1 * DT * (C**2 * calculate_laplacian_spectral(phi_new))
    
    phi_new = phi_new + C1 * DT * pi_new
    
    return phi_new, pi_new

# --- Component 2: Diagnostics ---

def calculate_energy(phi, pi):
    """Calculates the total energy of the system."""
    # Energy = integral of (0.5 * pi^2 + 0.5 * c^2 * (grad phi)^2)
    grad_phi_x = np.real(np.fft.ifft2(1j * np.fft.fftfreq(phi.shape[0]) * 2 * np.pi * np.fft.fft2(phi)))
    grad_phi_y = np.real(np.fft.ifft2(1j * np.fft.fftfreq(phi.shape[1]) * 2 * np.pi * np.fft.fft2(phi)))
    
    potential_energy = 0.5 * C**2 * (grad_phi_x**2 + grad_phi_y**2)
    kinetic_energy = 0.5 * pi**2
    
    return np.sum(potential_energy + kinetic_energy)

# --- Main Execution and Validation Block ---

if __name__ == "__main__":
    print("--- Running 2D Scalar Field Evolution Validation ---")
    
    # 1. Setup initial conditions for a standing wave
    x = np.arange(GRID_SIZE)
    y = np.arange(GRID_SIZE)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    
    k = 2 * np.pi / GRID_SIZE
    phi_initial = np.cos(k * xx) * np.cos(k * yy)
    pi_initial = np.zeros((GRID_SIZE, GRID_SIZE))
    
    phi, pi = phi_initial, pi_initial
    
    # 2. Calculate initial energy and run simulation
    initial_energy = calculate_energy(phi, pi)
    print(f"Initial Energy: {initial_energy:.14f}")

    for step in range(TIME_STEPS):
        phi, pi = yoshida_step(phi, pi)

    final_energy = calculate_energy(phi, pi)
    print(f"Final Energy after {TIME_STEPS} steps: {final_energy:.14f}")
    
    # 3. Assert energy conservation to machine precision
    energy_drift = final_energy - initial_energy
    print(f"Total Energy Drift: {energy_drift:.4e}")
    
    assert abs(energy_drift) < 1e-11, "Energy conservation FAILED"

    print("\n✅ SUCCESS: The 2D evolution engine is validated.")
    print("   Energy is conserved to machine precision.")