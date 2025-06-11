#
# Bray's Loops 4.0 - Phase II, Milestone 5: Complex Field Systems
# This script corrects a broadcasting error in the SU(2) force calculation
# and implements the full energy diagnostic for the Principal Chiral Model.
#

import numpy as np
import cupy as cp

# --- Configuration ---
GRID_SIZE_3D = 32
TIME_STEPS = 1000
DT = 0.005

# --- Yoshida 4th Order Symplectic Integrator Coefficients ---
C1 = 0.6756035959798288
C2 = -0.1756035959798288
D1 = 1.3512071919596576
D2 = -1.7024143839193152

# --- SU(2) Algebra Helpers ---
def get_su2_dagger(phi_gpu):
    """Calculates the conjugate transpose of an SU(2) matrix field."""
    return cp.conj(phi_gpu).transpose(0, 1, 2, 4, 3)

# --- Component 1: SU(2) Physics Engine ---

def get_su2_force(phi_gpu):
    """
    Calculates the force term for the SU(2) Principal Chiral Model.
    The force is proportional to the Laplacian of the field.
    """
    # CORRECTED IMPLEMENTATION:
    # 1. Fourier transform the SU(2) field
    phi_k = cp.fft.fftn(phi_gpu, axes=(0,1,2))
    
    # 2. Get 3D wave numbers
    k = cp.fft.fftfreq(phi_gpu.shape[0]) * 2 * np.pi
    kx, ky, kz = cp.meshgrid(k, k, k, indexing='ij')
    ksquared = kx**2 + ky**2 + kz**2
    
    # 3. Reshape ksquared to (G, G, G, 1, 1) to allow broadcasting with phi_k (G, G, G, 2, 2)
    ksquared_reshaped = ksquared[..., cp.newaxis, cp.newaxis]
    
    # 4. The Laplacian in Fourier space is -k^2 * phi_k
    laplacian_k = -ksquared_reshaped * phi_k
    
    # 5. Inverse transform to get the force in real space
    return cp.real(cp.fft.ifftn(laplacian_k, axes=(0,1,2)))

def yoshida_step_su2(phi_gpu, pi_gpu):
    """Performs a single GPU-accelerated time step for the SU(2) system."""
    # Using an approximate matrix exponential for the update: phi_new = phi * exp(dT * pi)
    # For small dT, this is phi * (I + dT*pi)
    phi_new = phi_gpu @ (cp.eye(2, dtype=np.complex128) + C1 * DT * pi_gpu)
    pi_new = pi_gpu + D1 * DT * get_su2_force(phi_new)

    phi_new = phi_new @ (cp.eye(2, dtype=np.complex128) + C2 * DT * pi_new)
    pi_new = pi_new + D2 * DT * get_su2_force(phi_new)
    
    phi_new = phi_new @ (cp.eye(2, dtype=np.complex128) + C2 * DT * pi_new)
    pi_new = pi_new + D1 * DT * get_su2_force(phi_new)
    
    phi_new = phi_new @ (cp.eye(2, dtype=np.complex128) + C1 * DT * pi_new)
    
    return phi_new, pi_new

# --- Component 2: SU(2) Diagnostics ---

def calculate_energy_su2(phi_gpu, pi_gpu):
    """Calculates the total energy for the SU(2) Principal Chiral Model."""
    # H = 0.5 * integral Tr(pi_dagger*pi + d_i(phi)_dagger*d_i(phi))
    
    # Kinetic Term
    pi_dagger = get_su2_dagger(pi_gpu)
    kinetic_energy = 0.5 * cp.real(cp.trace(pi_dagger @ pi_gpu, axis1=3, axis2=4))
    
    # Potential Term (Gradient Energy) - CORRECTED
    phi_k = cp.fft.fftn(phi_gpu, axes=(0,1,2))
    k = cp.fft.fftfreq(phi_gpu.shape[0]) * 2 * np.pi
    kx, ky, kz = cp.meshgrid(k, k, k, indexing='ij')

    # Calculate d_i(phi) in Fourier space
    d_phi_dx_k = 1j * kx[..., cp.newaxis, cp.newaxis] * phi_k
    d_phi_dy_k = 1j * ky[..., cp.newaxis, cp.newaxis] * phi_k
    d_phi_dz_k = 1j * kz[..., cp.newaxis, cp.newaxis] * phi_k

    # Transform back to real space
    d_phi_dx = cp.fft.ifftn(d_phi_dx_k, axes=(0,1,2))
    d_phi_dy = cp.fft.ifftn(d_phi_dy_k, axes=(0,1,2))
    d_phi_dz = cp.fft.ifftn(d_phi_dz_k, axes=(0,1,2))
    
    # Calculate Tr(d_i(phi)_dagger * d_i(phi))
    term_x = cp.real(cp.trace(get_su2_dagger(d_phi_dx) @ d_phi_dx, axis1=3, axis2=4))
    term_y = cp.real(cp.trace(get_su2_dagger(d_phi_dy) @ d_phi_dy, axis1=3, axis2=4))
    term_z = cp.real(cp.trace(get_su2_dagger(d_phi_dz) @ d_phi_dz, axis1=3, axis2=4))
    
    potential_energy = 0.5 * (term_x + term_y + term_z)
    
    return cp.sum(kinetic_energy + potential_energy)

# --- Main Execution Block ---

if __name__ == "__main__":
    print("--- Running SU(2) Field Evolution Validation (Corrected) ---")
    
    phi_initial_cpu = np.tile(np.eye(2, dtype=np.complex128), (GRID_SIZE_3D, GRID_SIZE_3D, GRID_SIZE_3D, 1, 1))
    # Add a small initial twist to create non-zero potential energy
    x = np.linspace(0, 2*np.pi, GRID_SIZE_3D, endpoint=False)
    _, yy, _ = np.meshgrid(x, x, x, indexing='ij')
    twist = np.cos(yy/4)
    s_vector = np.sin(yy/4)
    phi_initial_cpu[..., 0, 0] = twist + 1j*0
    phi_initial_cpu[..., 0, 1] = 0 + 1j*s_vector
    phi_initial_cpu[..., 1, 0] = 0 + 1j*s_vector
    phi_initial_cpu[..., 1, 1] = twist - 1j*0

    pi_initial_cpu = np.zeros_like(phi_initial_cpu)
    
    print("Moving data to GPU...")
    phi = cp.asarray(phi_initial_cpu)
    pi = cp.asarray(pi_initial_cpu)
    
    initial_energy = calculate_energy_su2(phi, pi)
    print(f"Initial SU(2) System Energy: {initial_energy.get():.14f}")

    print(f"Running {TIME_STEPS} steps...")
    for step in range(TIME_STEPS):
        phi, pi = yoshida_step_su2(phi, pi)

    cp.cuda.Stream.null.synchronize()
    print("Simulation complete.")

    final_energy = calculate_energy_su2(phi, pi)
    print(f"Final SU(2) System Energy: {final_energy.get():.14f}")
    
    energy_drift = final_energy - initial_energy
    print(f"Total Energy Drift: {energy_drift.get():.4e}")
    
    assert abs(energy_drift.get()) < 1e-9, "Energy conservation FAILED for SU(2) system"

    print("\n✅ SUCCESS: The 3D engine is validated for SU(2) field dynamics.")