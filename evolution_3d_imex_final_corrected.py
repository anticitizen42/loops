#
# Bray's Loops 4.0 - Final Corrected Script
# This version corrects the final NameError typo.
# This stands as the final, functional result of the engine development phase.
#

import numpy as np
import cupy as cp

# --- Configuration ---
GRID_SIZE_3D = 32
TIME_STEPS = 10000
DT = 0.005

# --- Component 1: Stable IMEX Physics Engine ---

def imex_step_su2(phi_gpu, pi_gpu):
    """
    Performs one stable time step using the validated IMEX logic.
    """
    # 1. Go to Fourier space
    phi_k = cp.fft.fftn(phi_gpu, axes=(0,1,2))
    pi_k = cp.fft.fftn(pi_gpu, axes=(0,1,2))
    
    # 2. Isolate the k=0 components for their separate, simpler evolution
    phi_k_0 = phi_k[0, 0, 0].copy()
    pi_k_0 = pi_k[0, 0, 0].copy()

    # 3. Calculate frequencies for the k>0 rotational update
    k = cp.fft.fftfreq(GRID_SIZE_3D) * 2 * np.pi
    kx, ky, kz = cp.meshgrid(k, k, k, indexing='ij')
    omega = cp.sqrt(kx**2 + ky**2 + kz**2)
    omega_reshaped = omega[..., cp.newaxis, cp.newaxis]

    # 4. Evolve k>0 modes using the stable rotational formula
    cos_dt = cp.cos(omega_reshaped * DT)
    # CORRECTED: Define sin_dt explicitly before using it
    sin_dt = cp.sin(omega_reshaped * DT)
    sin_dt_over_omega = cp.where(omega_reshaped > 1e-9, sin_dt / omega_reshaped, 0)

    phi_k_rotated = phi_k * cos_dt + pi_k * sin_dt_over_omega
    pi_k_rotated = -phi_k * omega_reshaped * sin_dt + pi_k * cos_dt

    # 5. Evolve k=0 mode with its simple, non-rotational update
    phi_k_0_new = phi_k_0 + DT * pi_k_0
    pi_k_0_new = pi_k_0 

    # 6. Combine the results
    final_phi_k = phi_k_rotated
    final_pi_k = pi_k_rotated
    final_phi_k[0, 0, 0] = phi_k_0_new
    final_pi_k[0, 0, 0] = pi_k_0_new
    
    # 7. Transform back to real space
    phi_new = cp.fft.ifftn(final_phi_k, axes=(0,1,2))
    pi_new = cp.fft.ifftn(final_pi_k, axes=(0,1,2))

    return phi_new, pi_new

# --- Component 2: SU(2) Diagnostics ---

def get_su2_dagger(phi_gpu):
    return cp.conj(phi_gpu).transpose(0, 1, 2, 4, 3)

def calculate_energy_su2(phi_gpu, pi_gpu):
    pi_dagger = get_su2_dagger(pi_gpu)
    kinetic_energy = 0.5 * cp.real(cp.trace(pi_dagger @ pi_gpu, axis1=3, axis2=4))
    phi_k = cp.fft.fftn(phi_gpu, axes=(0,1,2))
    k = cp.fft.fftfreq(phi_gpu.shape[0]) * 2 * np.pi
    kx, ky, kz = cp.meshgrid(k, k, k, indexing='ij')
    d_phi_dx = cp.fft.ifftn(1j * kx[..., cp.newaxis, cp.newaxis] * phi_k, axes=(0,1,2))
    d_phi_dy = cp.fft.ifftn(1j * ky[..., cp.newaxis, cp.newaxis] * phi_k, axes=(0,1,2))
    d_phi_dz = cp.fft.ifftn(1j * kz[..., cp.newaxis, cp.newaxis] * phi_k, axes=(0,1,2))
    term_x = cp.real(cp.trace(get_su2_dagger(d_phi_dx) @ d_phi_dx, axis1=3, axis2=4))
    term_y = cp.real(cp.trace(get_su2_dagger(d_phi_dy) @ d_phi_dy, axis1=3, axis2=4))
    term_z = cp.real(cp.trace(get_su2_dagger(d_phi_dz) @ d_phi_dz, axis1=3, axis2=4))
    potential_energy = 0.5 * (term_x + term_y + term_z)
    return cp.sum(kinetic_energy + potential_energy)

# --- Main Execution Block ---

if __name__ == "__main__":
    print("--- Final Run of the Corrected IMEX SU(2) Engine ---")
    
    phi_initial_cpu = np.tile(np.eye(2, dtype=np.complex128), (GRID_SIZE_3D, GRID_SIZE_3D, GRID_SIZE_3D, 1, 1))
    pi_initial_cpu = np.tile(np.array([[0.01j, 0.01], [-0.01, -0.01j]], dtype=np.complex128), (GRID_SIZE_3D, GRID_SIZE_3D, GRID_SIZE_3D, 1, 1))
    
    phi = cp.asarray(phi_initial_cpu)
    pi = cp.asarray(pi_initial_cpu)
    
    initial_energy = calculate_energy_su2(phi, pi)
    print(f"Initial SU(2) System Energy: {initial_energy.get():.14f}")

    print(f"Running {TIME_STEPS} steps...")
    for step in range(TIME_STEPS):
        phi, pi = imex_step_su2(phi, pi)

    cp.cuda.Stream.null.synchronize()
    print("Simulation complete.")

    final_energy = calculate_energy_su2(phi, pi)
    print(f"Final SU(2) System Energy: {final_energy.get():.14f}")
    
    energy_drift = final_energy - initial_energy
    print(f"Total Energy Drift: {energy_drift.get():.4e}")
    
    assert abs(energy_drift.get()) < 1e-9, "Energy conservation FAILED for SU(2) system"

    print("\n✅ SUCCESS: The IMEX engine is stable and conserves energy to machine precision.")