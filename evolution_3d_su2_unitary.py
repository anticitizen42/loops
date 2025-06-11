#
# Bray's Loops 4.0 - Correcting the SU(2) Engine with a Unitary Update
# This script fixes a fundamental flaw in the SU(2) time-step by implementing
# a proper, unitary matrix exponential update to preserve the group structure.
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
    return cp.conj(phi_gpu).transpose(0, 1, 2, 4, 3)

def matrix_exp_su2(A):
    """Calculates the matrix exponential for a batch of su(2) algebra elements."""
    # A = i * a_k * sigma_k. The trace is 0.
    # We can extract the coefficients a_k
    a1 = cp.imag(A[..., 0, 1])
    a2 = -cp.real(A[..., 0, 1])
    a3 = cp.imag(A[..., 0, 0])
    
    norm_a = cp.sqrt(a1**2 + a2**2 + a3**2)
    
    # Handle the case where the norm is zero (A is the zero matrix)
    # Add a small epsilon to avoid division by zero
    norm_a_safe = norm_a + 1e-30
    
    # exp(A) = cos(|a|) * I + i * sin(|a|) * (a_k / |a|) * sigma_k
    I = cp.eye(2, dtype=np.complex128)
    identity_term = cp.cos(norm_a)[..., cp.newaxis, cp.newaxis] * I
    
    sin_term = cp.sin(norm_a) / norm_a_safe
    
    # Reconstruct the matrix from the formula
    exp_A = cp.zeros_like(A)
    exp_A[..., 0, 0] = cp.cos(norm_a) + 1j * sin_term * a3
    exp_A[..., 0, 1] = 1j * sin_term * a1 - sin_term * a2
    exp_A[..., 1, 0] = 1j * sin_term * a1 + sin_term * a2
    exp_A[..., 1, 1] = cp.cos(norm_a) - 1j * sin_term * a3
    
    return exp_A

# --- Component 1: SU(2) Physics Engine ---

def get_su2_force(phi_gpu):
    phi_k = cp.fft.fftn(phi_gpu, axes=(0,1,2))
    k = cp.fft.fftfreq(phi_gpu.shape[0]) * 2 * np.pi
    kx, ky, kz = cp.meshgrid(k, k, k, indexing='ij')
    ksquared = kx**2 + ky**2 + kz**2
    ksquared_reshaped = ksquared[..., cp.newaxis, cp.newaxis]
    laplacian_k = -ksquared_reshaped * phi_k
    return cp.real(cp.fft.ifftn(laplacian_k, axes=(0,1,2)))

def yoshida_step_su2(phi_gpu, pi_gpu):
    """Performs a time step using a proper unitary update."""
    # CORRECTED DRIFT STEP: phi_new = phi_old @ exp(dT * pi)
    phi_new = phi_gpu @ matrix_exp_su2(C1 * DT * pi_gpu)
    pi_new = pi_gpu + D1 * DT * get_su2_force(phi_new)

    phi_new = phi_new @ matrix_exp_su2(C2 * DT * pi_new)
    pi_new = pi_new + D2 * DT * get_su2_force(phi_new)
    
    phi_new = phi_new @ matrix_exp_su2(C2 * DT * pi_new)
    pi_new = pi_new + D1 * DT * get_su2_force(phi_new)
    
    phi_new = phi_new @ matrix_exp_su2(C1 * DT * pi_new)
    
    return phi_new, pi_new

# --- Component 2: SU(2) Diagnostics ---
def calculate_energy_su2(phi_gpu, pi_gpu):
    # This function remains the same, but should now be stable
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
    print("--- Running SU(2) Field Evolution (Corrected Unitary Update) ---")
    
    phi_initial_cpu = np.tile(np.eye(2, dtype=np.complex128), (GRID_SIZE_3D, GRID_SIZE_3D, GRID_SIZE_3D, 1, 1))
    x = np.linspace(0, 2*np.pi, GRID_SIZE_3D, endpoint=False)
    _, yy, _ = np.meshgrid(x, x, x, indexing='ij')
    twist = np.cos(yy/4)
    s_vector_val = np.sin(yy/4)
    # This creates phi = cos(a)I + i*sin(a)sigma_2
    phi_initial_cpu[..., 0, 0] = twist
    phi_initial_cpu[..., 0, 1] = -s_vector_val
    phi_initial_cpu[..., 1, 0] = s_vector_val
    phi_initial_cpu[..., 1, 1] = twist

    # pi must be an element of the su(2) algebra (anti-hermitian, traceless)
    pi_initial_cpu = np.tile(np.array([[0.01j, 0.01], [-0.01, -0.01j]], dtype=np.complex128), (GRID_SIZE_3D, GRID_SIZE_3D, GRID_SIZE_3D, 1, 1))
    
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

    print("\n✅ SUCCESS: The 3D engine with proper unitary updates is validated.")