"""
3D Skyrme-Term Scalar Field Evolution (SU(2) Doublet)

Potential: V = ½ m²|ψ|² + ¼ λ|ψ|⁴ + Skyrme term (κ)
4th-order Yoshida integrator, periodic BCs.

Usage:
    python simulation3d_skyrme.py

Outputs:
    - {prefix}_energy.png            : Energy vs time plot
    - {prefix}_slice_z0.25.png       : 2D slice at z/L=0.25
    - {prefix}_slice_z0.50.png       : 2D slice at z/L=0.50
    - {prefix}_slice_z0.75.png       : 2D slice at z/L=0.75
    - Performance and memory usage printed to console
"""

import numpy as np
import json
import time
import os
import matplotlib.pyplot as plt

def load_params(fname='simulation3d_skyrme_params.json'):
    with open(fname) as f:
        return json.load(f)

def laplacian3d(field, dx):
    return (
        np.roll(field,  1, axis=0) + np.roll(field, -1, axis=0) +
        np.roll(field,  1, axis=1) + np.roll(field, -1, axis=1) +
        np.roll(field,  1, axis=2) + np.roll(field, -1, axis=2) -
        6 * field
    ) / (dx * dx)

def compute_force_nonlinear(psi, dx, m, lam):
    # psi: shape (2, N, N, N), complex
    norm2 = np.abs(psi[0])**2 + np.abs(psi[1])**2
    F = np.zeros_like(psi)
    for i in (0,1):
        F[i] = laplacian3d(psi[i], dx) - (m*m * psi[i] + lam * norm2 * psi[i])
    return F

def compute_skyrme_force(psi, dx, kappa):
    # Placeholder: proper Skyrme force goes here
    # For now returns zero array; set kappa=0 to reproduce previous results.
    return np.zeros_like(psi)

def compute_force_total(psi, dx, m, lam, kappa):
    F = compute_force_nonlinear(psi, dx, m, lam)
    if kappa != 0:
        F += compute_skyrme_force(psi, dx, kappa)
    return F

def yoshida_step(psi, pi, dt, dx, m, lam, kappa):
    alpha = 1.0 / (2 - 2**(1/3))
    c1 = alpha / 2
    c2 = (1 - 2**(1/3)) * alpha / 2
    d1 = alpha
    d2 = -2**(1/3) * alpha

    # Step sequence
    psi = psi + c1 * dt * pi
    pi  = pi  + d1 * dt * compute_force_total(psi, dx, m, lam, kappa)
    psi = psi + c2 * dt * pi
    pi  = pi  + d2 * dt * compute_force_total(psi, dx, m, lam, kappa)
    psi = psi + c2 * dt * pi
    pi  = pi  + d1 * dt * compute_force_total(psi, dx, m, lam, kappa)
    psi = psi + c1 * dt * pi

    return psi, pi

def main():
    p = load_params()
    m     = p['m']
    lam   = p['lambda']
    kappa = p['kappa']
    L     = p['L']
    N     = p['N']
    dt    = p['dt']
    n_steps      = p['n_steps']
    output_every = p['output_every']
    sigma  = p['sigma']
    amp    = p['amplitude']
    z_slices_frac = p['z_slices']
    prefix = p['output_prefix']

    dx = L / N
    x  = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='xy')

    # Initialize SU(2) doublet: both components with same Gaussian amplitude
    psi = np.zeros((2, N, N, N), dtype=np.complex128)
    gaussian = amp * np.exp(-((X-L/2)**2 + (Y-L/2)**2 + (Z-L/2)**2) / (2 * sigma**2))
    psi[0] = gaussian
    psi[1] = gaussian
    pi = np.zeros_like(psi)

    energies, times = [], []
    slice_idxs = [int(frac * N) for frac in z_slices_frac]

    t0 = time.time()
    for step in range(n_steps):
        psi, pi = yoshida_step(psi, pi, dt, dx, m, lam, kappa)
        if step % output_every == 0:
            t = step * dt
            kinetic = 0.5 * (np.abs(pi)**2).sum(axis=0)
            grad_sq = np.zeros_like(psi[0], dtype=float)
            for i in (0,1):
                for ax in (0,1,2):
                    grad = (np.roll(psi[i], -1, axis=ax) - psi[i]) / dx
                    grad_sq += 0.5 * np.abs(grad)**2
            potential = 0.5 * m*m * (np.abs(psi)**2).sum(axis=0) +                         0.25 * lam * ((np.abs(psi)**2)**2).sum(axis=0)
            E_field = kinetic + grad_sq + potential
            E = np.sum(E_field) * (dx**3)
            energies.append(E)
            times.append(t)
    t1 = time.time()

    # Performance
    total_time = t1 - t0
    avg_step   = total_time / n_steps
    mem_MB = (psi.nbytes + pi.nbytes) / 1024**2

    # Save energy plot
    plt.figure()
    plt.plot(times, energies, '-o', markersize=3)
    drift = energies[-1] - energies[0]
    plt.xlabel('Time'); plt.ylabel('Total Energy')
    plt.title(f'3D Skyrme Drift = {drift:.2e}')
    plt.grid(True)
    fname_e = f"{prefix}_energy.png"
    plt.savefig(fname_e)
    plt.close()

    # Save slices
    for frac, idx in zip(z_slices_frac, slice_idxs):
        plt.figure()
        plt.imshow(np.real(psi[0,:,:,idx]), origin='lower',
                   extent=[0,L,0,L], cmap='viridis')
        plt.colorbar(label='Re(ψ₁)')
        plt.title(f"ψ₁ slice at z/L={frac}")
        f_s = f"{prefix}_slice_z{frac:.2f}.png"
        plt.savefig(f_s)
        plt.close()

    # Report
    print(f"Run time: {total_time:.2f}s ({avg_step*1e3:.3f}ms/step)")
    print(f"Memory: {mem_MB:.1f} MB for ψ+π arrays")
    print(f"Energy drift: {drift:.2e}")

if __name__ == '__main__':
    main()
