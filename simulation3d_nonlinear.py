"""
3D Nonlinear Scalar Field Evolution with φ⁴ Interaction

Potential: V(φ) = ½ m² φ² + ¼ λ φ⁴
4th-order Yoshida integrator, periodic BCs.

Usage:
    python simulation3d_nonlinear.py

Outputs:
    - {prefix}_energy.png        : Energy vs time plot
    - {prefix}_slice_z{f1}.png   : 2D slice at z=f1 (fraction of L)
    - {prefix}_slice_z{f2}.png   : 2D slice at z=f2
    - {prefix}_slice_z{f3}.png   : 2D slice at z=f3
    - Performance and memory usage printed to console
"""

import numpy as np
import json
import time
import os
import matplotlib.pyplot as plt

def load_params(fname='simulation3d_nonlinear_params.json'):
    with open(fname) as f:
        return json.load(f)

def laplacian3d(phi, dx):
    return (
        np.roll(phi,  1, axis=0) + np.roll(phi, -1, axis=0) +
        np.roll(phi,  1, axis=1) + np.roll(phi, -1, axis=1) +
        np.roll(phi,  1, axis=2) + np.roll(phi, -1, axis=2) -
        6 * phi
    ) / (dx * dx)

def compute_force(phi, dx, m, lam):
    return laplacian3d(phi, dx) - (m*m*phi + lam*phi**3)

def yoshida_step(phi, pi, dt, dx, m, lam):
    alpha = 1.0 / (2 - 2**(1/3))
    c1 = alpha / 2
    c2 = (1 - 2**(1/3)) * alpha / 2
    d1 = alpha
    d2 = -2**(1/3) * alpha

    phi = phi + c1 * dt * pi
    pi  = pi  + d1 * dt * compute_force(phi, dx, m, lam)
    phi = phi + c2 * dt * pi
    pi  = pi  + d2 * dt * compute_force(phi, dx, m, lam)
    phi = phi + c2 * dt * pi
    pi  = pi  + d1 * dt * compute_force(phi, dx, m, lam)
    phi = phi + c1 * dt * pi

    return phi, pi

def main():
    p = load_params()
    m = p['m']
    lam = p['lambda']
    L = p['L']
    N = p['N']
    dt = p['dt']
    n_steps = p['n_steps']
    output_every = p['output_every']
    sigma = p['sigma']
    amp = p['amplitude']
    z_slices_frac = p['z_slices']
    prefix = p['output_prefix']

    dx = L / N
    x = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='xy')

    # Initial Gaussian pulse
    phi = amp * np.exp(-((X - L/2)**2 + (Y - L/2)**2 + (Z - L/2)**2) / (2 * sigma**2))
    pi  = np.zeros_like(phi)

    energies, times = [], []
    # prepare slices storage: for final time only
    slice_indices = [int(frac * N) for frac in z_slices_frac]

    # performance timing
    t0 = time.time()

    for step in range(n_steps):
        phi, pi = yoshida_step(phi, pi, dt, dx, m, lam)
        if step % output_every == 0:
            t = step * dt
            kin = 0.5 * pi**2
            # gradient terms
            grad_sq = 0.0
            # we approximate gradient squared via nearest neighbor difference sums
            for axis in [0,1,2]:
                grad = (np.roll(phi, -1, axis=axis) - phi) / dx
                grad_sq += 0.5 * grad**2
            pot = 0.5*(m*m)*phi**2 + 0.25*lam*phi**4
            E = np.sum(kin + grad_sq + pot) * dx**3
            energies.append(E)
            times.append(t)

    t1 = time.time()
    total_time = t1 - t0
    avg_step = total_time / n_steps

    # Memory footprint approximation
    phi_mem = phi.nbytes / (1024**2)
    pi_mem  = pi.nbytes  / (1024**2)
    total_mem = phi_mem + pi_mem

    # Save energy plot
    plt.figure()
    plt.plot(times, energies, '-o', markersize=3)
    drift = energies[-1] - energies[0]
    plt.xlabel('Time'); plt.ylabel('Total Energy')
    plt.title(f'3D Nonlinear Drift = {drift:.2e}')
    plt.grid(True)
    fname_energy = f"{prefix}_energy.png"
    plt.savefig(fname_energy)
    plt.close()

    # Save slice plots at final φ
    phi_final = phi
    for frac, idx in zip(z_slices_frac, slice_indices):
        plt.figure()
        plt.imshow(phi_final[:,:,idx], origin='lower', extent=[0,L,0,L], cmap='viridis')
        plt.colorbar(label='φ')
        plt.title(f"Slice at z/L={frac}")
        fname_slice = f"{prefix}_slice_z{frac:.2f}.png"
        plt.savefig(fname_slice)
        plt.close()

    # Print performance and memory
    print(f"Total run time: {total_time:.2f} s")
    print(f"Average per step: {avg_step*1e3:.4f} ms")
    print(f"Memory footprint: Field arrays = {total_mem:.2f} MB (phi+pi)")
    print(f"Energy drift: {drift:.2e}")

if __name__ == '__main__':
    main()
