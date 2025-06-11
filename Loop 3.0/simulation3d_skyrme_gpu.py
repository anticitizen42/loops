# Write the complete CPU script with full Skyrme force

#!/usr/bin/env python3

import cupy as cp
import numpy as np
import json, time
import matplotlib.pyplot as plt

print("▶ Starting 3D Skyrme CUDA simulation …")

# --- Load parameters ---
with open('simulation3d_skyrme_params.json') as f:
    p = json.load(f)
m      = p['m']
lam    = p['lambda']
kappa  = p['kappa']
L      = p['L']
N      = p['N']
dt     = p['dt']
n_steps = p['n_steps']
out    = p['output_every']
sigma  = p['sigma']
amp    = p['amplitude']
prefix = p['output_prefix']

# Override for initial test
if N > 64:
    print("⚠️ Overriding N to 64 for initial test.")
    N = 64

# Grid setup
dx = L / N
x_cpu = np.linspace(0, L, N, endpoint=False)
X_cpu, Y_cpu, Z_cpu = np.meshgrid(x_cpu, x_cpu, x_cpu, indexing='xy')

# Initialize fields on CPU first
psi_cpu = np.zeros((2, N, N, N), dtype=np.complex128)
gauss_cpu = amp * np.exp(-((X_cpu - L/2)**2 + (Y_cpu - L/2)**2 + (Z_cpu - L/2)**2)/(2*sigma**2))
psi_cpu[0] = gauss_cpu
psi_cpu[1] = gauss_cpu

# Move data to GPU
psi = cp.asarray(psi_cpu)
pi  = cp.zeros_like(psi)


# Spectral Laplacian precompute
kx_cpu = 2 * np.pi * np.fft.fftfreq(N, d=dx)
KX_cpu, KY_cpu, KZ_cpu = np.meshgrid(kx_cpu, kx_cpu, kx_cpu, indexing='ij')
k2_cpu = KX_cpu**2 + KY_cpu**2 + KZ_cpu**2
k2 = cp.asarray(k2_cpu)


def spectral_lap(phi):
    phi_hat = cp.fft.fftn(phi)
    return cp.fft.ifftn(-k2 * phi_hat)

def compute_force(psi):
    norm2 = cp.abs(psi[0])**2 + cp.abs(psi[1])**2
    F = cp.empty_like(psi)
    for i in (0,1):
        lap = spectral_lap(psi[i])
        F[i] = lap - (m*m * psi[i] + lam * norm2 * psi[i])
    return F

def compute_skyrme_force(psi):
    # Exact Skyrme via CP1 (n-vector) construction
    psi0, psi1 = psi[0], psi[1]
    norm2 = cp.abs(psi0)**2 + cp.abs(psi1)**2
    # n-field components
    n1 = (psi0.conj()*psi1 + psi1.conj()*psi0) / norm2
    n2 = (psi0.conj()*psi1 - psi1.conj()*psi0)/(1j*norm2)
    n3 = (cp.abs(psi0)**2 - cp.abs(psi1)**2) / norm2
    n = cp.stack([n1.real, n2.real, n3.real], axis=0)
    # spatial derivatives ∂i n
    dn = []
    for ax in range(3):
        dn.append((cp.roll(n, -1, axis=ax+1) - cp.roll(n, 1, axis=ax+1)) / (2*dx))
    # compute divergence of (dn_i × dn_j)
    Fsk_n = cp.zeros_like(n)
    for i in range(3):
        # build cross for each j
        C = cp.zeros_like(n)
        for j in range(3):
            C += cp.cross(dn[i], dn[j], axis=0)
        # divergence ∂j C_{ij}
        div = cp.zeros_like(n)
        for j in range(3):
            plus  = cp.roll(C, -1, axis=j+1)
            minus = cp.roll(C,  1, axis=j+1)
            div += (plus - minus) / (2*dx)
        Fsk_n += -kappa * div
    # project back onto psi: Fψ = σ^a ψ · Fsk_n^a
    Fpsi = cp.zeros_like(psi)
    # Pauli matrices - these can stay as numpy arrays and be used in cupy kernels
    sigmas = [
        np.array([[1,0],[0,-1]], dtype=complex),
        np.array([[0,1],[1,0]], dtype=complex),
        np.array([[0,-1j],[1j,0]], dtype=complex)
    ]
    for a, σ in enumerate(sigmas):
        F0 = σ[0,0]*psi0 + σ[0,1]*psi1
        F1 = σ[1,0]*psi0 + σ[1,1]*psi1
        Fpsi[0] += Fsk_n[a] * F0
        Fpsi[1] += Fsk_n[a] * F1
    return Fpsi


# 4th-order Yoshida coefficients
alpha = 1.0/(2 - 2**(1/3))
c1, c2 = alpha/2, (1 - 2**(1/3))*alpha/2
d1, d2 = alpha, -2**(1/3)*alpha

# Diagnostics
energies, times = [], []

t0 = time.time()
for step in range(n_steps):
    # step 1
    psi += c1 * dt * pi
    # step 2
    F   = compute_force(psi)
    Fsk = compute_skyrme_force(psi) if kappa!=0 else 0
    pi  += d1 * dt * (F + Fsk)
    # step 3
    psi += c2 * dt * pi
    # step 4
    F   = compute_force(psi)
    Fsk = compute_skyrme_force(psi) if kappa!=0 else 0
    pi  += d2 * dt * (F + Fsk)
    # step 5
    psi += c2 * dt * pi
    # step 6
    F   = compute_force(psi)
    Fsk = compute_skyrme_force(psi) if kappa!=0 else 0
    pi  += d1 * dt * (F + Fsk)
    # step 7
    psi += c1 * dt * pi

    if step % out == 0:
        t = step * dt
        norm2 = cp.abs(psi[0])**2 + cp.abs(psi[1])**2
        E = 0.0
        for i in (0,1):
            E += 0.5 * cp.sum(cp.abs(pi[i])**2) * dx**3
            lap = spectral_lap(psi[i])
            E += -0.5 * cp.real(cp.vdot(psi[i], lap)) * dx**3
        E += 0.5 * m*m * cp.sum(norm2) * dx**3
        E += 0.25 * lam * cp.sum(norm2**2) * dx**3
        energies.append(E.get()) # .get() moves data from GPU to CPU
        times.append(t)

t1 = time.time()
drift = energies[-1] - energies[0]

plt.figure()
plt.plot(times, energies, '-o', markersize=3)
plt.title(f'{prefix} CUDA Spectral Skyrme: Drift = {drift:.2e}')
plt.xlabel('Time'); plt.ylabel('Total Energy'); plt.grid(True)
plt.savefig(f'{prefix}_energy3d_cuda.png')
plt.show()

print(f"▶ Done: {t1-t0:.2f}s total, energy drift = {drift:.2e}")