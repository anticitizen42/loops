"""
2D Scalar Field Evolution Simulator with 4th-Order Symplectic Integrator (Yoshida)

Usage:
    python simulation2d.py

Outputs:
    - energy2d.png    : Energy vs time plot
    - evolution2d.gif : Heatmap animation of φ(x,y,t)
"""

import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def load_params(fname='simulation2d_params.json'):
    with open(fname) as f:
        return json.load(f)

def laplacian2d(phi, dx):
    return (
        np.roll(phi,  1, axis=0) + np.roll(phi, -1, axis=0) +
        np.roll(phi,  1, axis=1) + np.roll(phi, -1, axis=1) -
        4 * phi
    ) / (dx * dx)

def compute_force(phi, dx, m):
    return laplacian2d(phi, dx) - m * phi

def yoshida_step(phi, pi, dt, dx, m):
    # 4th-order Yoshida coefficients
    alpha = 1.0 / (2 - 2**(1/3))
    c1 = alpha / 2
    c2 = (1 - 2**(1/3)) * alpha / 2
    d1 = alpha
    d2 = -2**(1/3) * alpha

    phi = phi + c1 * dt * pi
    pi  = pi  + d1 * dt * compute_force(phi, dx, m)
    phi = phi + c2 * dt * pi
    pi  = pi  + d2 * dt * compute_force(phi, dx, m)
    phi = phi + c2 * dt * pi
    pi  = pi  + d1 * dt * compute_force(phi, dx, m)
    phi = phi + c1 * dt * pi

    return phi, pi

def main():
    # Load simulation parameters
    p = load_params()
    m = p.get('m', 1.0)
    L = p.get('L', 10.0)
    N = p.get('N', 256)
    dt = p.get('dt', 0.001)
    n_steps = p.get('n_steps', 5000)
    output_every = p.get('output_every', 50)
    sigma = p.get('sigma', 0.5)
    amp = p.get('amplitude', 1.0)
    gif_name = p.get('output_gif', 'evolution2d.gif')

    dx = L / N
    x = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing='xy')

    # Initial conditions: Gaussian pulse
    phi = amp * np.exp(-((X - L/2)**2 + (Y - L/2)**2) / (2 * sigma**2))
    pi  = np.zeros_like(phi)

    energies = []
    times = []
    snapshots = []

    # Time-stepping loop
    for step in range(n_steps):
        phi, pi = yoshida_step(phi, pi, dt, dx, m)
        if step % output_every == 0:
            t = step * dt
            kinetic = 0.5 * pi**2
            gradx = (np.roll(phi, -1, axis=0) - phi) / dx
            grady = (np.roll(phi, -1, axis=1) - phi) / dx
            potential = 0.5 * m * phi**2
            E = np.sum(kinetic + 0.5 * (gradx**2 + grady**2) + potential) * (dx * dx)
            energies.append(E)
            times.append(t)
            snapshots.append(phi.copy())

    # Plot energy vs time
    plt.figure()
    plt.plot(times, energies, '-o', markersize=3)
    plt.xlabel('Time')
    plt.ylabel('Total Energy')
    drift = energies[-1] - energies[0]
    plt.title(f'2D Energy Drift = {drift:.2e}')
    plt.grid(True)
    plt.savefig('energy2d.png')
    plt.show()

    # Animate field snapshots as heatmaps
    fig, ax = plt.subplots()
    vmin, vmax = np.min(snapshots), np.max(snapshots)
    im = ax.imshow(
        snapshots[0],
        origin='lower',
        extent=[0, L, 0, L],
        vmin=vmin,
        vmax=vmax,
        cmap='viridis'
    )
    ax.set_title(f'Time = {times[0]:.2f}')
    plt.colorbar(im, ax=ax)

    def update(i):
        im.set_data(snapshots[i])
        ax.set_title(f'Time = {times[i]:.2f}')
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=100)
    ani.save(gif_name)
    print(f"Saved animation : {gif_name}")
    print(f"Saved energy plot: energy2d.png")
    print(f"Energy drift: {drift:.2e}")

if __name__ == '__main__':
    main()
