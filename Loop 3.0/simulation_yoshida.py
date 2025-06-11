"""
Updated 1D Scalar Field Evolution Simulator with 4th-Order Symplectic Integrator (Yoshida)

Leapfrog has been replaced by the 4th-order splitting scheme to drastically reduce energy drift.

Usage:
    python simulation_yoshida.py

Outputs:
    - Energy vs time plot
    - Animation GIF of φ(x,t)
    - Printed energy drift
"""

import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def load_params(fname='simulation_params.json'):
    with open(fname) as f:
        return json.load(f)

def compute_force(phi, dx, m):
    """Compute spatial force: ∂²φ/∂x² - m φ (periodic BCs)."""
    lap = np.roll(phi, -1) - 2*phi + np.roll(phi, 1)
    return lap / dx**2 - m * phi

def main():
    # Load parameters
    p = load_params()
    m = p.get('m', 1.0)
    L = p.get('L', 10.0)
    N = p.get('N', 256)
    dt = p.get('dt', 0.001)
    n_steps = p.get('n_steps', 10000)
    output_every = p.get('output_every', 100)
    gif_name = p.get('output_gif', 'field_evolution_yoshida.gif')

    dx = L / N
    x = np.linspace(0, L, N, endpoint=False)

    # Initial conditions
    phi = np.exp(- (x - L/2)**2)
    pi = np.zeros_like(phi)

    # Yoshida 4th-order coefficients
    alpha = 1.0 / (2 - 2**(1/3))
    c1 =  alpha/2
    c2 = (1 - 2**(1/3))*alpha/2
    d1 = alpha
    d2 = -2**(1/3)*alpha

    energies, times, snapshots = [], [], []

    for step in range(n_steps):
        # 1)
        phi += c1 * dt * pi

        # 2)
        F = compute_force(phi, dx, m)
        pi += d1 * dt * F

        # 3)
        phi += c2 * dt * pi

        # 4)
        F = compute_force(phi, dx, m)
        pi += d2 * dt * F

        # 5)
        phi += c2 * dt * pi

        # 6)
        F = compute_force(phi, dx, m)
        pi += d1 * dt * F

        # 7)
        phi += c1 * dt * pi

        # Record outputs
        if step % output_every == 0:
            t = step * dt
            kinetic = 0.5 * pi**2
            grad = (np.roll(phi, -1) - phi) / dx
            potential = 0.5 * m * phi**2
            E = np.sum(kinetic + 0.5*grad**2 + potential) * dx
            energies.append(E)
            times.append(t)
            snapshots.append(phi.copy())

    # Energy plot
    plt.figure()
    plt.plot(times, energies, marker='.', markersize=4)
    plt.xlabel('Time')
    plt.ylabel('Total Energy')
    drift = energies[-1] - energies[0]
    plt.title(f'Yoshida 4th-order: Energy Drift = {drift:.2e}')
    plt.grid(True)
    plt.savefig('energy_yoshida.png')
    plt.show()

    # Animation
    fig, ax = plt.subplots()
    line, = ax.plot(x, snapshots[0])
    ax.set_xlim(0, L)
    ax.set_ylim(np.min(snapshots), np.max(snapshots))
    ax.set_xlabel('x')
    ax.set_ylabel('φ(x)')

    def update(i):
        line.set_ydata(snapshots[i])
        ax.set_title(f'Time = {times[i]:.2f}')
        return (line,)

    ani = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=50)
    ani.save(gif_name)
    print(f"Animation saved as {gif_name} in {os.getcwd()}")

    print("Energy drift over simulation: {:.2e}".format(drift))
    print("Done. Files: energy_yoshida.png,", gif_name)

if __name__ == '__main__':
    main()

