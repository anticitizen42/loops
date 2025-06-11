# Updated simulation.py with relative paths and default writer

import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def load_params(fname='simulation_params.json'):
    with open(fname) as f:
        return json.load(f)

def main():
    # Load parameters
    p = load_params()
    m = p.get('m', 1.0)
    L = p.get('L', 10.0)
    N = p.get('N', 256)
    dt = p.get('dt', 0.005)
    n_steps = p.get('n_steps', 10000)
    output_every = p.get('output_every', 100)
    output_gif = p.get('output_gif', 'field_evolution.gif')

    dx = L / N
    x = np.linspace(0, L, N, endpoint=False)

    # Initial conditions
    phi = np.exp(- (x - L/2)**2)
    pi = np.zeros_like(phi)

    energies = []
    times = []
    snapshots = []

    # Time evolution
    for step in range(n_steps):
        # Half-step for pi
        lap = np.roll(phi, -1) - 2*phi + np.roll(phi, 1)
        pi += dt * (lap / dx**2 - m * phi)

        # Full-step for phi
        phi += dt * pi

        if step % output_every == 0:
            t = step * dt
            kinetic = 0.5 * pi**2
            grad = (np.roll(phi, -1) - phi) / dx
            grad_sq = 0.5 * grad**2
            potential = 0.5 * m * phi**2
            E = np.sum(kinetic + grad_sq + potential) * dx
            energies.append(E)
            times.append(t)
            snapshots.append(phi.copy())

    # Energy plot
    plt.figure()
    plt.plot(times, energies)
    plt.xlabel('Time')
    plt.ylabel('Total Energy')
    plt.title(f'Energy vs Time (drift = {energies[-1]-energies[0]:.2e})')
    plt.grid(True)
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

    # Save animation in current directory
    ani.save(output_gif)
    print(f"Animation saved as {output_gif} in {os.getcwd()}")

    # Output drift
    print("Energy drift over simulation: {:.2e}".format(energies[-1] - energies[0]))
    print("Simulation complete. Files saved in current directory.")

if __name__ == '__main__':
    main()

