"""
2D Nonlinear Scalar Field Evolution with φ⁴ Interaction

Potential: V(φ) = ½ m² φ² + ¼ λ φ⁴
Uses 4th-order Yoshida integrator.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def load_params(fname='simulation2d_nonlinear_params.json'):
    with open(fname) as f:
        return json.load(f)

def laplacian2d(phi, dx):
    return (
        np.roll(phi,  1, axis=0) + np.roll(phi, -1, axis=0) +
        np.roll(phi,  1, axis=1) + np.roll(phi, -1, axis=1) -
        4 * phi
    ) / (dx * dx)

def compute_force(phi, dx, m, lam):
    # ∂²φ - dV/dφ with V = ½ m²φ² + ¼ λφ⁴
    return laplacian2d(phi, dx) - (m*m*phi + lam*phi**3)

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
    m   = p['m']
    lam = p['lambda']
    L   = p['L']
    N   = p['N']
    dt  = p['dt']
    n_steps       = p['n_steps']
    output_every  = p['output_every']
    sigma         = p['sigma']
    amp           = p['amplitude']
    gif_name      = p['output_gif']

    dx = L / N
    x  = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing='xy')

    # Initial Gaussian pulse
    phi = amp * np.exp(-((X-L/2)**2 + (Y-L/2)**2)/(2*sigma**2))
    pi  = np.zeros_like(phi)

    energies, times, snapshots = [], [], []

    for step in range(n_steps):
        phi, pi = yoshida_step(phi, pi, dt, dx, m, lam)
        if step % output_every == 0:
            t = step*dt
            kin = 0.5 * pi**2
            gradx = (np.roll(phi,-1,0) - phi)/dx
            grady = (np.roll(phi,-1,1) - phi)/dx
            pot = 0.5*(m*m)*phi**2 + 0.25*lam*phi**4
            E = np.sum(kin + 0.5*(gradx**2+grady**2) + pot) * dx*dx
            energies.append(E)
            times.append(t)
            snapshots.append(phi.copy())

    # Plot energy drift
    plt.figure()
    plt.plot(times, energies, '-o', markersize=3)
    drift = energies[-1] - energies[0]
    plt.xlabel('Time'); plt.ylabel('Total Energy')
    plt.title(f'2D Nonlinear Energy Drift = {drift:.2e}')
    plt.grid(True)
    plt.savefig('energy2d_nl.png')
    plt.show()

    # Animate field
    fig, ax = plt.subplots()
    vmin, vmax = np.min(snapshots), np.max(snapshots)
    im = ax.imshow(snapshots[0], origin='lower',
                   extent=[0,L,0,L], vmin=vmin, vmax=vmax,
                   cmap='viridis')
    plt.colorbar(im, ax=ax)
    ax.set_title(f'Time = {times[0]:.2f}')

    def update(i):
        im.set_data(snapshots[i])
        ax.set_title(f'Time = {times[i]:.2f}')
        return [im]

    ani = animation.FuncAnimation(fig, update,
                                  frames=len(snapshots),
                                  interval=100)
    ani.save(gif_name)
    print(f"Saved energy plot: energy2d_nl.png")
    print(f"Saved animation : {gif_name}")
    print(f"Final energy drift: {drift:.2e}")

if __name__ == '__main__':
    main()
