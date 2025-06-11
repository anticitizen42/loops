#!/usr/bin/env python3
"""
Test with boundary-compatible initial conditions
"""

import numpy as np
import matplotlib.pyplot as plt

def create_boundary_compatible_gaussian(x, L, width=1.0, amplitude=1.0):
    """Create Gaussian that naturally goes to zero at boundaries"""
    x0 = L / 2
    
    # Basic Gaussian
    gaussian = amplitude * np.exp(-((x - x0) / width)**2)
    
    # Multiply by sin function that ensures zero boundaries
    # Use sin(πx/L) which is zero at x=0 and x=L
    boundary_factor = np.sin(np.pi * x / L)
    
    return gaussian * boundary_factor

def test_compatible_gaussian():
    """Test Gaussian that naturally satisfies boundary conditions"""
    print("=== Boundary-Compatible Gaussian Test ===")
    
    N = 128
    L = 10.0
    dx = L / (N - 1)
    dt = 0.001
    m = 1.0
    
    x = np.linspace(0, L, N)
    
    # Create boundary-compatible Gaussian
    phi = create_boundary_compatible_gaussian(x, L, width=1.0, amplitude=0.1)
    pi = np.zeros(N)
    
    print(f"Boundary values: φ(0)={phi[0]:.2e}, φ(L)={phi[-1]:.2e}")
    
    times = []
    energies = []
    snapshots = []
    
    n_steps = 10000
    
    for step in range(n_steps):
        # Energy calculation
        kinetic = 0.5 * np.trapz(pi**2, dx=dx)
        grad_phi = np.gradient(phi, dx)
        gradient = 0.5 * np.trapz(grad_phi**2, dx=dx)
        potential = 0.5 * m**2 * np.trapz(phi**2, dx=dx)
        E = kinetic + gradient + potential
        
        if step % 1000 == 0:
            times.append(step * dt)
            energies.append(E)
            snapshots.append(phi.copy())
            print(f"Step {step:5d}: E={E:.6f}, boundary: φ(0)={phi[0]:.2e}, φ(L)={phi[-1]:.2e}")
        
        # Evolution
        lap = np.zeros(N)
        lap[1:-1] = (phi[2:] - 2*phi[1:-1] + phi[:-2]) / dx**2
        force = lap - m**2 * phi
        
        # Leapfrog step
        pi += 0.5 * dt * force
        phi += dt * pi
        # Note: NO boundary enforcement needed since field naturally satisfies them
        
        lap[1:-1] = (phi[2:] - 2*phi[1:-1] + phi[:-2]) / dx**2
        force = lap - m**2 * phi
        pi += 0.5 * dt * force
    
    # Check energy conservation
    energy_drift = abs(energies[-1] - energies[0]) / energies[0]
    print(f"\nEnergy drift: {energy_drift:.2e}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Energy plot
    ax1.plot(times, energies, 'b-', linewidth=2)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Total Energy')
    ax1.set_title(f'Energy Conservation\n(drift = {energy_drift:.2e})')
    ax1.grid(True, alpha=0.3)
    
    # Field evolution
    for i, t in enumerate(times):
        ax2.plot(x, snapshots[i], label=f't={t:.1f}')
    ax2.set_xlabel('x')
    ax2.set_ylabel('φ(x)')
    ax2.set_title('Field Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('compatible_gaussian_test.png', dpi=150)
    plt.show()
    
    if energy_drift < 1e-10:
        print("✅ EXCELLENT: Energy conserved to machine precision")
        return True
    elif energy_drift < 1e-6:
        print("✅ GOOD: Energy well conserved")
        return True
    else:
        print("❌ POOR: Energy not conserved")
        return False

def test_multiple_modes():
    """Test superposition of boundary-compatible modes"""
    print("\n=== Multiple Mode Superposition Test ===")
    
    N = 128
    L = 10.0
    dx = L / (N - 1)
    dt = 0.0005  # Smaller timestep for stability
    m = 1.0
    
    x = np.linspace(0, L, N)
    
    # Superposition of first few sine modes
    phi = (0.1 * np.sin(np.pi * x / L) + 
           0.05 * np.sin(2 * np.pi * x / L) + 
           0.02 * np.sin(3 * np.pi * x / L))
    pi = np.zeros(N)
    
    print(f"Boundary values: φ(0)={phi[0]:.2e}, φ(L)={phi[-1]:.2e}")
    
    times = []
    energies = []
    
    n_steps = 20000
    
    for step in range(n_steps):
        # Energy calculation
        kinetic = 0.5 * np.trapz(pi**2, dx=dx)
        grad_phi = np.gradient(phi, dx)
        gradient = 0.5 * np.trapz(grad_phi**2, dx=dx)
        potential = 0.5 * m**2 * np.trapz(phi**2, dx=dx)
        E = kinetic + gradient + potential
        
        if step % 2000 == 0:
            times.append(step * dt)
            energies.append(E)
            print(f"Step {step:5d}: E={E:.6f}")
        
        # Evolution
        lap = np.zeros(N)
        lap[1:-1] = (phi[2:] - 2*phi[1:-1] + phi[:-2]) / dx**2
        force = lap - m**2 * phi
        
        # Leapfrog step
        pi += 0.5 * dt * force
        phi += dt * pi
        
        lap[1:-1] = (phi[2:] - 2*phi[1:-1] + phi[:-2]) / dx**2
        force = lap - m**2 * phi
        pi += 0.5 * dt * force
    
    # Check energy conservation
    energy_drift = abs(energies[-1] - energies[0]) / energies[0]
    print(f"\nEnergy drift: {energy_drift:.2e}")
    
    if energy_drift < 1e-10:
        print("✅ EXCELLENT: Energy conserved to machine precision")
        return True
    elif energy_drift < 1e-6:
        print("✅ GOOD: Energy well conserved")
        return True
    else:
        print("❌ POOR: Energy not conserved")
        return False

def main():
    print("Boundary-Compatible Initial Conditions Test")
    print("=" * 50)
    
    # Test boundary-compatible Gaussian
    test1 = test_compatible_gaussian()
    
    # Test multiple modes
    test2 = test_multiple_modes()
    
    print(f"\n=== RESULTS ===")
    print(f"Compatible Gaussian: {'✅' if test1 else '❌'}")
    print(f"Multiple modes: {'✅' if test2 else '❌'}")
    
    if test1 and test2:
        print("\n🎉 SOLUTION FOUND!")
        print("The key is using initial conditions that naturally satisfy boundary conditions")
        print("Next: Build Loop 4.0 with proper boundary-compatible field generation")
    else:
        print("\n❌ Still have issues to debug")

if __name__ == "__main__":
    main()