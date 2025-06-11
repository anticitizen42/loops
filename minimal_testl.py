#!/usr/bin/env python3
"""
Minimal test to isolate the energy conservation problem
"""

import numpy as np
import matplotlib.pyplot as plt

def test_zero_field():
    """Test evolution of zero field - should stay zero with zero energy"""
    print("=== Test 1: Zero Field Evolution ===")
    
    N = 64
    L = 10.0
    dx = L / (N - 1)
    dt = 0.001
    m = 1.0
    
    # Zero field
    phi = np.zeros(N)
    pi = np.zeros(N)
    
    # Evolve for a few steps
    for step in range(1000):
        # Laplacian of zero is zero
        lap = np.zeros(N)
        lap[1:-1] = (phi[2:] - 2*phi[1:-1] + phi[:-2]) / dx**2
        
        # Force
        force = lap - m**2 * phi  # Should be zero
        
        # Leapfrog step
        pi += 0.5 * dt * force
        phi += dt * pi
        phi[0] = phi[-1] = 0.0  # Fixed boundaries
        pi[0] = pi[-1] = 0.0
        
        force = lap - m**2 * phi
        pi += 0.5 * dt * force
        pi[0] = pi[-1] = 0.0
    
    max_phi = np.max(np.abs(phi))
    max_pi = np.max(np.abs(pi))
    
    print(f"After 1000 steps: max|φ|={max_phi:.2e}, max|π|={max_pi:.2e}")
    
    if max_phi < 1e-14 and max_pi < 1e-14:
        print("✅ Zero field stays zero")
        return True
    else:
        print("❌ Zero field doesn't stay zero - numerical instability")
        return False

def test_constant_field():
    """Test evolution of constant field - should oscillate with known frequency"""
    print("\n=== Test 2: Constant Field Evolution ===")
    
    N = 64
    L = 10.0
    dx = L / (N - 1)
    dt = 0.001
    m = 1.0
    
    # Small constant field (zero at boundaries)
    phi = np.ones(N) * 0.01
    phi[0] = phi[-1] = 0.0
    pi = np.zeros(N)
    
    # For constant field in interior, Laplacian ≈ 0
    # So we get φ_tt ≈ -m²φ, which oscillates with frequency m
    
    times = []
    center_values = []
    energies = []
    
    for step in range(int(2 * np.pi / m / dt)):  # One period
        # Energy
        kinetic = 0.5 * np.trapz(pi**2, dx=dx)
        grad_phi = np.gradient(phi, dx)
        gradient = 0.5 * np.trapz(grad_phi**2, dx=dx)
        potential = 0.5 * m**2 * np.trapz(phi**2, dx=dx)
        E = kinetic + gradient + potential
        
        times.append(step * dt)
        center_values.append(phi[N//2])
        energies.append(E)
        
        # Laplacian
        lap = np.zeros(N)
        lap[1:-1] = (phi[2:] - 2*phi[1:-1] + phi[:-2]) / dx**2
        
        # Force
        force = lap - m**2 * phi
        
        # Leapfrog step
        pi += 0.5 * dt * force
        phi += dt * pi
        phi[0] = phi[-1] = 0.0
        pi[0] = pi[-1] = 0.0
        
        lap[1:-1] = (phi[2:] - 2*phi[1:-1] + phi[:-2]) / dx**2
        force = lap - m**2 * phi
        pi += 0.5 * dt * force
        pi[0] = pi[-1] = 0.0
    
    # Check energy conservation
    energy_drift = abs(energies[-1] - energies[0]) / energies[0]
    
    # Check oscillation
    center_values = np.array(center_values)
    initial_value = center_values[0]
    final_value = center_values[-1]
    return_error = abs(final_value - initial_value) / abs(initial_value)
    
    print(f"Energy drift: {energy_drift:.2e}")
    print(f"Field return error: {return_error:.2e}")
    
    if energy_drift < 1e-6:
        print("✅ Good energy conservation")
        return True
    else:
        print("❌ Poor energy conservation")
        return False

def test_single_mode():
    """Test single Fourier mode - should have exact solution"""
    print("\n=== Test 3: Single Fourier Mode ===")
    
    N = 128
    L = 10.0
    dx = L / (N - 1)
    dt = 0.0001  # Smaller timestep
    m = 1.0
    
    # Single sine mode that satisfies boundary conditions
    x = np.linspace(0, L, N)
    k = np.pi / L  # Fundamental mode
    phi = 0.01 * np.sin(k * x)  # Small amplitude
    pi = np.zeros(N)
    
    # For sin(kx), we get φ_tt = -k²φ - m²φ = -(k² + m²)φ
    # So oscillation frequency is ω = √(k² + m²)
    omega = np.sqrt(k**2 + m**2)
    period = 2 * np.pi / omega
    
    print(f"Expected frequency: {omega:.4f}")
    print(f"Expected period: {period:.4f}")
    
    times = []
    energies = []
    
    n_steps = int(period / dt)
    
    for step in range(n_steps):
        # Energy calculation
        kinetic = 0.5 * np.trapz(pi**2, dx=dx)
        grad_phi = np.gradient(phi, dx)
        gradient = 0.5 * np.trapz(grad_phi**2, dx=dx)
        potential = 0.5 * m**2 * np.trapz(phi**2, dx=dx)
        E = kinetic + gradient + potential
        
        times.append(step * dt)
        energies.append(E)
        
        # Laplacian
        lap = np.zeros(N)
        lap[1:-1] = (phi[2:] - 2*phi[1:-1] + phi[:-2]) / dx**2
        
        # Force
        force = lap - m**2 * phi
        
        # Leapfrog step
        pi += 0.5 * dt * force
        phi += dt * pi
        phi[0] = phi[-1] = 0.0
        pi[0] = pi[-1] = 0.0
        
        lap[1:-1] = (phi[2:] - 2*phi[1:-1] + phi[:-2]) / dx**2
        force = lap - m**2 * phi
        pi += 0.5 * dt * force
        pi[0] = pi[-1] = 0.0
    
    # Check energy conservation
    energy_drift = abs(energies[-1] - energies[0]) / energies[0]
    print(f"Energy drift: {energy_drift:.2e}")
    
    # Plot energy evolution
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(times, energies)
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title(f'Energy vs Time\n(drift = {energy_drift:.2e})')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(times, np.array(energies) - energies[0])
    plt.xlabel('Time')
    plt.ylabel('Energy - E₀')
    plt.title('Energy Drift')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('energy_test.png', dpi=150)
    plt.show()
    
    if energy_drift < 1e-10:
        print("✅ Excellent energy conservation")
        return True
    elif energy_drift < 1e-6:
        print("✅ Good energy conservation")
        return True
    else:
        print("❌ Poor energy conservation")
        return False

def main():
    print("Minimal Energy Conservation Tests")
    print("=" * 40)
    
    test1 = test_zero_field()
    test2 = test_constant_field()
    test3 = test_single_mode()
    
    print(f"\n=== RESULTS ===")
    print(f"Zero field: {'✅' if test1 else '❌'}")
    print(f"Constant field: {'✅' if test2 else '❌'}")
    print(f"Single mode: {'✅' if test3 else '❌'}")
    
    if test1 and test2 and test3:
        print("\n🎉 All basic tests pass - ready for complex fields!")
    else:
        print("\n❌ Basic tests fail - fix fundamentals first")

if __name__ == "__main__":
    main()