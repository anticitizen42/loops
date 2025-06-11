#!/usr/bin/env python3
"""
Exact reproduction of the test that worked (3.45e-13 drift)
vs the test that didn't work
"""

import numpy as np

def original_working_test():
    """Exact copy of the test that gave 3.45e-13 energy drift"""
    print("=== ORIGINAL WORKING TEST (from minimal_test.py) ===")
    
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
    print(f"dt = {dt}, N = {N}, amplitude = {phi.max()}")
    
    energies = []
    
    n_steps = int(period / dt)  # Exactly one period
    print(f"Running for {n_steps} steps ({n_steps * dt:.3f} time units)")
    
    for step in range(n_steps):
        # Energy calculation
        kinetic = 0.5 * np.trapz(pi**2, dx=dx)
        grad_phi = np.gradient(phi, dx)
        gradient = 0.5 * np.trapz(grad_phi**2, dx=dx)
        potential = 0.5 * m**2 * np.trapz(phi**2, dx=dx)
        E = kinetic + gradient + potential
        
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
    
    return energy_drift

def timestep_test_reproduction():
    """Same parameters as timestep test for comparison"""
    print("\n=== TIMESTEP TEST REPRODUCTION ===")
    
    N = 128
    L = 10.0
    dx = L / (N - 1)
    dt = 0.0001  # Same small timestep
    m = 1.0
    total_time = 1.0  # This is different - longer time!
    
    x = np.linspace(0, L, N)
    
    # Single sine mode (same as original)
    phi = 0.01 * np.sin(np.pi * x / L)
    pi = np.zeros(N)
    
    print(f"dt = {dt}, total_time = {total_time}")
    
    n_steps = int(total_time / dt)
    print(f"Running for {n_steps} steps")
    
    E_initial = None
    
    for step in range(n_steps):
        # Energy calculation (only at start and end)
        if step == 0 or step == n_steps - 1:
            kinetic = 0.5 * np.trapz(pi**2, dx=dx)
            grad_phi = np.gradient(phi, dx)
            gradient = 0.5 * np.trapz(grad_phi**2, dx=dx)
            potential = 0.5 * m**2 * np.trapz(phi**2, dx=dx)
            E = kinetic + gradient + potential
            
            if step == 0:
                E_initial = E
                print(f"Initial energy: {E:.6f}")
            else:
                E_final = E
                print(f"Final energy: {E:.6f}")
        
        # Evolution - same as original
        lap = np.zeros(N)
        lap[1:-1] = (phi[2:] - 2*phi[1:-1] + phi[:-2]) / dx**2
        force = lap - m**2 * phi
        
        pi += 0.5 * dt * force
        phi += dt * pi
        phi[0] = phi[-1] = 0.0
        pi[0] = pi[-1] = 0.0
        
        lap[1:-1] = (phi[2:] - 2*phi[1:-1] + phi[:-2]) / dx**2
        force = lap - m**2 * phi
        pi += 0.5 * dt * force
        pi[0] = pi[-1] = 0.0
    
    # Energy drift
    energy_drift = abs(E_final - E_initial) / E_initial
    print(f"Energy drift: {energy_drift:.2e}")
    
    return energy_drift

def investigate_accumulation():
    """Check if energy drift accumulates with time"""
    print("\n=== ENERGY DRIFT ACCUMULATION TEST ===")
    
    N = 128
    L = 10.0
    dx = L / (N - 1)
    dt = 0.0001
    m = 1.0
    
    x = np.linspace(0, L, N)
    phi = 0.01 * np.sin(np.pi * x / L)
    pi = np.zeros(N)
    
    # Test different evolution times
    times = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    
    for total_time in times:
        # Reset field
        phi = 0.01 * np.sin(np.pi * x / L)
        pi = np.zeros(N)
        
        n_steps = int(total_time / dt)
        
        # Initial energy
        kinetic = 0.5 * np.trapz(pi**2, dx=dx)
        grad_phi = np.gradient(phi, dx)
        gradient = 0.5 * np.trapz(grad_phi**2, dx=dx)
        potential = 0.5 * m**2 * np.trapz(phi**2, dx=dx)
        E_initial = kinetic + gradient + potential
        
        # Evolve
        for step in range(n_steps):
            lap = np.zeros(N)
            lap[1:-1] = (phi[2:] - 2*phi[1:-1] + phi[:-2]) / dx**2
            force = lap - m**2 * phi
            
            pi += 0.5 * dt * force
            phi += dt * pi
            phi[0] = phi[-1] = 0.0
            pi[0] = pi[-1] = 0.0
            
            lap[1:-1] = (phi[2:] - 2*phi[1:-1] + phi[:-2]) / dx**2
            force = lap - m**2 * phi
            pi += 0.5 * dt * force
            pi[0] = pi[-1] = 0.0
        
        # Final energy
        kinetic = 0.5 * np.trapz(pi**2, dx=dx)
        grad_phi = np.gradient(phi, dx)
        gradient = 0.5 * np.trapz(grad_phi**2, dx=dx)
        potential = 0.5 * m**2 * np.trapz(phi**2, dx=dx)
        E_final = kinetic + gradient + potential
        
        energy_drift = abs(E_final - E_initial) / E_initial
        
        print(f"Time {total_time:4.1f}: {n_steps:5d} steps, drift = {energy_drift:.2e}")

def main():
    print("Exact Reproduction and Investigation")
    print("=" * 40)
    
    # Reproduce the original working test
    drift1 = original_working_test()
    
    # Reproduce the timestep test conditions
    drift2 = timestep_test_reproduction()
    
    # Investigate accumulation
    investigate_accumulation()
    
    print(f"\n=== ANALYSIS ===")
    print(f"Original test (1 period): {drift1:.2e}")
    print(f"Timestep test (1 second): {drift2:.2e}")
    
    if drift1 < 1e-10 and drift2 > 1e-6:
        print("\n💡 INSIGHT: Energy drift accumulates over long times!")
        print("   Even excellent short-term conservation becomes poor over long evolution")
        print("   This is a fundamental limitation of finite precision numerics")
    elif drift1 > 1e-10:
        print("\n❌ Original test doesn't reproduce - there's a bug somewhere")
    else:
        print("\n✅ Both tests give similar results")

if __name__ == "__main__":
    main()