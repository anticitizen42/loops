#!/usr/bin/env python3
"""
Test timestep convergence for energy conservation
"""

import numpy as np
import matplotlib.pyplot as plt

def test_timestep_convergence():
    """Test how energy conservation depends on timestep"""
    print("=== Timestep Convergence Test ===")
    
    timesteps = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    
    N = 128
    L = 10.0
    dx = L / (N - 1)
    m = 1.0
    total_time = 1.0  # Fixed total evolution time
    
    x = np.linspace(0, L, N)
    
    results = []
    
    for dt in timesteps:
        print(f"\nTesting dt = {dt}")
        
        # Single sine mode (we know this should work)
        phi = 0.1 * np.sin(np.pi * x / L)
        pi = np.zeros(N)
        
        n_steps = int(total_time / dt)
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
                else:
                    E_final = E
            
            # Evolution - Leapfrog
            lap = np.zeros(N)
            lap[1:-1] = (phi[2:] - 2*phi[1:-1] + phi[:-2]) / dx**2
            force = lap - m**2 * phi
            
            pi += 0.5 * dt * force
            phi += dt * pi
            
            lap[1:-1] = (phi[2:] - 2*phi[1:-1] + phi[:-2]) / dx**2
            force = lap - m**2 * phi
            pi += 0.5 * dt * force
        
        # Energy drift
        energy_drift = abs(E_final - E_initial) / E_initial
        results.append(energy_drift)
        
        print(f"  Steps: {n_steps}, Energy drift: {energy_drift:.2e}")
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.loglog(timesteps, results, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Timestep dt')
    plt.ylabel('Relative Energy Drift')
    plt.title('Energy Conservation vs Timestep\n(Single Sine Mode)')
    plt.grid(True, alpha=0.3)
    
    # Add reference lines
    plt.loglog(timesteps, np.array(timesteps)**2 * results[0] / timesteps[0]**2, 'r--', 
               label='dt² scaling', alpha=0.7)
    plt.loglog(timesteps, np.array(timesteps)**4 * results[0] / timesteps[0]**4, 'g--', 
               label='dt⁴ scaling', alpha=0.7)
    
    plt.legend()
    plt.savefig('timestep_convergence.png', dpi=150)
    plt.show()
    
    # Find acceptable timestep
    acceptable_drift = 1e-10
    good_timesteps = [dt for dt, drift in zip(timesteps, results) if drift < acceptable_drift]
    
    if good_timesteps:
        best_dt = max(good_timesteps)  # Largest timestep that gives good conservation
        print(f"\n✅ Found acceptable timestep: dt = {best_dt}")
        print(f"   Gives energy drift: {results[timesteps.index(best_dt)]:.2e}")
        return best_dt
    else:
        print(f"\n❌ No timestep gives drift < {acceptable_drift}")
        best_dt = timesteps[np.argmin(results)]
        print(f"   Best timestep: dt = {best_dt}")
        print(f"   Gives energy drift: {min(results):.2e}")
        return best_dt

def test_yoshida_vs_leapfrog(dt):
    """Compare Yoshida vs Leapfrog with optimal timestep"""
    print(f"\n=== Yoshida vs Leapfrog (dt={dt}) ===")
    
    N = 128
    L = 10.0
    dx = L / (N - 1)
    m = 1.0
    total_time = 5.0  # Longer evolution
    
    x = np.linspace(0, L, N)
    
    # Test with boundary-compatible Gaussian
    gaussian = 0.1 * np.exp(-((x - L/2) / 1.0)**2)
    boundary_factor = np.sin(np.pi * x / L)
    phi_init = gaussian * boundary_factor
    
    integrators = ['leapfrog', 'yoshida4']
    results = {}
    
    for integrator in integrators:
        print(f"\nTesting {integrator}:")
        
        phi = phi_init.copy()
        pi = np.zeros(N)
        
        n_steps = int(total_time / dt)
        E_initial = None
        
        for step in range(n_steps):
            # Energy calculation
            if step == 0 or step == n_steps - 1:
                kinetic = 0.5 * np.trapz(pi**2, dx=dx)
                grad_phi = np.gradient(phi, dx)
                gradient = 0.5 * np.trapz(grad_phi**2, dx=dx)
                potential = 0.5 * m**2 * np.trapz(phi**2, dx=dx)
                E = kinetic + gradient + potential
                
                if step == 0:
                    E_initial = E
                    print(f"  Initial energy: {E:.6f}")
                else:
                    E_final = E
                    print(f"  Final energy: {E:.6f}")
            
            # Evolution
            if integrator == 'leapfrog':
                # Leapfrog
                lap = np.zeros(N)
                lap[1:-1] = (phi[2:] - 2*phi[1:-1] + phi[:-2]) / dx**2
                force = lap - m**2 * phi
                
                pi += 0.5 * dt * force
                phi += dt * pi
                
                lap[1:-1] = (phi[2:] - 2*phi[1:-1] + phi[:-2]) / dx**2
                force = lap - m**2 * phi
                pi += 0.5 * dt * force
                
            else:  # yoshida4
                # Yoshida coefficients
                w0 = -2**(1/3) / (2 - 2**(1/3))
                w1 = 1 / (2 - 2**(1/3))
                c1 = c4 = w1 / 2
                c2 = c3 = (w0 + w1) / 2
                d1 = d3 = w1
                d2 = w0
                
                # Yoshida steps
                phi += c1 * dt * pi
                lap = np.zeros(N)
                lap[1:-1] = (phi[2:] - 2*phi[1:-1] + phi[:-2]) / dx**2
                force = lap - m**2 * phi
                pi += d1 * dt * force
                
                phi += c2 * dt * pi
                lap[1:-1] = (phi[2:] - 2*phi[1:-1] + phi[:-2]) / dx**2
                force = lap - m**2 * phi
                pi += d2 * dt * force
                
                phi += c3 * dt * pi
                lap[1:-1] = (phi[2:] - 2*phi[1:-1] + phi[:-2]) / dx**2
                force = lap - m**2 * phi
                pi += d3 * dt * force
                
                phi += c4 * dt * pi
        
        # Energy drift
        energy_drift = abs(E_final - E_initial) / E_initial
        results[integrator] = energy_drift
        
        print(f"  Energy drift: {energy_drift:.2e}")
    
    # Compare
    print(f"\n--- COMPARISON ---")
    for integrator in integrators:
        drift = results[integrator]
        if drift < 1e-12:
            status = "EXCELLENT"
        elif drift < 1e-10:
            status = "VERY GOOD"
        elif drift < 1e-6:
            status = "GOOD"
        else:
            status = "POOR"
        print(f"{integrator:10s}: {drift:.2e} ({status})")
    
    return results

def main():
    print("Systematic Timestep and Integrator Analysis")
    print("=" * 50)
    
    # Find optimal timestep
    best_dt = test_timestep_convergence()
    
    # Test integrators with optimal timestep
    integrator_results = test_yoshida_vs_leapfrog(best_dt)
    
    print(f"\n=== FINAL RECOMMENDATIONS ===")
    print(f"Optimal timestep: dt = {best_dt}")
    
    if integrator_results['yoshida4'] < 1e-10:
        print("✅ Yoshida integrator with optimal timestep gives excellent conservation")
        print("🎉 Ready to build Loop 4.0 with these settings!")
    else:
        print("⚠️  Still need smaller timesteps or better methods")

if __name__ == "__main__":
    main()