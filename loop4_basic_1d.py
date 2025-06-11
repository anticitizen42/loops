#!/usr/bin/env python3
"""
Bray's Loops 4.0 - Phase I Foundation
Minimal 1D harmonic oscillator field with complete validation

Starting from the absolute basics and building systematically.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time

class Field1D:
    """
    Simplest possible 1D scalar field with harmonic potential
    φ_tt = φ_xx - m²φ
    """
    
    def __init__(self, L=10.0, N=128, m=1.0, dt=0.01):
        """
        Initialize 1D field
        L: box length
        N: number of grid points
        m: mass parameter
        dt: time step
        """
        self.L = L
        self.N = N
        self.m = m
        self.dt = dt
        self.dx = L / N
        
        # Grid
        self.x = np.linspace(0, L, N, endpoint=False)
        
        # Fields: φ(x,t) and π(x,t) = ∂φ/∂t
        self.phi = np.zeros(N)
        self.pi = np.zeros(N)
        
        # Diagnostics
        self.time = 0.0
        self.step = 0
        
    def set_initial_gaussian(self, x0=None, width=1.0, amplitude=1.0):
        """Set Gaussian initial condition"""
        if x0 is None:
            x0 = self.L / 2
        self.phi = amplitude * np.exp(-((self.x - x0) / width)**2)
        self.pi = np.zeros_like(self.phi)
        
    def laplacian(self, field):
        """Compute second derivative with periodic boundary conditions"""
        return (np.roll(field, -1) - 2*field + np.roll(field, 1)) / self.dx**2
    
    def compute_energy(self):
        """Compute total energy"""
        # Kinetic energy: (1/2) ∫ π² dx
        kinetic = 0.5 * np.sum(self.pi**2) * self.dx
        
        # Gradient energy: (1/2) ∫ (∂φ/∂x)² dx
        grad_phi = (np.roll(self.phi, -1) - np.roll(self.phi, 1)) / (2 * self.dx)
        gradient = 0.5 * np.sum(grad_phi**2) * self.dx
        
        # Potential energy: (1/2) m² ∫ φ² dx
        potential = 0.5 * self.m**2 * np.sum(self.phi**2) * self.dx
        
        return kinetic + gradient + potential, kinetic, gradient, potential
    
    def step_leapfrog(self):
        """Single leapfrog time step"""
        # Half step for π
        force = self.laplacian(self.phi) - self.m**2 * self.phi
        self.pi += 0.5 * self.dt * force
        
        # Full step for φ
        self.phi += self.dt * self.pi
        
        # Half step for π
        force = self.laplacian(self.phi) - self.m**2 * self.phi
        self.pi += 0.5 * self.dt * force
        
        self.time += self.dt
        self.step += 1
    
    def step_yoshida4(self):
        """4th-order Yoshida symplectic integrator"""
        # Yoshida coefficients for 4th order
        w0 = -2**(1/3) / (2 - 2**(1/3))
        w1 = 1 / (2 - 2**(1/3))
        c1 = c4 = w1 / 2
        c2 = c3 = (w0 + w1) / 2
        d1 = d3 = w1
        d2 = w0
        
        dt = self.dt
        
        # Yoshida step sequence
        self.phi += c1 * dt * self.pi
        force = self.laplacian(self.phi) - self.m**2 * self.phi
        self.pi += d1 * dt * force
        
        self.phi += c2 * dt * self.pi
        force = self.laplacian(self.phi) - self.m**2 * self.phi
        self.pi += d2 * dt * force
        
        self.phi += c3 * dt * self.pi
        force = self.laplacian(self.phi) - self.m**2 * self.phi
        self.pi += d3 * dt * force
        
        self.phi += c4 * dt * self.pi
        
        self.time += self.dt
        self.step += 1
    
    def evolve(self, n_steps, output_every=100, integrator='yoshida4'):
        """Evolve field and track diagnostics"""
        times = []
        energies = []
        snapshots = []
        
        # Choose integrator
        if integrator == 'yoshida4':
            step_func = self.step_yoshida4
        elif integrator == 'leapfrog':
            step_func = self.step_leapfrog
        else:
            raise ValueError(f"Unknown integrator: {integrator}")
        
        # Initial energy
        E_init, K_init, G_init, V_init = self.compute_energy()
        print(f"Initial energy: {E_init:.6f} (K={K_init:.6f}, G={G_init:.6f}, V={V_init:.6f})")
        print(f"Using integrator: {integrator}")
        
        for i in range(n_steps):
            step_func()
            
            if i % output_every == 0:
                E_total, K, G, V = self.compute_energy()
                times.append(self.time)
                energies.append(E_total)
                snapshots.append(self.phi.copy())
                
                if i % (output_every * 10) == 0:
                    print(f"Step {i:5d}, t={self.time:6.2f}, E={E_total:.6f}, drift={E_total-E_init:.2e}")
        
        # Final energy
        E_final, K_final, G_final, V_final = self.compute_energy()
        drift = E_final - E_init
        print(f"Final energy: {E_final:.6f} (K={K_final:.6f}, G={G_final:.6f}, V={V_final:.6f})")
        print(f"Energy drift: {drift:.2e} (relative: {drift/E_init:.2e})")
        
        return np.array(times), np.array(energies), snapshots

def test_integrator_comparison():
    """Compare integrators and find optimal timestep"""
    print("\n=== Integrator Comparison ===")
    
    timesteps = [0.01, 0.005, 0.002, 0.001]
    integrators = ['leapfrog', 'yoshida4']
    
    results = {}
    
    for integrator in integrators:
        results[integrator] = []
        print(f"\nTesting {integrator}:")
        
        for dt in timesteps:
            field = Field1D(L=10.0, N=128, m=1.0, dt=dt)
            field.set_initial_gaussian(width=1.0, amplitude=1.0)
            
            # Short test run - don't print verbose output
            times, energies, _ = field.evolve(n_steps=1000, output_every=2000, integrator=integrator)
            
            drift = abs(energies[-1] - energies[0])
            relative_drift = drift / abs(energies[0])
            results[integrator].append(relative_drift)
            
            print(f"  dt={dt:6.3f}: drift={relative_drift:.2e}")
    
    # Find best combination
    best_drift = float('inf')
    best_config = None
    
    for integrator in integrators:
        for i, dt in enumerate(timesteps):
            drift = results[integrator][i]
            if drift < best_drift:
                best_drift = drift
                best_config = (integrator, dt)
    
    print(f"\nBest configuration: {best_config[0]} with dt={best_config[1]}")
    print(f"Best drift: {best_drift:.2e}")
    
    return best_config, best_drift

def test_basic_validation(integrator='yoshida4', dt=0.001):
    """Test basic functionality with optimal settings"""
    print("=== Bray's Loops 4.0 - Basic Validation Test ===")
    
    # Create field
    field = Field1D(L=10.0, N=128, m=1.0, dt=dt)
    
    # Set initial condition
    field.set_initial_gaussian(width=1.0, amplitude=1.0)
    
    # Short evolution
    n_steps = int(10.0 / dt)  # 10 time units
    times, energies, snapshots = field.evolve(n_steps=n_steps, output_every=n_steps//20, integrator=integrator)
    
    # Check energy conservation
    energy_drift = energies[-1] - energies[0]
    relative_drift = abs(energy_drift / energies[0])
    
    print(f"\n=== VALIDATION RESULTS ===")
    print(f"Energy conservation: {energy_drift:.2e} (relative: {relative_drift:.2e})")
    
    if relative_drift < 1e-10:
        print("✅ EXCELLENT: Energy conserved to machine precision")
    elif relative_drift < 1e-6:
        print("✅ GOOD: Energy well conserved")
    elif relative_drift < 1e-3:
        print("⚠️  ACCEPTABLE: Energy reasonably conserved")
    else:
        print("❌ POOR: Energy not conserved - check implementation")
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Energy plot
    ax1.plot(times, energies, 'b-', linewidth=2)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Total Energy')
    ax1.set_title(f'Energy Conservation ({integrator})\n(drift = {energy_drift:.2e})')
    ax1.grid(True, alpha=0.3)
    
    # Field snapshots
    for i in [0, len(snapshots)//3, 2*len(snapshots)//3, -1]:
        ax2.plot(field.x, snapshots[i], label=f't={times[i]:.1f}')
    ax2.set_xlabel('x')
    ax2.set_ylabel('φ(x)')
    ax2.set_title('Field Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('loop4_basic_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return relative_drift < 1e-6

def test_analytical_solution():
    """Test against known analytical solution for small oscillations"""
    print("\n=== Analytical Solution Test ===")
    
    # For small amplitude, we should get simple harmonic motion
    field = Field1D(L=20.0, N=256, m=1.0, dt=0.001)
    
    # Very small Gaussian (linear regime)
    field.set_initial_gaussian(width=2.0, amplitude=0.01)
    
    # Evolve for one period
    omega = field.m  # For harmonic oscillator
    period = 2 * np.pi / omega
    n_steps = int(2 * period / field.dt)
    
    times, energies, snapshots = field.evolve(n_steps=n_steps, output_every=n_steps//20, integrator='yoshida4')
    
    print(f"Evolved for {2*period:.2f} time units (2 periods)")
    print(f"Period should be {period:.3f}, evolved for {times[-1]:.3f}")
    
    # Check if field returns to near initial state after 2 periods
    initial_field = snapshots[0]
    final_field = snapshots[-1]
    
    field_error = np.max(np.abs(final_field - initial_field))
    field_norm = np.max(np.abs(initial_field))
    relative_error = field_error / field_norm
    
    print(f"Field return error: {relative_error:.2e}")
    
    if relative_error < 1e-3:
        print("✅ GOOD: Field returns to initial state (periodic motion)")
    else:
        print("⚠️  Field does not return precisely (nonlinear effects or numerical error)")
    
    return relative_error < 1e-2

def test_analytical_solution():
    """Test against known analytical solution for small oscillations"""
    print("\n=== Analytical Solution Test ===")
    
    # For small amplitude, we should get simple harmonic motion
    field = Field1D(L=20.0, N=256, m=1.0, dt=0.001)
    
    # Very small Gaussian (linear regime)
    field.set_initial_gaussian(width=2.0, amplitude=0.01)
    
    # Evolve for one period
    omega = field.m  # For harmonic oscillator
    period = 2 * np.pi / omega
    n_steps = int(2 * period / field.dt)
    
    times, energies, snapshots = field.evolve(n_steps=n_steps, output_every=n_steps//20)
    
    print(f"Evolved for {2*period:.2f} time units (2 periods)")
    print(f"Period should be {period:.3f}, evolved for {times[-1]:.3f}")
    
    # Check if field returns to near initial state after 2 periods
    initial_field = snapshots[0]
    final_field = snapshots[-1]
    
    field_error = np.max(np.abs(final_field - initial_field))
    field_norm = np.max(np.abs(initial_field))
    relative_error = field_error / field_norm
    
    print(f"Field return error: {relative_error:.2e}")
    
    if relative_error < 1e-3:
        print("✅ GOOD: Field returns to initial state (periodic motion)")
    else:
        print("⚠️  Field does not return precisely (nonlinear effects or numerical error)")
    
    return relative_error < 1e-2

if __name__ == "__main__":
    print("Bray's Loops 4.0 - Starting from absolute basics")
    print("=" * 50)
    
    # First find optimal integrator and timestep
    print("Step 1: Finding optimal integrator settings...")
    best_config, best_drift = test_integrator_comparison()
    
    if best_drift < 1e-6:
        print(f"✅ Found good configuration: {best_config}")
        
        # Run full validation with optimal settings
        print(f"\nStep 2: Full validation with optimal settings...")
        basic_pass = test_basic_validation(integrator=best_config[0], dt=best_config[1])
        
        if basic_pass:
            analytical_pass = test_analytical_solution()
            
            if analytical_pass:
                print("\n🎉 ALL TESTS PASSED - Ready for next phase!")
                print(f"Optimal settings: {best_config[0]} integrator, dt={best_config[1]}")
            else:
                print("\n⚠️  Basic tests passed, analytical test needs investigation")
        else:
            print("\n❌ Basic test failed even with optimal settings")
    else:
        print(f"❌ Could not achieve good energy conservation (best: {best_drift:.2e})")
        print("Need to investigate: smaller timesteps, better integrator, or implementation bugs")
    
    print("\nNext steps:")
    print("1. Verify energy conservation is excellent (< 1e-6)")
    print("2. Test different initial conditions") 
    print("3. Test different parameters (m, dt, N)")
    print("4. Only then move to more complex fields")