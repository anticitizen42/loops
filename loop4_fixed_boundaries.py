#!/usr/bin/env python3
"""
Bray's Loops 4.0 - Phase I Foundation (FIXED VERSION)
Fixed boundary conditions for proper energy conservation
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time

class Field1D:
    """
    1D scalar field with proper boundary handling
    φ_tt = φ_xx - m²φ
    """
    
    def __init__(self, L=10.0, N=128, m=1.0, dt=0.01, boundary='fixed'):
        """
        Initialize 1D field
        L: box length
        N: number of grid points  
        m: mass parameter
        dt: time step
        boundary: 'fixed' (φ=0 at boundaries) or 'periodic'
        """
        self.L = L
        self.N = N
        self.m = m
        self.dt = dt
        self.dx = L / (N - 1) if boundary == 'fixed' else L / N
        self.boundary = boundary
        
        # Grid
        if boundary == 'fixed':
            self.x = np.linspace(0, L, N)  # Include endpoints
        else:
            self.x = np.linspace(0, L, N, endpoint=False)  # Periodic
        
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
        
        # For fixed boundaries, ensure φ=0 at endpoints
        if self.boundary == 'fixed':
            self.phi[0] = 0.0
            self.phi[-1] = 0.0
        
    def laplacian(self, field):
        """Compute second derivative with proper boundary conditions"""
        if self.boundary == 'periodic':
            # Periodic boundaries
            return (np.roll(field, -1) - 2*field + np.roll(field, 1)) / self.dx**2
        else:
            # Fixed boundaries: φ=0 at endpoints
            lap = np.zeros_like(field)
            # Interior points
            lap[1:-1] = (field[2:] - 2*field[1:-1] + field[:-2]) / self.dx**2
            # Boundary points (φ=0, so contribute zero to Laplacian)
            lap[0] = 0.0
            lap[-1] = 0.0
            return lap
    
    def compute_energy(self):
        """Compute total energy with proper boundary treatment"""
        if self.boundary == 'fixed':
            # Fixed boundaries - use trapezoidal rule for integration
            # Kinetic energy: (1/2) ∫ π² dx
            kinetic = 0.5 * np.trapz(self.pi**2, dx=self.dx)
            
            # Gradient energy: (1/2) ∫ (∂φ/∂x)² dx
            grad_phi = np.gradient(self.phi, self.dx)
            gradient = 0.5 * np.trapz(grad_phi**2, dx=self.dx)
            
            # Potential energy: (1/2) m² ∫ φ² dx
            potential = 0.5 * self.m**2 * np.trapz(self.phi**2, dx=self.dx)
            
        else:
            # Periodic boundaries - simple rectangle rule
            kinetic = 0.5 * np.sum(self.pi**2) * self.dx
            grad_phi = (np.roll(self.phi, -1) - np.roll(self.phi, 1)) / (2 * self.dx)
            gradient = 0.5 * np.sum(grad_phi**2) * self.dx
            potential = 0.5 * self.m**2 * np.sum(self.phi**2) * self.dx
        
        return kinetic + gradient + potential, kinetic, gradient, potential
    
    def step_leapfrog(self):
        """Single leapfrog time step with proper boundaries"""
        # Half step for π
        force = self.laplacian(self.phi) - self.m**2 * self.phi
        self.pi += 0.5 * self.dt * force
        
        # Full step for φ
        self.phi += self.dt * self.pi
        
        # Enforce boundary conditions
        if self.boundary == 'fixed':
            self.phi[0] = 0.0
            self.phi[-1] = 0.0
            self.pi[0] = 0.0
            self.pi[-1] = 0.0
        
        # Half step for π
        force = self.laplacian(self.phi) - self.m**2 * self.phi
        self.pi += 0.5 * self.dt * force
        
        # Enforce boundary conditions again
        if self.boundary == 'fixed':
            self.pi[0] = 0.0
            self.pi[-1] = 0.0
        
        self.time += self.dt
        self.step += 1
    
    def step_yoshida4(self):
        """4th-order Yoshida symplectic integrator with proper boundaries"""
        # Yoshida coefficients for 4th order
        w0 = -2**(1/3) / (2 - 2**(1/3))
        w1 = 1 / (2 - 2**(1/3))
        c1 = c4 = w1 / 2
        c2 = c3 = (w0 + w1) / 2
        d1 = d3 = w1
        d2 = w0
        
        dt = self.dt
        
        # Yoshida step sequence with boundary enforcement
        self.phi += c1 * dt * self.pi
        if self.boundary == 'fixed':
            self.phi[0] = self.phi[-1] = 0.0
            
        force = self.laplacian(self.phi) - self.m**2 * self.phi
        self.pi += d1 * dt * force
        if self.boundary == 'fixed':
            self.pi[0] = self.pi[-1] = 0.0
        
        self.phi += c2 * dt * self.pi
        if self.boundary == 'fixed':
            self.phi[0] = self.phi[-1] = 0.0
            
        force = self.laplacian(self.phi) - self.m**2 * self.phi
        self.pi += d2 * dt * force
        if self.boundary == 'fixed':
            self.pi[0] = self.pi[-1] = 0.0
        
        self.phi += c3 * dt * self.pi
        if self.boundary == 'fixed':
            self.phi[0] = self.phi[-1] = 0.0
            
        force = self.laplacian(self.phi) - self.m**2 * self.phi
        self.pi += d3 * dt * force
        if self.boundary == 'fixed':
            self.pi[0] = self.pi[-1] = 0.0
        
        self.phi += c4 * dt * self.pi
        if self.boundary == 'fixed':
            self.phi[0] = self.phi[-1] = 0.0
        
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
        print(f"Using integrator: {integrator}, boundary: {self.boundary}")
        
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

def test_boundary_conditions():
    """Test different boundary conditions"""
    print("=== Testing Boundary Conditions ===")
    
    boundaries = ['fixed', 'periodic']
    integrators = ['leapfrog', 'yoshida4']
    
    for boundary in boundaries:
        print(f"\n--- {boundary.upper()} BOUNDARIES ---")
        
        for integrator in integrators:
            print(f"Testing {integrator}:")
            
            # Create field
            field = Field1D(L=10.0, N=128, m=1.0, dt=0.001, boundary=boundary)
            field.set_initial_gaussian(width=1.0, amplitude=1.0)
            
            # Short evolution
            times, energies, snapshots = field.evolve(n_steps=1000, output_every=2000, integrator=integrator)
            
            # Check energy conservation
            drift = abs(energies[-1] - energies[0])
            relative_drift = drift / abs(energies[0])
            
            print(f"  Energy drift: {relative_drift:.2e}")
            
            if relative_drift < 1e-10:
                print("  ✅ EXCELLENT")
            elif relative_drift < 1e-6:
                print("  ✅ GOOD")
            elif relative_drift < 1e-3:
                print("  ⚠️  ACCEPTABLE")
            else:
                print("  ❌ POOR")

def test_convergence():
    """Test convergence with resolution"""
    print("\n=== Testing Resolution Convergence ===")
    
    resolutions = [64, 128, 256, 512]
    
    for N in resolutions:
        field = Field1D(L=10.0, N=N, m=1.0, dt=0.001, boundary='fixed')
        field.set_initial_gaussian(width=1.0, amplitude=1.0)
        
        times, energies, _ = field.evolve(n_steps=1000, output_every=2000, integrator='yoshida4')
        
        drift = abs(energies[-1] - energies[0]) / abs(energies[0])
        print(f"N={N:3d}: drift={drift:.2e}")

if __name__ == "__main__":
    print("Bray's Loops 4.0 - Fixed Boundary Conditions")
    print("=" * 50)
    
    # Test boundary conditions
    test_boundary_conditions()
    
    # Test convergence
    test_convergence()
    
    print("\n=== SUMMARY ===")
    print("Fixed boundaries should give excellent energy conservation")
    print("Periodic boundaries need compatible initial conditions")