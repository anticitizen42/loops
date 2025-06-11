#!/usr/bin/env python3
"""
Debug energy conservation - systematic investigation
"""

import numpy as np
import matplotlib.pyplot as plt

def test_energy_components():
    """Test individual energy components"""
    print("=== Testing Energy Components ===")
    
    # Simple test field
    L = 10.0
    N = 64
    dx = L / N
    x = np.linspace(0, L, N, endpoint=False)
    
    # Gaussian field
    phi = np.exp(-((x - L/2)/1.0)**2)
    pi = np.zeros_like(phi)
    m = 1.0
    
    # Test kinetic energy
    E_kinetic = 0.5 * np.sum(pi**2) * dx
    print(f"Kinetic energy (should be 0): {E_kinetic:.6f}")
    
    # Test gradient energy - compare methods
    # Method 1: Forward difference
    grad1 = (np.roll(phi, -1) - phi) / dx
    E_grad1 = 0.5 * np.sum(grad1**2) * dx
    
    # Method 2: Central difference  
    grad2 = (np.roll(phi, -1) - np.roll(phi, 1)) / (2 * dx)
    E_grad2 = 0.5 * np.sum(grad2**2) * dx
    
    # Method 3: Analytical for Gaussian
    grad_analytical = -2 * (x - L/2) / 1.0**2 * phi
    E_grad_analytical = 0.5 * np.sum(grad_analytical**2) * dx
    
    print(f"Gradient energy (forward): {E_grad1:.6f}")
    print(f"Gradient energy (central): {E_grad2:.6f}")
    print(f"Gradient energy (analytical): {E_grad_analytical:.6f}")
    
    # Test potential energy
    E_potential = 0.5 * m**2 * np.sum(phi**2) * dx
    print(f"Potential energy: {E_potential:.6f}")
    
    # Total
    E_total = E_kinetic + E_grad2 + E_potential
    print(f"Total energy: {E_total:.6f}")
    
    return E_grad2, E_potential

def test_force_calculation():
    """Test force calculation"""
    print("\n=== Testing Force Calculation ===")
    
    L = 10.0
    N = 64
    dx = L / N
    x = np.linspace(0, L, N, endpoint=False)
    
    # Simple sine wave (has analytical second derivative)
    k = 2 * np.pi / L  # One wavelength fits in box
    phi = np.sin(k * x)
    m = 1.0
    
    # Numerical Laplacian
    lap_numerical = (np.roll(phi, -1) - 2*phi + np.roll(phi, 1)) / dx**2
    
    # Analytical Laplacian for sin(kx) is -k²sin(kx)
    lap_analytical = -k**2 * phi
    
    # Compare
    error = np.max(np.abs(lap_numerical - lap_analytical))
    print(f"Laplacian error: {error:.2e}")
    
    # Total force
    force_numerical = lap_numerical - m**2 * phi
    force_analytical = -k**2 * phi - m**2 * phi
    
    force_error = np.max(np.abs(force_numerical - force_analytical))
    print(f"Force error: {force_error:.2e}")
    
    if error < 1e-10:
        print("✅ Laplacian calculation is correct")
    else:
        print("❌ Laplacian calculation has errors")
        
    return error < 1e-10

def test_simple_harmonic_oscillator():
    """Test against exact harmonic oscillator solution"""
    print("\n=== Testing Simple Harmonic Oscillator ===")
    
    # For very small fields, we get φ_tt = -m²φ (ignore spatial derivatives)
    # Solution: φ(t) = A cos(mt + φ₀)
    
    m = 1.0
    omega = m
    A = 0.01  # Small amplitude
    
    # Single point test
    phi_0 = A
    pi_0 = 0.0  # Start at maximum displacement, zero velocity
    
    dt = 0.001
    n_steps = int(2 * np.pi / omega / dt)  # One full period
    
    # Analytical solution
    times = np.arange(n_steps) * dt
    phi_analytical = A * np.cos(omega * times)
    pi_analytical = -A * omega * np.sin(omega * times)
    
    # Numerical integration (simple leapfrog for single oscillator)
    phi = phi_0
    pi = pi_0
    
    phi_numerical = [phi]
    pi_numerical = [pi]
    
    for i in range(n_steps - 1):
        # Leapfrog for ṗ = -m²φ, φ̇ = p
        pi += -0.5 * dt * m**2 * phi  # Half step for p
        phi += dt * pi                  # Full step for φ  
        pi += -0.5 * dt * m**2 * phi  # Half step for p
        
        phi_numerical.append(phi)
        pi_numerical.append(pi)
    
    phi_numerical = np.array(phi_numerical)
    pi_numerical = np.array(pi_numerical)
    
    # Compare final values (should return to initial)
    phi_error = abs(phi_numerical[-1] - phi_numerical[0])
    pi_error = abs(pi_numerical[-1] - pi_numerical[0])
    
    print(f"φ return error: {phi_error:.2e}")
    print(f"π return error: {pi_error:.2e}")
    
    # Energy conservation
    energy_numerical = 0.5 * pi_numerical**2 + 0.5 * m**2 * phi_numerical**2
    energy_drift = abs(energy_numerical[-1] - energy_numerical[0])
    relative_drift = energy_drift / energy_numerical[0]
    
    print(f"Energy drift: {relative_drift:.2e}")
    
    if relative_drift < 1e-12:
        print("✅ Simple harmonic oscillator conserves energy perfectly")
        return True
    else:
        print("❌ Simple harmonic oscillator doesn't conserve energy")
        return False

def main():
    print("Systematic Energy Conservation Debug")
    print("=" * 40)
    
    # Test 1: Energy components
    test_energy_components()
    
    # Test 2: Force calculation
    force_ok = test_force_calculation()
    
    # Test 3: Simple harmonic oscillator
    sho_ok = test_simple_harmonic_oscillator()
    
    print(f"\n=== SUMMARY ===")
    print(f"Force calculation: {'✅' if force_ok else '❌'}")
    print(f"Simple harmonic oscillator: {'✅' if sho_ok else '❌'}")
    
    if force_ok and sho_ok:
        print("Basic components work - problem is likely in field code integration")
    else:
        print("Fundamental issues found - fix these first")

if __name__ == "__main__":
    main()