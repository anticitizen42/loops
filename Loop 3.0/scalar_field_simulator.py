import json
import numpy as np
import matplotlib.pyplot as plt

def load_parameters(filename="parameters.json"):
    """Loads simulation parameters from a JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def initialize_field(nx, dx):
    """
    Initializes the scalar field 'phi' and its time derivative 'pi'.
    """
    x = np.arange(nx) * dx
    phi = 0.5 * np.exp(-((x - x.mean())**2) / 2.0)
    pi = np.zeros(nx)
    return phi, pi

# --- CORRECTED LINE 1 ---
# The function now accepts 'dx' as an argument.
def get_force(phi, m_squared, dx):
    """Calculates the force term F = d^2(phi)/dx^2 - V'(phi)."""
    V_prime = m_squared * phi
    phi_laplacian = np.gradient(np.gradient(phi, dx, edge_order=2), dx, edge_order=2)
    return phi_laplacian - V_prime

def integrator_step(phi, pi, dt, m_squared, dx):
    """
    Performs a single time step using a 4th-order Forest-Ruth symplectic integrator.
    """
    # Forest-Ruth integrator coefficients
    theta = 1. / (2. - 2.**(1./3.))
    
    # 1st step
    phi += theta * 0.5 * dt * pi
    # --- CORRECTED LINE 2 ---
    # We now pass 'dx' to the function.
    force = get_force(phi, m_squared, dx)
    pi += theta * dt * force
    
    # 2nd step
    phi += (1. - theta) * 0.5 * dt * pi
    # --- CORRECTED LINE 3 ---
    # We now pass 'dx' to the function here as well.
    force = get_force(phi, m_squared, dx)
    pi += (1. - 2. * theta) * dt * force
    
    # 3rd step
    phi += (1. - theta) * 0.5 * dt * pi
    force = get_force(phi, m_squared, dx)
    pi += theta * dt * force
    
    # 4th step
    phi += theta * 0.5 * dt * pi
    
    return phi, pi

def calculate_energy(phi, pi, dx, m_squared):
    """
    Calculates the total energy of the scalar field.
    """
    kinetic_energy = 0.5 * np.sum(pi**2) * dx
    gradient_phi = np.gradient(phi, dx, edge_order=2)
    gradient_energy = 0.5 * np.sum(gradient_phi**2) * dx
    potential_energy = np.sum(0.5 * m_squared * phi**2) * dx
    return kinetic_energy + gradient_energy + potential_energy

def main():
    """
    Main function to run the scalar field evolution simulation.
    """
    params = load_parameters()
    nx = params['nx']
    nt = params['nt']
    dt = params['dt']
    dx = params['dx']
    m_squared = params['m_squared']
    
    phi, pi = initialize_field(nx, dx)
    energies = []
    
    # --- Main simulation loop ---
    for i in range(nt):
        # Store energy at the beginning of the step
        total_energy = calculate_energy(phi, pi, dx, m_squared)
        energies.append(total_energy)
        
        # Perform one full 4th-order integration step
        phi, pi = integrator_step(phi, pi, dt, m_squared, dx)

    # --- Energy Conservation Plot ---
    energies = np.array(energies)
    initial_energy = energies[0]
    energy_drift = (energies - initial_energy) / initial_energy
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(nt), energy_drift)
    plt.title("Energy Conservation (4th-Order Symplectic Integrator)")
    plt.xlabel("Time Step")
    plt.ylabel("Fractional Energy Drift (E(t) - E(0)) / E(0)")
    plt.grid(True)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.savefig("energy_conservation_4th_order.png")
    plt.close()
    
    print("Energy conservation plot saved as energy_conservation_4th_order.png")
    # Use the last energy value for the final drift calculation
    final_drift = abs(energy_drift[-1]) if len(energy_drift) > 0 else 0.0
    print(f"Final fractional energy drift: {final_drift:.2e}")


if __name__ == "__main__":
    main()