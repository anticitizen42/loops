#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eye_of_terror.py

A systematic data collection platform for studying vortex formation in 3D Skyrme models.
This script is an advanced evolution of the original 'simulation3d_skyrme_gpu.py'.
"""

import cupy as cp
import cupyx.scipy.ndimage
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
import argparse
from itertools import product
from multiprocessing import Pool, cpu_count, Value
from tqdm import tqdm
import warnings
import os

# --- Core Physics and Simulation Functions ---

def setup_grid_and_k(p):
    """Pre-computes grid coordinates and Fourier space variables."""
    dx = p['L'] / p['N']
    x_cpu = np.linspace(0, p['L'], p['N'], endpoint=False)
    X_cpu, Y_cpu, Z_cpu = np.meshgrid(x_cpu, x_cpu, x_cpu, indexing='xy')
    
    kx_cpu = 2 * np.pi * np.fft.fftfreq(p['N'], d=dx)
    KX_cpu, KY_cpu, KZ_cpu = np.meshgrid(kx_cpu, kx_cpu, kx_cpu, indexing='ij')
    k2_cpu = KX_cpu**2 + KY_cpu**2 + KZ_cpu**2
    
    grid = {'X': cp.asarray(X_cpu), 'Y': cp.asarray(Y_cpu), 'Z': cp.asarray(Z_cpu)}
    k2 = cp.asarray(k2_cpu)
    
    # Derivatives in Fourier space
    d_dx = 1j * cp.asarray(KX_cpu)
    d_dy = 1j * cp.asarray(KY_cpu)
    d_dz = 1j * cp.asarray(KZ_cpu)
    
    derivs = {'x': d_dx, 'y': d_dy, 'z': d_dz}
    return grid, k2, derivs, dx

def spectral_lap(phi, k2):
    """Computes the Laplacian of a field in Fourier space."""
    return cp.fft.ifftn(-k2 * cp.fft.fftn(phi))

def spectral_deriv(phi, k_deriv):
    """Computes a spatial derivative of a field in Fourier space."""
    return cp.fft.ifftn(k_deriv * cp.fft.fftn(phi))

def get_n_vector(psi):
    """Computes the n-vector field from the CP1 field psi."""
    psi0, psi1 = psi[0], psi[1]
    norm2 = cp.abs(psi0)**2 + cp.abs(psi1)**2
    norm2[norm2 == 0] = 1e-16 # Avoid division by zero
    
    n1 = (psi0.conj() * psi1 + psi1.conj() * psi0) / norm2
    n2 = (psi0.conj() * psi1 - psi1.conj() * psi0) / (1j * norm2)
    n3 = (cp.abs(psi0)**2 - cp.abs(psi1)**2) / norm2
    return cp.stack([n1.real, n2.real, n3.real], axis=0)

def compute_force(psi, p, k2):
    """Computes the standard potential force."""
    norm2 = cp.abs(psi[0])**2 + cp.abs(psi[1])**2
    F = cp.empty_like(psi)
    for i in (0, 1):
        lap = spectral_lap(psi[i], k2)
        F[i] = lap - (p['m']**2 * psi[i] + p['lambda'] * norm2 * psi[i])
    return F

def compute_skyrme_force(psi, p, derivs):
    """Computes the Skyrme force term."""
    if p['kappa'] == 0:
        return 0
        
    n = get_n_vector(psi)
    dn = {axis: [spectral_deriv(n[comp], d) for comp in range(3)] for axis, d in derivs.items()}

    Fsk_n_x = (dn['y'][1]*dn['z'][2] - dn['y'][2]*dn['z'][1]) - (dn['z'][1]*dn['y'][2] - dn['z'][2]*dn['y'][1])
    Fsk_n_y = (dn['z'][1]*dn['x'][2] - dn['z'][2]*dn['x'][1]) - (dn['x'][1]*dn['z'][2] - dn['x'][2]*dn['z'][1])
    Fsk_n_z = (dn['x'][1]*dn['y'][2] - dn['x'][2]*dn['y'][1]) - (dn['y'][1]*dn['x'][2] - dn['y'][2]*dn['x'][1])
    
    Fsk_n = cp.stack([Fsk_n_x.real, Fsk_n_y.real, Fsk_n_z.real], axis=0)
    Fsk_n *= -2*p['kappa']**2

    # Project back onto psi
    Fpsi = cp.zeros_like(psi)
    sigmas = {
        'x': cp.array([[0, 1], [1, 0]], dtype=complex),
        'y': cp.array([[0, -1j], [1j, 0]], dtype=complex),
        'z': cp.array([[1, 0], [0, -1]], dtype=complex)
    }
    
    for i, axis in enumerate(['x', 'y', 'z']):
        sigma_psi = cp.tensordot(sigmas[axis], psi, axes=1)
        Fpsi += Fsk_n[i] * sigma_psi
        
    return Fpsi
    
# --- Diagnostics ---

def calculate_energies(psi, pi, p, k2, derivs, dx):
    """Calculates kinetic, potential, and Skyrme energies."""
    vol_element = dx**3
    
    # Kinetic Energy
    E_kin = 0.5 * cp.sum(cp.abs(pi[0])**2 + cp.abs(pi[1])**2) * vol_element
    
    # Potential Energy (Gradient + Mass + Self-interaction)
    E_grad = 0
    E_mass = 0
    E_lam = 0
    
    norm2 = cp.abs(psi[0])**2 + cp.abs(psi[1])**2
    E_mass = 0.5 * p['m']**2 * cp.sum(norm2) * vol_element
    E_lam = 0.25 * p['lambda'] * cp.sum(norm2**2) * vol_element

    for i in (0, 1):
        lap = spectral_lap(psi[i], k2)
        E_grad += -0.5 * cp.real(cp.vdot(psi[i], lap)) * vol_element
        
    E_pot = E_grad + E_mass + E_lam
    
    # Skyrme Energy
    # For simplicity, we calculate this from the force term.
    # Note: This is an approximation. A more direct calculation is preferred for rigor.
    F_sk = compute_skyrme_force(psi, p, derivs)
    E_sk = -0.5 * cp.real(cp.vdot(psi[0], F_sk[0]) + cp.vdot(psi[1], F_sk[1])) * vol_element if p['kappa'] != 0 else cp.array(0.0)
    
    return {
        "energy_kinetic": E_kin,
        "energy_potential": E_pot,
        "energy_skyrme": E_sk,
        "energy_total": E_kin + E_pot + E_sk
    }

def calculate_topological_charge(n, dx):
    """
    Calculates the topological charge (Baryon number) B on the lattice.
    This uses a robust geometric method based on summing solid angles.
    """
    # Shift fields to get neighbors
    n_xp1 = cp.roll(n, -1, axis=1)
    n_yp1 = cp.roll(n, -1, axis=2)
    n_xp1_yp1 = cp.roll(n_xp1, -1, axis=2)

    # For each plaquette in the xy plane, calculate the solid angle
    # subtended by the four n-vectors at its corners.
    v1 = n
    v2 = n_xp1
    v3 = n_xp1_yp1
    v4 = n_yp1
    
    # Solid angle of a spherical quadrilateral
    N1 = cp.cross(v1, v2, axis=0)
    N2 = cp.cross(v2, v3, axis=0)
    N3 = cp.cross(v3, v4, axis=0)
    N4 = cp.cross(v4, v1, axis=0)
    
    # Normalize
    N1 /= (cp.linalg.norm(N1, axis=0) + 1e-16)
    N2 /= (cp.linalg.norm(N2, axis=0) + 1e-16)
    N3 /= (cp.linalg.norm(N3, axis=0) + 1e-16)
    N4 /= (cp.linalg.norm(N4, axis=0) + 1e-16)

    # Angles between normals
    a1 = cp.arccos(cp.sum(N1 * N2, axis=0))
    a2 = cp.arccos(cp.sum(N2 * N3, axis=0))
    a3 = cp.arccos(cp.sum(N3 * N4, axis=0))
    a4 = cp.arccos(cp.sum(N4 * N1, axis=0))
    
    charge_density = (a1 + a2 + a3 + a4 - 2 * np.pi) / (4 * np.pi)
    
    # Integrate over one z-slice and multiply by Lz/dz as an approximation
    # A full 3D lattice calculation is much more involved.
    total_charge = cp.sum(charge_density[:,:,0]) * dx
    return total_charge

def calculate_diagnostics(psi, pi, p, k2, derivs, dx):
    """Calculates all specified diagnostic quantities."""
    diags = {}
    
    # Energy calculations
    energies = calculate_energies(psi, pi, p, k2, derivs, dx)
    diags.update(energies)
    
    # Field properties
    norm2 = cp.abs(psi[0])**2 + cp.abs(psi[1])**2
    diags["field_norm"] = cp.sum(norm2) * dx**3
    diags["max_amplitude"] = cp.max(norm2)
    
    n_vec = get_n_vector(psi)
    diags["topological_charge"] = calculate_topological_charge(n_vec, dx)
    
    # Structure counting
    # Threshold on energy density (potential part)
    energy_density = 0.5 * p['m']**2 * norm2 + 0.25 * p['lambda'] * norm2**2
    mean_energy = cp.mean(energy_density)
    labeled_array, num_features = cupyx.scipy.ndimage.label(energy_density > 2 * mean_energy)
    diags["structure_count"] = num_features
    
    # Correlation length placeholder
    diags["correlation_length"] = 0.0 # Requires more complex calculation
    
    return {k: v.get() if hasattr(v, 'get') else v for k, v in diags.items()}


# --- Initialization Modes ---

def init_random(p, grid):
    """Initializes fields with random noise."""
    cp.random.seed(p['init_params']['seed'])
    amp = p['init_params']['amplitude']
    dist = p['init_params']['distribution']
    
    psi = cp.zeros((2, p['N'], p['N'], p['N']), dtype=cp.complex128)
    if dist == "gaussian":
        noise = cp.random.randn(p['N'], p['N'], p['N']) + 1j * cp.random.randn(p['N'], p['N'], p['N'])
    else: # uniform
        noise = cp.random.rand(p['N'], p['N'], p['N']) + 1j * cp.random.rand(p['N'], p['N'], p['N'])

    psi[0] = amp * noise
    psi[1] = amp * (cp.random.randn(p['N'], p['N'], p['N']) + 1j * cp.random.randn(p['N'], p['N'], p['N']))

    # Normalize to start near the vacuum manifold
    norm = cp.sqrt(cp.abs(psi[0])**2 + cp.abs(psi[1])**2)
    norm[norm == 0] = 1.0
    psi[0] /= norm
    psi[1] /= norm
    
    return psi, cp.zeros_like(psi)

def init_ansatz(p, grid):
    """Initializes fields with a 'hedgehog' ansatz."""
    params = p['init_params']
    width = params.get('width', 5.0)
    center = params.get('center', [p['L']/2, p['L']/2, p['L']/2])
    
    r = cp.sqrt((grid['X'] - center[0])**2 + (grid['Y'] - center[1])**2 + (grid['Z'] - center[2])**2)
    
    # Hedgehog profile function F(r)
    F_r = np.pi * cp.exp(-r / width)
    
    # Convert n-vector to psi
    nx = (grid['X'] - center[0]) / (r + 1e-16)
    ny = (grid['Y'] - center[1]) / (r + 1e-16)
    
    psi = cp.zeros((2, p['N'], p['N'], p['N']), dtype=cp.complex128)
    psi[0] = cp.cos(F_r / 2)
    psi[1] = (nx + 1j * ny) * cp.sin(F_r / 2)
    
    return psi, cp.zeros_like(psi)


# --- Main Simulation Runner ---
counter = Value('i', 0)

def run_simulation(run_params):
    """
    Executes a single simulation run with a given set of parameters.
    Designed to be called by a multiprocessing Pool.
    """
    global counter
    
    # Unpack parameters
    p, run_id, output_dir, device_id = run_params
    
    # Assign GPU device for this worker process
    cp.cuda.Device(device_id).use()

    # Setup directories for this run's output
    ts_dir = output_dir / "timeseries"
    snap_dir = output_dir / "snapshots"
    final_dir = output_dir / "final_states"
    ts_dir.mkdir(exist_ok=True)
    snap_dir.mkdir(exist_ok=True)
    final_dir.mkdir(exist_ok=True)
    
    # --- Setup ---
    grid, k2, derivs, dx = setup_grid_and_k(p)
    
    if p['mode'] == 'random':
        psi, pi = init_random(p, grid)
    elif p['mode'] == 'ansatz':
        psi, pi = init_ansatz(p, grid)
    else:
        raise ValueError(f"Unknown initialization mode: {p['mode']}")

    # --- Time-stepping (4th-order Yoshida) ---
    alpha = 1.0 / (2 - 2**(1/3))
    c = [alpha/2, (1 - 2*alpha)/2, (1 - 2*alpha)/2, alpha/2]
    d = [alpha, 1 - 2*alpha, alpha, 0]
    
    dt = p['dt']

    diagnostics_data = []
    
    for step in range(p['n_steps']):
        for i in range(4):
            psi += c[i] * dt * pi
            
            F = compute_force(psi, p, k2)
            Fsk = compute_skyrme_force(psi, p, derivs) if p['kappa'] != 0 else 0
            pi += d[i] * dt * (F + Fsk)
        
        # --- Diagnostics & Output ---
        if step % p['output_every'] == 0:
            diags = calculate_diagnostics(psi, pi, p, k2, derivs, dx)
            diags['time'] = step * p['dt']
            diags['step'] = step
            diagnostics_data.append(diags)

            if step % p['snapshot_every'] == 0:
                cp.savez_compressed(
                    snap_dir / f"snap_{run_id}_step_{step:06d}.npz",
                    psi=psi.get(),
                    pi=pi.get()
                )

    # --- Finalize ---
    # Save final state
    cp.savez_compressed(final_dir / f"final_{run_id}.npz", psi=psi.get())
    
    # Save timeseries data
    df = pd.DataFrame(diagnostics_data)
    df.to_csv(ts_dir / f"timeseries_{run_id}.csv", index=False)
    
    with counter.get_lock():
        counter.value += 1
    
    return f"Run {run_id} completed."


# --- Argument Parsing and Main Driver ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Systematic data collection for Skyrme vortex formation.")
    parser.add_argument('--config', type=str, default='params.json', help='Path to the JSON parameter file.')
    parser.add_argument('--mode', type=str, choices=['random', 'ansatz'], help='Override initialization mode.')
    parser.add_argument('--trials', type=int, help='Override number of trials for random mode.')
    parser.add_argument('--workers', type=int, default=max(1, cpu_count() - 1), help='Number of parallel processes to run.')
    args = parser.parse_args()

    # --- Load configuration ---
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Override from command line
    if args.mode:
        config['base_params']['mode'] = args.mode
    if args.trials:
        config['scan_params']['trials'] = args.trials

    # --- Setup output directory ---
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = Path(f"scan_{config['base_params']['mode']}_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "analysis").mkdir(exist_ok=True)

    print(f"🚀 Starting new scan. Output will be saved in: {output_dir}")

    # --- Generate parameter sets for all runs ---
    scan_keys = list(config['scan_params'].keys())
    scan_values = [v if isinstance(v, list) else [v] for v in config['scan_params'].values()]
    
    param_combinations = list(product(*scan_values))
    
    run_list = []
    run_id_counter = 0
    
    num_gpus = cp.cuda.runtime.getDeviceCount()
    print(f"Found {num_gpus} CUDA devices.")
    print(f"Using {args.workers} parallel workers.")

    for combo in param_combinations:
        run_p = config['base_params'].copy()
        
        # Update params with scan values
        for i, key in enumerate(scan_keys):
            if key != 'trials':
                run_p[key] = combo[i]
        
        # Handle trials for random mode
        num_trials = config['scan_params'].get('trials', 1) if run_p['mode'] == 'random' else 1
        
        for trial in range(num_trials):
            trial_p = run_p.copy()
            run_id = f"{run_id_counter:04d}"
            
            if trial_p['mode'] == 'random':
                trial_p['init_params'] = trial_p.get('init_params', {})
                trial_p['init_params']['seed'] = run_p.get('init_params', {}).get('seed', 42) + trial
                run_id += f"_trial{trial:03d}"
            
            device_id = run_id_counter % num_gpus
            run_list.append((trial_p, run_id, output_dir, device_id))
            run_id_counter += 1

    # Save metadata
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump({
            "config": config,
            "num_runs": len(run_list),
            "command_args": vars(args)
        }, f, indent=4)
        
    print(f"Generated {len(run_list)} simulation runs.")

    # --- Execute runs in parallel ---
    # Suppress CuPy experimental warnings for ndimage
    warnings.filterwarnings("ignore", message="cupyx.scipy.ndimage is experimental")

    with Pool(processes=args.workers) as pool:
        # Use tqdm to create a progress bar
        list(tqdm(pool.imap_unordered(run_simulation, run_list), total=len(run_list), desc="Simulations"))

    print("\n✅ All simulations completed.")

    # --- Basic Post-Analysis ---
    print("📊 Performing basic post-analysis...")
    all_timeseries = []
    ts_dir = output_dir / "timeseries"
    for csv_file in ts_dir.glob("*.csv"):
        run_id = csv_file.stem.replace("timeseries_", "")
        df = pd.read_csv(csv_file)
        df['run_id'] = run_id
        all_timeseries.append(df)
        
    if all_timeseries:
        full_df = pd.concat(all_timeseries)
        summary = full_df.groupby('run_id').last().describe()
        summary.to_csv(output_dir / "analysis" / "summary_stats.csv")
        print(f"Saved summary statistics to {output_dir / 'analysis' / 'summary_stats.csv'}")
    else:
        print("No timeseries data found to analyze.")
        
    print("✨ Data collection complete.")