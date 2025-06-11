#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
foam_analyzer.py

An analysis tool to process 3D Skyrme field configurations (.npz files).
It calculates and compares global vs. absolute topological charge to test the 
"proto-soliton foam" hypothesis, where a near-zero net charge can hide a
complex structure of positive and negative charge pairs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import label
import re
import json

# --- Core Physics Calculation ---

def compute_local_charge_density(psi):
    """
    Computes the 3D topological charge density field from the psi field.
    The method uses a geometric calculation on 2D plaquettes stacked in 3D,
    which is a robust method for lattice field theory.
    """
    # 1. Calculate the n-vector field from psi
    psi0, psi1 = psi[0], psi[1]
    norm2 = np.abs(psi0)**2 + np.abs(psi1)**2
    # Avoid division by zero in empty regions
    norm2[norm2 == 0] = 1.0 
    
    n_vec = np.stack([
        ((psi0.conj() * psi1 + psi1.conj() * psi0) / norm2).real,
        ((psi0.conj() * psi1 - psi1.conj() * psi0) / (1j * norm2)).real,
        ((np.abs(psi0)**2 - np.abs(psi1)**2) / norm2)
    ], axis=0)

    # 2. Use the geometric formula on 2D plaquettes (stacked in z)
    # Get neighboring vectors in the x and y directions
    n_xp1 = np.roll(n_vec, -1, axis=1)  # x+1 neighbor
    n_yp1 = np.roll(n_vec, -1, axis=2)  # y+1 neighbor
    n_xp1_yp1 = np.roll(n_xp1, -1, axis=2) # (x+1, y+1) neighbor

    # The four corners of the plaquette
    v1, v2, v3, v4 = n_vec, n_xp1, n_xp1_yp1, n_yp1
    
    # Normals to the triangles forming the spherical quadrilateral
    N1 = np.cross(v1, v2, axis=0)
    N2 = np.cross(v2, v3, axis=0)
    N3 = np.cross(v3, v4, axis=0)
    N4 = np.cross(v4, v1, axis=0)
    
    # Normalize normals, adding a small epsilon to avoid division by zero
    N1 /= (np.linalg.norm(N1, axis=0) + 1e-16)
    N2 /= (np.linalg.norm(N2, axis=0) + 1e-16)
    N3 /= (np.linalg.norm(N3, axis=0) + 1e-16)
    N4 /= (np.linalg.norm(N4, axis=0) + 1e-16)

    # Angles between normals, clipping to handle floating point inaccuracies
    a1 = np.arccos(np.clip(np.sum(N1 * N2, axis=0), -1.0, 1.0))
    a2 = np.arccos(np.clip(np.sum(N2 * N3, axis=0), -1.0, 1.0))
    a3 = np.arccos(np.clip(np.sum(N3 * N4, axis=0), -1.0, 1.0))
    a4 = np.arccos(np.clip(np.sum(N4 * N1, axis=0), -1.0, 1.0))
    
    # The charge density is the spherical excess of the quadrilateral
    charge_density = (a1 + a2 + a3 + a4 - 2 * np.pi) / (4 * np.pi)
    
    return charge_density

def calculate_topological_charges(psi_field, dx):
    """
    Calculates both global and absolute topological charges from the charge density.
    """
    charge_density = compute_local_charge_density(psi_field)
    
    volume_element = dx**3
    
    # Global charge is the standard net integral
    global_charge = np.sum(charge_density) * volume_element
    
    # Absolute charge is the integral of the absolute value of the density
    absolute_charge = np.sum(np.abs(charge_density)) * volume_element
    
    return global_charge, absolute_charge, charge_density

# --- Batch Processing and Visualization ---

def analyze_simulation_run(npz_directory, output_csv_path, L, N):
    """
    Processes all NPZ files from a simulation run directory, calculates
    key metrics, and saves them to a CSV file.
    """
    dx = L / N
    npz_files = sorted(list(npz_directory.glob('*.npz')))
    
    if not npz_files:
        print(f"Warning: No .npz files found in {npz_directory}")
        return pd.DataFrame()

    analysis_results = []

    for npz_path in tqdm(npz_files, desc=f"Analyzing {npz_directory.name}"):
        try:
            # 1. Load data and metadata
            data = np.load(npz_path)
            psi = data['psi']
            
            # Extract timestep from filename using regex
            match = re.search(r'step_(\d+)', npz_path.name)
            timestep = int(match.group(1)) if match else -1

            # 2. Calculate charges
            global_charge, absolute_charge, charge_density = calculate_topological_charges(psi, dx)
            
            # 3. Calculate other metrics
            # Add a small epsilon to avoid division by zero
            charge_ratio = absolute_charge / (np.abs(global_charge) + 1e-9)
            max_local_density = np.max(np.abs(charge_density))
            
            # Count structures by labeling regions with high charge density
            threshold = 0.5 * max_local_density
            _, structure_count = label(np.abs(charge_density) > threshold)

            analysis_results.append({
                'filename': npz_path.name,
                'timestep': timestep,
                'global_charge': global_charge,
                'absolute_charge': absolute_charge,
                'charge_ratio': charge_ratio,
                'max_local_density': max_local_density,
                'structure_count': structure_count
            })
        except Exception as e:
            tqdm.write(f"Error processing {npz_path.name}: {e}")

    df = pd.DataFrame(analysis_results)
    df.to_csv(output_csv_path, index=False)
    print(f"\nAnalysis complete. Results saved to {output_csv_path}")
    return df

def visualize_analysis_results(df, charge_density_example, analysis_dir):
    """
    Creates plots visualizing the charge evolution and saves them.
    """
    if df.empty:
        print("Skipping visualization because no data was generated.")
        return

    # Plot 1: Global vs. Absolute Charge Evolution
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['timestep'], df['global_charge'], '-o', markersize=4, label='Global Charge (Net)')
    ax.plot(df['timestep'], df['absolute_charge'], '-s', markersize=4, label='Absolute Charge (Sum of |density|)')
    ax.set_title('Topological Charge Evolution', fontsize=16)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Topological Charge')
    ax.legend()
    ax.set_yscale('symlog', linthresh=1e-2) # Use symlog for better visibility near zero
    plt.savefig(analysis_dir / 'charge_evolution.png', dpi=150)
    plt.close(fig)

    # Plot 2: Charge Ratio Evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['timestep'], df['charge_ratio'], 'r-^', markersize=4)
    ax.set_title('Charge Ratio (Absolute / |Global|)', fontsize=16)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Ratio')
    ax.set_yscale('log') # Ratio is expected to grow, so log scale is useful
    plt.savefig(analysis_dir / 'charge_ratio.png', dpi=150)
    plt.close(fig)
    
    # Plot 3: 2D slice of charge density
    if charge_density_example is not None:
        fig, ax = plt.subplots(figsize=(7, 6))
        N = charge_density_example.shape[0]
        slice_2d = charge_density_example[:, :, N // 2]
        
        vmax = np.max(np.abs(slice_2d))
        im = ax.imshow(slice_2d.T, cmap='bwr', origin='lower', vmin=-vmax, vmax=vmax)
        ax.set_title(f'Charge Density Slice at Timestep {df["timestep"].iloc[-1]}', fontsize=14)
        ax.axis('off')
        fig.colorbar(im, ax=ax, label='Local Charge Density')
        plt.savefig(analysis_dir / 'final_charge_density_slice.png', dpi=150)
        plt.close(fig)

    print(f"Visualizations saved in {analysis_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze topological charge in Skyrme simulation snapshots.")
    parser.add_argument('run_directory', type=Path, help='Path to the main run directory (e.g., scan_random_20250610-...).')
    args = parser.parse_args()

    # --- Validation and Setup ---
    if not args.run_directory.is_dir():
        print(f"Error: Run directory not found at '{args.run_directory}'")
        return
        
    npz_dir = args.run_directory / 'snapshots'
    if not npz_dir.is_dir():
        print(f"Error: Snapshots directory not found at '{npz_dir}'")
        return

    analysis_dir = args.run_directory / 'analysis'
    analysis_dir.mkdir(exist_ok=True)
    
    output_csv = analysis_dir / 'topological_charge_analysis.csv'

    # --- Load Metadata ---
    metadata_path = args.run_directory / 'metadata.json'
    if not metadata_path.exists():
        print(f"Error: metadata.json not found in {args.run_directory}")
        return
        
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    L = metadata['config']['base_params']['L']
    N = metadata['config']['base_params']['N']

    # --- Run Analysis ---
    df = analyze_simulation_run(npz_dir, output_csv, L, N)

    # --- Visualize Results ---
    # Load a final state for charge density visualization
    charge_density_final = None
    if not df.empty:
        last_npz_filename = df['filename'].iloc[-1]
        last_npz_path = npz_dir / last_npz_filename
        if last_npz_path.exists():
            psi_final = np.load(last_npz_path)['psi']
            _, _, charge_density_final = calculate_topological_charges(psi_final, L/N)

    visualize_analysis_results(df, charge_density_final, analysis_dir)

if __name__ == '__main__':
    main()