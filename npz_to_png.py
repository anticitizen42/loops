#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
npz_to_png.py

A utility script to convert simulation snapshots (.npz files) into images (.png).
It visualizes a 2D slice of a selected 3D physical quantity from the data.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from tqdm import tqdm

def calculate_nz(psi):
    """Calculates the n_z component of the n-vector field."""
    psi0, psi1 = psi[0], psi[1]
    norm2 = np.abs(psi0)**2 + np.abs(psi1)**2
    # Avoid division by zero in empty regions
    norm2[norm2 == 0] = 1.0
    
    n_z = (np.abs(psi0)**2 - np.abs(psi1)**2) / norm2
    return n_z

def calculate_charge_density(psi):
    """Calculates the topological charge density for each 2D plaquette."""
    # First, get the n-vector
    psi0, psi1 = psi[0], psi[1]
    norm2 = np.abs(psi0)**2 + np.abs(psi1)**2
    norm2[norm2 == 0] = 1.0
    n = np.stack([
        ((psi0.conj() * psi1 + psi1.conj() * psi0) / norm2).real,
        ((psi0.conj() * psi1 - psi1.conj() * psi0) / (1j * norm2)).real,
        ((np.abs(psi0)**2 - np.abs(psi1)**2) / norm2)
    ], axis=0)
    
    # Calculate charge density on each plaquette using solid angles
    n_xp1 = np.roll(n, -1, axis=1)
    n_yp1 = np.roll(n, -1, axis=2)
    n_xp1_yp1 = np.roll(n_xp1, -1, axis=2)
    
    v1, v2, v3, v4 = n, n_xp1, n_xp1_yp1, n_yp1
    
    N1 = np.cross(v1, v2, axis=0)
    N2 = np.cross(v2, v3, axis=0)
    N3 = np.cross(v3, v4, axis=0)
    N4 = np.cross(v4, v1, axis=0)
    
    # Normalize normals, adding a small epsilon to avoid division by zero
    N1 /= (np.linalg.norm(N1, axis=0) + 1e-16)
    N2 /= (np.linalg.norm(N2, axis=0) + 1e-16)
    N3 /= (np.linalg.norm(N3, axis=0) + 1e-16)
    N4 /= (np.linalg.norm(N4, axis=0) + 1e-16)

    # Angles between normals, clipping to avoid domain errors from precision issues
    a1 = np.arccos(np.clip(np.sum(N1 * N2, axis=0), -1.0, 1.0))
    a2 = np.arccos(np.clip(np.sum(N2 * N3, axis=0), -1.0, 1.0))
    a3 = np.arccos(np.clip(np.sum(N3 * N4, axis=0), -1.0, 1.0))
    a4 = np.arccos(np.clip(np.sum(N4 * N1, axis=0), -1.0, 1.0))
    
    # The charge density is the spherical excess of the quadrilateral
    charge_density = (a1 + a2 + a3 + a4 - 2 * np.pi) / (4 * np.pi)
    return charge_density


def main():
    parser = argparse.ArgumentParser(description="Convert Skyrme simulation .npz snapshots to .png images.")
    parser.add_argument('input_dir', type=Path, help='Directory containing the .npz snapshot files.')
    parser.add_argument('output_dir', type=Path, help='Directory where .png images will be saved.')
    parser.add_argument('--plot-type', type=str, choices=['nz', 'charge'], default='nz',
                        help="The physical quantity to plot. 'nz': z-component of the n-vector. 'charge': topological charge density.")
    parser.add_argument('--slice-axis', type=str, choices=['x', 'y', 'z'], default='z',
                        help='The axis along which to take the 2D slice.')
    parser.add_argument('--slice-index', type=int, default=-1,
                        help='The index of the slice to visualize. -1 means the center slice.')
    parser.add_argument('--cmap', type=str, default='bwr',
                        help='Matplotlib colormap to use for the plot (e.g., bwr, viridis, plasma).')
    
    args = parser.parse_args()

    # --- Validation and Setup ---
    if not args.input_dir.is_dir():
        print(f"Error: Input directory not found at '{args.input_dir}'")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Plotting quantity: {args.plot_type.upper()}")

    files_to_process = list(args.input_dir.glob('*.npz'))
    if not files_to_process:
        print(f"No .npz files found in '{args.input_dir}'.")
        return

    # --- Processing Loop ---
    for npz_file in tqdm(files_to_process, desc="Converting files"):
        try:
            with np.load(npz_file) as data:
                # Ensure the 'psi' field exists in the npz file
                if 'psi' not in data:
                    tqdm.write(f"Warning: 'psi' field not found in {npz_file.name}. Skipping.")
                    continue
                psi = data['psi']

            # --- Calculate the 3D scalar field to plot ---
            if args.plot_type == 'nz':
                scalar_field_3d = calculate_nz(psi)
                clim = (-1, 1)
                cbar_label = r'$n_z$'
            elif args.plot_type == 'charge':
                scalar_field_3d = calculate_charge_density(psi)
                max_abs_val = np.max(np.abs(scalar_field_3d))
                clim = (-max_abs_val, max_abs_val) if max_abs_val > 0 else (-1, 1)
                cbar_label = 'Topological Charge Density'
            
            # --- Select the 2D slice ---
            N = scalar_field_3d.shape[0]
            slice_idx = args.slice_index if args.slice_index != -1 else N // 2
            
            if args.slice_axis == 'x':
                slice_2d = scalar_field_3d[slice_idx, :, :]
            elif args.slice_axis == 'y':
                slice_2d = scalar_field_3d[:, slice_idx, :]
            else: # 'z'
                slice_2d = scalar_field_3d[:, :, slice_idx]

            # --- Plotting ---
            fig, ax = plt.subplots(figsize=(6, 5))
            
            im = ax.imshow(slice_2d.T, cmap=args.cmap, origin='lower', vmin=clim[0], vmax=clim[1])
            ax.axis('off')
            ax.set_title(f"{args.plot_type.upper()} of {npz_file.name}", fontsize=10)
            
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(cbar_label)
            
            # --- Save and Cleanup ---
            output_path = args.output_dir / (npz_file.stem + '.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            tqdm.write(f"Error processing {npz_file.name}: {e}")

    print("\n✅ Conversion complete.")


if __name__ == '__main__':
    main()