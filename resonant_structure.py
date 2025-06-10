import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def create_resonant_structure(grid_size=128):
    """
    Creates a 3D grid representing a multi-lobed 'hourglass' structure.
    """
    print("Generating the higher-dimensional structure...")
    # Create the coordinate grid
    x = np.linspace(-5, 5, grid_size)
    y = np.linspace(-5, 5, grid_size)
    z = np.linspace(-15, 15, grid_size)
    X, Y, Z = np.meshgrid(x, y, z)

    # Define the "hourglass" radius as a function of Z
    # It's wide at the ends (Z^2) and narrow in the middle.
    hourglass_radius = 0.5 + 0.1 * Z**2

    # Add "multiple lobes" by modulating the radius with a sine wave along Z
    # This creates bulges and constrictions along the polar axis.
    multi_lobed_radius = hourglass_radius * (1.5 + np.sin(Z * 0.8))

    # Define the radial distance from the Z-axis for every point
    R = np.sqrt(X**2 + Y**2)

    # The structure's density is defined by a Gaussian falloff from the radius
    # This creates a "soft" structure rather than a hard-edged one.
    structure = np.exp(-(R - multi_lobed_radius)**2 * 2.0)
    
    # Add a central "core" along the Z-axis
    core = np.exp(-R**2 * 5.0)
    
    # Combine the core and the lobes
    final_structure = structure + core
    
    # Normalize the structure's density to be between 0 and 1
    final_structure /= np.max(final_structure)
    
    print("Structure created.")
    return final_structure

# --- Main Visualization Block ---
if __name__ == "__main__":
    GRID_SIZE = 128
    
    # 1. Create our higher-dimensional object
    structure_3d = create_resonant_structure(grid_size=GRID_SIZE)

    # 2. Set up the plotting environment
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Initialize the plot with the central slice (our "present moment" at time=0)
    initial_slice = structure_3d[:, :, GRID_SIZE // 2]
    im = ax.imshow(initial_slice, cmap='magma', interpolation='spline16', vmin=0, vmax=1)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add a colorbar to represent the "density" of the structure
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Structure Density at Cross-Section')

    # Title will be updated for each frame of the animation
    title = ax.set_title("", fontsize=14)

    # 3. Define the animation function
    def update(frame):
        """
        This function is called for each frame of the animation.
        It updates the data of the image plot to show a new slice.
        """
        # The 'frame' variable acts as our "time"
        # We move our 2D membrane along the Z-axis
        slice_index = frame
        
        # Get the 2D cross-section at the new position
        cross_section = structure_3d[:, :, slice_index]
        
        # Update the image data
        im.set_data(cross_section)
        
        # Update the title to show the current time/position
        current_time = (slice_index - GRID_SIZE / 2) / (GRID_SIZE / 2)
        title.set_text(f"Our Perception (2D Cross-Section)\nTime / Position in Higher Dimension = {current_time:.2f}")
        
        return [im, title]

    # 4. Create and save the animation
    # We will animate slicing through the middle half of the structure
    start_frame = GRID_SIZE // 4
    end_frame = 3 * GRID_SIZE // 4
    num_frames = end_frame - start_frame

    print(f"Creating animation from {num_frames} frames...")
    # Use FuncAnimation to create the animation object
    anim = FuncAnimation(fig, update, frames=np.arange(start_frame, end_frame), blit=True)

    # Save the animation as a GIF
    output_filename = "resonant_structure_visualization.gif"
    writer = PillowWriter(fps=15)
    anim.save(output_filename, writer=writer)
    
    print(f"\nAnimation successfully saved to: {output_filename}")
    # To see the animation, open the file 'resonant_structure_visualization.gif'