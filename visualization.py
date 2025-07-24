"""
visualization.py: Visualization utilities for TopoFluid2D
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection, PolyCollection
import matplotlib.cm as cm
from matplotlib.colors import Normalize


def plot_voronoi_mesh(ax, cells, solid_segments, show_orphaned=True):
    """
    Plot the Voronoi mesh with solid boundaries

    Parameters:
    -----------
    ax : matplotlib axis
        Axis to plot on
    cells : dict
        Voronoi cells
    solid_segments : list
        Solid boundary segments
    show_orphaned : bool
        Whether to highlight orphaned cells
    """
    # Clear axis
    ax.clear()

    # Plot Voronoi cells
    for idx, cell in cells.items():
        if 'vertices' not in cell or not cell['vertices']:
            continue

        vertices = np.array(cell['vertices'])

        # Color based on cell type
        if not cell.get('contains_source', True):
            # Orphaned cell
            color = 'lightcoral' if show_orphaned else 'lightblue'
            alpha = 0.3
        else:
            # Valid cell
            color = 'lightblue'
            alpha = 0.2

        # Create polygon
        if len(vertices) > 2:
            poly = Polygon(vertices, facecolor=color,
                           edgecolor='black', alpha=alpha, linewidth=0.5)
            ax.add_patch(poly)

    # Plot source points
    for idx, cell in cells.items():
        if 'source_position' in cell:
            pos = cell['source_position']
            if cell.get('contains_source', True):
                ax.plot(pos[0], pos[1], 'ko', markersize=3)
            else:
                ax.plot(pos[0], pos[1], 'ro', markersize=3)

    # Plot solid boundaries
    for segment in solid_segments:
        start = segment['start']
        end = segment['end']
        ax.plot([start[0], end[0]], [start[1], end[1]],
                'k-', linewidth=2)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)


def plot_pressure_field(ax, positions, pressure, domain_bounds,
                        n_grid=50, cmap='RdBu_r'):
    """
    Plot pressure field using scattered data interpolation

    Parameters:
    -----------
    ax : matplotlib axis
        Axis to plot on
    positions : array
        Particle positions
    pressure : array
        Pressure values
    domain_bounds : tuple
        Domain boundaries
    n_grid : int
        Grid resolution for interpolation
    cmap : str
        Colormap name
    """
    from scipy.interpolate import griddata

    # Create grid
    x_min, x_max = domain_bounds[0]
    y_min, y_max = domain_bounds[1]

    xi = np.linspace(x_min, x_max, n_grid)
    yi = np.linspace(y_min, y_max, n_grid)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate pressure
    zi = griddata(positions, pressure, (xi, yi), method='linear')

    # Plot
    im = ax.contourf(xi, yi, zi, levels=20, cmap=cmap)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Pressure')

    # Scatter plot of particles
    scatter = ax.scatter(positions[:, 0], positions[:, 1],
                         c=pressure, s=20, cmap=cmap,
                         edgecolors='black', linewidth=0.5)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)


def plot_velocity_field(ax, positions, velocities, domain_bounds,
                        scale=1.0, density=1):
    """
    Plot velocity field using arrows

    Parameters:
    -----------
    ax : matplotlib axis
        Axis to plot on
    positions : array
        Particle positions
    velocities : array
        Velocity vectors (u, v)
    domain_bounds : tuple
        Domain boundaries
    scale : float
        Arrow scale factor
    density : int
        Show every nth particle
    """
    # Subsample for clarity
    indices = np.arange(0, len(positions), density)
    pos_sub = positions[indices]
    vel_sub = velocities[indices]

    # Plot arrows
    ax.quiver(pos_sub[:, 0], pos_sub[:, 1],
              vel_sub[:, 0], vel_sub[:, 1],
              scale=scale, scale_units='xy', angles='xy',
              color='black', alpha=0.6)

    ax.set_xlim(domain_bounds[0])
    ax.set_ylim(domain_bounds[1])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)


def plot_density_field(ax, positions, density, domain_bounds,
                       n_grid=50, cmap='viridis'):
    """
    Plot density field
    """
    from scipy.interpolate import griddata

    # Create grid
    x_min, x_max = domain_bounds[0]
    y_min, y_max = domain_bounds[1]

    xi = np.linspace(x_min, x_max, n_grid)
    yi = np.linspace(y_min, y_max, n_grid)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate density
    zi = griddata(positions, density, (xi, yi), method='linear')

    # Plot
    im = ax.contourf(xi, yi, zi, levels=20, cmap=cmap)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Density')

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)


def create_animation(simulation_data, output_file='topofluid2d.mp4',
                     fps=30, dpi=100):
    """
    Create animation from simulation data

    Parameters:
    -----------
    simulation_data : list
        List of dictionaries containing simulation state at each frame
    output_file : str
        Output filename
    fps : int
        Frames per second
    dpi : int
        Resolution
    """
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    # Setup figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    def update_frame(frame_idx):
        data = simulation_data[frame_idx]

        # Clear all axes
        for ax in axes:
            ax.clear()

        # Plot Voronoi mesh
        plot_voronoi_mesh(axes[0], data['cells'], data['solid_segments'])
        axes[0].set_title(f"Voronoi Mesh (t={data['time']:.3f})")

        # Plot pressure
        positions = data['positions']
        pressure = data['pressure']
        domain_bounds = data['domain_bounds']

        plot_pressure_field(axes[1], positions, pressure, domain_bounds)
        axes[1].set_title("Pressure Field")

        # Plot density
        plot_density_field(axes[2], positions, data['density'], domain_bounds)
        axes[2].set_title("Density Field")

        # Plot velocity
        velocities = np.column_stack([data['velocity_u'], data['velocity_v']])
        plot_velocity_field(axes[3], positions, velocities, domain_bounds)
        axes[3].set_title("Velocity Field")

        plt.tight_layout()

    # Create animation
    anim = FuncAnimation(fig, update_frame, frames=len(simulation_data),
                         interval=1000 / fps, blit=False)

    # Save animation
    writer = FFMpegWriter(fps=fps, metadata=dict(artist='TopoFluid2D'),
                          bitrate=1800)
    anim.save(output_file, writer=writer, dpi=dpi)

    print(f"Animation saved to {output_file}")


def plot_simulation_diagnostics(time_history, dt_history, energy_history):
    """
    Plot simulation diagnostics

    Parameters:
    -----------
    time_history : list
        Time values
    dt_history : list
        Timestep values
    energy_history : list
        Total energy values
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

    # Timestep history
    ax1.plot(time_history[:-1], dt_history)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Timestep (dt)')
    ax1.set_title('Timestep Evolution')
    ax1.grid(True)
    ax1.set_yscale('log')

    # Energy conservation
    ax2.plot(time_history, energy_history)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Total Energy')
    ax2.set_title('Energy Conservation')
    ax2.grid(True)

    # Energy deviation
    initial_energy = energy_history[0]
    energy_deviation = [(e - initial_energy) / initial_energy * 100
                        for e in energy_history]
    ax3.plot(time_history, energy_deviation)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Energy Deviation (%)')
    ax3.set_title('Relative Energy Error')
    ax3.grid(True)

    plt.tight_layout()
    plt.show()


def plot_shock_structure(positions, state, x_range, y_center, width=0.1):
    """
    Plot 1D slice through shock structure

    Parameters:
    -----------
    positions : array
        Particle positions
    state : dict
        Fluid state
    x_range : tuple
        (x_min, x_max) for plot
    y_center : float
        y-coordinate for slice
    width : float
        Width of slice
    """
    # Extract particles in slice
    mask = np.abs(positions[:, 1] - y_center) < width / 2
    x_slice = positions[mask, 0]

    # Sort by x-coordinate
    sort_idx = np.argsort(x_slice)
    x_sorted = x_slice[sort_idx]

    # Extract variables
    rho = state['rho'][mask][sort_idx]
    u = state['rho_u'][mask][sort_idx] / rho
    p = compute_pressure_from_state(state, mask)[sort_idx]

    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    ax1.plot(x_sorted, rho, 'o-')
    ax1.set_ylabel('Density')
    ax1.grid(True)

    ax2.plot(x_sorted, u, 'o-')
    ax2.set_ylabel('Velocity')
    ax2.grid(True)

    ax3.plot(x_sorted, p, 'o-')
    ax3.set_ylabel('Pressure')
    ax3.set_xlabel('x')
    ax3.grid(True)

    ax1.set_title('Shock Structure')
    ax1.set_xlim(x_range)

    plt.tight_layout()
    plt.show()


def compute_pressure_from_state(state, mask=None):
    """
    Helper function to compute pressure from state
    """
    if mask is None:
        mask = np.ones(len(state['rho']), dtype=bool)

    rho = state['rho'][mask]
    rho_u = state['rho_u'][mask]
    rho_v = state['rho_v'][mask]
    rho_e = state['rho_e'][mask]
    gamma = state['gamma']

    # Compute pressure
    u = rho_u / rho
    v = rho_v / rho
    e_kinetic = 0.5 * (u ** 2 + v ** 2)
    e_internal = rho_e / rho - e_kinetic
    pressure = (gamma - 1) * rho * e_internal

    return pressure