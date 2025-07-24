"""
visualization.py: Improved visualization utilities for TopoFluid2D
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection, PolyCollection
import matplotlib.cm as cm
from matplotlib.colors import Normalize


def plot_voronoi_mesh(ax, cells, solid_segments, show_particles=True, show_cells=True):
    """
    Plot the Voronoi mesh with solid boundaries - IMPROVED VERSION
    
    Parameters:
    -----------
    ax : matplotlib axis
        Axis to plot on
    cells : dict
        Voronoi cells
    solid_segments : list
        Solid boundary segments
    show_particles : bool
        Whether to show particle positions
    show_cells : bool
        Whether to show cell boundaries
    """
    # Clear axis
    ax.clear()

    # Plot Voronoi cells if requested
    if show_cells:
        for idx, cell in cells.items():
            if 'vertices' not in cell or not cell['vertices']:
                continue

            vertices = np.array(cell['vertices'])

            # Color based on cell type
            if not cell.get('contains_source', True):
                # Orphaned cell
                color = 'lightcoral'
                alpha = 0.3
            else:
                # Valid cell
                color = 'lightblue'
                alpha = 0.1

            # Create polygon
            if len(vertices) > 2:
                poly = Polygon(vertices, facecolor=color,
                               edgecolor='gray', alpha=alpha, linewidth=0.3)
                ax.add_patch(poly)

    # Plot source points
    if show_particles:
        for idx, cell in cells.items():
            if 'source_position' in cell:
                pos = cell['source_position']
                if cell.get('contains_source', True):
                    ax.plot(pos[0], pos[1], 'ko', markersize=4, markerfacecolor='black')
                else:
                    ax.plot(pos[0], pos[1], 'ro', markersize=4, markerfacecolor='red')

    # Plot solid boundaries - HIGHLIGHTED
    for segment in solid_segments:
        start = segment['start']
        end = segment['end']
        ax.plot([start[0], end[0]], [start[1], end[1]],
                'r-', linewidth=4, alpha=0.8, label='Solid Boundary')

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)


def plot_pressure_field_clean(ax, positions, pressure, domain_bounds, title="Pressure Field"):
    """
    Plot pressure field with clean visualization - NO BARS
    
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
    title : str
        Plot title
    """
    # Clear axis
    ax.clear()
    
    # Determine pressure range for consistent coloring
    p_min, p_max = np.min(pressure), np.max(pressure)
    
    # Create scatter plot with clean appearance
    scatter = ax.scatter(positions[:, 0], positions[:, 1],
                        c=pressure, s=50, cmap='RdBu_r',
                        vmin=p_min, vmax=p_max,
                        edgecolors='black', linewidths=0.5,
                        alpha=0.8)
    
    # Set domain bounds
    ax.set_xlim(domain_bounds[0])
    ax.set_ylim(domain_bounds[1])
    ax.set_aspect('equal')
    ax.set_title(title)
    
    # Add clean colorbar WITHOUT extra bars
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Pressure', rotation=270, labelpad=15)
    
    return scatter


def plot_density_field_clean(ax, positions, density, domain_bounds, title="Density Field"):
    """
    Plot density field with clean visualization
    """
    # Clear axis
    ax.clear()
    
    # Determine density range
    rho_min, rho_max = np.min(density), np.max(density)
    
    # Create scatter plot
    scatter = ax.scatter(positions[:, 0], positions[:, 1],
                        c=density, s=50, cmap='viridis',
                        vmin=rho_min, vmax=rho_max,
                        edgecolors='black', linewidths=0.5,
                        alpha=0.8)
    
    ax.set_xlim(domain_bounds[0])
    ax.set_ylim(domain_bounds[1])
    ax.set_aspect('equal')
    ax.set_title(title)
    
    # Clean colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Density', rotation=270, labelpad=15)
    
    return scatter


def plot_velocity_field(ax, positions, velocities, domain_bounds, title="Velocity Field", 
                       scale=None, density=2):
    """
    Plot velocity field using arrows
    """
    ax.clear()
    
    # Subsample for clarity
    indices = np.arange(0, len(positions), density)
    pos_sub = positions[indices]
    vel_sub = velocities[indices]
    
    # Compute velocity magnitude for coloring
    vel_mag = np.linalg.norm(vel_sub, axis=1)
    
    # Auto-scale if not provided
    if scale is None:
        max_vel = np.max(vel_mag) if len(vel_mag) > 0 else 1.0
        scale = max_vel * 10  # Adjust scaling factor
    
    # Plot arrows
    quiver = ax.quiver(pos_sub[:, 0], pos_sub[:, 1],
                      vel_sub[:, 0], vel_sub[:, 1],
                      vel_mag, scale=scale, scale_units='xy', 
                      angles='xy', cmap='plasma',
                      alpha=0.7)
    
    ax.set_xlim(domain_bounds[0])
    ax.set_ylim(domain_bounds[1])
    ax.set_aspect('equal')
    ax.set_title(title)
    
    # Clean colorbar
    cbar = plt.colorbar(quiver, ax=ax, shrink=0.8)
    cbar.set_label('Velocity Magnitude', rotation=270, labelpad=15)


def create_physics_diagnostic_plot(state, t, step):
    """
    Create diagnostic plots showing key physics quantities
    Following the paper's approach for validation
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    positions = state['positions']
    rho = state['rho']
    rho_u = state['rho_u']
    rho_v = state['rho_v'] 
    rho_e = state['rho_e']
    gamma = state['gamma']
    
    # Compute derived quantities safely
    u = np.where(rho > 1e-15, rho_u / rho, 0.0)
    v = np.where(rho > 1e-15, rho_v / rho, 0.0)
    e_kinetic = 0.5 * (u**2 + v**2)
    e_internal = np.where(rho > 1e-15, rho_e / rho - e_kinetic, 1e-10)
    pressure = np.maximum((gamma - 1) * rho * e_internal, 1e-10)
    
    # Sound speed
    sound_speed = np.sqrt(gamma * pressure / rho)
    
    # Mach number
    vel_magnitude = np.sqrt(u**2 + v**2)
    mach_number = vel_magnitude / sound_speed
    
    domain_bounds = ((-1.0, 1.0), (-1.0, 1.0))
    
    # Plot 1: Density
    plot_density_field_clean(ax1, positions, rho, domain_bounds, 
                            f"Density (t={t:.4f})")
    
    # Plot 2: Pressure  
    plot_pressure_field_clean(ax2, positions, pressure, domain_bounds,
                             f"Pressure (t={t:.4f})")
    
    # Plot 3: Velocity vectors
    velocities = np.column_stack([u, v])
    plot_velocity_field(ax3, positions, velocities, domain_bounds,
                       f"Velocity Field (t={t:.4f})")
    
    # Plot 4: Mach number
    ax4.clear()
    scatter = ax4.scatter(positions[:, 0], positions[:, 1],
                         c=mach_number, s=50, cmap='jet',
                         edgecolors='black', linewidths=0.5,
                         alpha=0.8)
    ax4.set_xlim(domain_bounds[0])
    ax4.set_ylim(domain_bounds[1])
    ax4.set_aspect('equal')
    ax4.set_title(f"Mach Number (t={t:.4f})")
    cbar = plt.colorbar(scatter, ax=ax4, shrink=0.8)
    cbar.set_label('Mach Number', rotation=270, labelpad=15)
    
    plt.tight_layout()
    return fig


def plot_1d_slice(state, y_center=0.0, width=0.2):
    """
    Plot 1D slice through the domain for comparison with theory
    This matches the paper's validation approach
    """
    positions = state['positions']
    rho = state['rho']
    rho_u = state['rho_u']
    rho_e = state['rho_e']
    gamma = state['gamma']
    
    # Find particles in the slice
    mask = np.abs(positions[:, 1] - y_center) < width/2
    if np.sum(mask) == 0:
        print("No particles found in slice")
        return
        
    x_slice = positions[mask, 0]
    rho_slice = rho[mask]
    u_slice = np.where(rho_slice > 1e-15, rho_u[mask] / rho_slice, 0.0)
    
    # Compute pressure
    e_kinetic = 0.5 * u_slice**2
    e_internal = np.where(rho_slice > 1e-15, rho_e[mask] / rho_slice - e_kinetic, 1e-10)
    p_slice = np.maximum((gamma - 1) * rho_slice * e_internal, 1e-10)
    
    # Sort by x coordinate
    sort_idx = np.argsort(x_slice)
    x_sorted = x_slice[sort_idx]
    rho_sorted = rho_slice[sort_idx]
    u_sorted = u_slice[sort_idx]
    p_sorted = p_slice[sort_idx]
    
    # Create 1D plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(x_sorted, rho_sorted, 'o-', markersize=4)
    ax1.set_ylabel('Density')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('1D Slice Through Shock Tube')
    
    ax2.plot(x_sorted, u_sorted, 'o-', markersize=4, color='red')
    ax2.set_ylabel('Velocity')
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(x_sorted, p_sorted, 'o-', markersize=4, color='green')
    ax3.set_ylabel('Pressure')
    ax3.set_xlabel('x')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig