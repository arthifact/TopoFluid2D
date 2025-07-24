#!/usr/bin/env python3
"""
TopoFluid2D: Topology-Preserving Coupling of Compressible Fluids and Thin Deformables
Main simulation script for 2D implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Check if modules exist, if not, create placeholder functions
try:
    from voronoi_utils import (
        compute_voronoi_diagram,
        clip_voronoi_by_solids,
        stitch_orphaned_cells,
        compute_interface_geometry
    )
except ImportError:
    print("Warning: voronoi_utils not found. Using placeholder functions.")


    def compute_voronoi_diagram(positions):
        from scipy.spatial import Voronoi
        return Voronoi(positions)


    def clip_voronoi_by_solids(vor, solid_segments):
        # Placeholder - return unclipped cells
        cells = {}
        for i in range(len(vor.points)):
            if vor.point_region[i] >= 0:
                region = vor.regions[vor.point_region[i]]
                if -1 not in region and len(region) > 0:
                    cells[i] = {
                        'vertices': [vor.vertices[j] for j in region],
                        'solid_faces': [],
                        'contains_source': True,
                        'source_position': vor.points[i]
                    }
        return cells


    def stitch_orphaned_cells(clipped_cells, positions):
        return clipped_cells


    def compute_interface_geometry(cells):
        # Placeholder - return empty interfaces
        return []

try:
    from fluid_solver import (
        compute_numerical_flux,
        update_fluid_state,
        compute_timestep,
        apply_boundary_conditions
    )
except ImportError:
    print("Warning: fluid_solver not found. Using placeholder functions.")


    def compute_numerical_flux(interfaces, state):
        return []


    def update_fluid_state(state, fluxes, interfaces, dt):
        return state


    def compute_timestep(state, interfaces, cfl):
        return 0.001


    def apply_boundary_conditions(interfaces, solid_segments, state):
        return interfaces

try:
    from solid_handler import (
        update_solid_positions,
        compute_fluid_solid_coupling
    )
except ImportError:
    print("Warning: solid_handler not found. Using placeholder functions.")


    def update_solid_positions(solid_segments, dt):
        return solid_segments


    def compute_fluid_solid_coupling(interfaces, state, solid_segments):
        return {}

try:
    from visualization import (
        plot_voronoi_mesh,
        plot_pressure_field,
        create_animation
    )
except ImportError:
    print("Warning: visualization not found. Using basic plotting.")


    def plot_voronoi_mesh(ax, cells, solid_segments):
        ax.clear()
        # Basic scatter plot
        for idx, cell in cells.items():
            if 'source_position' in cell:
                pos = cell['source_position']
                ax.plot(pos[0], pos[1], 'ko', markersize=3)
        ax.set_aspect('equal')
        ax.grid(True)


    def plot_pressure_field(ax, positions, pressure, domain_bounds):
        ax.clear()
        scatter = ax.scatter(positions[:, 0], positions[:, 1],
                             c=pressure, cmap='RdBu_r', s=50)
        ax.set_xlim(domain_bounds[0])
        ax.set_ylim(domain_bounds[1])
        ax.set_aspect('equal')
        plt.colorbar(scatter, ax=ax)


def initialize_fluid_particles(domain_bounds, n_particles, initial_state):
    """
    Initialize fluid particles with positions and state variables

    Parameters:
    -----------
    domain_bounds : tuple
        ((xmin, xmax), (ymin, ymax))
    n_particles : int
        Number of fluid particles
    initial_state : dict
        Initial conditions for density, velocity, pressure

    Returns:
    --------
    positions : array (n_particles, 2)
        Particle positions
    state : dict
        Fluid state variables (density, momentum, energy)
    """
    xmin, xmax = domain_bounds[0]
    ymin, ymax = domain_bounds[1]

    # Create regular grid of particles
    nx = int(np.sqrt(n_particles * (xmax - xmin) / (ymax - ymin)))
    ny = int(n_particles / nx)

    x = np.linspace(xmin + 0.05 * (xmax - xmin), xmax - 0.05 * (xmax - xmin), nx)
    y = np.linspace(ymin + 0.05 * (ymax - ymin), ymax - 0.05 * (ymax - ymin), ny)
    xx, yy = np.meshgrid(x, y)

    positions = np.column_stack([xx.ravel(), yy.ravel()])

    # Initialize conservative variables
    n = len(positions)
    rho = np.full(n, initial_state['density'])
    u = np.full(n, initial_state['velocity_x'])
    v = np.full(n, initial_state['velocity_y'])
    p = np.full(n, initial_state['pressure'])

    # Convert to conservative form
    gamma = 1.4  # Adiabatic index for diatomic gas
    e_internal = p / ((gamma - 1) * rho)
    e_kinetic = 0.5 * (u ** 2 + v ** 2)
    e_total = e_internal + e_kinetic

    state = {
        'rho': rho,  # Density
        'rho_u': rho * u,  # x-momentum
        'rho_v': rho * v,  # y-momentum
        'rho_e': rho * e_total,  # Total energy
        'positions': positions,
        'gamma': gamma
    }

    return positions, state


def initialize_solids(solid_type='thin_sheet'):
    """
    Initialize solid boundaries

    Parameters:
    -----------
    solid_type : str
        Type of solid boundary to create

    Returns:
    --------
    solid_segments : list
        List of line segments defining solid boundaries
    """
    solid_segments = []

    if solid_type == 'thin_sheet':
        # Horizontal thin sheet in the middle
        solid_segments.append({
            'start': np.array([-0.3, 0.0]),
            'end': np.array([0.3, 0.0]),
            'velocity': np.array([0.0, 0.0]),
            'is_deformable': False
        })
    elif solid_type == 'box':
        # Box boundaries
        corners = [
            [-0.5, -0.5], [0.5, -0.5],
            [0.5, 0.5], [-0.5, 0.5]
        ]
        for i in range(4):
            solid_segments.append({
                'start': np.array(corners[i]),
                'end': np.array(corners[(i + 1) % 4]),
                'velocity': np.array([0.0, 0.0]),
                'is_deformable': False
            })

    return solid_segments


def main():
    """
    Main simulation loop
    """
    # Simulation parameters
    domain_bounds = ((-1.0, 1.0), (-1.0, 1.0))
    n_particles = 400  # 20x20 grid
    t_final = 0.5  # Shorter simulation for testing
    cfl = 0.5

    # Initial conditions (Sod shock tube in x-direction)
    left_state = {
        'density': 1.0,
        'velocity_x': 0.0,
        'velocity_y': 0.0,
        'pressure': 1.0
    }

    right_state = {
        'density': 0.125,
        'velocity_x': 0.0,
        'velocity_y': 0.0,
        'pressure': 0.1
    }

    # Initialize particles
    positions, state = initialize_fluid_particles(domain_bounds, n_particles, left_state)

    # Apply initial discontinuity (Sod shock tube)
    mask = positions[:, 0] > 0.0
    state['rho'][mask] = right_state['density']
    state['rho_u'][mask] = right_state['density'] * right_state['velocity_x']
    state['rho_v'][mask] = right_state['density'] * right_state['velocity_y']

    gamma = state['gamma']
    p_right = right_state['pressure']
    rho_right = right_state['density']
    e_internal = p_right / ((gamma - 1) * rho_right)
    state['rho_e'][mask] = rho_right * e_internal

    # Initialize solids
    solid_segments = initialize_solids('thin_sheet')

    # Visualization setup
    plt.ion()  # Interactive mode
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Time stepping
    t = 0.0
    step = 0
    dt_history = []

    print(f"Starting TopoFluid2D simulation...")
    print(f"Domain: {domain_bounds}")
    print(f"Particles: {n_particles}")
    print(f"CFL: {cfl}")

    # Store initial state for visualization
    cells = {}
    for i in range(len(positions)):
        cells[i] = {
            'source_position': positions[i],
            'contains_source': True
        }

    while t < t_final:
        # 1. Compute Voronoi diagram
        vor = compute_voronoi_diagram(positions)

        # 2. Clip Voronoi cells by solid boundaries
        clipped_cells = clip_voronoi_by_solids(vor, solid_segments)

        # 3. Stitch orphaned cells
        stitched_cells = stitch_orphaned_cells(clipped_cells, positions)

        # 4. Compute interface geometry
        interfaces = compute_interface_geometry(stitched_cells)

        # 5. Apply boundary conditions (reflected particles)
        bc_interfaces = apply_boundary_conditions(interfaces, solid_segments, state)

        # 6. Compute numerical fluxes
        fluxes = compute_numerical_flux(bc_interfaces, state)

        # 7. Compute timestep
        dt = compute_timestep(state, interfaces, cfl)
        dt = min(dt, t_final - t)
        dt_history.append(dt)

        # 8. Update fluid state
        state = update_fluid_state(state, fluxes, interfaces, dt)

        # 9. Update particle positions (Lagrangian motion)
        u = state['rho_u'] / state['rho']
        v = state['rho_v'] / state['rho']
        positions += dt * np.column_stack([u, v])
        state['positions'] = positions

        # 10. Update time
        t += dt
        step += 1

        # Visualization every 10 steps
        if step % 10 == 0:
            ax1.clear()
            ax2.clear()

            # Plot Voronoi mesh
            plot_voronoi_mesh(ax1, stitched_cells, solid_segments)
            ax1.set_title(f'Voronoi Mesh (t={t:.3f})')
            ax1.set_xlim(domain_bounds[0])
            ax1.set_ylim(domain_bounds[1])

            # Plot pressure field
            rho = state['rho']
            rho_e = state['rho_e']
            rho_u = state['rho_u']
            rho_v = state['rho_v']
            e_kinetic = 0.5 * (rho_u ** 2 + rho_v ** 2) / rho
            e_internal = rho_e / rho - e_kinetic
            pressure = (gamma - 1) * rho * e_internal

            plot_pressure_field(ax2, positions, pressure, domain_bounds)
            ax2.set_title(f'Pressure Field (t={t:.3f})')

            plt.pause(0.01)

        # Progress report
        if step % 100 == 0:
            avg_dt = np.mean(dt_history[-100:]) if len(dt_history) > 100 else np.mean(dt_history)
            print(f"Step {step}: t={t:.4f}, dt={dt:.2e}, avg_dt={avg_dt:.2e}")

    print(f"\nSimulation complete!")
    print(f"Total steps: {step}")
    print(f"Average timestep: {np.mean(dt_history):.2e}")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()