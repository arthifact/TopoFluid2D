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

# Check if modules exist
from voronoi_utils import (
    compute_voronoi_diagram,
    clip_voronoi_by_solids,
    stitch_orphaned_cells,
    compute_interface_geometry
)
from fluid_solver import (
    compute_numerical_flux,
    update_fluid_state,
    compute_timestep,
    apply_boundary_conditions
)

from solid_handler import (
    update_solid_positions,
    compute_fluid_solid_coupling
)

from visualization import (
    plot_voronoi_mesh,
    plot_pressure_field,
    create_animation
)

def initialize_fluid_particles(domain_bounds, n_particles, initial_state):
    """
    Initialize fluid particles with positions and state variables

    Parameters:
    -----------
    domain_bounds : tuple
        ((x_min, x_max), (y_min, y_max))
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
    x_min, x_max = domain_bounds[0]
    y_min, y_max = domain_bounds[1]

    # Create regular grid of particles
    nx = int(np.sqrt(n_particles * (x_max - x_min) / (y_max - y_min)))
    ny = int(n_particles / nx)

    x = np.linspace(x_min + 0.05 * (x_max - x_min), x_max - 0.05 * (x_max - x_min), nx)
    y = np.linspace(y_min + 0.05 * (y_max - y_min), y_max - 0.05 * (y_max - y_min), ny)
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
