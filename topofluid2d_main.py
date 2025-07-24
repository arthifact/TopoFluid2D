#!/usr/bin/env python3
"""
TopoFluid2D: Topology-Preserving Coupling of Compressible Fluids and Thin Deformables
Main simulation script for 2D implementation - FIXED VERSION
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


def check_for_nan_values(state, step):
    """Check for NaN values in the state and report them"""
    nan_found = False
    
    for key, values in state.items():
        if key == 'positions':
            if np.any(np.isnan(values)):
                print(f"Step {step}: NaN found in {key}")
                nan_positions = np.where(np.isnan(values))
                print(f"  Positions with NaN: {nan_positions}")
                nan_found = True
        elif key != 'gamma' and hasattr(values, '__iter__'):
            if np.any(np.isnan(values)):
                print(f"Step {step}: NaN found in {key}")
                nan_indices = np.where(np.isnan(values))[0]
                print(f"  Indices with NaN: {nan_indices[:10]}...")  # Show first 10
                nan_found = True
    
    return nan_found


def sanitize_state(state):
    """Replace NaN and Inf values with safe defaults"""
    print("Sanitizing state to remove NaN/Inf values...")
    
    # Replace NaN/Inf in density with minimum safe value
    rho_min = 1e-10
    mask_rho = ~np.isfinite(state['rho']) | (state['rho'] <= 0)
    if np.any(mask_rho):
        print(f"  Fixed {np.sum(mask_rho)} density values")
        state['rho'][mask_rho] = rho_min
    
    # Replace NaN/Inf in momentum
    mask_rho_u = ~np.isfinite(state['rho_u'])
    if np.any(mask_rho_u):
        print(f"  Fixed {np.sum(mask_rho_u)} x-momentum values")
        state['rho_u'][mask_rho_u] = 0.0
    
    mask_rho_v = ~np.isfinite(state['rho_v'])
    if np.any(mask_rho_v):
        print(f"  Fixed {np.sum(mask_rho_v)} y-momentum values")
        state['rho_v'][mask_rho_v] = 0.0
    
    # Replace NaN/Inf in energy
    mask_rho_e = ~np.isfinite(state['rho_e'])
    if np.any(mask_rho_e):
        print(f"  Fixed {np.sum(mask_rho_e)} energy values")
        # Set to minimum internal energy
        gamma = state['gamma']
        p_min = 1e-10
        e_internal_min = p_min / ((gamma - 1) * state['rho'][mask_rho_e])
        state['rho_e'][mask_rho_e] = state['rho'][mask_rho_e] * e_internal_min
    
    # Replace NaN/Inf in positions
    mask_pos = ~np.isfinite(state['positions'])
    if np.any(mask_pos):
        print(f"  Fixed {np.sum(mask_pos)} position values")
        state['positions'][mask_pos] = 0.0
    
    return state


def initialize_fluid_particles(domain_bounds, n_particles, initial_state):
    """
    Initialize fluid particles with positions and state variables
    """
    xmin, xmax = domain_bounds[0]
    ymin, ymax = domain_bounds[1]

    # Create regular grid of particles with some randomization to avoid degeneracies
    nx = int(np.sqrt(n_particles * (xmax - xmin) / (ymax - ymin)))
    ny = int(n_particles / nx)

    x = np.linspace(xmin + 0.1 * (xmax - xmin), xmax - 0.1 * (xmax - xmin), nx)
    y = np.linspace(ymin + 0.1 * (ymax - ymin), ymax - 0.1 * (ymax - ymin), ny)
    xx, yy = np.meshgrid(x, y)

    positions = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Add small random perturbation to avoid perfect alignment
    perturbation = 0.01 * np.min([xmax - xmin, ymax - ymin])
    positions += perturbation * (np.random.random(positions.shape) - 0.5)

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


def compute_interface_geometry(cells):
    """
    Compute geometric properties of interfaces between cells - IMPROVED VERSION
    """
    interfaces = []
    processed_pairs = set()
    
    # Get positions for all cells
    positions = {}
    for idx, cell in cells.items():
        if 'source_position' in cell:
            positions[idx] = cell['source_position']
    
    # Create interfaces between neighboring particles
    # For simplicity, use distance-based neighborhood
    max_distance = 0.3  # Adjust based on particle spacing
    
    for i in positions.keys():
        for j in positions.keys():
            if i >= j:
                continue
                
            if (i, j) in processed_pairs:
                continue
                
            # Check if particles are neighbors (within max_distance)
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < max_distance:
                # Compute interface properties
                midpoint = 0.5 * (positions[i] + positions[j])
                
                # Normal vector points from i to j
                direction = positions[j] - positions[i]
                if np.linalg.norm(direction) > 1e-10:
                    normal = direction / np.linalg.norm(direction)
                else:
                    normal = np.array([1.0, 0.0])
                
                # Interface area (length in 2D) - estimate based on distance
                area = max(0.1, 0.5 * dist)  # Simple estimate
                
                interfaces.append({
                    'cell_i': i,
                    'cell_j': j,
                    'area': area,
                    'normal': normal,
                    'midpoint': midpoint,
                    'is_solid': False
                })
                
                processed_pairs.add((i, j))
    
    # Add solid interfaces if any
    for i, cell in cells.items():
        for solid_face in cell.get('solid_faces', []):
            # Compute normal pointing into fluid
            edge = solid_face['end'] - solid_face['start']
            if np.linalg.norm(edge) > 1e-10:
                normal = np.array([-edge[1], edge[0]])  # Rotate 90 degrees
                normal = normal / np.linalg.norm(normal)
                
                interfaces.append({
                    'cell_i': i,
                    'cell_j': -1,  # Solid boundary
                    'area': np.linalg.norm(edge),
                    'normal': normal,
                    'midpoint': 0.5 * (solid_face['start'] + solid_face['end']),
                    'is_solid': True,
                    'solid_velocity': solid_face['solid_ref'].get('velocity', np.array([0.0, 0.0]))
                })
    
    return interfaces


def main():
    """
    Main simulation loop - IMPROVED VERSION with better shock tube
    """
    # Simulation parameters
    domain_bounds = ((-1.0, 1.0), (-1.0, 1.0))
    n_particles = 100
    t_final = 0.2  # Longer simulation to see shock development
    cfl = 0.3

    # More pronounced initial conditions for visible shock
    left_state = {
        'density': 1.0,
        'velocity_x': 0.0,
        'velocity_y': 0.0,
        'pressure': 1.0
    }

    right_state = {
        'density': 0.3,  # More pronounced density difference
        'velocity_x': 0.0,
        'velocity_y': 0.0,
        'pressure': 0.3  # More pronounced pressure difference
    }

    # Initialize particles
    positions, state = initialize_fluid_particles(domain_bounds, n_particles, left_state)

    # Apply initial discontinuity at x = 0
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
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Time stepping
    t = 0.0
    step = 0
    dt_history = []
    max_steps = 500

    print(f"Starting TopoFluid2D simulation...")
    print(f"Domain: {domain_bounds}")
    print(f"Particles: {n_particles}")
    print(f"CFL: {cfl}")
    print(f"Initial shock: left (ρ={left_state['density']}, P={left_state['pressure']}) | right (ρ={right_state['density']}, P={right_state['pressure']})")

    while t < t_final and step < max_steps:
        # Check for NaN values before computation
        if check_for_nan_values(state, step):
            print(f"NaN detected at step {step}, attempting to sanitize...")
            state = sanitize_state(state)
            positions = state['positions']

        try:
            # 1. Compute Voronoi diagram
            vor = compute_voronoi_diagram(positions)

            # 2. Clip Voronoi cells by solid boundaries
            clipped_cells = clip_voronoi_by_solids(vor, solid_segments)

            # 3. Stitch orphaned cells
            stitched_cells = stitch_orphaned_cells(clipped_cells, positions)

            # 4. Compute interface geometry (IMPROVED)
            interfaces = compute_interface_geometry(stitched_cells)

            # 5. Apply boundary conditions
            bc_interfaces = apply_boundary_conditions(interfaces, solid_segments, state)

            # 6. Compute numerical fluxes
            fluxes = compute_numerical_flux(bc_interfaces, state)

            # 7. Compute timestep
            dt = compute_timestep(state, interfaces, cfl)
            dt = min(dt, t_final - t, 5e-4)  # Slightly larger max timestep
            dt_history.append(dt)

            # 8. Update fluid state
            state = update_fluid_state(state, fluxes, interfaces, dt)

            # 9. Update particle positions (Lagrangian motion) with safety checks
            u = np.where(state['rho'] > 1e-15, state['rho_u'] / state['rho'], 0.0)
            v = np.where(state['rho'] > 1e-15, state['rho_v'] / state['rho'], 0.0)
            
            # Limit maximum velocity to prevent runaway
            max_vel = 5.0  # Reduced from 10.0
            u = np.clip(u, -max_vel, max_vel)
            v = np.clip(v, -max_vel, max_vel)
            
            displacement = dt * np.column_stack([u, v])
            positions += displacement
            state['positions'] = positions

            # 10. Update time
            t += dt
            step += 1

        except Exception as e:
            print(f"Error at step {step}: {e}")
            print("Attempting to recover...")
            state = sanitize_state(state)
            positions = state['positions']
            dt = 1e-5
            t += dt
            step += 1
            continue

        # Visualization every 20 steps
        if step % 20 == 0:
            try:
                ax1.clear()
                ax2.clear()

                # Plot Voronoi mesh with solid boundary
                plot_voronoi_mesh(ax1, stitched_cells, solid_segments)
                ax1.set_title(f'Voronoi Mesh (t={t:.4f})')
                ax1.set_xlim(domain_bounds[0])
                ax1.set_ylim(domain_bounds[1])
                
                # Add solid boundary visualization
                for segment in solid_segments:
                    start, end = segment['start'], segment['end']
                    ax1.plot([start[0], end[0]], [start[1], end[1]], 'r-', linewidth=3, label='Solid')

                # Plot pressure field
                rho = state['rho']
                rho_e = state['rho_e']
                rho_u = state['rho_u']
                rho_v = state['rho_v']
                
                # Safe pressure computation
                u_safe = np.where(rho > 1e-15, rho_u / rho, 0.0)
                v_safe = np.where(rho > 1e-15, rho_v / rho, 0.0)
                e_kinetic = 0.5 * (u_safe ** 2 + v_safe ** 2)
                e_internal = np.where(rho > 1e-15, rho_e / rho - e_kinetic, 1e-10)
                pressure = np.maximum((gamma - 1) * rho * e_internal, 1e-10)

                # Create pressure plot with better color range
                scatter = ax2.scatter(positions[:, 0], positions[:, 1],
                                     c=pressure, cmap='RdBu_r', s=60,
                                     vmin=0.25, vmax=1.1)  # Fixed color range to see variation
                ax2.set_xlim(domain_bounds[0])
                ax2.set_ylim(domain_bounds[1])
                ax2.set_aspect('equal')
                ax2.set_title(f'Pressure Field (t={t:.4f})')
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax2)
                cbar.set_label('Pressure')
                
                # Add interface visualization
                if len(interfaces) > 0:
                    print(f"Step {step}: {len(interfaces)} interfaces, {len(fluxes)} fluxes")

                plt.tight_layout()
                plt.pause(0.05)
                
            except Exception as e:
                print(f"Visualization error: {e}")

        # Progress report
        if step % 50 == 0:
            avg_dt = np.mean(dt_history[-50:]) if len(dt_history) > 50 else np.mean(dt_history)
            min_p, max_p = np.min(pressure), np.max(pressure)
            min_rho, max_rho = np.min(rho), np.max(rho)
            print(f"Step {step}: t={t:.5f}, dt={dt:.3e}, P=[{min_p:.3f},{max_p:.3f}], ρ=[{min_rho:.3f},{max_rho:.3f}]")

    print(f"\nSimulation complete!")
    print(f"Total steps: {step}")
    print(f"Final time: {t:.5f}")
    if dt_history:
        print(f"Average timestep: {np.mean(dt_history):.3e}")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()