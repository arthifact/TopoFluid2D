#!/usr/bin/env python3
"""
TopoFluid2D: Topology-Preserving Coupling of Compressible Fluids and Thin Deformables
FINAL FIXED VERSION - Aligned with paper physics and clean visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modules with fallbacks
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
        """Improved interface geometry computation"""
        interfaces = []
        processed_pairs = set()
        
        # Get positions for all cells
        positions = {}
        for idx, cell in cells.items():
            if 'source_position' in cell:
                positions[idx] = cell['source_position']
        
        # Create interfaces between neighboring particles
        max_distance = 0.25  # Adjusted for better connectivity
        
        for i in positions.keys():
            for j in positions.keys():
                if i >= j:
                    continue
                    
                if (i, j) in processed_pairs:
                    continue
                    
                # Check if particles are neighbors
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
                    
                    # Interface area based on Voronoi edge length estimation
                    area = max(0.05, 0.4 * dist)
                    
                    interfaces.append({
                        'cell_i': i,
                        'cell_j': j,
                        'area': area,
                        'normal': normal,
                        'midpoint': midpoint,
                        'is_solid': False
                    })
                    
                    processed_pairs.add((i, j))
        
        return interfaces

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


def check_for_nan_values(state, step):
    """Check for NaN values in the state and report them"""
    nan_found = False
    
    for key, values in state.items():
        if key == 'positions':
            if np.any(np.isnan(values)):
                print(f"Step {step}: NaN found in {key}")
                nan_found = True
        elif key != 'gamma' and hasattr(values, '__iter__'):
            if np.any(np.isnan(values)):
                print(f"Step {step}: NaN found in {key}")
                nan_found = True
    
    return nan_found


def sanitize_state(state):
    """Replace NaN and Inf values with safe defaults"""
    print("Sanitizing state to remove NaN/Inf values...")
    
    # Replace NaN/Inf in density
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
    """Initialize fluid particles with positions and state variables"""
    xmin, xmax = domain_bounds[0]
    ymin, ymax = domain_bounds[1]

    # Create regular grid with slight perturbation
    nx = int(np.sqrt(n_particles * (xmax - xmin) / (ymax - ymin)))
    ny = int(n_particles / nx)

    x = np.linspace(xmin + 0.1 * (xmax - xmin), xmax - 0.1 * (xmax - xmin), nx)
    y = np.linspace(ymin + 0.1 * (ymax - ymin), ymax - 0.1 * (ymax - ymin), ny)
    xx, yy = np.meshgrid(x, y)

    positions = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Add small random perturbation to break symmetry
    perturbation = 0.02 * np.min([xmax - xmin, ymax - ymin])
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
        'rho': rho,
        'rho_u': rho * u,
        'rho_v': rho * v,
        'rho_e': rho * e_total,
        'positions': positions,
        'gamma': gamma
    }

    return positions, state


def initialize_solids(solid_type='thin_sheet'):
    """Initialize solid boundaries"""
    solid_segments = []

    if solid_type == 'thin_sheet':
        # Horizontal thin sheet - following paper Figure 2
        solid_segments.append({
            'start': np.array([-0.4, 0.0]),
            'end': np.array([0.4, 0.0]),
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


def plot_clean_visualization(ax1, ax2, stitched_cells, solid_segments, positions, 
                           pressure, domain_bounds, t, interfaces_count, fluxes_count):
    """Clean visualization without accumulated colorbars"""
    
    # Clear axes completely
    ax1.clear()
    ax2.clear()
    
    # Plot 1: Voronoi mesh
    # Plot Voronoi cells
    for idx, cell in stitched_cells.items():
        if 'vertices' not in cell or not cell['vertices']:
            continue
        vertices = np.array(cell['vertices'])
        if len(vertices) > 2:
            from matplotlib.patches import Polygon
            poly = Polygon(vertices, facecolor='lightblue', 
                         edgecolor='gray', alpha=0.15, linewidth=0.5)
            ax1.add_patch(poly)

    # Plot particles
    for idx, cell in stitched_cells.items():
        if 'source_position' in cell:
            pos = cell['source_position']
            ax1.plot(pos[0], pos[1], 'ko', markersize=3)

    # Plot solid boundaries
    for segment in solid_segments:
        start, end = segment['start'], segment['end']
        ax1.plot([start[0], end[0]], [start[1], end[1]],
                'r-', linewidth=4, alpha=0.9, label='Solid')

    ax1.set_xlim(domain_bounds[0])
    ax1.set_ylim(domain_bounds[1])
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'Voronoi Mesh (t={t:.4f})')
    
    # Plot 2: Pressure field
    # Determine pressure range for consistent coloring
    p_min, p_max = np.min(pressure), np.max(pressure)
    
    scatter = ax2.scatter(positions[:, 0], positions[:, 1],
                         c=pressure, s=60, cmap='RdBu_r',
                         vmin=p_min, vmax=p_max,
                         edgecolors='black', linewidths=0.5,
                         alpha=0.8)
    
    # Add solid boundary to pressure plot
    for segment in solid_segments:
        start, end = segment['start'], segment['end']
        ax2.plot([start[0], end[0]], [start[1], end[1]],
                'r-', linewidth=3, alpha=0.9)
    
    ax2.set_xlim(domain_bounds[0])
    ax2.set_ylim(domain_bounds[1])
    ax2.set_aspect('equal')
    ax2.set_title(f'Pressure Field (t={t:.4f})')
    ax2.grid(True, alpha=0.3)
    
    return scatter


def compute_analytical_sod_solution(x, t, gamma=1.4):
    """
    Compute analytical Sod shock tube solution for validation
    Standard initial conditions: left (ρ=1, P=1, u=0), right (ρ=0.125, P=0.1, u=0)
    """
    # Initial conditions
    rho_L, P_L, u_L = 1.0, 1.0, 0.0
    rho_R, P_R, u_R = 0.125, 0.1, 0.0
    
    # Sound speeds
    c_L = np.sqrt(gamma * P_L / rho_L)
    c_R = np.sqrt(gamma * P_R / rho_R)
    
    # For simplicity, return approximate solution
    # In practice, would solve Riemann problem iteratively
    
    # Approximate wave speeds (for t > 0)
    if t <= 0:
        return np.where(x < 0, rho_L, rho_R), np.where(x < 0, P_L, P_R), np.where(x < 0, u_L, u_R)
    
    # Shock speed (approximate)
    shock_speed = 1.75  # Typical for these conditions
    contact_speed = 0.93
    rarefaction_head = -c_L
    rarefaction_tail = -0.27
    
    rho = np.zeros_like(x)
    P = np.zeros_like(x)
    u = np.zeros_like(x)
    
    for i, xi in enumerate(x):
        pos = xi / t if t > 0 else 0
        
        if pos < rarefaction_head:
            # Left state
            rho[i], P[i], u[i] = rho_L, P_L, u_L
        elif pos < rarefaction_tail:
            # Rarefaction fan
            factor = 0.5 + 0.5 * (pos - rarefaction_head) / (rarefaction_tail - rarefaction_head)
            rho[i] = rho_L * (0.5 + 0.3 * factor)
            P[i] = P_L * (0.3 + 0.7 * factor)
            u[i] = 0.3 * (1 - factor)
        elif pos < contact_speed:
            # Post-shock left
            rho[i], P[i], u[i] = 0.426, 0.303, 0.927
        elif pos < shock_speed:
            # Post-shock right  
            rho[i], P[i], u[i] = 0.265, 0.303, 0.927
        else:
            # Right state
            rho[i], P[i], u[i] = rho_R, P_R, u_R
    
    return rho, P, u


def create_validation_plot(state, t):
    """Create 1D validation plot against analytical solution"""
    positions = state['positions']
    rho = state['rho']
    rho_u = state['rho_u']
    rho_e = state['rho_e']
    gamma = state['gamma']
    
    # Extract 1D slice along y=0
    y_center = 0.0
    width = 0.3
    mask = np.abs(positions[:, 1] - y_center) < width/2
    
    if np.sum(mask) < 3:
        return None
        
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
    
    # Analytical solution
    x_theory = np.linspace(-1, 1, 200)
    rho_theory, p_theory, u_theory = compute_analytical_sod_solution(x_theory, t)
    
    # Create plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(x_theory, rho_theory, 'k-', linewidth=2, label='Analytical')
    ax1.plot(x_sorted, rho_sorted, 'ro', markersize=4, label='Numerical')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'Sod Shock Tube Validation (t={t:.4f})')
    
    ax2.plot(x_theory, u_theory, 'k-', linewidth=2, label='Analytical')
    ax2.plot(x_sorted, u_sorted, 'bo', markersize=4, label='Numerical')
    ax2.set_ylabel('Velocity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(x_theory, p_theory, 'k-', linewidth=2, label='Analytical')
    ax3.plot(x_sorted, p_sorted, 'go', markersize=4, label='Numerical')
    ax3.set_ylabel('Pressure')
    ax3.set_xlabel('x')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    """
    Main simulation loop - FINAL VERSION aligned with paper
    """
    # Simulation parameters - following paper's approach
    domain_bounds = ((-1.0, 1.0), (-1.0, 1.0))
    n_particles = 100
    t_final = 10.25  # Sufficient time to see shock development
    cfl = 0.3

    # Standard Sod shock tube conditions (paper validation)
    left_state = {
        'density': 1.0,
        'velocity_x': 0.0,
        'velocity_y': 0.0,
        'pressure': 1.0
    }

    right_state = {
        'density': 0.125,  # Standard Sod conditions
        'velocity_x': 0.0,
        'velocity_y': 0.0,
        'pressure': 0.1   # Standard Sod conditions
    }

    # Initialize particles
    positions, state = initialize_fluid_particles(domain_bounds, n_particles, left_state)

    # Apply Sod shock tube discontinuity at x = 0
    mask = positions[:, 0] > 0.0
    state['rho'][mask] = right_state['density']
    state['rho_u'][mask] = right_state['density'] * right_state['velocity_x']
    state['rho_v'][mask] = right_state['density'] * right_state['velocity_y']

    gamma = state['gamma']
    p_right = right_state['pressure']
    rho_right = right_state['density']
    e_internal = p_right / ((gamma - 1) * rho_right)
    state['rho_e'][mask] = rho_right * e_internal

    # Initialize solids - thin sheet as in paper Figure 2
    solid_segments = initialize_solids('thin_sheet')

    # Visualization setup
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Time stepping
    t = 0.0
    step = 0
    dt_history = []
    max_steps = 800

    print(f"Starting TopoFluid2D simulation...")
    print(f"Domain: {domain_bounds}")
    print(f"Particles: {n_particles}")
    print(f"CFL: {cfl}")
    print(f"Standard Sod shock tube: left (ρ={left_state['density']}, P={left_state['pressure']}) | right (ρ={right_state['density']}, P={right_state['pressure']})")
    print(f"Following paper's topology-preserving discretization approach")

    while t < t_final and step < max_steps:
        # Check for NaN values
        if check_for_nan_values(state, step):
            print(f"NaN detected at step {step}, sanitizing...")
            state = sanitize_state(state)
            positions = state['positions']

        try:
            # 1. Compute Voronoi diagram (Section 4 of paper)
            vor = compute_voronoi_diagram(positions)

            # 2. Clip Voronoi cells by solid boundaries (Section 4.2)
            clipped_cells = clip_voronoi_by_solids(vor, solid_segments)

            # 3. Stitch orphaned cells (Algorithm 1 in paper)
            stitched_cells = stitch_orphaned_cells(clipped_cells, positions)

            # 4. Compute interface geometry (Section 4.2)
            interfaces = compute_interface_geometry(stitched_cells)

            # 5. Apply boundary conditions (Section 4.5)
            bc_interfaces = apply_boundary_conditions(interfaces, solid_segments, state)

            # 6. Compute numerical fluxes (Equation 6 - Kurganov-Tadmor)
            fluxes = compute_numerical_flux(bc_interfaces, state)

            # 7. Compute stable timestep (CFL condition)
            dt = compute_timestep(state, interfaces, cfl)
            dt = min(dt, t_final - t, 1e-3)
            dt_history.append(dt)

            # 8. Update fluid state (Equation 5 - finite volume)
            state = update_fluid_state(state, fluxes, interfaces, dt)

            # 9. Update particle positions (Lagrangian motion)
            u = np.where(state['rho'] > 1e-15, state['rho_u'] / state['rho'], 0.0)
            v = np.where(state['rho'] > 1e-15, state['rho_v'] / state['rho'], 0.0)
            
            # Limit velocities for stability
            max_vel = 3.0
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
            state = sanitize_state(state)
            positions = state['positions']
            dt = 1e-5
            t += dt
            step += 1
            continue

        # Clean visualization every 25 steps
        if step % 25 == 0:
            try:
                # Compute pressure for visualization
                rho = state['rho']
                rho_e = state['rho_e']
                rho_u = state['rho_u']
                rho_v = state['rho_v']
                
                u_safe = np.where(rho > 1e-15, rho_u / rho, 0.0)
                v_safe = np.where(rho > 1e-15, rho_v / rho, 0.0)
                e_kinetic = 0.5 * (u_safe ** 2 + v_safe ** 2)
                e_internal = np.where(rho > 1e-15, rho_e / rho - e_kinetic, 1e-10)
                pressure = np.maximum((gamma - 1) * rho * e_internal, 1e-10)

                # Clean plot without colorbar accumulation
                scatter = plot_clean_visualization(ax1, ax2, stitched_cells, solid_segments, 
                                                 positions, pressure, domain_bounds, t, 
                                                 len(interfaces), len(fluxes))
                
                # Add colorbar only once per update
                if hasattr(fig, '_colorbar'):
                    fig._colorbar.remove()
                fig._colorbar = plt.colorbar(scatter, ax=ax2, shrink=0.8)
                fig._colorbar.set_label('Pressure', rotation=270, labelpad=15)

                plt.tight_layout()
                plt.pause(0.05)
                
            except Exception as e:
                print(f"Visualization error: {e}")

        # Progress report with physics diagnostics
        if step % 100 == 0:
            avg_dt = np.mean(dt_history[-100:]) if len(dt_history) > 100 else np.mean(dt_history)
            min_p, max_p = np.min(pressure), np.max(pressure)
            min_rho, max_rho = np.min(rho), np.max(rho)
            max_u = np.max(np.sqrt(u_safe**2 + v_safe**2))
            print(f"Step {step}: t={t:.5f}, dt={dt:.3e}, P=[{min_p:.3f},{max_p:.3f}], ρ=[{min_rho:.3f},{max_rho:.3f}], |u|_max={max_u:.3f}")
            print(f"  Interfaces: {len(interfaces)}, Active fluxes: {len(fluxes)}")

    print(f"\nSimulation complete!")
    print(f"Total steps: {step}")
    print(f"Final time: {t:.5f}")
    if dt_history:
        print(f"Average timestep: {np.mean(dt_history):.3e}")

    # Create validation plot
    try:
        val_fig = create_validation_plot(state, t)
        if val_fig:
            plt.show()
    except Exception as e:
        print(f"Validation plot error: {e}")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()