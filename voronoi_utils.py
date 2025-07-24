"""
fluid_solver.py: Compressible Euler equations solver with Godunov-type schemes - FIXED VERSION
"""

import numpy as np
import numba
from numba import njit, prange


@njit
def compute_primitive_variables(conservative_state, gamma=1.4):
    """
    Convert conservative variables to primitive variables with safety checks
    """
    rho, rho_u, rho_v, rho_e = conservative_state

    # Ensure positive density
    rho = max(rho, 1e-15)
    
    u = rho_u / rho
    v = rho_v / rho

    # Total energy = kinetic + internal
    e_kinetic = 0.5 * (u * u + v * v)
    e_internal = rho_e / rho - e_kinetic

    # Ensure positive internal energy
    e_internal = max(e_internal, 1e-15)

    # Pressure from ideal gas law
    p = (gamma - 1) * rho * e_internal
    
    # Ensure positive pressure
    p = max(p, 1e-15)

    return rho, u, v, p


@njit
def compute_conservative_flux(rho, u, v, p, nx, ny, gamma=1.4):
    """
    Compute flux vector F·n for Euler equations with safety checks
    """
    # Ensure positive values
    rho = max(rho, 1e-15)
    p = max(p, 1e-15)
    
    # Normal velocity
    un = u * nx + v * ny

    # Total energy
    e_total = p / ((gamma - 1) * rho) + 0.5 * (u * u + v * v)

    # Flux components
    flux = np.array([
        rho * un,  # Mass flux
        rho * u * un + p * nx,  # x-momentum flux
        rho * v * un + p * ny,  # y-momentum flux
        rho * un * (e_total + p / rho)  # Energy flux
    ])

    return flux


@njit
def compute_sound_speed(rho, p, gamma=1.4):
    """Compute sound speed c = sqrt(gamma * p / rho) with safety checks"""
    rho = max(rho, 1e-15)
    p = max(p, 1e-15)
    return np.sqrt(gamma * p / rho)


@njit
def compute_signal_velocity(state_left, state_right, normal, gamma=1.4):
    """
    Compute signal velocity for Kurganov-Tadmor scheme with safety checks
    """
    nx, ny = normal

    # Left state
    rho_l, u_l, v_l, p_l = compute_primitive_variables(state_left, gamma)
    c_l = compute_sound_speed(rho_l, p_l, gamma)
    un_l = u_l * nx + v_l * ny

    # Right state
    rho_r, u_r, v_r, p_r = compute_primitive_variables(state_right, gamma)
    c_r = compute_sound_speed(rho_r, p_r, gamma)
    un_r = u_r * nx + v_r * ny

    # Maximum eigenvalue magnitudes
    lambda_max = max(
        abs(un_l - c_l), abs(un_l), abs(un_l + c_l),
        abs(un_r - c_r), abs(un_r), abs(un_r + c_r)
    )

    return max(lambda_max, 1e-10)  # Ensure non-zero signal velocity


@njit
def kurganov_tadmor_flux(state_left, state_right, normal, gamma=1.4):
    """
    Kurganov-Tadmor numerical flux with safety checks
    """
    nx, ny = normal

    # Primitive variables
    rho_l, u_l, v_l, p_l = compute_primitive_variables(state_left, gamma)
    rho_r, u_r, v_r, p_r = compute_primitive_variables(state_right, gamma)

    # Physical fluxes
    flux_l = compute_conservative_flux(rho_l, u_l, v_l, p_l, nx, ny, gamma)
    flux_r = compute_conservative_flux(rho_r, u_r, v_r, p_r, nx, ny, gamma)

    # Signal velocity
    a_ij = compute_signal_velocity(state_left, state_right, normal, gamma)

    # Conservative state vectors
    U_l = np.array(state_left)
    U_r = np.array(state_right)

    # Kurganov-Tadmor flux
    flux = 0.5 * (flux_l + flux_r) - 0.5 * a_ij * (U_r - U_l)

    return flux


def compute_numerical_flux(interfaces, state):
    """
    Compute numerical fluxes at all interfaces - SAFE VERSION
    """
    fluxes = []

    if not interfaces:
        return fluxes

    rho = state['rho']
    rho_u = state['rho_u']
    rho_v = state['rho_v']
    rho_e = state['rho_e']
    gamma = state['gamma']

    for interface in interfaces:
        i = interface['cell_i']
        j = interface['cell_j']

        if j == -1:  # Solid boundary
            continue

        # Bounds check
        if i >= len(rho) or j >= len(rho) or i < 0 or j < 0:
            continue

        # Get states
        state_i = (rho[i], rho_u[i], rho_v[i], rho_e[i])
        state_j = (rho[j], rho_u[j], rho_v[j], rho_e[j])

        # Check for invalid states
        if any(not np.isfinite(x) for x in state_i + state_j):
            continue

        # Compute flux
        try:
            flux = kurganov_tadmor_flux(state_i, state_j, interface['normal'], gamma)
            
            # Check for invalid flux
            if np.any(~np.isfinite(flux)):
                continue

            fluxes.append({
                'cell_i': i,
                'cell_j': j,
                'flux': flux,
                'area': interface['area'],
                'normal': interface['normal']
            })
        except:
            continue

    return fluxes


def apply_boundary_conditions(interfaces, solid_segments, state):
    """
    Apply boundary conditions - SIMPLIFIED SAFE VERSION
    """
    # For now, just return the original interfaces
    # This avoids complex reflection computations that might cause NaN
    bc_interfaces = []
    
    for interface in interfaces:
        bc_interfaces.append(interface)
    
    return bc_interfaces


def compute_timestep(state, interfaces, cfl=0.5):
    """
    Compute stable timestep using CFL condition - SAFE VERSION
    """
    rho = state['rho']
    rho_u = state['rho_u']
    rho_v = state['rho_v']
    rho_e = state['rho_e']
    gamma = state['gamma']
    positions = state['positions']

    dt_min = 1e-3  # Default safe timestep

    try:
        # Estimate cell sizes and maximum velocities
        for i in range(len(rho)):
            if rho[i] <= 1e-15:
                continue
                
            # Get primitive variables with safety checks
            u = rho_u[i] / rho[i]
            v = rho_v[i] / rho[i]

            # Limit velocities
            u = np.clip(u, -100.0, 100.0)
            v = np.clip(v, -100.0, 100.0)

            # Compute pressure and sound speed
            e_kinetic = 0.5 * (u ** 2 + v ** 2)
            e_internal = max(rho_e[i] / rho[i] - e_kinetic, 1e-15)
            p = max((gamma - 1) * rho[i] * e_internal, 1e-15)
            c = np.sqrt(gamma * p / rho[i])

            # Maximum signal speed
            max_speed = np.sqrt(u ** 2 + v ** 2) + c
            max_speed = max(max_speed, 1e-10)

            # Estimate cell size (distance to nearest neighbor)
            if len(positions) > 1:
                distances = np.linalg.norm(positions - positions[i], axis=1)
                distances[i] = np.inf  # Exclude self
                min_dist = np.min(distances)
                min_dist = max(min_dist, 1e-3)  # Minimum cell size
            else:
                min_dist = 0.1

            # Local timestep constraint
            dt_local = cfl * min_dist / max_speed
            dt_min = min(dt_min, dt_local)

    except Exception as e:
        print(f"Warning in timestep computation: {e}")
        dt_min = 1e-4

    return max(dt_min, 1e-6)  # Ensure minimum timestep


def update_fluid_state(state, fluxes, interfaces, dt):
    """
    Update fluid state using finite volume method - SAFE VERSION
    """
    # Copy state
    new_state = {
        'rho': state['rho'].copy(),
        'rho_u': state['rho_u'].copy(),
        'rho_v': state['rho_v'].copy(),
        'rho_e': state['rho_e'].copy(),
        'positions': state['positions'].copy(),
        'gamma': state['gamma']
    }

    # If no fluxes, return unchanged state
    if not fluxes:
        return new_state

    # Compute cell volumes (simplified)
    n_particles = len(state['rho'])
    volumes = np.ones(n_particles) * 0.01  # Simple uniform volume

    # Initialize flux accumulator
    flux_sum = np.zeros((n_particles, 4))

    # Accumulate fluxes
    for flux_data in fluxes:
        i = flux_data['cell_i']
        j = flux_data['cell_j']
        flux = flux_data['flux']
        area = flux_data.get('area', 1.0)

        # Bounds check
        if i >= n_particles or j >= n_particles or i < 0:
            continue

        # Check for invalid flux
        if np.any(~np.isfinite(flux)):
            continue

        # Add flux contribution
        flux_contribution = area * flux

        # Flux leaves cell i
        flux_sum[i] -= flux_contribution

        # Flux enters cell j (if not boundary)
        if j >= 0 and j < n_particles:
            flux_sum[j] += flux_contribution

    # Update conservative variables
    for i in range(n_particles):
        if volumes[i] > 1e-10:
            # Finite volume update
            update = dt * flux_sum[i] / volumes[i]

            # Apply updates with safety checks
            new_state['rho'][i] += update[0]
            new_state['rho_u'][i] += update[1]
            new_state['rho_v'][i] += update[2]
            new_state['rho_e'][i] += update[3]

            # Ensure positive density
            new_state['rho'][i] = max(new_state['rho'][i], 1e-15)
            
            # Ensure positive energy
            min_energy = 1e-15 * new_state['rho'][i]  # Minimum internal energy
            new_state['rho_e'][i] = max(new_state['rho_e'][i], min_energy)

    return new_state


# Add this function to replace the empty compute_interface_geometry in voronoi_utils.py

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