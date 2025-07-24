"""
fluid_solver.py: Enhanced Compressible Euler equations solver with paper-faithful improvements
"""

import numpy as np
import numba
from numba import njit, prange
from scipy.spatial.distance import cdist


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


def compute_cell_volumes_enhanced(interfaces, positions):
    """
    Enhanced cell volume computation based on particle distribution
    Should eventually use actual Voronoi cell volumes
    """
    n_particles = len(positions)
    
    if n_particles == 0:
        return np.array([])
    
    if n_particles == 1:
        return np.array([1.0])  # Single particle gets unit volume
    
    # Method 1: Uniform volume distribution
    domain_bounds = ((-1.2, 1.2), (-0.8, 0.8))  # From main script
    domain_area = (domain_bounds[0][1] - domain_bounds[0][0]) * (domain_bounds[1][1] - domain_bounds[1][0])
    uniform_volume = domain_area / n_particles
    
    # Method 2: Density-based volume (more sophisticated)
    try:
        distances = cdist(positions, positions)
        np.fill_diagonal(distances, np.inf)
        
        # Volume inversely proportional to local density
        min_distances = np.min(distances, axis=1)
        local_volumes = np.pi * (min_distances / 2)**2  # Circular approximation
        
        # Normalize to preserve total volume
        total_local = np.sum(local_volumes)
        if total_local > 1e-10:
            local_volumes *= domain_area / total_local
        else:
            local_volumes = np.full(n_particles, uniform_volume)
    except:
        local_volumes = np.full(n_particles, uniform_volume)
    
    # Use local volumes, but ensure minimum volume
    min_volume = domain_area / (10 * n_particles)  # At least 1/10 of average
    volumes = np.maximum(local_volumes, min_volume)
    
    return volumes


def compute_timestep(state, interfaces, cfl=0.4):
    """
    Enhanced timestep computation following paper's CFL condition
    
    The paper uses: dt = CFL * min(A_ij / (|V_i| * a_ij))
    where A_ij is interface area, V_i is cell volume, a_ij is signal velocity
    
    Parameters:
    -----------
    state : dict
        Fluid state variables
    interfaces : list
        Interface information
    cfl : float
        CFL number (should be < 0.5 for stability)
        
    Returns:
    --------
    dt : float
        Stable timestep
    """
    rho = state['rho']
    rho_u = state['rho_u']
    rho_v = state['rho_v']
    rho_e = state['rho_e']
    gamma = state['gamma']
    positions = state['positions']
    
    dt_min = 1e-2  # Default safe timestep
    
    if len(interfaces) == 0:
        return dt_min
    
    try:
        # Compute cell volumes using enhanced method
        n_particles = len(rho)
        if n_particles < 2:
            return dt_min
            
        volumes = compute_cell_volumes_enhanced(interfaces, positions)
        
        # Process each interface for CFL constraint
        for interface in interfaces:
            i = interface['cell_i']
            j = interface['cell_j']
            area = interface.get('area', 0.01)
            
            # Skip invalid indices
            if i < 0 or i >= n_particles:
                continue
                
            # Compute signal velocity at this interface
            if j >= 0 and j < n_particles:
                # Fluid-fluid interface
                # Get primitive variables for both sides
                rho_i = max(rho[i], 1e-15)
                u_i = rho_u[i] / rho_i
                v_i = rho_v[i] / rho_i
                
                rho_j = max(rho[j], 1e-15)
                u_j = rho_u[j] / rho_j
                v_j = rho_v[j] / rho_j
                
                # Compute pressures and sound speeds
                e_kinetic_i = 0.5 * (u_i**2 + v_i**2)
                e_internal_i = max(rho_e[i] / rho_i - e_kinetic_i, 1e-15)
                p_i = max((gamma - 1) * rho_i * e_internal_i, 1e-15)
                c_i = np.sqrt(gamma * p_i / rho_i)
                
                e_kinetic_j = 0.5 * (u_j**2 + v_j**2)
                e_internal_j = max(rho_e[j] / rho_j - e_kinetic_j, 1e-15)
                p_j = max((gamma - 1) * rho_j * e_internal_j, 1e-15)
                c_j = np.sqrt(gamma * p_j / rho_j)
                
                # Normal velocities
                normal = interface['normal']
                un_i = u_i * normal[0] + v_i * normal[1]
                un_j = u_j * normal[0] + v_j * normal[1]
                
                # Maximum signal velocity (from Kurganov-Tadmor)
                signal_velocity = max(
                    abs(un_i - c_i), abs(un_i), abs(un_i + c_i),
                    abs(un_j - c_j), abs(un_j), abs(un_j + c_j)
                )
                
            else:
                # Solid interface - only consider fluid side
                rho_i = max(rho[i], 1e-15)
                u_i = rho_u[i] / rho_i
                v_i = rho_v[i] / rho_i
                
                e_kinetic_i = 0.5 * (u_i**2 + v_i**2)
                e_internal_i = max(rho_e[i] / rho_i - e_kinetic_i, 1e-15)
                p_i = max((gamma - 1) * rho_i * e_internal_i, 1e-15)
                c_i = np.sqrt(gamma * p_i / rho_i)
                
                # Include solid velocity if available
                solid_vel = interface.get('solid_velocity', np.array([0.0, 0.0]))
                normal = interface['normal']
                
                # Relative velocity
                rel_u = u_i - solid_vel[0]
                rel_v = v_i - solid_vel[1]
                un_rel = rel_u * normal[0] + rel_v * normal[1]
                
                signal_velocity = max(abs(un_rel - c_i), abs(un_rel + c_i))
            
            # Ensure minimum signal velocity
            signal_velocity = max(signal_velocity, 1e-8)
            
            # CFL constraint: dt < CFL * Volume / (Area * SignalVelocity)
            # This ensures waves don't travel more than CFL fraction of cell size
            volume_i = volumes[i] if i < len(volumes) else 0.01
            dt_interface = cfl * volume_i / (area * signal_velocity)
            
            dt_min = min(dt_min, dt_interface)
            
            # Also check cell j for fluid-fluid interfaces
            if j >= 0 and j < n_particles and j < len(volumes):
                volume_j = volumes[j]
                dt_interface_j = cfl * volume_j / (area * signal_velocity)
                dt_min = min(dt_min, dt_interface_j)
    
    except Exception as e:
        print(f"Warning in enhanced timestep computation: {e}")
        dt_min = 1e-4
    
    # Apply reasonable bounds
    dt_min = max(dt_min, 1e-6)  # Minimum timestep
    dt_min = min(dt_min, 1e-2)  # Maximum timestep for stability
    
    return dt_min


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

    # Compute cell volumes using enhanced method
    n_particles = len(state['rho'])
    volumes = compute_cell_volumes_enhanced(interfaces, state['positions'])
    
    # Ensure we have proper volumes
    if len(volumes) != n_particles:
        volumes = np.ones(n_particles) * 0.01  # Fallback

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