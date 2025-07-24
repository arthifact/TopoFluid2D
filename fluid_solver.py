"""
fluid_solver.py: Compressible Euler equations solver with Godunov-type schemes
"""

import numpy as np
import numba
from numba import njit, prange


@njit
def compute_primitive_variables(conservative_state, gamma=1.4):
    """
    Convert conservative variables to primitive variables

    Parameters:
    -----------
    conservative_state : tuple
        (rho, rho_u, rho_v, rho_e)
    gamma : float
        Adiabatic index

    Returns:
    --------
    primitive_state : tuple
        (rho, u, v, p)
    """
    rho, rho_u, rho_v, rho_e = conservative_state

    u = rho_u / rho
    v = rho_v / rho

    # Total energy = kinetic + internal
    e_kinetic = 0.5 * (u * u + v * v)
    e_internal = rho_e / rho - e_kinetic

    # Pressure from ideal gas law
    p = (gamma - 1) * rho * e_internal

    return rho, u, v, p


@njit
def compute_conservative_flux(rho, u, v, p, nx, ny, gamma=1.4):
    """
    Compute flux vector F·n for Euler equations

    Parameters:
    -----------
    rho, u, v, p : float
        Primitive variables
    nx, ny : float
        Normal vector components
    gamma : float
        Adiabatic index

    Returns:
    --------
    flux : array
        Conservative flux vector
    """
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
    """Compute sound speed c = sqrt(gamma * p / rho)"""
    return np.sqrt(gamma * p / rho)


@njit
def compute_signal_velocity(state_left, state_right, normal, gamma=1.4):
    """
    Compute signal velocity for Kurganov-Tadmor scheme

    Parameters:
    -----------
    state_left, state_right : tuple
        Conservative states (rho, rho_u, rho_v, rho_e)
    normal : array
        Normal vector (nx, ny)

    Returns:
    --------
    a_ij : float
        Maximum signal velocity
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

    return lambda_max


@njit
def kurganov_tadmor_flux(state_left, state_right, normal, gamma=1.4):
    """
    Kurganov-Tadmor numerical flux (Equation 6 from paper)

    Parameters:
    -----------
    state_left, state_right : tuple
        Conservative states
    normal : array
        Interface normal vector

    Returns:
    --------
    flux : array
        Numerical flux at interface
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
    Compute numerical fluxes at all interfaces

    Parameters:
    -----------
    interfaces : list
        List of interface dictionaries
    state : dict
        Fluid state variables

    Returns:
    --------
    fluxes : list
        List of flux dictionaries
    """
    fluxes = []

    rho = state['rho']
    rho_u = state['rho_u']
    rho_v = state['rho_v']
    rho_e = state['rho_e']
    gamma = state['gamma']

    for interface in interfaces:
        i = interface['cell_i']
        j = interface['cell_j']

        if j == -1:  # Solid boundary
            # Handle boundary condition in apply_boundary_conditions
            continue

        # Get states
        state_i = (rho[i], rho_u[i], rho_v[i], rho_e[i])
        state_j = (rho[j], rho_u[j], rho_v[j], rho_e[j])

        # Compute flux
        flux = kurganov_tadmor_flux(state_i, state_j, interface['normal'], gamma)

        fluxes.append({
            'cell_i': i,
            'cell_j': j,
            'flux': flux,
            'area': interface['area'],
            'normal': interface['normal']
        })

    return fluxes


def apply_boundary_conditions(interfaces, solid_segments, state):
    """
    Apply boundary conditions by creating reflected particles
    (Section 4.5 of the paper)

    Parameters:
    -----------
    interfaces : list
        Original interfaces
    solid_segments : list
        Solid boundary segments
    state : dict
        Fluid state

    Returns:
    --------
    bc_interfaces : list
        Interfaces with boundary conditions applied
    """
    bc_interfaces = interfaces.copy()

    rho = state['rho']
    rho_u = state['rho_u']
    rho_v = state['rho_v']
    rho_e = state['rho_e']
    gamma = state['gamma']

    # Add reflected states for solid boundaries
    for interface in interfaces:
        if interface['is_solid']:
            i = interface['cell_i']
            normal = interface['normal']
            solid_vel = interface['solid_velocity']

            # Get fluid state
            state_f = (rho[i], rho_u[i], rho_v[i], rho_e[i])
            rho_f, u_f, v_f, p_f = compute_primitive_variables(state_f, gamma)

            # Velocity in solid frame
            u_rel = u_f - solid_vel[0]
            v_rel = v_f - solid_vel[1]

            # Reflect velocity (Equation 8 from paper)
            dot = 2.0 * (u_rel * normal[0] + v_rel * normal[1])
            u_reflected = u_f - dot * normal[0]
            v_reflected = v_f - dot * normal[1]

            # Create reflected conservative state
            rho_reflected = rho_f  # Same density
            rho_u_reflected = rho_reflected * u_reflected
            rho_v_reflected = rho_reflected * v_reflected

            # Same pressure/energy
            e_total = p_f / ((gamma - 1) * rho_reflected) + \
                      0.5 * (u_reflected ** 2 + v_reflected ** 2)
            rho_e_reflected = rho_reflected * e_total

            state_reflected = (rho_reflected, rho_u_reflected,
                               rho_v_reflected, rho_e_reflected)

            # Compute flux between real and reflected states
            flux = kurganov_tadmor_flux(state_f, state_reflected, normal, gamma)

            bc_interfaces.append({
                'cell_i': i,
                'cell_j': -1,  # Boundary
                'flux': flux,
                'area': interface['area'],
                'normal': normal,
                'is_boundary': True
            })

    return bc_interfaces


def compute_timestep(state, interfaces, cfl=0.5):
    """
    Compute stable timestep using CFL condition

    Parameters:
    -----------
    state : dict
        Fluid state
    interfaces : list
        Interface information
    cfl : float
        CFL number

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

    dt_min = 1e10

    # Estimate cell sizes and maximum velocities
    for i in range(len(rho)):
        # Get primitive variables
        u = rho_u[i] / rho[i]
        v = rho_v[i] / rho[i]

        # Compute pressure and sound speed
        e_kinetic = 0.5 * (u ** 2 + v ** 2)
        e_internal = rho_e[i] / rho[i] - e_kinetic
        p = (gamma - 1) * rho[i] * e_internal
        c = np.sqrt(gamma * p / rho[i])

        # Maximum signal speed
        max_speed = np.sqrt(u ** 2 + v ** 2) + c

        # Estimate cell size (distance to nearest neighbor)
        distances = np.linalg.norm(positions - positions[i], axis=1)
        distances[i] = np.inf  # Exclude self
        min_dist = np.min(distances)

        # Local timestep constraint
        dt_local = cfl * min_dist / max_speed
        dt_min = min(dt_min, dt_local)

    return dt_min


def update_fluid_state(state, fluxes, interfaces, dt):
    """
    Update fluid state using finite volume method (Equation 5)

    Parameters:
    -----------
    state : dict
        Current fluid state
    fluxes : list
        Numerical fluxes at interfaces
    interfaces : list
        Interface information
    dt : float
        Timestep

    Returns:
    --------
    new_state : dict
        Updated fluid state
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

    # Compute cell volumes (areas in 2D)
    n_particles = len(state['rho'])
    volumes = compute_cell_volumes(interfaces, n_particles)

    # Initialize flux accumulator
    flux_sum = np.zeros((n_particles, 4))

    # Accumulate fluxes
    for flux_data in fluxes:
        i = flux_data['cell_i']
        j = flux_data['cell_j']
        flux = flux_data['flux']
        area = flux_data['area']

        # Add flux contribution
        flux_contribution = area * flux

        # Flux leaves cell i
        flux_sum[i] -= flux_contribution

        # Flux enters cell j (if not boundary)
        if j >= 0:
            flux_sum[j] += flux_contribution

    # Update conservative variables
    for i in range(n_particles):
        if volumes[i] > 1e-10:  # Avoid division by zero
            # Finite volume update (Equation 5)
            update = dt * flux_sum[i] / volumes[i]

            new_state['rho'][i] += update[0]
            new_state['rho_u'][i] += update[1]
            new_state['rho_v'][i] += update[2]
            new_state['rho_e'][i] += update[3]

            # Ensure positive density and pressure
            if new_state['rho'][i] < 1e-10:
                new_state['rho'][i] = 1e-10

    return new_state


def compute_cell_volumes(interfaces, n_particles):
    """
    Compute cell volumes from interface data

    Simple approximation: use interface areas to estimate volumes
    """
    volumes = np.ones(n_particles) * 0.1  # Default small volume

    # Count interfaces per cell
    interface_count = np.zeros(n_particles)
    total_area = np.zeros(n_particles)

    for interface in interfaces:
        i = interface['cell_i']
        if i >= 0:
            interface_count[i] += 1
            total_area[i] += interface['area']

    # Estimate volume from total interface area
    # For regular polygons: V ≈ (perimeter * apothem) / 2
    # Rough approximation: V ≈ perimeter² / (4π)
    for i in range(n_particles):
        if total_area[i] > 0:
            volumes[i] = total_area[i] ** 2 / (4 * np.pi)

    return volumes