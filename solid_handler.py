"""
solid_handler.py: Handle solid boundaries and fluid-solid coupling
"""

import numpy as np
import numba


def update_solid_positions(solid_segments, dt):
    """
    Update positions of moving solid boundaries

    Parameters:
    -----------
    solid_segments : list
        List of solid segment dictionaries
    dt : float
        Timestep

    Returns:
    --------
    updated_segments : list
        Updated solid segments
    """
    updated_segments = []

    for segment in solid_segments:
        # Copy segment
        new_segment = segment.copy()

        # Update positions based on velocity
        velocity = segment['velocity']
        new_segment['start'] = segment['start'] + dt * velocity
        new_segment['end'] = segment['end'] + dt * velocity

        # Handle deformable solids if needed
        if segment.get('is_deformable', False):
            # Placeholder for deformable solid physics
            # Could implement spring-mass system or FEM here
            pass

        updated_segments.append(new_segment)

    return updated_segments


def compute_fluid_solid_coupling(interfaces, state, solid_segments):
    """
    Compute forces between fluid and solid

    Parameters:
    -----------
    interfaces : list
        Interface information including solid boundaries
    state : dict
        Fluid state
    solid_segments : list
        Solid boundaries

    Returns:
    --------
    forces : dict
        Forces on each solid segment
    """
    forces = {}

    rho = state['rho']
    rho_u = state['rho_u']
    rho_v = state['rho_v']
    rho_e = state['rho_e']
    gamma = state['gamma']

    # Compute pressure at each particle
    pressures = []
    for i in range(len(rho)):
        u = rho_u[i] / rho[i]
        v = rho_v[i] / rho[i]
        e_kinetic = 0.5 * (u ** 2 + v ** 2)
        e_internal = rho_e[i] / rho[i] - e_kinetic
        p = (gamma - 1) * rho[i] * e_internal
        pressures.append(p)

    # Accumulate forces on solid segments
    for idx, segment in enumerate(solid_segments):
        total_force = np.zeros(2)

        # Find interfaces touching this solid
        for interface in interfaces:
            if interface.get('is_solid', False):
                # Check if this interface belongs to current solid segment
                # (In practice, would need better bookkeeping)

                i = interface['cell_i']
                if i >= 0 and i < len(pressures):
                    # Pressure force: F = p * A * n
                    pressure = pressures[i]
                    area = interface['area']
                    normal = interface['normal']

                    force = pressure * area * normal
                    total_force += force

        forces[idx] = total_force

    return forces


def apply_solid_forces(solid_segments, forces, dt):
    """
    Apply computed forces to solid segments

    Parameters:
    -----------
    solid_segments : list
        Solid boundaries
    forces : dict
        Forces on each segment
    dt : float
        Timestep

    Returns:
    --------
    updated_segments : list
        Segments with updated velocities
    """
    updated_segments = []

    for idx, segment in enumerate(solid_segments):
        new_segment = segment.copy()

        if idx in forces and segment.get('mass', None) is not None:
            # F = ma -> a = F/m
            mass = segment['mass']
            acceleration = forces[idx] / mass

            # Update velocity
            new_segment['velocity'] = segment['velocity'] + dt * acceleration

        updated_segments.append(new_segment)

    return updated_segments


def create_balloon_geometry(center, radius, n_segments=32):
    """
    Create a closed balloon geometry

    Parameters:
    -----------
    center : array
        Center position
    radius : float
        Balloon radius
    n_segments : int
        Number of segments to approximate circle

    Returns:
    --------
    segments : list
        List of solid segments forming balloon
    """
    segments = []

    angles = np.linspace(0, 2 * np.pi, n_segments + 1)

    for i in range(n_segments):
        start = center + radius * np.array([np.cos(angles[i]), np.sin(angles[i])])
        end = center + radius * np.array([np.cos(angles[i + 1]), np.sin(angles[i + 1])])

        segments.append({
            'start': start,
            'end': end,
            'velocity': np.zeros(2),
            'is_deformable': True,
            'rest_length': np.linalg.norm(end - start),
            'stiffness': 100.0,
            'mass': 0.1 / n_segments  # Total mass distributed
        })

    return segments


def update_balloon_physics(segments, dt):
    """
    Update balloon using spring model

    Parameters:
    -----------
    segments : list
        Balloon segments
    dt : float
        Timestep

    Returns:
    --------
    updated_segments : list
        Updated balloon geometry
    """
    n = len(segments)
    forces = [np.zeros(2) for _ in range(n)]

    # Spring forces between consecutive segments
    for i in range(n):
        j = (i + 1) % n

        # Current length
        current_start = segments[i]['end']
        current_end = segments[j]['start']
        vec = current_end - current_start
        length = np.linalg.norm(vec)

        if length > 1e-10:
            # Spring force
            rest_length = segments[i]['rest_length']
            stiffness = segments[i]['stiffness']
            force_mag = stiffness * (length - rest_length)
            force_dir = vec / length

            # Apply to both endpoints
            forces[i] -= force_mag * force_dir * 0.5
            forces[j] += force_mag * force_dir * 0.5

    # Update positions
    updated_segments = []
    for i, segment in enumerate(segments):
        new_segment = segment.copy()

        if segment.get('is_deformable', False):
            # Simple integration
            mass = segment.get('mass', 0.1)
            acceleration = forces[i] / mass

            # Update velocity and position
            new_velocity = segment['velocity'] + dt * acceleration
            displacement = dt * new_velocity

            new_segment['start'] = segment['start'] + displacement
            new_segment['end'] = segment['end'] + displacement
            new_segment['velocity'] = new_velocity

        updated_segments.append(new_segment)

    return updated_segments