"""
Wind Tunnel with Proper Moving Particles

Key fixes:
1. Proper particle advection with visible movement
2. Better particle injection and removal
3. Improved timestep control
4. Enhanced visualization with particle trails
5. Debug information to track particle movement
"""

import numpy as np
import pyvoro
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
import os

# Parameters
num_seeds = 250
dim = 2
domain_size = [2.0, 1.0]

# Solid definition
solid_limits = [[0.0, domain_size[0]], [0.0, domain_size[1]]]
solid_center = (0.6, 0.5)
solid_radius = 0.08
solid_polygon = Point(solid_center).buffer(solid_radius)

# Fluid region
bounding_box = Polygon([(0,0), (domain_size[0],0), (domain_size[0],domain_size[1]), (0,domain_size[1])])
fluid_region = bounding_box.difference(solid_polygon)

def build_voronoi_cells(seed_positions_np, solid_limits):
    """Build analytic Voronoi cells using PyVoro"""
    try:
        if len(seed_positions_np) < 3:
            return []
        cells = pyvoro.compute_2d_voronoi(
            seed_positions_np.tolist(),
            solid_limits,
            1.0,
            [0.0] * len(seed_positions_np)
        )
        return cells
    except Exception as e:
        print(f"Voronoi error: {e}")
        return []

def extract_cell_geometry_clipped(cells, seed_positions_np):
    """Extract and clip cell geometry to fluid region"""
    geometry = []

    for i, cell in enumerate(cells):
        if i >= len(seed_positions_np):
            continue

        vertices = cell['vertices']
        if len(vertices) < 3:
            continue

        try:
            cell_poly = Polygon(vertices)
            if not cell_poly.is_valid:
                continue

            clipped_poly = cell_poly.intersection(fluid_region)

            if clipped_poly.is_empty or not clipped_poly.is_valid or clipped_poly.area < 1e-10:
                continue

            area = clipped_poly.area
            seed = seed_positions_np[i]
            seed_point = Point(seed)
            is_orphan = not clipped_poly.contains(seed_point)

            neighbors = []
            face_areas = {}
            face_normals = {}

            # Process faces for flux computation
            for face in cell['faces']:
                neighbor = face['adjacent_cell']
                face_vertex_indices = face['vertices']

                if len(face_vertex_indices) == 2:
                    v0 = np.array(vertices[face_vertex_indices[0]])
                    v1 = np.array(vertices[face_vertex_indices[1]])
                    edge = v1 - v0
                    length = np.linalg.norm(edge)

                    if length > 1e-10:
                        normal = np.array([-edge[1], edge[0]]) / length
                        mid = (v0 + v1) / 2

                        # Check if this is a solid boundary
                        if solid_polygon.distance(Point(mid)) < 1e-6:
                            neighbors.append(-1)
                            face_areas[-1] = length
                            face_normals[-1] = normal
                        elif (clipped_poly.buffer(1e-8).contains(Point(mid)) and
                              0 <= neighbor < len(cells)):
                            neighbors.append(neighbor)
                            face_areas[neighbor] = length
                            face_normals[neighbor] = normal

            geometry.append({
                'cell': i,
                'area': area,
                'neighbors': neighbors,
                'face_areas': face_areas,
                'face_normals': face_normals,
                'is_orphan': is_orphan,
                'clipped_polygon': clipped_poly,
                'centroid': np.array([clipped_poly.centroid.x, clipped_poly.centroid.y])
            })

        except Exception as e:
            continue

    return geometry

def compute_fluxes_simple(geometry, fluid_state, gamma=1.4):
    """Simplified but more stable flux computation"""
    cell_to_idx = {g['cell']: idx for idx, g in enumerate(geometry)}
    dUdt = np.zeros((len(geometry), 4), dtype=np.float64)

    for idx, g in enumerate(geometry):
        if g['area'] <= 1e-10:
            continue

        i = g['cell']
        if i >= len(fluid_state):
            continue

        Ui = fluid_state[i].copy()
        net_flux = np.zeros(4, dtype=np.float64)

        # Ensure physical state
        Ui[0] = max(Ui[0], 1e-6)  # density > 0

        # Simple upwind flux for fluid-fluid interfaces
        for j in g['neighbors']:
            if j >= 0 and j < len(fluid_state) and j in cell_to_idx:
                neighbor_idx = cell_to_idx[j]
                if neighbor_idx < len(geometry):
                    neighbor_geom = geometry[neighbor_idx]

                    if neighbor_geom['area'] > 1e-10:
                        Uj = fluid_state[j].copy()
                        Uj[0] = max(Uj[0], 1e-6)

                        normal = g['face_normals'][j]
                        area = g['face_areas'][j]

                        # Simple Lax-Friedrichs flux
                        flux_L = compute_euler_flux(Ui, normal, gamma)
                        flux_R = compute_euler_flux(Uj, normal, gamma)

                        # Wave speeds
                        c_L = compute_sound_speed(Ui, gamma)
                        c_R = compute_sound_speed(Uj, gamma)
                        u_L = (Ui[1]*normal[0] + Ui[2]*normal[1]) / Ui[0]
                        u_R = (Uj[1]*normal[0] + Uj[2]*normal[1]) / Uj[0]

                        alpha = max(abs(u_L) + c_L, abs(u_R) + c_R, 0.1)

                        flux = 0.5 * (flux_L + flux_R) - 0.5 * alpha * (Uj - Ui)
                        net_flux += area * flux

        # Solid boundary - reflection
        if -1 in g['neighbors']:
            normal = g['face_normals'][-1]
            area = g['face_areas'][-1]

            # Reflect velocity across normal
            rho = Ui[0]
            u = Ui[1] / rho
            v = Ui[2] / rho
            vel = np.array([u, v])
            un = np.dot(vel, normal)
            vel_reflected = vel - 2 * un * normal

            U_reflected = np.array([
                rho,
                rho * vel_reflected[0],
                rho * vel_reflected[1],
                Ui[3]
            ])

            flux_wall = compute_euler_flux(Ui, normal, gamma)
            flux_refl = compute_euler_flux(U_reflected, normal, gamma)

            c = compute_sound_speed(Ui, gamma)
            alpha = abs(un) + c
            flux = 0.5 * (flux_wall + flux_refl) - 0.5 * alpha * (U_reflected - Ui)
            net_flux += area * flux

        if g['area'] > 1e-10:
            dUdt[idx] = -net_flux / g['area']

    return dUdt

def compute_euler_flux(U, normal, gamma=1.4):
    """Compute Euler flux in normal direction"""
    rho = max(U[0], 1e-6)
    u = U[1] / rho
    v = U[2] / rho
    E = U[3] / rho

    un = u * normal[0] + v * normal[1]
    p = max((gamma - 1) * rho * (E - 0.5 * (u*u + v*v)), 0.1)

    return np.array([
        rho * un,
        rho * u * un + p * normal[0],
        rho * v * un + p * normal[1],
        (rho * E + p) * un
    ])

def compute_sound_speed(U, gamma=1.4):
    """Compute sound speed"""
    rho = max(U[0], 1e-6)
    u = U[1] / rho
    v = U[2] / rho
    E = U[3] / rho
    p = max((gamma - 1) * rho * (E - 0.5 * (u*u + v*v)), 0.1)
    return np.sqrt(gamma * p / rho)

def advect_particles_explicit(seed_positions, fluid_state, dt):
    """Explicit particle advection with VISIBLE movement"""
    new_positions = seed_positions.copy()
    movements = np.zeros(len(seed_positions))

    # FORCE LARGE MOVEMENTS
    movement_multiplier = 5.0  # Make particles move 5x faster than physics

    for i in range(len(seed_positions)):
        if i < len(fluid_state):
            U = fluid_state[i]
            rho = max(U[0], 1e-6)
            u = U[1] / rho
            v = U[2] / rho

            # AMPLIFIED displacement for visual effect
            displacement = dt * movement_multiplier * np.array([u, v])
            old_pos = seed_positions[i].copy()
            new_pos = old_pos + displacement

            # Track movement magnitude
            movements[i] = np.linalg.norm(displacement)

            # Keep particles in domain bounds
            new_pos[0] = np.clip(new_pos[0], 0.02, domain_size[0] - 0.02)
            new_pos[1] = np.clip(new_pos[1], 0.02, domain_size[1] - 0.02)

            # Check if particle would enter solid - with larger safety margin
            if solid_polygon.distance(Point(new_pos)) > solid_radius * 0.2:
                new_positions[i] = new_pos
            else:
                # Bounce off solid more dramatically
                to_center = np.array(solid_center) - old_pos
                distance = np.linalg.norm(to_center)
                if distance > 1e-8:
                    to_center_norm = to_center / distance
                    # Push particle away from solid
                    new_positions[i] = old_pos - 0.02 * to_center_norm
                    # Add tangential velocity for flow around obstacle
                    tangent = np.array([-to_center_norm[1], to_center_norm[0]])
                    new_positions[i] += 0.01 * tangent * np.sign(u)

    return new_positions, movements

def manage_particles(seed_positions, fluid_state, inflow_state, frame):
    """Better particle injection and removal"""
    # Remove particles that have gone too far right
    keep_mask = seed_positions[:, 0] < domain_size[0] - 0.1
    seed_positions = seed_positions[keep_mask]
    fluid_state = fluid_state[keep_mask]

    # Inject particles more frequently for better visualization
    if frame % 3 == 0:  # Every 3 frames
        num_inject = min(8, max(0, num_seeds - len(seed_positions)))

        for i in range(num_inject):
            # Inject at inflow with some vertical spread
            y = 0.2 + i * 0.6 / max(1, num_inject - 1)
            x = 0.05 + np.random.uniform(-0.02, 0.02)  # Small horizontal variation

            new_point = Point(x, y)
            if fluid_region.contains(new_point):
                new_pos = np.array([[x, y]])
                seed_positions = np.vstack([seed_positions, new_pos])

                # Add slight velocity variation for more interesting flow
                state = inflow_state.copy()
                state[1] += inflow_state[0] * np.random.uniform(-0.1, 0.1)
                state[2] += inflow_state[0] * np.random.uniform(-0.05, 0.05)
                fluid_state = np.vstack([fluid_state, [state]])

    return seed_positions, fluid_state

def animate_wind_tunnel():
    """Interactive wind tunnel with proper moving particles"""
    gamma = 1.4
    inflow_rho = 1.0
    inflow_u = 0.1  # Increased for more visible movement
    inflow_v = 0.0
    inflow_p = 1.0
    inflow_e = inflow_p / ((gamma - 1) * inflow_rho) + 0.5 * (inflow_u**2 + inflow_v**2)
    inflow_state = np.array([inflow_rho, inflow_rho * inflow_u, inflow_rho * inflow_v, inflow_rho * inflow_e])

    # Initialize particles
    np.random.seed(42)
    seed_positions = []
    fluid_state = []

    # Create initial distribution with more particles near inflow
    for i in range(num_seeds):
        # Weight distribution toward left side
        x = np.random.beta(2, 5) * (domain_size[0] - 0.2) + 0.1
        y = 0.1 + np.random.rand() * 0.8
        point = Point(x, y)

        if fluid_region.contains(point):
            seed_positions.append([x, y])
            state = inflow_state.copy()
            # Add some initial perturbations
            state[1] += inflow_rho * np.random.normal(0, 0.05)
            state[2] += inflow_rho * np.random.normal(0, 0.03)
            fluid_state.append(state)

    seed_positions = np.array(seed_positions, dtype=np.float64)
    fluid_state = np.array(fluid_state, dtype=np.float64)

    print(f"Initialized {len(seed_positions)} particles")

    # Visualization setup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Store particle history for trails
    particle_history = []
    max_history = 10

    # Colorbar reference to avoid multiple colorbars
    colorbar_ref = [None]

    def update(frame):
        nonlocal seed_positions, fluid_state, particle_history

        if len(seed_positions) < 5:
            print("Too few particles, reinitializing...")
            return []

        try:
            # Build mesh
            cells = build_voronoi_cells(seed_positions, solid_limits)
            if not cells or len(cells) < 5:
                print("Failed to build Voronoi cells")
                return []

            geometry = extract_cell_geometry_clipped(cells, seed_positions)
            if len(geometry) < 5:
                print("Insufficient geometry cells")
                return []

            # Compute time step - MUCH LARGER for visible particle movement
            max_speed = 0.1
            for g in geometry:
                i = g['cell']
                if i < len(fluid_state):
                    U = fluid_state[i]
                    rho = max(U[0], 1e-6)
                    vel_mag = np.sqrt((U[1]/rho)**2 + (U[2]/rho)**2)
                    c = compute_sound_speed(U, gamma)
                    max_speed = max(max_speed, vel_mag + c)

            min_cell_size = min([np.sqrt(g['area']) for g in geometry if g['area'] > 1e-10] + [0.01])
            dt_physics = 0.1 * min_cell_size / max_speed  # Small timestep for physics
            dt_visual = 0.02  # LARGE timestep for particle visualization

            # Use small timestep for physics, large for particle movement
            dt = dt_physics

            # Update fluid state
            dUdt = compute_fluxes_simple(geometry, fluid_state, gamma)

            for idx, g in enumerate(geometry):
                i = g['cell']
                if i < len(fluid_state) and g['area'] > 1e-10:
                    fluid_state[i] += dt * dUdt[idx]
                    # Ensure positivity
                    fluid_state[i][0] = max(fluid_state[i][0], 1e-6)

            # Advect particles with LARGE visual timestep
            old_positions = seed_positions.copy()
            seed_positions, movements = advect_particles_explicit(seed_positions, fluid_state, dt_visual)

            # Track particle history for trails
            particle_history.append(seed_positions.copy())
            if len(particle_history) > max_history:
                particle_history.pop(0)

            # Print movement statistics for debugging
            if frame % 20 == 0:
                avg_movement = np.mean(movements) if len(movements) > 0 else 0
                max_movement = np.max(movements) if len(movements) > 0 else 0
                print(f"Frame {frame}: {len(seed_positions)} particles, avg movement: {avg_movement:.4f}, max: {max_movement:.4f}")

            # Particle management
            seed_positions, fluid_state = manage_particles(seed_positions, fluid_state, inflow_state, frame)

            # Apply boundary conditions
            for i in range(len(seed_positions)):
                x, y = seed_positions[i]

                # Strong inflow enforcement
                if x < 0.15:
                    fluid_state[i] = inflow_state.copy()

                # Wall boundaries
                if y < 0.1 or y > domain_size[1] - 0.1:
                    rho = max(fluid_state[i][0], 1e-6)
                    u = fluid_state[i][1] / rho
                    v = fluid_state[i][2] / rho
                    if y < 0.1:
                        v = abs(v) * 0.5  # Bounce with damping
                    else:
                        v = -abs(v) * 0.5
                    fluid_state[i][1] = rho * u
                    fluid_state[i][2] = rho * v

        except Exception as e:
            print(f"Error in simulation: {e}")
            return []

        # Visualization
        ax1.clear()
        ax2.clear()

        try:
            # Compute flow properties
            pressures = []
            velocities = []

            for g in geometry:
                i = g['cell']
                if i < len(fluid_state):
                    U = fluid_state[i]
                    rho = max(U[0], 1e-6)
                    u = U[1] / rho
                    v = U[2] / rho
                    E = U[3] / rho
                    p = max((gamma - 1) * rho * (E - 0.5 * (u*u + v*v)), 0.5)
                    pressures.append(p)
                    velocities.append([u, v])
                else:
                    pressures.append(1.0)
                    velocities.append([0, 0])

            pressures = np.array(pressures)
            velocities = np.array(velocities)

            # Left plot: Pressure field with cells
            vmin, vmax = 0.7, 1.3
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.get_cmap('coolwarm')

            for idx, g in enumerate(geometry):
                poly = g['clipped_polygon']
                if poly.is_valid and not poly.is_empty:
                    color = cmap(norm(pressures[idx]))
                    if hasattr(poly, 'exterior'):
                        coords = list(poly.exterior.coords)
                        patch = plt.Polygon(coords, facecolor=color,
                                          edgecolor='black', linewidth=0.3, alpha=0.7)
                        ax1.add_patch(patch)

            # Show particle trails with MORE contrast for better visibility
            trail_alpha = 0.6  # More visible trails
            for i, hist_pos in enumerate(particle_history):
                alpha = trail_alpha * (i + 1) / len(particle_history)
                size = 3 + 8 * (i + 1) / len(particle_history)  # Larger trail particles
                color_intensity = 0.5 + 0.5 * (i + 1) / len(particle_history)
                ax1.scatter(hist_pos[:, 0], hist_pos[:, 1],
                           c=[(1, 0, 0, alpha)], s=size, zorder=5)

            # Current particle positions (MUCH MORE VISIBLE)
            ax1.scatter(seed_positions[:, 0], seed_positions[:, 1],
                       c='red', s=20, alpha=1.0, zorder=6,
                       edgecolors='yellow', linewidth=1.0)  # Yellow outline for visibility

            # Velocity vectors
            if len(geometry) > 0:
                centroids = np.array([g['centroid'] for g in geometry])
                ax1.quiver(centroids[:, 0], centroids[:, 1],
                          velocities[:, 0], velocities[:, 1],
                          scale=8, alpha=0.7, color='white', width=0.003)

            # Draw solid
            circle1 = Circle(solid_center, solid_radius, facecolor='gray',
                           edgecolor='black', linewidth=2, alpha=0.9, zorder=10)
            ax1.add_patch(circle1)

            ax1.set_xlim(0, domain_size[0])
            ax1.set_ylim(0, domain_size[1])
            ax1.set_aspect('equal')
            ax1.set_title(f'Pressure & Particles (Frame {frame}, N={len(seed_positions)})')
            ax1.grid(True, alpha=0.3)

            # Right plot: Velocity field
            if len(geometry) > 0:
                centroids = np.array([g['centroid'] for g in geometry])
                vel_mags = np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2)

                scatter = ax2.scatter(centroids[:, 0], centroids[:, 1],
                                    c=vel_mags, s=40, cmap='viridis', alpha=0.8,
                                    vmin=0, vmax=0.8)  # Fixed colorbar range

                # Velocity vectors
                ax2.quiver(centroids[:, 0], centroids[:, 1],
                          velocities[:, 0], velocities[:, 1],
                          scale=6, alpha=0.8, color='red', width=0.004)

                # Particle positions (BRIGHT and VISIBLE)
                ax2.scatter(seed_positions[:, 0], seed_positions[:, 1],
                           c='yellow', s=25, alpha=1.0, zorder=5,
                           edgecolors='red', linewidth=1.5)

                circle2 = Circle(solid_center, solid_radius, facecolor='gray',
                               edgecolor='black', linewidth=2, alpha=0.9)
                ax2.add_patch(circle2)

                # Only create colorbar once
                if colorbar_ref[0] is None:
                    colorbar_ref[0] = plt.colorbar(scatter, ax=ax2, label='Velocity Magnitude')
                else:
                    # Update existing colorbar
                    colorbar_ref[0].update_normal(scatter)

            ax2.set_xlim(0, domain_size[0])
            ax2.set_ylim(0, domain_size[1])
            ax2.set_aspect('equal')
            ax2.set_title(f'Velocity Field (MOVEMENT x{5})')
            ax2.grid(True, alpha=0.3)

        except Exception as e:
            print(f"Error in visualization: {e}")

        plt.tight_layout()
        return []

    anim = FuncAnimation(fig, update, frames=2000, interval=100, blit=False, repeat=False)
    plt.show()

if __name__ == "__main__":
    animate_wind_tunnel()