"""
TopoFluid2D - Step 3: Interface Flux Computation
Educational implementation of topology-preserving compressible fluid simulation.

Based on: "Topology-Preserving Coupling of Compressible Fluids and Thin Deformables"

This step implements:
- Godunov-type finite volume flux computation (Equation 5)
- Kurganov-Tadmor numerical flux (Equation 6)
- Interface Riemann problems between neighboring cells
- Time step computation with CFL condition
- Conservative state updates
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist

class FluidParticle:
    """
    Lagrangian fluid particle with full finite-volume capabilities.
    """
    def __init__(self, x, y, density=1.0, velocity_x=0.0, velocity_y=0.0, pressure=1.0, gamma=1.4):
        # Lagrangian position (moves with fluid)
        self.x = x
        self.y = y

        # Conservative state variables (Equation 2 from paper)
        self.density = density  # ρ
        self.momentum_x = density * velocity_x  # ρu_x
        self.momentum_y = density * velocity_y  # ρu_y

        # Compute total energy density: ρe_T = ρ(e + 1/2||u||²)
        kinetic_energy = 0.5 * (velocity_x**2 + velocity_y**2)
        internal_energy = pressure / ((gamma - 1) * density)
        total_energy_per_mass = internal_energy + kinetic_energy  # e_T
        self.energy_total = density * total_energy_per_mass  # ρe_T

        # Voronoi cell properties
        self.volume = 0.0  # V_i in Equation 5
        self.cell_vertices = []
        self.neighbors = []
        self.interface_areas = {}  # A_ij for each neighbor j
        self.interface_normals = {}  # n_ij for each neighbor j
        self.interface_midpoints = {}  # Interface centers

        # Flux computation
        self.total_flux = np.zeros(4)  # [ρ, ρu_x, ρu_y, ρe_T] flux accumulator
        self.max_wave_speed = 0.0  # For CFL condition

        self.gamma = gamma
        self.particle_id = id(self)  # Unique identifier

    @property
    def velocity_x(self):
        return self.momentum_x / self.density if self.density > 0 else 0.0

    @property
    def velocity_y(self):
        return self.momentum_y / self.density if self.density > 0 else 0.0

    @property
    def speed(self):
        return np.sqrt(self.velocity_x**2 + self.velocity_y**2)

    @property
    def sound_speed(self):
        return np.sqrt(self.gamma * self.pressure / self.density)

    @property
    def pressure(self):
        kinetic_energy = 0.5 * (self.velocity_x**2 + self.velocity_y**2)
        total_energy_per_mass = self.energy_total / self.density if self.density > 0 else 0
        internal_energy = total_energy_per_mass - kinetic_energy
        return max(0.0, (self.gamma - 1) * self.density * internal_energy)

    @property
    def conservative_state(self):
        """Return conservative variables U = [ρ, ρu_x, ρu_y, ρe_T]"""
        return np.array([self.density, self.momentum_x, self.momentum_y, self.energy_total])

    def set_conservative_state(self, U):
        """Set state from conservative variables"""
        self.density = max(1e-10, U[0])  # Prevent negative density
        self.momentum_x = U[1]
        self.momentum_y = U[2]
        self.energy_total = max(1e-10, U[3])  # Prevent negative energy

    def reset_flux(self):
        """Reset flux accumulator for new time step"""
        self.total_flux = np.zeros(4)
        self.max_wave_speed = 0.0

    def __repr__(self):
        return f"FluidParticle(pos=({self.x:.2f},{self.y:.2f}), ρ={self.density:.3f}, P={self.pressure:.3f})"

class RiemannSolver:
    """
    Riemann solver for computing numerical fluxes between fluid states.
    Implements Kurganov-Tadmor method (Equation 6 from paper).
    """

    @staticmethod
    def euler_flux(U, normal, gamma=1.4):
        """
        Compute Euler flux F(U) · n for conservative state U
        Following Equation 3 from the paper
        """
        rho = U[0]
        rho_u = U[1]
        rho_v = U[2]
        rho_E = U[3]

        if rho <= 0:
            return np.zeros(4)

        u = rho_u / rho
        v = rho_v / rho

        # Compute pressure using ideal gas law
        kinetic_energy = 0.5 * (u**2 + v**2)
        total_energy_per_mass = rho_E / rho
        internal_energy = total_energy_per_mass - kinetic_energy
        P = max(0.0, (gamma - 1) * rho * internal_energy)

        # Normal velocity u_n = u · n
        u_n = u * normal[0] + v * normal[1]

        # Euler flux vector F · n (Equation 3)
        flux = np.array([
            rho * u_n,                           # Mass flux
            rho_u * u_n + P * normal[0],         # x-momentum flux
            rho_v * u_n + P * normal[1],         # y-momentum flux
            rho_E * u_n + P * u_n                # Energy flux
        ])

        return flux

    @staticmethod
    def max_wave_speed(U, normal, gamma=1.4):
        """
        Compute maximum wave speed for CFL condition
        Returns max|u_n ± c| where c is sound speed
        """
        rho = U[0]
        if rho <= 0:
            return 0.0

        u = U[1] / rho
        v = U[2] / rho

        # Compute pressure and sound speed
        kinetic_energy = 0.5 * (u**2 + v**2)
        total_energy_per_mass = U[3] / rho
        internal_energy = total_energy_per_mass - kinetic_energy
        P = max(0.0, (gamma - 1) * rho * internal_energy)
        c = np.sqrt(gamma * P / rho)

        # Normal velocity
        u_n = u * normal[0] + v * normal[1]

        # Maximum characteristic speed
        return abs(u_n) + c

    @staticmethod
    def kurganov_tadmor_flux(U_L, U_R, normal, gamma=1.4):
        """
        Kurganov-Tadmor numerical flux (Equation 6 from paper)

        Args:
            U_L: Left state [ρ, ρu_x, ρu_y, ρe_T]
            U_R: Right state [ρ, ρu_x, ρu_y, ρe_T]
            normal: Interface normal vector

        Returns:
            numerical_flux: F_ij for interface
            max_speed: Maximum wave speed for CFL
        """
        # Compute fluxes at left and right states
        F_L = RiemannSolver.euler_flux(U_L, normal, gamma)
        F_R = RiemannSolver.euler_flux(U_R, normal, gamma)

        # Compute maximum wave speeds
        a_L = RiemannSolver.max_wave_speed(U_L, normal, gamma)
        a_R = RiemannSolver.max_wave_speed(U_R, normal, gamma)
        a_max = max(a_L, a_R)

        # Kurganov-Tadmor flux (Equation 6)
        numerical_flux = 0.5 * (F_L + F_R) - 0.5 * a_max * (U_R - U_L)

        return numerical_flux, a_max

class VoronoiUtils:
    """Voronoi utilities (same as Step 2)"""

    @staticmethod
    def compute_voronoi_tessellation(particles, domain_bounds):
        if len(particles) < 3:
            raise ValueError("Need at least 3 particles for Voronoi tessellation")

        points = np.array([[p.x, p.y] for p in particles])

        xmin, xmax, ymin, ymax = domain_bounds
        margin = 0.1 * max(xmax - xmin, ymax - ymin)

        boundary_points = [
            [xmin - margin, ymin - margin], [xmax + margin, ymin - margin],
            [xmax + margin, ymax + margin], [xmin - margin, ymax + margin],
            [xmin - margin, (ymin + ymax) / 2], [xmax + margin, (ymin + ymax) / 2],
            [(xmin + xmax) / 2, ymin - margin], [(xmin + xmax) / 2, ymax + margin],
        ]

        all_points = np.vstack([points, boundary_points])
        vor = Voronoi(all_points)

        return vor

    @staticmethod
    def clip_cell_to_domain(vertices, domain_bounds):
        xmin, xmax, ymin, ymax = domain_bounds

        if len(vertices) == 0:
            return []

        vertices = np.array(vertices)

        for boundary in ['left', 'right', 'bottom', 'top']:
            if len(vertices) == 0:
                break

            clipped = []

            for i in range(len(vertices)):
                current = vertices[i]
                previous = vertices[i-1]

                if boundary == 'left':
                    curr_inside = current[0] >= xmin
                    prev_inside = previous[0] >= xmin
                    boundary_val = xmin
                    axis = 0
                elif boundary == 'right':
                    curr_inside = current[0] <= xmax
                    prev_inside = previous[0] <= xmax
                    boundary_val = xmax
                    axis = 0
                elif boundary == 'bottom':
                    curr_inside = current[1] >= ymin
                    prev_inside = previous[1] >= ymin
                    boundary_val = ymin
                    axis = 1
                else:  # top
                    curr_inside = current[1] <= ymax
                    prev_inside = previous[1] <= ymax
                    boundary_val = ymax
                    axis = 1

                if curr_inside:
                    if not prev_inside:
                        t = (boundary_val - previous[axis]) / (current[axis] - previous[axis])
                        intersection = previous + t * (current - previous)
                        clipped.append(intersection)
                    clipped.append(current)
                elif prev_inside:
                    t = (boundary_val - previous[axis]) / (current[axis] - previous[axis])
                    intersection = previous + t * (current - previous)
                    clipped.append(intersection)

            vertices = np.array(clipped) if clipped else np.array([])

        return vertices.tolist() if len(vertices) > 0 else []

    @staticmethod
    def compute_polygon_area(vertices):
        if len(vertices) < 3:
            return 0.0

        vertices = np.array(vertices)
        x = vertices[:, 0]
        y = vertices[:, 1]

        return 0.5 * abs(sum(x[i]*y[i+1] - x[i+1]*y[i] for i in range(-1, len(x)-1)))

    @staticmethod
    def compute_interface_properties(cell1_vertices, cell2_vertices):
        if len(cell1_vertices) < 3 or len(cell2_vertices) < 3:
            return 0.0, np.array([0.0, 0.0]), np.array([0.0, 0.0])

        cell1 = np.array(cell1_vertices)
        cell2 = np.array(cell2_vertices)

        min_dist = float('inf')
        best_edge1 = None
        best_edge2 = None

        for i in range(len(cell1)):
            edge1_start = cell1[i]
            edge1_end = cell1[(i+1) % len(cell1)]
            edge1_mid = (edge1_start + edge1_end) / 2

            for j in range(len(cell2)):
                edge2_start = cell2[j]
                edge2_end = cell2[(j+1) % len(cell2)]
                edge2_mid = (edge2_start + edge2_end) / 2

                dist = np.linalg.norm(edge1_mid - edge2_mid)
                if dist < min_dist:
                    min_dist = dist
                    best_edge1 = (edge1_start, edge1_end)
                    best_edge2 = (edge2_start, edge2_end)

        if best_edge1 is None:
            return 0.0, np.array([0.0, 0.0]), np.array([0.0, 0.0])

        edge_vector = best_edge1[1] - best_edge1[0]
        interface_length = np.linalg.norm(edge_vector)

        if interface_length > 0:
            tangent = edge_vector / interface_length
            normal = np.array([-tangent[1], tangent[0]])
        else:
            normal = np.array([1.0, 0.0])

        midpoint = (best_edge1[0] + best_edge1[1]) / 2

        return interface_length, normal, midpoint

class TopoFluid2D:
    """
    Main simulation class with full flux computation capability.
    """
    def __init__(self, domain_size=(2.0, 2.0), cfl_factor=0.3):
        self.domain_width, self.domain_height = domain_size
        self.domain_bounds = (0.0, domain_size[0], 0.0, domain_size[1])
        self.particles = []
        self.voronoi = None
        self.time = 0.0
        self.dt = 0.01
        self.cfl_factor = cfl_factor  # CFL safety factor

    def add_particle(self, x, y, **kwargs):
        particle = FluidParticle(x, y, **kwargs)
        self.particles.append(particle)
        return particle

    def create_shock_tube(self, separation_x=1.0):
        """Create classical Sod shock tube setup"""
        self.particles.clear()

        rho_L, u_L, P_L = 1.0, 0.0, 1.0
        rho_R, u_R, P_R = 0.125, 0.0, 0.1

        nx = 40
        margin = 0.1
        dx = (self.domain_width - 2*margin) / nx

        for i in range(nx):
            x = margin + i * dx
            y = self.domain_height / 2

            if x < separation_x:
                self.add_particle(x, y, density=rho_L, pressure=P_L, velocity_x=u_L)
            else:
                self.add_particle(x, y, density=rho_R, pressure=P_R, velocity_x=u_R)

        self._shock_separation = separation_x
        print(f"Created Sod shock tube with {len(self.particles)} particles")

    def compute_voronoi_tessellation(self):
        """Compute Voronoi tessellation and cell properties"""
        if len(self.particles) < 3:
            return

        self.voronoi = VoronoiUtils.compute_voronoi_tessellation(self.particles, self.domain_bounds)
        n_fluid_particles = len(self.particles)

        for i, particle in enumerate(self.particles):
            region_index = self.voronoi.point_region[i]
            vertex_indices = self.voronoi.regions[region_index]

            if -1 in vertex_indices or len(vertex_indices) == 0:
                particle.cell_vertices = []
                particle.volume = 0.0
                continue

            vertices = [self.voronoi.vertices[vi] for vi in vertex_indices if vi >= 0]
            clipped_vertices = VoronoiUtils.clip_cell_to_domain(vertices, self.domain_bounds)

            particle.cell_vertices = clipped_vertices
            particle.volume = VoronoiUtils.compute_polygon_area(clipped_vertices)

            # Find neighbors and compute interface properties
            particle.neighbors = []
            particle.interface_areas = {}
            particle.interface_normals = {}
            particle.interface_midpoints = {}

            for ridge_points, ridge_vertices in zip(self.voronoi.ridge_points, self.voronoi.ridge_vertices):
                if i in ridge_points and -1 not in ridge_vertices:
                    neighbor_idx = ridge_points[0] if ridge_points[1] == i else ridge_points[1]

                    if neighbor_idx < n_fluid_particles:
                        neighbor = self.particles[neighbor_idx]
                        particle.neighbors.append(neighbor)

                        area, normal, midpoint = VoronoiUtils.compute_interface_properties(
                            particle.cell_vertices, neighbor.cell_vertices)

                        particle.interface_areas[neighbor.particle_id] = area
                        particle.interface_normals[neighbor.particle_id] = normal
                        particle.interface_midpoints[neighbor.particle_id] = midpoint

    def compute_fluxes(self):
        """
        Compute fluxes between all neighboring particles.
        Implements Equation 5 from the paper.
        """
        # Reset flux accumulators
        for particle in self.particles:
            particle.reset_flux()

        # Compute interface fluxes
        processed_pairs = set()

        for i, particle in enumerate(self.particles):
            for neighbor in particle.neighbors:
                j = self.particles.index(neighbor)

                # Avoid double-counting interfaces
                pair = tuple(sorted([i, j]))
                if pair in processed_pairs:
                    continue
                processed_pairs.add(pair)

                # Get interface properties
                neighbor_id = neighbor.particle_id
                if neighbor_id not in particle.interface_areas:
                    continue

                area = particle.interface_areas[neighbor_id]
                normal = particle.interface_normals[neighbor_id]

                if area <= 0:
                    continue

                # Get conservative states
                U_L = particle.conservative_state
                U_R = neighbor.conservative_state

                # Compute numerical flux
                flux, max_speed = RiemannSolver.kurganov_tadmor_flux(U_L, U_R, normal)

                # Apply flux to both particles (Newton's 3rd law)
                particle.total_flux += area * flux
                neighbor.total_flux -= area * flux  # Opposite direction

                # Track maximum wave speed for CFL
                particle.max_wave_speed = max(particle.max_wave_speed, max_speed)
                neighbor.max_wave_speed = max(neighbor.max_wave_speed, max_speed)

    def compute_time_step(self):
        """
        Compute stable time step using CFL condition.
        """
        if not self.particles:
            return self.dt

        min_dt = float('inf')

        for particle in self.particles:
            if particle.volume > 0 and particle.max_wave_speed > 0:
                # Estimate cell size as sqrt(volume)
                cell_size = np.sqrt(particle.volume)
                dt_local = self.cfl_factor * cell_size / particle.max_wave_speed
                min_dt = min(min_dt, dt_local)

        self.dt = min_dt if min_dt < float('inf') else 0.001
        return self.dt

    def update_particles(self, dt):
        """
        Update particle states using computed fluxes.
        Implements conservative finite volume update (Equation 5).
        """
        for particle in self.particles:
            if particle.volume > 0:
                # Conservative update: dU/dt = -1/V * sum(A_ij * F_ij)
                dU_dt = -particle.total_flux / particle.volume

                # Forward Euler time integration
                U_old = particle.conservative_state
                U_new = U_old + dt * dU_dt

                # Update particle state
                particle.set_conservative_state(U_new)

                # Move particle with fluid velocity (Lagrangian)
                particle.x += dt * particle.velocity_x
                particle.y += dt * particle.velocity_y

    def evolve_one_step(self):
        """
        Perform one complete time step of the simulation.
        """
        # 1. Compute Voronoi tessellation
        self.compute_voronoi_tessellation()

        # 2. Compute interface fluxes
        self.compute_fluxes()

        # 3. Determine stable time step
        dt = self.compute_time_step()

        # 4. Update particle states
        self.update_particles(dt)

        # 5. Update simulation time
        self.time += dt

        return dt

    def visualize_fluxes(self, figsize=(12, 8)):
        """
        Visualize flux computation results.
        """
        if not self.particles:
            return

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Extract data
        positions = np.array([[p.x, p.y] for p in self.particles])
        densities = np.array([p.density for p in self.particles])
        pressures = np.array([p.pressure for p in self.particles])
        velocities_x = np.array([p.velocity_x for p in self.particles])
        speeds = np.array([p.speed for p in self.particles])
        volumes = np.array([p.volume for p in self.particles])

        # Plot 1: Pressure field
        ax1 = axes[0,0]
        scatter1 = ax1.scatter(positions[:, 0], positions[:, 1], c=pressures,
                             s=50, cmap='viridis', alpha=0.8)

        # Draw flux arrows
        for particle in self.particles:
            if abs(particle.total_flux[0]) > 1e-10:  # Mass flux threshold
                flux_magnitude = np.linalg.norm(particle.total_flux[:2])
                if flux_magnitude > 0:
                    arrow_scale = 0.1 * flux_magnitude / max(abs(particle.total_flux).max(), 1e-10)
                    ax1.arrow(particle.x, particle.y,
                             arrow_scale * particle.total_flux[1] / particle.density,
                             arrow_scale * particle.total_flux[2] / particle.density,
                             head_width=0.02, head_length=0.02, fc='red', ec='red', alpha=0.7)

        ax1.set_title('Pressure & Flux Vectors')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        plt.colorbar(scatter1, ax=ax1, label='Pressure')

        # Plot 2: Density field
        ax2 = axes[0,1]
        scatter2 = ax2.scatter(positions[:, 0], positions[:, 1], c=densities,
                             s=50, cmap='plasma', alpha=0.8)
        ax2.set_title('Density Field')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        plt.colorbar(scatter2, ax=ax2, label='Density')

        # Plot 3: Velocity field
        ax3 = axes[1,0]
        ax3.quiver(positions[:, 0], positions[:, 1], velocities_x,
                  np.zeros_like(velocities_x), speeds, cmap='coolwarm', alpha=0.8)
        ax3.set_title('Velocity Field')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')

        # Plot 4: Cell volumes
        ax4 = axes[1,1]
        scatter4 = ax4.scatter(positions[:, 0], positions[:, 1], c=volumes,
                             s=50, cmap='viridis', alpha=0.8)
        ax4.set_title('Cell Volumes')
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        plt.colorbar(scatter4, ax=ax4, label='Volume')

        for ax in axes.flat:
            ax.set_xlim(self.domain_bounds[0], self.domain_bounds[1])
            ax.set_ylim(self.domain_bounds[2], self.domain_bounds[3])
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.suptitle(f'Flux Computation Results (t={self.time:.4f})', y=1.02)
        plt.show()

        # Print flux statistics
        total_mass_flux = sum(abs(p.total_flux[0]) for p in self.particles)
        max_speed = max(p.max_wave_speed for p in self.particles) if self.particles else 0

        print(f"\n=== Flux Computation Statistics ===")
        print(f"Simulation time: {self.time:.4f}")
        print(f"Time step: {self.dt:.6f}")
        print(f"Total mass flux: {total_mass_flux:.6f}")
        print(f"Max wave speed: {max_speed:.4f}")
        print(f"CFL number: {max_speed * self.dt / np.sqrt(np.mean(volumes)):.4f}")

def main():
    """
    Test Step 3: Interface flux computation
    """
    print("=== TopoFluid2D Step 3: Interface Flux Computation ===")

    # Create simulation
    sim = TopoFluid2D(domain_size=(2.0, 1.0), cfl_factor=0.3)

    print("\nTest: Shock tube flux computation")
    sim.create_shock_tube()

    print("Initial state:")
    sim.visualize_fluxes()

    print("\nPerforming one time step...")
    dt = sim.evolve_one_step()
    print(f"Computed time step: {dt:.6f}")

    print("After one time step:")
    sim.visualize_fluxes()

    print("\nPerforming multiple time steps...")
    for step in range(5):
        dt = sim.evolve_one_step()
        print(f"Step {step+1}: dt={dt:.6f}, t={sim.time:.4f}")

    print("After multiple time steps:")
    sim.visualize_fluxes()

if __name__ == "__main__":
    main()