"""
TopoFluid2D - Step 2: Voronoi Tessellation
Educational implementation of topology-preserving compressible fluid simulation.

Based on: "Topology-Preserving Coupling of Compressible Fluids and Thin Deformables"

This step implements:
- Voronoi tessellation around Lagrangian particles (following Springel 2010)
- Cell volume computation (V_i in Equation 5)
- Interface area computation (A_ij in Equation 5)
- Visualization of the finite-volume mesh
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
    Lagrangian fluid particle with Voronoi cell properties.
    Following the paper's approach where each particle induces a Voronoi cell.
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

        # Voronoi cell properties (will be computed)
        self.volume = 0.0  # V_i in Equation 5
        self.cell_vertices = []  # Vertices of Voronoi cell
        self.neighbors = []  # Neighboring particles
        self.interface_areas = {}  # A_ij for each neighbor j
        self.interface_normals = {}  # n_ij for each neighbor j

        self.gamma = gamma

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
        return (self.gamma - 1) * self.density * internal_energy

    def __repr__(self):
        return f"FluidParticle(pos=({self.x:.2f},{self.y:.2f}), ρ={self.density:.3f}, V={self.volume:.3f})"

class VoronoiUtils:
    """
    Utility functions for Voronoi tessellation and geometric computations.
    Following the finite-volume approach from the paper.
    """

    @staticmethod
    def compute_voronoi_tessellation(particles, domain_bounds):
        """
        Compute Voronoi tessellation for fluid particles with domain clipping.

        Args:
            particles: List of FluidParticle objects
            domain_bounds: (xmin, xmax, ymin, ymax)

        Returns:
            vor: scipy.spatial.Voronoi object
        """
        if len(particles) < 3:
            raise ValueError("Need at least 3 particles for Voronoi tessellation")

        # Extract particle positions
        points = np.array([[p.x, p.y] for p in particles])

        # Add boundary points to ensure proper clipping
        xmin, xmax, ymin, ymax = domain_bounds
        margin = 0.1 * max(xmax - xmin, ymax - ymin)

        boundary_points = [
            [xmin - margin, ymin - margin],  # Bottom-left
            [xmax + margin, ymin - margin],  # Bottom-right
            [xmax + margin, ymax + margin],  # Top-right
            [xmin - margin, ymax + margin],  # Top-left
            # Add more boundary points for better tessellation
            [xmin - margin, (ymin + ymax) / 2],  # Left-center
            [xmax + margin, (ymin + ymax) / 2],  # Right-center
            [(xmin + xmax) / 2, ymin - margin],  # Bottom-center
            [(xmin + xmax) / 2, ymax + margin],  # Top-center
        ]

        all_points = np.vstack([points, boundary_points])

        # Compute Voronoi tessellation
        vor = Voronoi(all_points)

        return vor

    @staticmethod
    def clip_cell_to_domain(vertices, domain_bounds):
        """
        Clip a Voronoi cell to the domain boundaries.
        Using Sutherland-Hodgman clipping algorithm.
        """
        xmin, xmax, ymin, ymax = domain_bounds

        if len(vertices) == 0:
            return []

        # Convert to numpy array for easier manipulation
        vertices = np.array(vertices)

        # Clip against each domain boundary
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
                        # Entering the domain
                        t = (boundary_val - previous[axis]) / (current[axis] - previous[axis])
                        intersection = previous + t * (current - previous)
                        clipped.append(intersection)
                    clipped.append(current)
                elif prev_inside:
                    # Leaving the domain
                    t = (boundary_val - previous[axis]) / (current[axis] - previous[axis])
                    intersection = previous + t * (current - previous)
                    clipped.append(intersection)

            vertices = np.array(clipped) if clipped else np.array([])

        return vertices.tolist() if len(vertices) > 0 else []

    @staticmethod
    def compute_polygon_area(vertices):
        """Compute area of polygon using shoelace formula"""
        if len(vertices) < 3:
            return 0.0

        vertices = np.array(vertices)
        x = vertices[:, 0]
        y = vertices[:, 1]

        return 0.5 * abs(sum(x[i]*y[i+1] - x[i+1]*y[i] for i in range(-1, len(x)-1)))

    @staticmethod
    def compute_interface_properties(cell1_vertices, cell2_vertices):
        """
        Compute interface area and normal between two Voronoi cells.
        Returns (area, normal_vector, midpoint)
        """
        if len(cell1_vertices) < 3 or len(cell2_vertices) < 3:
            return 0.0, np.array([0.0, 0.0]), np.array([0.0, 0.0])

        # Find shared edge between cells (simplified approach)
        # In practice, this would use more sophisticated geometric algorithms
        cell1 = np.array(cell1_vertices)
        cell2 = np.array(cell2_vertices)

        # Find closest approach between cell edges
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

        # Compute interface properties
        edge_vector = best_edge1[1] - best_edge1[0]
        interface_length = np.linalg.norm(edge_vector)

        # Normal vector (perpendicular to edge, pointing from cell1 to cell2)
        if interface_length > 0:
            tangent = edge_vector / interface_length
            normal = np.array([-tangent[1], tangent[0]])  # Rotate 90 degrees
        else:
            normal = np.array([1.0, 0.0])

        midpoint = (best_edge1[0] + best_edge1[1]) / 2

        return interface_length, normal, midpoint

class TopoFluid2D:
    """
    Main simulation class with Voronoi tessellation capability.
    """
    def __init__(self, domain_size=(2.0, 2.0)):
        self.domain_width, self.domain_height = domain_size
        self.domain_bounds = (0.0, domain_size[0], 0.0, domain_size[1])
        self.particles = []
        self.voronoi = None
        self.time = 0.0
        self.dt = 0.01

    def add_particle(self, x, y, **kwargs):
        """Add a fluid particle at position (x, y)"""
        particle = FluidParticle(x, y, **kwargs)
        self.particles.append(particle)
        return particle

    def create_uniform_grid(self, nx, ny, density=1.0, pressure=1.0, velocity=(0.0, 0.0)):
        """Create a uniform grid of fluid particles"""
        self.particles.clear()

        # Create spacing slightly inside domain boundaries
        margin = 0.1
        dx = (self.domain_width - 2*margin) / (nx - 1) if nx > 1 else 0
        dy = (self.domain_height - 2*margin) / (ny - 1) if ny > 1 else 0

        for i in range(nx):
            for j in range(ny):
                x = margin + i * dx
                y = margin + j * dy
                self.add_particle(x, y, density=density, pressure=pressure,
                                velocity_x=velocity[0], velocity_y=velocity[1])

        print(f"Created {len(self.particles)} fluid particles in {nx}x{ny} grid")

    def create_shock_tube(self, separation_x=1.0):
        """Create classical Sod shock tube setup"""
        self.particles.clear()

        # Standard Sod shock tube parameters
        rho_L, u_L, P_L = 1.0, 0.0, 1.0      # Left state
        rho_R, u_R, P_R = 0.125, 0.0, 0.1    # Right state

        # Create particles along a horizontal line
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
        """
        Compute Voronoi tessellation and update particle cell properties.
        This implements the spatial discretization from the paper.
        """
        if len(self.particles) < 3:
            print("Need at least 3 particles for Voronoi tessellation")
            return

        # Compute Voronoi diagram
        self.voronoi = VoronoiUtils.compute_voronoi_tessellation(self.particles, self.domain_bounds)

        # Process each particle's Voronoi cell
        n_fluid_particles = len(self.particles)

        for i, particle in enumerate(self.particles):
            # Get vertices of this particle's Voronoi cell
            region_index = self.voronoi.point_region[i]
            vertex_indices = self.voronoi.regions[region_index]

            if -1 in vertex_indices or len(vertex_indices) == 0:
                # Unbounded cell - use domain clipping
                particle.cell_vertices = []
                particle.volume = 0.0
                continue

            # Get actual vertex coordinates
            vertices = [self.voronoi.vertices[vi] for vi in vertex_indices if vi >= 0]

            # Clip to domain
            clipped_vertices = VoronoiUtils.clip_cell_to_domain(vertices, self.domain_bounds)

            # Store cell properties
            particle.cell_vertices = clipped_vertices
            particle.volume = VoronoiUtils.compute_polygon_area(clipped_vertices)

            # Find neighbors and compute interface properties
            particle.neighbors = []
            particle.interface_areas = {}
            particle.interface_normals = {}

            # Find neighboring particles through Voronoi ridge structure
            for ridge_points, ridge_vertices in zip(self.voronoi.ridge_points, self.voronoi.ridge_vertices):
                if i in ridge_points and -1 not in ridge_vertices:
                    # This ridge connects particle i to another particle
                    neighbor_idx = ridge_points[0] if ridge_points[1] == i else ridge_points[1]

                    if neighbor_idx < n_fluid_particles:  # Only fluid particles, not boundary
                        neighbor = self.particles[neighbor_idx]
                        particle.neighbors.append(neighbor)

                        # Compute interface properties
                        area, normal, midpoint = VoronoiUtils.compute_interface_properties(
                            particle.cell_vertices, neighbor.cell_vertices)

                        particle.interface_areas[neighbor_idx] = area
                        particle.interface_normals[neighbor_idx] = normal

        print(f"Computed Voronoi tessellation for {len(self.particles)} particles")
        print(f"Average cell volume: {np.mean([p.volume for p in self.particles]):.4f}")
        print(f"Total domain coverage: {sum(p.volume for p in self.particles):.4f}")

    def visualize_voronoi(self, color_by='pressure', figsize=(12, 8)):
        """
        Visualize the Voronoi tessellation with fluid properties.
        """
        if self.voronoi is None:
            print("No Voronoi tessellation computed! Call compute_voronoi_tessellation() first.")
            return

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Extract particle properties for coloring
        positions = np.array([[p.x, p.y] for p in self.particles])

        if color_by == 'pressure':
            colors = [p.pressure for p in self.particles]
            color_label = 'Pressure'
        elif color_by == 'density':
            colors = [p.density for p in self.particles]
            color_label = 'Density'
        elif color_by == 'volume':
            colors = [p.volume for p in self.particles]
            color_label = 'Cell Volume'
        else:
            colors = [p.pressure for p in self.particles]
            color_label = 'Pressure'

        # Plot 1: Voronoi diagram with colored cells
        ax1 = axes[0]

        # Plot Voronoi diagram structure
        voronoi_plot_2d(self.voronoi, ax=ax1, show_vertices=False, line_colors='gray', line_width=1)

        # Color each cell by fluid property
        for i, particle in enumerate(self.particles):
            if len(particle.cell_vertices) >= 3:
                poly = Polygon(particle.cell_vertices, alpha=0.6,
                             facecolor=plt.cm.viridis(plt.Normalize(vmin=min(colors), vmax=max(colors))(colors[i])))
                ax1.add_patch(poly)

        # Plot particle positions
        scatter = ax1.scatter(positions[:, 0], positions[:, 1],
                            c=colors, s=30, alpha=1.0, cmap='viridis',
                            edgecolors='black', linewidths=0.5, zorder=5)

        ax1.set_xlim(self.domain_bounds[0], self.domain_bounds[1])
        ax1.set_ylim(self.domain_bounds[2], self.domain_bounds[3])
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_title(f'Voronoi Tessellation (colored by {color_label})')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        plt.colorbar(scatter, ax=ax1, label=color_label)

        # Plot 2: Statistics and connectivity
        ax2 = axes[1]

        # Show particle connectivity
        for i, particle in enumerate(self.particles):
            for neighbor in particle.neighbors:
                j = self.particles.index(neighbor)
                ax2.plot([particle.x, neighbor.x], [particle.y, neighbor.y],
                        'gray', alpha=0.3, linewidth=0.5)

        # Plot particles with size proportional to cell volume
        volumes = np.array([p.volume for p in self.particles])
        if np.max(volumes) > 0:
            sizes = 50 + 200 * volumes / np.max(volumes)
        else:
            sizes = 50

        ax2.scatter(positions[:, 0], positions[:, 1],
                   s=sizes, alpha=0.7, c=colors, cmap='viridis',
                   edgecolors='black', linewidths=0.5)

        ax2.set_xlim(self.domain_bounds[0], self.domain_bounds[1])
        ax2.set_ylim(self.domain_bounds[2], self.domain_bounds[3])
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        ax2.set_title('Particle Connectivity (size ∝ cell volume)')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')

        plt.tight_layout()
        plt.show()

        # Print tessellation statistics
        volumes = [p.volume for p in self.particles]
        print(f"\n=== Voronoi Tessellation Statistics ===")
        print(f"Total particles: {len(self.particles)}")
        print(f"Domain area: {self.domain_width * self.domain_height:.4f}")
        print(f"Total cell volume: {sum(volumes):.4f}")
        print(f"Coverage efficiency: {sum(volumes)/(self.domain_width * self.domain_height)*100:.1f}%")
        print(f"Average cell volume: {np.mean(volumes):.4f}")
        print(f"Cell volume std: {np.std(volumes):.4f}")
        print(f"Average neighbors per particle: {np.mean([len(p.neighbors) for p in self.particles]):.1f}")

def main():
    """
    Test Step 2: Voronoi tessellation around fluid particles
    """
    print("=== TopoFluid2D Step 2: Voronoi Tessellation ===")

    # Create simulation
    sim = TopoFluid2D(domain_size=(2.0, 1.0))

    print("\nTest 1: Uniform grid Voronoi tessellation")
    sim.create_uniform_grid(6, 4, density=1.0, pressure=1.0)
    sim.compute_voronoi_tessellation()
    sim.visualize_voronoi(color_by='volume')

    print("\nTest 2: Shock tube Voronoi tessellation")
    sim.create_shock_tube()
    sim.compute_voronoi_tessellation()
    sim.visualize_voronoi(color_by='pressure')

    print("\nTest 3: Different particle arrangement")
    sim.create_uniform_grid(8, 3, density=0.8, pressure=1.5)
    sim.compute_voronoi_tessellation()
    sim.visualize_voronoi(color_by='density')

if __name__ == "__main__":
    main()