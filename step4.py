"""
TopoFluid2D - Step 4: Simple Solid Boundaries
Educational implementation of topology-preserving compressible fluid simulation.

Based on: "Topology-Preserving Coupling of Compressible Fluids and Thin Deformables"

This step implements:
- Simple solid representation as series of points
- Deformable balloon (elastic membrane)
- Rigid oval "bunny"
- Basic solid-fluid boundary setup
- Visualization of solid boundaries with fluid
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist

class SolidPoint:
    """
    Represents a point on a solid boundary.
    Can be part of rigid or deformable structures.
    """
    def __init__(self, x, y, solid_id=0, is_deformable=False):
        # Position
        self.x = x
        self.y = y
        self.x0 = x  # Initial position (for deformable reference)
        self.y0 = y

        # Solid properties
        self.solid_id = solid_id  # Which solid this point belongs to
        self.is_deformable = is_deformable

        # Velocity (for deformable solids)
        self.velocity_x = 0.0
        self.velocity_y = 0.0

        # Forces (from fluid pressure)
        self.force_x = 0.0
        self.force_y = 0.0

        # Connectivity
        self.neighbors = []  # Connected solid points
        self.normal = np.array([0.0, 0.0])  # Outward normal vector

    def __repr__(self):
        solid_type = "deformable" if self.is_deformable else "rigid"
        return f"SolidPoint(pos=({self.x:.2f},{self.y:.2f}), {solid_type}, id={self.solid_id})"

class FluidParticle:
    """
    Enhanced fluid particle with solid boundary awareness.
    Same as Step 3 but with solid interaction capabilities.
    """
    def __init__(self, x, y, density=1.0, velocity_x=0.0, velocity_y=0.0, pressure=1.0, gamma=1.4):
        # Lagrangian position
        self.x = x
        self.y = y

        # Conservative state variables
        self.density = density
        self.momentum_x = density * velocity_x
        self.momentum_y = density * velocity_y

        kinetic_energy = 0.5 * (velocity_x**2 + velocity_y**2)
        internal_energy = pressure / ((gamma - 1) * density)
        total_energy_per_mass = internal_energy + kinetic_energy
        self.energy_total = density * total_energy_per_mass

        # Voronoi cell properties
        self.volume = 0.0
        self.cell_vertices = []
        self.neighbors = []  # Other fluid particles
        self.interface_areas = {}
        self.interface_normals = {}

        # Solid interaction
        self.solid_neighbors = []  # Nearby solid points
        self.solid_interfaces = {}  # Solid boundary interfaces

        # Flux computation
        self.total_flux = np.zeros(4)
        self.max_wave_speed = 0.0

        self.gamma = gamma
        self.particle_id = id(self)

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
        return np.array([self.density, self.momentum_x, self.momentum_y, self.energy_total])

    def set_conservative_state(self, U):
        self.density = max(1e-10, U[0])
        self.momentum_x = U[1]
        self.momentum_y = U[2]
        self.energy_total = max(1e-10, U[3])

    def reset_flux(self):
        self.total_flux = np.zeros(4)
        self.max_wave_speed = 0.0

    def __repr__(self):
        return f"FluidParticle(pos=({self.x:.2f},{self.y:.2f}), ρ={self.density:.3f}, P={self.pressure:.3f})"

class SolidFactory:
    """
    Factory class for creating different types of solid boundaries.
    """

    @staticmethod
    def create_balloon(center_x, center_y, radius, n_points=20):
        """
        Create a deformable balloon as a circular arrangement of points.

        Args:
            center_x, center_y: Center position
            radius: Initial radius
            n_points: Number of boundary points

        Returns:
            List of SolidPoint objects forming a balloon
        """
        balloon_points = []
        solid_id = 1  # Balloon ID

        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)

            point = SolidPoint(x, y, solid_id=solid_id, is_deformable=True)
            balloon_points.append(point)

        # Set up connectivity (each point connected to neighbors)
        for i, point in enumerate(balloon_points):
            prev_idx = (i - 1) % n_points
            next_idx = (i + 1) % n_points
            point.neighbors = [balloon_points[prev_idx], balloon_points[next_idx]]

            # Compute outward normal (pointing away from center)
            to_center = np.array([center_x - point.x, center_y - point.y])
            if np.linalg.norm(to_center) > 0:
                point.normal = -to_center / np.linalg.norm(to_center)  # Outward
            else:
                point.normal = np.array([1.0, 0.0])

        print(f"Created deformable balloon with {len(balloon_points)} points at ({center_x:.1f}, {center_y:.1f})")
        return balloon_points

    @staticmethod
    def create_oval_bunny(center_x, center_y, width, height, n_points=16):
        """
        Create a rigid oval "bunny" as an elliptical arrangement of points.

        Args:
            center_x, center_y: Center position
            width, height: Ellipse dimensions
            n_points: Number of boundary points

        Returns:
            List of SolidPoint objects forming an oval
        """
        bunny_points = []
        solid_id = 2  # Bunny ID

        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            x = center_x + (width / 2) * np.cos(angle)
            y = center_y + (height / 2) * np.sin(angle)

            point = SolidPoint(x, y, solid_id=solid_id, is_deformable=False)
            bunny_points.append(point)

        # Set up connectivity
        for i, point in enumerate(bunny_points):
            prev_idx = (i - 1) % n_points
            next_idx = (i + 1) % n_points
            point.neighbors = [bunny_points[prev_idx], bunny_points[next_idx]]

            # Compute outward normal
            to_center = np.array([center_x - point.x, center_y - point.y])
            if np.linalg.norm(to_center) > 0:
                point.normal = -to_center / np.linalg.norm(to_center)
            else:
                point.normal = np.array([1.0, 0.0])

        print(f"Created rigid oval bunny with {len(bunny_points)} points at ({center_x:.1f}, {center_y:.1f})")
        return bunny_points

    @staticmethod
    def create_thin_barrier(start_x, start_y, end_x, end_y, n_points=10):
        """
        Create a thin rigid barrier between two points.
        Perfect for testing leakage prevention.
        """
        barrier_points = []
        solid_id = 3  # Barrier ID

        for i in range(n_points):
            t = i / (n_points - 1) if n_points > 1 else 0
            x = start_x + t * (end_x - start_x)
            y = start_y + t * (end_y - start_y)

            point = SolidPoint(x, y, solid_id=solid_id, is_deformable=False)
            barrier_points.append(point)

        # Set up connectivity
        for i, point in enumerate(barrier_points):
            neighbors = []
            if i > 0:
                neighbors.append(barrier_points[i-1])
            if i < len(barrier_points) - 1:
                neighbors.append(barrier_points[i+1])
            point.neighbors = neighbors

            # Compute normal (perpendicular to barrier direction)
            barrier_dir = np.array([end_x - start_x, end_y - start_y])
            if np.linalg.norm(barrier_dir) > 0:
                barrier_dir = barrier_dir / np.linalg.norm(barrier_dir)
                point.normal = np.array([-barrier_dir[1], barrier_dir[0]])  # Perpendicular
            else:
                point.normal = np.array([0.0, 1.0])

        print(f"Created thin barrier with {n_points} points from ({start_x:.1f},{start_y:.1f}) to ({end_x:.1f},{end_y:.1f})")
        return barrier_points

class TopoFluid2D:
    """
    Main simulation class with solid boundary support.
    """
    def __init__(self, domain_size=(2.0, 2.0)):
        self.domain_width, self.domain_height = domain_size
        self.domain_bounds = (0.0, domain_size[0], 0.0, domain_size[1])
        self.particles = []  # Fluid particles
        self.solid_points = []  # All solid boundary points
        self.time = 0.0
        self.dt = 0.01

    def add_particle(self, x, y, **kwargs):
        """Add a fluid particle"""
        particle = FluidParticle(x, y, **kwargs)
        self.particles.append(particle)
        return particle

    def add_solid_points(self, points):
        """Add solid boundary points to the simulation"""
        self.solid_points.extend(points)
        return points

    def create_fluid_around_solids(self, density=1.0, pressure=1.0, velocity=(0.0, 0.0)):
        """
        Create fluid particles around existing solid boundaries.
        Automatically avoids placing particles inside solids.
        """
        self.particles.clear()

        # Create a grid of potential particle positions
        nx, ny = 25, 15
        margin = 0.1
        dx = (self.domain_width - 2*margin) / (nx - 1)
        dy = (self.domain_height - 2*margin) / (ny - 1)

        for i in range(nx):
            for j in range(ny):
                x = margin + i * dx
                y = margin + j * dy

                # Check if this position is inside any solid
                inside_solid = False
                for solid_id in self.get_solid_ids():
                    if self.point_inside_solid(x, y, solid_id):
                        inside_solid = True
                        break

                if not inside_solid:
                    self.add_particle(x, y, density=density, pressure=pressure,
                                    velocity_x=velocity[0], velocity_y=velocity[1])

        print(f"Created {len(self.particles)} fluid particles around solids")

    def get_solid_ids(self):
        """Get list of unique solid IDs"""
        return list(set(point.solid_id for point in self.solid_points))

    def point_inside_solid(self, x, y, solid_id):
        """
        Check if point (x,y) is inside the solid with given ID.
        Uses ray casting algorithm for closed boundaries.
        """
        solid_boundary = [p for p in self.solid_points if p.solid_id == solid_id]

        if len(solid_boundary) < 3:
            return False  # Need at least 3 points for closed boundary

        # Ray casting algorithm
        inside = False
        j = len(solid_boundary) - 1

        for i in range(len(solid_boundary)):
            xi, yi = solid_boundary[i].x, solid_boundary[i].y
            xj, yj = solid_boundary[j].x, solid_boundary[j].y

            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i

        return inside

    def create_shock_tube_with_barrier(self):
        """
        Create shock tube setup with a thin barrier in the middle.
        Perfect for testing leakage prevention!
        """
        self.particles.clear()
        self.solid_points.clear()

        # Create thin barrier in the middle
        barrier_y = self.domain_height / 2
        barrier_points = SolidFactory.create_thin_barrier(
            1.0, barrier_y - 0.2, 1.0, barrier_y + 0.2, n_points=8
        )
        self.add_solid_points(barrier_points)

        # Create fluid particles on both sides
        rho_L, u_L, P_L = 1.0, 0.0, 1.0      # Left state (high pressure)
        rho_R, u_R, P_R = 0.125, 0.0, 0.1    # Right state (low pressure)

        nx = 40
        margin = 0.1
        dx = (self.domain_width - 2*margin) / nx

        for i in range(nx):
            x = margin + i * dx
            y = self.domain_height / 2

            # Skip particles near the barrier
            if abs(x - 1.0) > 0.05:  # 0.05 buffer around barrier
                if x < 1.0:
                    self.add_particle(x, y, density=rho_L, pressure=P_L, velocity_x=u_L)
                else:
                    self.add_particle(x, y, density=rho_R, pressure=P_R, velocity_x=u_R)

        print(f"Created shock tube with barrier: {len(self.particles)} fluid particles")

    def visualize_system(self, figsize=(12, 8)):
        """
        Visualize the complete fluid-solid system.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Extract fluid data
        if self.particles:
            fluid_positions = np.array([[p.x, p.y] for p in self.particles])
            pressures = np.array([p.pressure for p in self.particles])
            densities = np.array([p.density for p in self.particles])
        else:
            fluid_positions = np.array([]).reshape(0, 2)
            pressures = np.array([])
            densities = np.array([])

        # Plot 1: System overview with pressure
        ax1 = axes[0]

        # Plot fluid particles colored by pressure
        if len(self.particles) > 0:
            scatter = ax1.scatter(fluid_positions[:, 0], fluid_positions[:, 1],
                                c=pressures, s=30, cmap='viridis', alpha=0.8,
                                edgecolors='white', linewidths=0.5, label='Fluid')
            plt.colorbar(scatter, ax=ax1, label='Pressure')

        # Plot solid boundaries
        solid_colors = ['red', 'blue', 'green', 'orange', 'purple']
        solid_names = {1: 'Balloon', 2: 'Bunny', 3: 'Barrier', 4: 'Solid4', 5: 'Solid5'}

        for solid_id in self.get_solid_ids():
            solid_boundary = [p for p in self.solid_points if p.solid_id == solid_id]

            if solid_boundary:
                # Plot solid points
                x_coords = [p.x for p in solid_boundary]
                y_coords = [p.y for p in solid_boundary]

                color = solid_colors[(solid_id - 1) % len(solid_colors)]
                solid_name = solid_names.get(solid_id, f'Solid{solid_id}')

                # Check if it's a closed boundary (more than 2 points)
                if len(solid_boundary) > 2:
                    # Close the loop for visualization
                    x_coords.append(x_coords[0])
                    y_coords.append(y_coords[0])

                ax1.plot(x_coords, y_coords, color=color, linewidth=3,
                        marker='o', markersize=4, label=solid_name)

                # Show normals for solid points
                for point in solid_boundary[::2]:  # Every other point to avoid clutter
                    normal_scale = 0.05
                    ax1.arrow(point.x, point.y,
                            normal_scale * point.normal[0],
                            normal_scale * point.normal[1],
                            head_width=0.02, head_length=0.02,
                            fc=color, ec=color, alpha=0.7)

        ax1.set_xlim(self.domain_bounds[0], self.domain_bounds[1])
        ax1.set_ylim(self.domain_bounds[2], self.domain_bounds[3])
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_title('Fluid-Solid System Overview')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_aspect('equal')

        # Plot 2: Solid properties
        ax2 = axes[1]

        # Show solid point details
        for solid_id in self.get_solid_ids():
            solid_boundary = [p for p in self.solid_points if p.solid_id == solid_id]

            if solid_boundary:
                x_coords = [p.x for p in solid_boundary]
                y_coords = [p.y for p in solid_boundary]

                color = solid_colors[(solid_id - 1) % len(solid_colors)]

                # Distinguish deformable vs rigid
                deformable_points = [p for p in solid_boundary if p.is_deformable]
                rigid_points = [p for p in solid_boundary if not p.is_deformable]

                if deformable_points:
                    ax2.scatter([p.x for p in deformable_points],
                              [p.y for p in deformable_points],
                              c=color, s=60, marker='o', alpha=0.8,
                              label=f'Solid {solid_id} (Deformable)')

                if rigid_points:
                    ax2.scatter([p.x for p in rigid_points],
                              [p.y for p in rigid_points],
                              c=color, s=60, marker='s', alpha=0.8,
                              label=f'Solid {solid_id} (Rigid)')

                # Connect the points
                if len(solid_boundary) > 1:
                    ax2.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.5)

        ax2.set_xlim(self.domain_bounds[0], self.domain_bounds[1])
        ax2.set_ylim(self.domain_bounds[2], self.domain_bounds[3])
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        ax2.set_title('Solid Boundary Details')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_aspect('equal')

        plt.tight_layout()
        plt.show()

        # Print system statistics
        print(f"\n=== Fluid-Solid System Statistics ===")
        print(f"Fluid particles: {len(self.particles)}")
        print(f"Solid points: {len(self.solid_points)}")
        print(f"Number of solids: {len(self.get_solid_ids())}")

        for solid_id in self.get_solid_ids():
            solid_boundary = [p for p in self.solid_points if p.solid_id == solid_id]
            deformable_count = sum(1 for p in solid_boundary if p.is_deformable)
            rigid_count = len(solid_boundary) - deformable_count
            print(f"  Solid {solid_id}: {len(solid_boundary)} points ({deformable_count} deformable, {rigid_count} rigid)")

def main():
    """
    Test Step 4: Simple solid boundaries
    """
    print("=== TopoFluid2D Step 4: Simple Solid Boundaries ===")

    # Create simulation
    sim = TopoFluid2D(domain_size=(2.0, 1.5))

    print("\nTest 1: Balloon and bunny with surrounding fluid")

    # Create a deformable balloon
    balloon = SolidFactory.create_balloon(0.6, 0.7, 0.2, n_points=16)
    sim.add_solid_points(balloon)

    # Create a rigid oval bunny
    bunny = SolidFactory.create_oval_bunny(1.4, 0.8, 0.3, 0.4, n_points=12)
    sim.add_solid_points(bunny)

    # Create fluid around the solids
    sim.create_fluid_around_solids(density=1.0, pressure=1.0, velocity=(0.1, 0.0))

    # Visualize the system
    sim.visualize_system()

    print("\nTest 2: Shock tube with thin barrier (leakage test)")

    # Create new simulation for shock tube
    sim2 = TopoFluid2D(domain_size=(2.0, 1.0))
    sim2.create_shock_tube_with_barrier()
    sim2.visualize_system()

    print("\nTest 3: Multiple solid types")

    # Create simulation with various solids
    sim3 = TopoFluid2D(domain_size=(2.5, 2.0))

    # Add multiple solids
    balloon1 = SolidFactory.create_balloon(0.8, 0.6, 0.15, n_points=12)
    sim3.add_solid_points(balloon1)

    balloon2 = SolidFactory.create_balloon(1.8, 1.2, 0.18, n_points=14)
    sim3.add_solid_points(balloon2)

    bunny = SolidFactory.create_oval_bunny(1.2, 1.4, 0.25, 0.35, n_points=10)
    sim3.add_solid_points(bunny)

    barrier = SolidFactory.create_thin_barrier(0.3, 0.2, 2.2, 0.4, n_points=12)
    sim3.add_solid_points(barrier)

    # Create fluid
    sim3.create_fluid_around_solids(density=0.8, pressure=1.2, velocity=(0.05, 0.02))

    # Visualize
    sim3.visualize_system()

if __name__ == "__main__":
    main()