"""
TopoFluid2D - Step 5: Clipped Voronoi Stitching
Educational implementation of topology-preserving compressible fluid simulation.

Based on: "Topology-Preserving Coupling of Compressible Fluids and Thin Deformables"

This step implements:
- Algorithm 1: Clipped Voronoi Stitching (THE core contribution!)
- Solid boundary clipping of Voronoi cells
- Orphaned cell detection and stitching
- Topology-preserving mesh reconstruction
- Leakproof boundary enforcement
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon as ShapelyPolygon, Point, LineString
from shapely.ops import unary_union
import warnings

warnings.filterwarnings('ignore')


class SolidPoint:
    """Solid boundary point (same as Step 4)"""

    def __init__(self, x, y, solid_id=0, is_deformable=False):
        self.x = x
        self.y = y
        self.x0 = x
        self.y0 = y
        self.solid_id = solid_id
        self.is_deformable = is_deformable
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.force_x = 0.0
        self.force_y = 0.0
        self.neighbors = []
        self.normal = np.array([0.0, 0.0])

    def __repr__(self):
        solid_type = "deformable" if self.is_deformable else "rigid"
        return f"SolidPoint(pos=({self.x:.2f},{self.y:.2f}), {solid_type}, id={self.solid_id})"


class FluidParticle:
    """Enhanced fluid particle with clipped Voronoi support"""

    def __init__(self, x, y, density=1.0, velocity_x=0.0, velocity_y=0.0, pressure=1.0, gamma=1.4):
        # Lagrangian position
        self.x = x
        self.y = y

        # Conservative state variables
        self.density = density
        self.momentum_x = density * velocity_x
        self.momentum_y = density * velocity_y

        kinetic_energy = 0.5 * (velocity_x ** 2 + velocity_y ** 2)
        internal_energy = pressure / ((gamma - 1) * density)
        total_energy_per_mass = internal_energy + kinetic_energy
        self.energy_total = density * total_energy_per_mass

        # Voronoi cell properties
        self.volume = 0.0
        self.cell_vertices = []  # Original Voronoi cell
        self.clipped_vertices = []  # After solid clipping
        self.final_vertices = []  # After stitching

        # Stitching properties
        self.is_orphaned = False  # Does cell contain its generating point?
        self.owner_particle = None  # If orphaned, which particle owns this cell
        self.orphaned_cells = []  # If this particle owns orphaned cells

        # Interface properties
        self.neighbors = []
        self.interface_areas = {}
        self.interface_normals = {}
        self.solid_interfaces = {}  # Interfaces with solid boundaries

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
    def pressure(self):
        kinetic_energy = 0.5 * (self.velocity_x ** 2 + self.velocity_y ** 2)
        total_energy_per_mass = self.energy_total / self.density if self.density > 0 else 0
        internal_energy = total_energy_per_mass - kinetic_energy
        return max(0.0, (self.gamma - 1) * self.density * internal_energy)

    @property
    def effective_volume(self):
        """Volume including owned orphaned cells"""
        total_vol = self.volume
        for orphan in self.orphaned_cells:
            total_vol += orphan.volume
        return total_vol

    def __repr__(self):
        status = "orphaned" if self.is_orphaned else "normal"
        return f"FluidParticle(pos=({self.x:.2f},{self.y:.2f}), {status}, V={self.volume:.3f})"


class ClippedVoronoiStitcher:
    """
    Implementation of Algorithm 1 from the paper.
    The core topology-preserving algorithm!
    """

    @staticmethod
    def clip_cell_by_solids(cell_vertices, solid_boundaries):
        """
        Clip a Voronoi cell by all solid boundaries.

        Args:
            cell_vertices: List of (x,y) vertices of Voronoi cell
            solid_boundaries: List of solid boundary polygons

        Returns:
            clipped_vertices: Vertices after clipping by solids
        """
        if len(cell_vertices) < 3:
            return []

        try:
            # Create Shapely polygon from cell
            cell_poly = ShapelyPolygon(cell_vertices)

            if not cell_poly.is_valid:
                return []

            # Clip against each solid boundary
            for boundary_points in solid_boundaries:
                if len(boundary_points) < 3:
                    continue  # Skip non-closed boundaries

                try:
                    # Create solid boundary polygon
                    solid_poly = ShapelyPolygon([(p.x, p.y) for p in boundary_points])

                    if solid_poly.is_valid:
                        # Remove solid interior from cell
                        cell_poly = cell_poly.difference(solid_poly)

                        if cell_poly.is_empty:
                            return []

                except Exception:
                    continue  # Skip problematic boundaries

            # Convert back to vertex list
            if hasattr(cell_poly, 'exterior'):
                return list(cell_poly.exterior.coords[:-1])  # Remove duplicate last point
            elif hasattr(cell_poly, 'geoms'):
                # Handle multipolygon (take largest piece)
                largest_area = 0
                largest_poly = None
                for geom in cell_poly.geoms:
                    if hasattr(geom, 'exterior') and geom.area > largest_area:
                        largest_area = geom.area
                        largest_poly = geom

                if largest_poly:
                    return list(largest_poly.exterior.coords[:-1])

            return []

        except Exception:
            return []

    @staticmethod
    def point_in_polygon(point, vertices):
        """Check if point is inside polygon using ray casting"""
        if len(vertices) < 3:
            return False

        x, y = point
        inside = False
        j = len(vertices) - 1

        for i in range(len(vertices)):
            xi, yi = vertices[i]
            xj, yj = vertices[j]

            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i

        return inside

    @staticmethod
    def compute_polygon_area(vertices):
        """Compute polygon area using shoelace formula"""
        if len(vertices) < 3:
            return 0.0

        vertices = np.array(vertices)
        x = vertices[:, 0]
        y = vertices[:, 1]

        return 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(-1, len(x) - 1)))

    @staticmethod
    def compute_shared_edge_length(vertices1, vertices2):
        """
        Compute length of shared edge between two polygons.
        Simplified approach: find closest edges and estimate overlap.
        """
        if len(vertices1) < 3 or len(vertices2) < 3:
            return 0.0

        max_overlap = 0.0

        # Check each edge of polygon 1 against each edge of polygon 2
        for i in range(len(vertices1)):
            edge1_start = np.array(vertices1[i])
            edge1_end = np.array(vertices1[(i + 1) % len(vertices1)])
            edge1_vec = edge1_end - edge1_start
            edge1_len = np.linalg.norm(edge1_vec)

            if edge1_len < 1e-10:
                continue

            for j in range(len(vertices2)):
                edge2_start = np.array(vertices2[j])
                edge2_end = np.array(vertices2[(j + 1) % len(vertices2)])
                edge2_vec = edge2_end - edge2_start
                edge2_len = np.linalg.norm(edge2_vec)

                if edge2_len < 1e-10:
                    continue

                # Check if edges are approximately parallel and close
                if edge1_len > 0 and edge2_len > 0:
                    edge1_unit = edge1_vec / edge1_len
                    edge2_unit = edge2_vec / edge2_len

                    # Check parallelism (dot product close to ±1)
                    dot_product = abs(np.dot(edge1_unit, edge2_unit))

                    if dot_product > 0.8:  # Approximately parallel
                        # Check proximity
                        mid1 = (edge1_start + edge1_end) / 2
                        mid2 = (edge2_start + edge2_end) / 2
                        distance = np.linalg.norm(mid1 - mid2)

                        if distance < 0.1:  # Close enough
                            overlap = min(edge1_len, edge2_len) * (dot_product - 0.8) / 0.2
                            max_overlap = max(max_overlap, overlap)

        return max_overlap

    @staticmethod
    def stitch_orphaned_cells(particles, max_iterations=10):
        """
        Algorithm 1: Clipped Voronoi Stitching

        This is the core algorithm from the paper!
        """
        print(f"\n=== Algorithm 1: Clipped Voronoi Stitching ===")

        # Step 1: Identify orphaned cells
        orphaned_particles = []
        valid_particles = []

        for particle in particles:
            if particle.is_orphaned:
                orphaned_particles.append(particle)
            else:
                valid_particles.append(particle)

        print(f"Initial orphaned cells: {len(orphaned_particles)}")
        print(f"Initial valid cells: {len(valid_particles)}")

        iteration = 0

        # Step 2: Iterative stitching
        while orphaned_particles and iteration < max_iterations:
            iteration += 1
            print(f"\nStitching iteration {iteration}:")

            stitched_this_iteration = []

            for orphan in orphaned_particles[:]:  # Copy list to modify during iteration
                if len(orphan.clipped_vertices) < 3:
                    orphaned_particles.remove(orphan)
                    continue

                # Find best valid neighbor to stitch to
                best_neighbor = None
                best_interface_area = 0.0

                for valid_particle in valid_particles:
                    if len(valid_particle.clipped_vertices) < 3:
                        continue

                    # Compute shared interface area
                    interface_area = ClippedVoronoiStitcher.compute_shared_edge_length(
                        orphan.clipped_vertices, valid_particle.clipped_vertices)

                    if interface_area > best_interface_area:
                        best_interface_area = interface_area
                        best_neighbor = valid_particle

                # Stitch to best neighbor if found
                if best_neighbor and best_interface_area > 1e-6:
                    # Transfer ownership
                    orphan.owner_particle = best_neighbor
                    best_neighbor.orphaned_cells.append(orphan)

                    # Remove from orphaned list
                    orphaned_particles.remove(orphan)
                    stitched_this_iteration.append(orphan)

                    print(
                        f"  Stitched orphan at ({orphan.x:.2f},{orphan.y:.2f}) to valid particle at ({best_neighbor.x:.2f},{best_neighbor.y:.2f}), area={best_interface_area:.4f}")

            print(f"  Stitched {len(stitched_this_iteration)} cells this iteration")

            # Check for newly valid cells (orphans that are no longer orphaned)
            newly_valid = []
            for orphan in orphaned_particles[:]:
                if not orphan.is_orphaned:  # Status might have changed
                    orphaned_particles.remove(orphan)
                    valid_particles.append(orphan)
                    newly_valid.append(orphan)

            if newly_valid:
                print(f"  {len(newly_valid)} orphaned cells became valid")

            if not stitched_this_iteration and not newly_valid:
                print(f"  No progress made, stopping early")
                break

        print(f"\nStitching complete after {iteration} iterations")
        print(f"Remaining orphaned cells: {len(orphaned_particles)}")
        print(f"Final valid cells: {len(valid_particles)}")

        # Update final volumes including stitched cells
        for particle in valid_particles:
            total_volume = particle.volume
            for orphan in particle.orphaned_cells:
                total_volume += orphan.volume
            particle.final_volume = total_volume

        return len(orphaned_particles) == 0  # True if fully stitched


class SolidFactory:
    """Factory for creating solid boundaries (same as Step 4)"""

    @staticmethod
    def create_balloon(center_x, center_y, radius, n_points=20):
        balloon_points = []
        solid_id = 1

        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)

            point = SolidPoint(x, y, solid_id=solid_id, is_deformable=True)
            balloon_points.append(point)

        for i, point in enumerate(balloon_points):
            prev_idx = (i - 1) % n_points
            next_idx = (i + 1) % n_points
            point.neighbors = [balloon_points[prev_idx], balloon_points[next_idx]]

            to_center = np.array([center_x - point.x, center_y - point.y])
            if np.linalg.norm(to_center) > 0:
                point.normal = -to_center / np.linalg.norm(to_center)
            else:
                point.normal = np.array([1.0, 0.0])

        return balloon_points

    @staticmethod
    def create_thin_barrier(start_x, start_y, end_x, end_y, n_points=10):
        barrier_points = []
        solid_id = 3

        for i in range(n_points):
            t = i / (n_points - 1) if n_points > 1 else 0
            x = start_x + t * (end_x - start_x)
            y = start_y + t * (end_y - start_y)

            point = SolidPoint(x, y, solid_id=solid_id, is_deformable=False)
            barrier_points.append(point)

        for i, point in enumerate(barrier_points):
            neighbors = []
            if i > 0:
                neighbors.append(barrier_points[i - 1])
            if i < len(barrier_points) - 1:
                neighbors.append(barrier_points[i + 1])
            point.neighbors = neighbors

            barrier_dir = np.array([end_x - start_x, end_y - start_y])
            if np.linalg.norm(barrier_dir) > 0:
                barrier_dir = barrier_dir / np.linalg.norm(barrier_dir)
                point.normal = np.array([-barrier_dir[1], barrier_dir[0]])
            else:
                point.normal = np.array([0.0, 1.0])

        return barrier_points


class TopoFluid2D:
    """
    Main simulation class with clipped Voronoi stitching capability.
    """

    def __init__(self, domain_size=(2.0, 2.0)):
        self.domain_width, self.domain_height = domain_size
        self.domain_bounds = (0.0, domain_size[0], 0.0, domain_size[1])
        self.particles = []
        self.solid_points = []
        self.voronoi = None
        self.time = 0.0
        self.dt = 0.01

    def add_particle(self, x, y, **kwargs):
        particle = FluidParticle(x, y, **kwargs)
        self.particles.append(particle)
        return particle

    def add_solid_points(self, points):
        self.solid_points.extend(points)
        return points

    def get_solid_boundaries(self):
        """Get solid boundaries grouped by solid ID"""
        boundaries = {}
        for solid_id in set(p.solid_id for p in self.solid_points):
            boundaries[solid_id] = [p for p in self.solid_points if p.solid_id == solid_id]
        return boundaries

    def create_shock_tube_with_barrier(self):
        """Create shock tube with thin barrier for leakage testing"""
        self.particles.clear()
        self.solid_points.clear()

        # Create thin barrier
        barrier_points = SolidFactory.create_thin_barrier(
            1.0, 0.3, 1.0, 0.7, n_points=6
        )
        self.add_solid_points(barrier_points)

        # Create fluid particles
        rho_L, P_L = 1.0, 1.0
        rho_R, P_R = 0.125, 0.1

        # Left side (high pressure)
        for i in range(15):
            x = 0.2 + i * 0.05
            y = 0.5
            self.add_particle(x, y, density=rho_L, pressure=P_L)

        # Right side (low pressure)
        for i in range(15):
            x = 1.05 + i * 0.05
            y = 0.5
            self.add_particle(x, y, density=rho_R, pressure=P_R)

        print(
            f"Created shock tube with barrier: {len(self.particles)} particles, {len(self.solid_points)} solid points")

    def compute_clipped_voronoi_tessellation(self):
        """
        Compute Voronoi tessellation with solid boundary clipping.
        Implements the first part of Algorithm 1.
        """
        if len(self.particles) < 3:
            return

        print(f"\n=== Computing Clipped Voronoi Tessellation ===")

        # Step 1: Compute standard Voronoi tessellation
        points = np.array([[p.x, p.y] for p in self.particles])

        # Add boundary points for proper tessellation
        xmin, xmax, ymin, ymax = self.domain_bounds
        margin = 0.1

        boundary_points = [
            [xmin - margin, ymin - margin], [xmax + margin, ymin - margin],
            [xmax + margin, ymax + margin], [xmin - margin, ymax + margin],
        ]

        all_points = np.vstack([points, boundary_points])
        self.voronoi = Voronoi(all_points)

        # Step 2: Get solid boundaries
        solid_boundaries = list(self.get_solid_boundaries().values())

        # Step 3: Process each particle's cell
        for i, particle in enumerate(self.particles):
            # Get original Voronoi cell
            region_index = self.voronoi.point_region[i]
            vertex_indices = self.voronoi.regions[region_index]

            if -1 in vertex_indices or len(vertex_indices) == 0:
                particle.cell_vertices = []
                particle.clipped_vertices = []
                particle.volume = 0.0
                continue

            # Get cell vertices
            vertices = []
            for vi in vertex_indices:
                if vi >= 0:
                    vertices.append(self.voronoi.vertices[vi])

            # Clip to domain
            vertices = self.clip_to_domain(vertices)
            particle.cell_vertices = vertices

            # Step 4: Clip by solid boundaries
            clipped_vertices = ClippedVoronoiStitcher.clip_cell_by_solids(vertices, solid_boundaries)
            particle.clipped_vertices = clipped_vertices
            particle.volume = ClippedVoronoiStitcher.compute_polygon_area(clipped_vertices)

            # Step 5: Check if cell is orphaned
            if len(clipped_vertices) >= 3:
                particle.is_orphaned = not ClippedVoronoiStitcher.point_in_polygon(
                    (particle.x, particle.y), clipped_vertices)
            else:
                particle.is_orphaned = True

        # Count orphaned vs valid cells
        orphaned_count = sum(1 for p in self.particles if p.is_orphaned)
        valid_count = len(self.particles) - orphaned_count

        print(f"Voronoi tessellation complete:")
        print(f"  Valid cells: {valid_count}")
        print(f"  Orphaned cells: {orphaned_count}")
        print(f"  Total cells: {len(self.particles)}")

    def clip_to_domain(self, vertices):
        """Clip polygon to domain boundaries"""
        xmin, xmax, ymin, ymax = self.domain_bounds

        if len(vertices) == 0:
            return []

        vertices = np.array(vertices)

        for boundary in ['left', 'right', 'bottom', 'top']:
            if len(vertices) == 0:
                break

            clipped = []

            for i in range(len(vertices)):
                current = vertices[i]
                previous = vertices[i - 1]

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

    def perform_stitching(self):
        """Perform the stitching part of Algorithm 1"""
        success = ClippedVoronoiStitcher.stitch_orphaned_cells(self.particles)

        if success:
            print("✅ Stitching successful - topology preserved!")
        else:
            print("⚠️  Some cells remain orphaned")

        return success

    def visualize_clipped_stitched(self, figsize=(15, 10)):
        """
        Visualize the complete clipped and stitched Voronoi tessellation.
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Extract data
        positions = np.array([[p.x, p.y] for p in self.particles])
        pressures = np.array([p.pressure for p in self.particles])

        # Plot 1: Original Voronoi cells
        ax1 = axes[0, 0]
        for i, particle in enumerate(self.particles):
            if len(particle.cell_vertices) >= 3:
                poly = Polygon(particle.cell_vertices, alpha=0.3, facecolor='lightblue', edgecolor='blue')
                ax1.add_patch(poly)

        ax1.scatter(positions[:, 0], positions[:, 1], c='blue', s=30, zorder=5)
        self.plot_solids(ax1)
        ax1.set_title('1. Original Voronoi Cells')
        ax1.set_xlim(self.domain_bounds[0], self.domain_bounds[1])
        ax1.set_ylim(self.domain_bounds[2], self.domain_bounds[3])
        ax1.grid(True, alpha=0.3)

        # Plot 2: Clipped cells (valid vs orphaned)
        ax2 = axes[0, 1]
        for particle in self.particles:
            if len(particle.clipped_vertices) >= 3:
                color = 'lightcoral' if particle.is_orphaned else 'lightgreen'
                edge_color = 'red' if particle.is_orphaned else 'green'
                poly = Polygon(particle.clipped_vertices, alpha=0.5, facecolor=color, edgecolor=edge_color)
                ax2.add_patch(poly)

        # Color particles by status
        valid_particles = [p for p in self.particles if not p.is_orphaned]
        orphaned_particles = [p for p in self.particles if p.is_orphaned]

        if valid_particles:
            valid_pos = np.array([[p.x, p.y] for p in valid_particles])
            ax2.scatter(valid_pos[:, 0], valid_pos[:, 1], c='green', s=30, label='Valid', zorder=5)

        if orphaned_particles:
            orphan_pos = np.array([[p.x, p.y] for p in orphaned_particles])
            ax2.scatter(orphan_pos[:, 0], orphan_pos[:, 1], c='red', s=30, label='Orphaned', zorder=5)

        self.plot_solids(ax2)
        ax2.set_title(f'2. Clipped Cells (Valid: {len(valid_particles)}, Orphaned: {len(orphaned_particles)})')
        ax2.legend()
        ax2.set_xlim(self.domain_bounds[0], self.domain_bounds[1])
        ax2.set_ylim(self.domain_bounds[2], self.domain_bounds[3])
        ax2.grid(True, alpha=0.3)

        # Plot 3: Stitching connections
        ax3 = axes[1, 0]
        for particle in self.particles:
            if len(particle.clipped_vertices) >= 3:
                color = 'lightcoral' if particle.is_orphaned else 'lightgreen'
                poly = Polygon(particle.clipped_vertices, alpha=0.3, facecolor=color)
                ax3.add_patch(poly)

            # Draw stitching connections
            for orphan in particle.orphaned_cells:
                ax3.plot([particle.x, orphan.x], [particle.y, orphan.y],
                         'purple', linewidth=2, alpha=0.7, zorder=4)
                ax3.plot([particle.x, orphan.x], [particle.y, orphan.y],
                         'o', color='purple', markersize=3, zorder=5)

        ax3.scatter(positions[:, 0], positions[:, 1], c=['red' if p.is_orphaned else 'green' for p in self.particles],
                    s=30, zorder=5)
        self.plot_solids(ax3)
        ax3.set_title('3. Stitching Connections')
        ax3.set_xlim(self.domain_bounds[0], self.domain_bounds[1])
        ax3.set_ylim(self.domain_bounds[2], self.domain_bounds[3])
        ax3.grid(True, alpha=0.3)

        # Plot 4: Final result with pressure
        ax4 = axes[1, 1]

        # Show effective domains (including stitched cells)
        for particle in self.particles:
            if not particle.is_orphaned and len(particle.clipped_vertices) >= 3:
                # Main cell
                poly = Polygon(particle.clipped_vertices, alpha=0.6,
                               facecolor=plt.cm.viridis(
                                   plt.Normalize(vmin=min(pressures), vmax=max(pressures))(particle.pressure)))
                ax4.add_patch(poly)

                # Owned orphaned cells
                for orphan in particle.orphaned_cells:
