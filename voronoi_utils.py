"""
voronoi_utils.py: Voronoi diagram construction and topology-preserving operations
"""

import numpy as np
import numba
from scipy.spatial import Voronoi, voronoi_plot_2d
from collections import defaultdict
import matplotlib.pyplot as plt


@numba.njit
def line_intersection(p1, p2, p3, p4):
    """
    Find intersection point between two line segments
    Line 1: p1 to p2
    Line 2: p3 to p4

    Returns:
    --------
    intersection : array or None
        Intersection point if it exists, None otherwise
    t : float
        Parameter for line 1 (intersection = p1 + t*(p2-p1))
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if abs(denom) < 1e-10:
        return None, -1.0

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return np.array([x, y]), t

    return None, -1.0


def compute_voronoi_diagram(positions, add_boundary_points=True):
    """
    Compute Voronoi diagram from particle positions

    Parameters:
    -----------
    positions : array (n_particles, 2)
        Particle positions
    add_boundary_points : bool
        Whether to add mirrored points for boundary handling

    Returns:
    --------
    vor : Voronoi object
        Scipy Voronoi diagram
    """
    if add_boundary_points:
        # Add mirrored points to handle boundaries properly
        # This helps with finite domain boundaries
        extended_positions = [positions]

        # Mirror across boundaries (simple box domain assumption)
        xmin, xmax = positions[:, 0].min() - 0.1, positions[:, 0].max() + 0.1
        ymin, ymax = positions[:, 1].min() - 0.1, positions[:, 1].max() + 0.1

        # Mirror left/right
        mirror_left = positions.copy()
        mirror_left[:, 0] = 2 * xmin - positions[:, 0]
        mirror_right = positions.copy()
        mirror_right[:, 0] = 2 * xmax - positions[:, 0]

        # Mirror top/bottom
        mirror_bottom = positions.copy()
        mirror_bottom[:, 1] = 2 * ymin - positions[:, 1]
        mirror_top = positions.copy()
        mirror_top[:, 1] = 2 * ymax - positions[:, 1]

        extended_positions.extend([mirror_left, mirror_right, mirror_bottom, mirror_top])
        all_positions = np.vstack(extended_positions)
    else:
        all_positions = positions

    vor = Voronoi(all_positions)
    return vor


def clip_voronoi_by_solids(vor, solid_segments):
    """
    Clip Voronoi cells by solid boundaries

    Parameters:
    -----------
    vor : Voronoi object
        Original Voronoi diagram
    solid_segments : list
        List of solid line segments

    Returns:
    --------
    clipped_cells : dict
        Dictionary mapping particle index to clipped cell vertices
    """
    clipped_cells = {}
    n_original = len(vor.points) // 5 if len(vor.points) > 100 else len(vor.points)

    for i in range(n_original):  # Only process original particles
        region_index = vor.point_region[i]
        region = vor.regions[region_index]

        if -1 in region or len(region) == 0:
            continue

        # Get cell vertices
        vertices = [vor.vertices[j] for j in region]

        # Clip cell by each solid segment
        clipped_vertices = vertices.copy()
        solid_faces_in_cell = []

        for solid in solid_segments:
            start = solid['start']
            end = solid['end']

            # Check intersections with cell edges
            new_vertices = []
            intersections = []

            for j in range(len(clipped_vertices)):
                v1 = clipped_vertices[j]
                v2 = clipped_vertices[(j + 1) % len(clipped_vertices)]

                # Check if edge intersects solid
                inter, t = line_intersection(v1, v2, start, end)

                # Check which side of solid the vertices are on
                # Using cross product to determine side
                def point_side(p):
                    return np.cross(end - start, p - start)

                side1 = point_side(v1)
                side2 = point_side(v2)

                # Keep v1 if it's on positive side
                if side1 >= 0:
                    new_vertices.append(v1)

                # Add intersection if edge crosses solid
                if inter is not None and side1 * side2 < 0:
                    new_vertices.append(inter)
                    intersections.append(inter)

            # If we had intersections, add solid face to cell
            if len(intersections) == 2:
                solid_faces_in_cell.append({
                    'start': intersections[0],
                    'end': intersections[1],
                    'solid_ref': solid
                })

            clipped_vertices = new_vertices

        clipped_cells[i] = {
            'vertices': clipped_vertices,
            'solid_faces': solid_faces_in_cell,
            'contains_source': True,
            'source_position': vor.points[i]
        }

    return clipped_cells


def stitch_orphaned_cells(clipped_cells, positions):
    """
    Stitch orphaned cells to valid cells based on largest shared interface
    Following Algorithm 1 from the paper

    Parameters:
    -----------
    clipped_cells : dict
        Clipped Voronoi cells
    positions : array
        Particle positions

    Returns:
    --------
    stitched_cells : dict
        Cells with orphaned cells properly assigned
    """
    # First, identify orphaned cells (cells that don't contain their source)
    orphaned = []
    valid_cells = {}

    for idx, cell in clipped_cells.items():
        if not cell['vertices']:  # Empty cell
            continue

        # Check if source point is inside polygon
        if is_point_in_polygon(cell['source_position'], cell['vertices']):
            cell['contains_source'] = True
            valid_cells[idx] = cell
        else:
            cell['contains_source'] = False
            orphaned.append((idx, cell))

    # Iteratively assign orphaned cells
    max_iterations = 100
    iteration = 0

    while orphaned and iteration < max_iterations:
        newly_assigned = []
        still_orphaned = []

        for orph_idx, orph_cell in orphaned:
            # Find neighboring valid cells
            best_neighbor = None
            best_area = 0.0

            # Check shared interfaces with valid cells
            for valid_idx, valid_cell in valid_cells.items():
                shared_area = compute_shared_interface_area(
                    orph_cell['vertices'],
                    valid_cell['vertices']
                )

                if shared_area > best_area:
                    best_area = shared_area
                    best_neighbor = valid_idx

            if best_neighbor is not None:
                # Assign orphaned cell to best neighbor
                orph_cell['assigned_to'] = best_neighbor
                newly_assigned.append((orph_idx, orph_cell))
            else:
                still_orphaned.append((orph_idx, orph_cell))

        # Update valid cells with newly assigned
        for idx, cell in newly_assigned:
            valid_cells[idx] = cell

        orphaned = still_orphaned
        iteration += 1

    # Combine all cells
    stitched_cells = valid_cells.copy()
    for idx, cell in orphaned:
        # Force assignment to nearest particle if still orphaned
        nearest_idx = find_nearest_valid_particle(cell['source_position'], positions, valid_cells)
        cell['assigned_to'] = nearest_idx
        stitched_cells[idx] = cell

    return stitched_cells


@numba.njit
def is_point_in_polygon(point, vertices):
    """
    Check if a point is inside a polygon using ray casting
    """
    x, y = point
    n = len(vertices)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]

        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside

        j = i

    return inside


def compute_shared_interface_area(vertices1, vertices2):
    """
    Compute the length of shared interface between two cells
    (In 2D, "area" of interface is actually length)
    """
    shared_length = 0.0

    # Check each edge of cell 1
    for i in range(len(vertices1)):
        v1a = vertices1[i]
        v1b = vertices1[(i + 1) % len(vertices1)]

        # Check against each edge of cell 2
        for j in range(len(vertices2)):
            v2a = vertices2[j]
            v2b = vertices2[(j + 1) % len(vertices2)]

            # Check if edges are colinear and overlap
            if edges_overlap(v1a, v1b, v2a, v2b):
                overlap_length = compute_edge_overlap_length(v1a, v1b, v2a, v2b)
                shared_length += overlap_length

    return shared_length


def edges_overlap(p1, p2, p3, p4):
    """
    Check if two line segments overlap (are colinear and have shared portion)
    """
    # Check if lines are parallel
    d1 = p2 - p1
    d2 = p4 - p3

    cross = np.cross(d1, d2)
    if abs(cross) > 1e-10:
        return False  # Not parallel

    # Check if colinear
    cross2 = np.cross(d1, p3 - p1)
    if abs(cross2) > 1e-10:
        return False  # Not colinear

    # Project onto line direction
    if abs(d1[0]) > abs(d1[1]):
        t1 = 0
        t2 = (p2[0] - p1[0]) / d1[0]
        t3 = (p3[0] - p1[0]) / d1[0]
        t4 = (p4[0] - p1[0]) / d1[0]
    else:
        t1 = 0
        t2 = (p2[1] - p1[1]) / d1[1]
        t3 = (p3[1] - p1[1]) / d1[1]
        t4 = (p4[1] - p1[1]) / d1[1]

    # Check overlap
    min1, max1 = min(t1, t2), max(t1, t2)
    min2, max2 = min(t3, t4), max(t3, t4)

    return max(min1, min2) < min(max1, max2)


def compute_edge_overlap_length(p1, p2, p3, p4):
    """
    Compute length of overlap between two colinear segments
    """
    # Project all points onto line p1-p2
    d = p2 - p1
    length = np.linalg.norm(d)
    if length < 1e-10:
        return 0.0

    d_norm = d / length

    # Project points
    t1 = 0
    t2 = length
    t3 = np.dot(p3 - p1, d_norm)
    t4 = np.dot(p4 - p1, d_norm)

    # Find overlap
    min1, max1 = min(t1, t2), max(t1, t2)
    min2, max2 = min(t3, t4), max(t3, t4)

    overlap_start = max(min1, min2)
    overlap_end = min(max1, max2)

    if overlap_start < overlap_end:
        return overlap_end - overlap_start
    else:
        return 0.0


def find_nearest_valid_particle(position, all_positions, valid_cells):
    """
    Find nearest valid particle to a given position
    """
    min_dist = float('inf')
    nearest_idx = 0

    for idx in valid_cells.keys():
        dist = np.linalg.norm(all_positions[idx] - position)
        if dist < min_dist:
            min_dist = dist
            nearest_idx = idx

    return nearest_idx


def compute_interface_geometry(cells):
    """
    Compute geometric properties of interfaces between cells

    Returns:
    --------
    interfaces : list
        List of interface dictionaries with area, normal, and neighboring cells
    """
    interfaces = []
    processed_pairs = set()

    # Build adjacency information
    for i, cell_i in cells.items():
        if not cell_i.get('vertices'):
            continue

        for j, cell_j in cells.items():
            if i >= j or not cell_j.get('vertices'):
                continue

            if (i, j) in processed_pairs:
                continue

            # Check for shared interface
            shared_length = compute_shared_interface_area(
                cell_i['vertices'],
                cell_j['vertices']
            )

            if shared_length > 1e-10:
                # Compute interface normal (from i to j)
                # Find the shared edge midpoint and normal
                midpoint, normal = compute_interface_normal(
                    cell_i['vertices'],
                    cell_j['vertices']
                )

                interfaces.append({
                    'cell_i': i,
                    'cell_j': j,
                    'area': shared_length,  # In 2D, this is length
                    'normal': normal,
                    'midpoint': midpoint,
                    'is_solid': False
                })

                processed_pairs.add((i, j))

    # Add solid interfaces
    for i, cell in cells.items():
        for solid_face in cell.get('solid_faces', []):
            # Compute normal pointing into fluid
            edge = solid_face['end'] - solid_face['start']
            normal = np.array([-edge[1], edge[0]])  # Rotate 90 degrees
            normal = normal / np.linalg.norm(normal)

            interfaces.append({
                'cell_i': i,
                'cell_j': -1,  # Solid boundary
                'area': np.linalg.norm(edge),
                'normal': normal,
                'midpoint': 0.5 * (solid_face['start'] + solid_face['end']),
                'is_solid': True,
                'solid_velocity': solid_face['solid_ref']['velocity']
            })

    return interfaces


def compute_interface_normal(vertices1, vertices2):
    """
    Compute the normal vector and midpoint of shared interface
    """
    # Find shared vertices/edges
    shared_points = []

    for i in range(len(vertices1)):
        v1a = vertices1[i]
        v1b = vertices1[(i + 1) % len(vertices1)]

        for j in range(len(vertices2)):
            v2a = vertices2[j]
            v2b = vertices2[(j + 1) % len(vertices2)]

            if edges_overlap(v1a, v1b, v2a, v2b):
                # Get overlap endpoints
                overlap_points = get_overlap_endpoints(v1a, v1b, v2a, v2b)
                shared_points.extend(overlap_points)

    if len(shared_points) >= 2:
        # Use first two points to define interface
        p1 = shared_points[0]
        p2 = shared_points[1]
        midpoint = 0.5 * (p1 + p2)

        # Normal points from cell 1 to cell 2
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        normal = normal / np.linalg.norm(normal)

        return midpoint, normal
    else:
        # Fallback: use cell centroids
        c1 = np.mean(vertices1, axis=0)
        c2 = np.mean(vertices2, axis=0)
        normal = c2 - c1
        normal = normal / np.linalg.norm(normal)
        return 0.5 * (c1 + c2), normal


def get_overlap_endpoints(p1, p2, p3, p4):
    """
    Get the endpoints of the overlap between two colinear segments
    """
    # Project all points onto line p1-p2
    d = p2 - p1
    length = np.linalg.norm(d)
    if length < 1e-10:
        return []

    d_norm = d / length

    # Project points
    t1 = 0
    t2 = length
    t3 = np.dot(p3 - p1, d_norm)
    t4 = np.dot(p4 - p1, d_norm)

    # Find overlap
    min1, max1 = min(t1, t2), max(t1, t2)
    min2, max2 = min(t3, t4), max(t3, t4)

    overlap_start = max(min1, min2)
    overlap_end = min(max1, max2)

    if overlap_start < overlap_end:
        # Convert back to points
        start_point = p1 + overlap_start * d_norm
        end_point = p1 + overlap_end * d_norm
        return [start_point, end_point]
    else:
        return []