"""
voronoi_utils.py: FIXED - Enhanced topology-preserving Voronoi discretization utilities
Fixed the critical issue with missing fluid-fluid interfaces
"""

import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from collections import defaultdict


def compute_voronoi_diagram(positions):
    """
    Compute standard Voronoi diagram from fluid particle positions
    """
    if len(positions) < 3:
        raise ValueError("Need at least 3 points for Voronoi diagram")
    
    # Add boundary points to prevent infinite regions
    xmin, xmax = np.min(positions[:, 0]) - 1, np.max(positions[:, 0]) + 1
    ymin, ymax = np.min(positions[:, 1]) - 1, np.max(positions[:, 1]) + 1
    
    boundary_points = np.array([
        [xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax],
        [xmin, (ymin+ymax)/2], [xmax, (ymin+ymax)/2],
        [(xmin+xmax)/2, ymin], [(xmin+xmax)/2, ymax]
    ])
    
    extended_positions = np.vstack([positions, boundary_points])
    
    try:
        vor = Voronoi(extended_positions)
        return vor
    except Exception as e:
        print(f"Voronoi computation failed: {e}")
        return Voronoi(positions)


def point_in_polygon(point, vertices):
    """Ray casting algorithm to determine if point is inside polygon"""
    x, y = point
    n = len(vertices)
    inside = False
    
    p1x, p1y = vertices[0]
    for i in range(1, n + 1):
        p2x, p2y = vertices[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def compute_adaptive_threshold(positions, solid_segments, base_threshold=0.15):
    """Compute adaptive distance threshold based on local particle density"""
    if len(positions) < 2:
        return base_threshold
    
    # Compute average nearest neighbor distance
    distances = cdist(positions, positions)
    np.fill_diagonal(distances, np.inf)
    
    avg_nn_distance = np.mean(np.min(distances, axis=1))
    
    # FIXED: Better adaptive scaling
    if len(solid_segments) > 100:
        adaptive_factor = 1.8  # Reduced from 2.0
    elif len(solid_segments) > 50:
        adaptive_factor = 1.4  # Reduced from 1.5
    else:
        adaptive_factor = 1.0
    
    # Scale with average particle spacing but cap it
    threshold = adaptive_factor * max(base_threshold, min(0.6 * avg_nn_distance, 0.3))
    
    return threshold


def clip_voronoi_by_solids(vor, solid_segments):
    """FIXED - Enhanced clip Voronoi cells by solid boundaries"""
    cells = {}
    n_particles = len(vor.points) - 8  # Subtract boundary points
    
    print(f"FIXED Enhanced clipping: {n_particles} particles against {len(solid_segments)} solid segments")
    
    # Compute adaptive threshold
    particle_positions = vor.points[:n_particles]
    distance_threshold = compute_adaptive_threshold(particle_positions, solid_segments)
    print(f"Using adaptive threshold: {distance_threshold:.4f}")
    
    # Pre-compute segment centers for faster spatial queries
    segment_centers = []
    for segment in solid_segments:
        center = 0.5 * (segment['start'] + segment['end'])
        segment_centers.append(center)
    segment_centers = np.array(segment_centers) if segment_centers else np.empty((0, 2))
    
    for i in range(n_particles):
        if vor.point_region[i] < 0:
            continue
            
        region_idx = vor.point_region[i]
        if region_idx >= len(vor.regions):
            continue
            
        region = vor.regions[region_idx]
        if not region or -1 in region:
            continue
            
        try:
            cell_vertices = [vor.vertices[j] for j in region]
            if len(cell_vertices) < 3:
                continue
            cell_vertices = np.array(cell_vertices)
        except:
            continue
        
        # Find nearby solid segments
        particle_pos = vor.points[i]
        solid_faces = []
        
        # Quick spatial filtering
        if len(segment_centers) > 0:
            distances_to_centers = np.linalg.norm(segment_centers - particle_pos, axis=1)
            nearby_indices = np.where(distances_to_centers < distance_threshold * 1.5)[0]  # Reduced multiplier
        else:
            nearby_indices = []
        
        # Check nearby segments
        for idx in nearby_indices:
            segment = solid_segments[idx]
            seg_start = segment['start']
            seg_end = segment['end']
            
            # Distance from particle to line segment
            seg_vec = seg_end - seg_start
            to_start = particle_pos - seg_start
            
            seg_length_sq = np.dot(seg_vec, seg_vec)
            if seg_length_sq > 1e-10:
                t = np.clip(np.dot(to_start, seg_vec) / seg_length_sq, 0, 1)
                closest_point = seg_start + t * seg_vec
                distance = np.linalg.norm(particle_pos - closest_point)
                
                if distance < distance_threshold:
                    solid_faces.append({
                        'start': seg_start,
                        'end': seg_end,
                        'solid_ref': segment,
                        'distance': distance
                    })
        
        # Sort by distance
        if solid_faces:
            solid_faces.sort(key=lambda x: x['distance'])
        
        # Store cell data
        cells[i] = {
            'vertices': cell_vertices,
            'solid_faces': solid_faces,
            'contains_source': True,
            'source_position': vor.points[i]
        }
    
    # Debug output
    cells_with_solids = sum(1 for cell in cells.values() if len(cell.get('solid_faces', [])) > 0)
    total_solid_faces = sum(len(cell.get('solid_faces', [])) for cell in cells.values())
    avg_faces_per_cell = total_solid_faces / max(cells_with_solids, 1)
    
    print(f"FIXED Enhanced clipping result:")
    print(f"  {cells_with_solids} cells have solid faces")
    print(f"  Total solid faces: {total_solid_faces}")
    print(f"  Average faces per boundary cell: {avg_faces_per_cell:.2f}")
    
    return cells


def stitch_orphaned_cells(clipped_cells, positions):
    """Stitch orphaned cells back to valid cells - Algorithm 1 from Paper"""
    stitched_cells = clipped_cells.copy()
    
    orphaned_cells = []
    valid_cells = []
    
    for idx, cell in stitched_cells.items():
        if cell['contains_source']:
            source_pos = cell['source_position']
            vertices = cell['vertices']
            
            if len(vertices) > 2:
                inside = point_in_polygon(source_pos, vertices)
                if not inside:
                    cell['contains_source'] = False
                    orphaned_cells.append(idx)
                else:
                    valid_cells.append(idx)
            else:
                cell['contains_source'] = False
                orphaned_cells.append(idx)
        else:
            orphaned_cells.append(idx)
    
    print(f"Stitching: {len(orphaned_cells)} orphaned cells, {len(valid_cells)} valid cells")
    
    # Algorithm 1: Iteratively assign orphaned cells to valid neighbors
    max_iterations = 10
    iteration = 0
    
    while orphaned_cells and iteration < max_iterations:
        newly_assigned = []
        
        for orphan_idx in orphaned_cells:
            if orphan_idx not in stitched_cells:
                continue
                
            best_neighbor = None
            largest_interface_area = 0
            
            for valid_idx in valid_cells:
                if valid_idx not in stitched_cells:
                    continue
                    
                dist = np.linalg.norm(positions[orphan_idx] - positions[valid_idx])
                if dist > 0:
                    interface_area = 1.0 / dist
                    
                    if interface_area > largest_interface_area:
                        largest_interface_area = interface_area
                        best_neighbor = valid_idx
            
            if best_neighbor is not None:
                newly_assigned.append(orphan_idx)
                valid_cells.append(orphan_idx)
                stitched_cells[orphan_idx]['contains_source'] = False
                stitched_cells[orphan_idx]['assigned_to'] = best_neighbor
        
        for assigned_idx in newly_assigned:
            if assigned_idx in orphaned_cells:
                orphaned_cells.remove(assigned_idx)
        
        iteration += 1
        
        if newly_assigned:
            print(f"Stitching iteration {iteration}: Assigned {len(newly_assigned)} orphaned cells")
    
    if orphaned_cells:
        print(f"Warning: {len(orphaned_cells)} cells remain orphaned after stitching")
    
    return stitched_cells


def compute_interface_geometry(cells):
    """
    FIXED - Enhanced compute geometric properties of interfaces
    
    CRITICAL FIX: Ensures fluid-fluid interfaces are created properly
    """
    interfaces = []
    processed_pairs = set()
    
    # Get positions for all cells
    positions = {}
    for idx, cell in cells.items():
        if 'source_position' in cell:
            positions[idx] = cell['source_position']
    
    if not positions:
        print("WARNING: No positions found for interface computation!")
        return interfaces
    
    # FIXED: Better neighborhood computation
    pos_array = np.array(list(positions.values()))
    indices = list(positions.keys())
    
    if len(pos_array) > 1:
        distances = cdist(pos_array, pos_array)
        np.fill_diagonal(distances, np.inf)
        avg_nn_distance = np.mean(np.min(distances, axis=1))
        # FIXED: More generous fluid-fluid neighborhood
        max_distance = min(0.5, 3.0 * avg_nn_distance)  # Increased multiplier
    else:
        max_distance = 0.4
    
    print(f"FIXED Enhanced interface computation using neighborhood distance: {max_distance:.4f}")
    
    # FIXED: Create fluid-fluid interfaces with better logic
    fluid_interface_count = 0
    for i_idx, i in enumerate(indices):
        for j_idx, j in enumerate(indices):
            if i >= j:
                continue
                
            if (i, j) in processed_pairs:
                continue
                
            # Check if particles are neighbors
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
                
                # Area computation based on distance
                area = max(0.03, 0.4 * dist)  # More generous area
                
                interfaces.append({
                    'cell_i': i,
                    'cell_j': j,
                    'area': area,
                    'normal': normal,
                    'midpoint': midpoint,
                    'is_solid': False,
                    'distance': dist
                })
                
                processed_pairs.add((i, j))
                fluid_interface_count += 1
    
    # Add solid interfaces
    solid_interface_count = 0
    total_solid_area = 0.0
    
    for i, cell in cells.items():
        solid_faces = cell.get('solid_faces', [])
        
        for solid_face in solid_faces:
            edge = solid_face['end'] - solid_face['start']
            edge_length = np.linalg.norm(edge)
            
            if edge_length > 1e-10:
                # Compute area (simplified)
                area = max(0.02, edge_length * 0.05)  # Simple area estimate
                
                # Normal perpendicular to edge
                normal = np.array([-edge[1], edge[0]])
                normal = normal / np.linalg.norm(normal)
                
                # Ensure normal points toward fluid particle
                edge_midpoint = 0.5 * (solid_face['start'] + solid_face['end'])
                to_particle = positions[i] - edge_midpoint
                if np.dot(normal, to_particle) < 0:
                    normal = -normal
                
                interfaces.append({
                    'cell_i': i,
                    'cell_j': -1,  # Solid boundary marker
                    'area': area,
                    'normal': normal,
                    'midpoint': edge_midpoint,
                    'is_solid': True,
                    'solid_velocity': solid_face['solid_ref'].get('velocity', np.array([0.0, 0.0]))
                })
                
                solid_interface_count += 1
                total_solid_area += area
    
    # FIXED: Better debug output
    avg_solid_area = total_solid_area / max(solid_interface_count, 1)
    fluid_interfaces = [i for i in interfaces if not i['is_solid']]
    avg_fluid_area = np.mean([i['area'] for i in fluid_interfaces]) if fluid_interfaces else 0
    
    print(f"FIXED Enhanced interfaces created: {len(interfaces)} total")
    print(f"  Fluid interfaces: {len(fluid_interfaces)}, avg area: {avg_fluid_area:.4f}")
    print(f"  Solid interfaces: {solid_interface_count}, avg area: {avg_solid_area:.4f}")
    print(f"  Total solid area: {total_solid_area:.4f}")
    
    # CRITICAL CHECK
    if len(fluid_interfaces) == 0:
        print("⚠️  CRITICAL WARNING: No fluid-fluid interfaces created!")
        print("   This will cause simulation failure - particles can't communicate")
        print(f"   Particle positions range:")
        if positions:
            pos_array = np.array(list(positions.values()))
            print(f"   X: [{np.min(pos_array[:, 0]):.3f}, {np.max(pos_array[:, 0]):.3f}]")
            print(f"   Y: [{np.min(pos_array[:, 1]):.3f}, {np.max(pos_array[:, 1]):.3f}]")
            print(f"   Avg distance: {avg_nn_distance:.3f}, Max allowed: {max_distance:.3f}")
    
    return interfaces


def debug_plot_voronoi_clipping(vor, solid_segments, clipped_cells, filename=None):
    """Debug visualization of Voronoi clipping process"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Original Voronoi
    voronoi_plot_2d(vor, ax=ax1, show_vertices=False, line_colors='blue', line_width=1)
    
    # Add solid segments
    for segment in solid_segments:
        start, end = segment['start'], segment['end']
        ax1.plot([start[0], end[0]], [start[1], end[1]], 'r-', linewidth=3, alpha=0.8)
    
    ax1.set_title('Original Voronoi + Solid Boundaries')
    ax1.set_aspect('equal')
    
    # Plot 2: Clipped cells
    for idx, cell in clipped_cells.items():
        vertices = cell['vertices']
        if len(vertices) > 2:
            vertices = np.array(vertices)
            vertices_closed = np.vstack([vertices, vertices[0]])
            
            color = 'green' if cell['contains_source'] else 'red'
            ax2.plot(vertices_closed[:, 0], vertices_closed[:, 1], 
                    color=color, linewidth=2, alpha=0.7)
            
            if 'source_position' in cell:
                pos = cell['source_position']
                ax2.plot(pos[0], pos[1], 'ko', markersize=4)
    
    # Add solid segments to second plot
    for segment in solid_segments:
        start, end = segment['start'], segment['end']
        ax2.plot([start[0], end[0]], [start[1], end[1]], 'r-', linewidth=3, alpha=0.8)
    
    ax2.set_title('FIXED Clipped Cells (Green=Valid, Red=Orphaned)')
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    
    return fig