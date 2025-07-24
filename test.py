#!/usr/bin/env python3
"""
Debug script to check why solid interfaces aren't being created
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi

# Import our modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from voronoi_utils import compute_voronoi_diagram, clip_voronoi_by_solids, compute_interface_geometry

def create_simple_test():
    """Create a simple test case with a few particles and a box boundary"""
    
    # Simple particle setup
    positions = np.array([
        [-0.3, 0.0],   # Outside box
        [0.0, 0.0],    # Inside box  
        [0.3, 0.0],    # Outside box
        [0.0, 0.3],    # Outside box
        [0.0, -0.3]    # Outside box
    ])
    
    # Simple box boundary
    box_size = 0.2
    solid_segments = [
        {'start': np.array([-box_size, -box_size]), 'end': np.array([box_size, -box_size]), 'velocity': np.zeros(2)},
        {'start': np.array([box_size, -box_size]), 'end': np.array([box_size, box_size]), 'velocity': np.zeros(2)},
        {'start': np.array([box_size, box_size]), 'end': np.array([-box_size, box_size]), 'velocity': np.zeros(2)},
        {'start': np.array([-box_size, box_size]), 'end': np.array([-box_size, -box_size]), 'velocity': np.zeros(2)}
    ]
    
    return positions, solid_segments

def debug_interface_creation():
    """Debug the interface creation process step by step"""
    
    print("=== Debugging Solid Interface Creation ===")
    
    # Create simple test case
    positions, solid_segments = create_simple_test()
    
    print(f"Test setup:")
    print(f"  Particles: {len(positions)}")
    print(f"  Solid segments: {len(solid_segments)}")
    
    # Step 1: Compute Voronoi
    print("\n1. Computing Voronoi diagram...")
    try:
        vor = compute_voronoi_diagram(positions)
        print(f"   Success: {len(vor.points)} points, {len(vor.regions)} regions")
    except Exception as e:
        print(f"   ERROR: {e}")
        return
    
    # Step 2: Clip by solids
    print("\n2. Clipping Voronoi by solids...")
    try:
        clipped_cells = clip_voronoi_by_solids(vor, solid_segments)
        print(f"   Success: {len(clipped_cells)} cells")
        
        # Check which cells have solid faces
        cells_with_solids = 0
        total_solid_faces = 0
        for idx, cell in clipped_cells.items():
            n_solid_faces = len(cell.get('solid_faces', []))
            if n_solid_faces > 0:
                cells_with_solids += 1
                total_solid_faces += n_solid_faces
                print(f"   Cell {idx}: {n_solid_faces} solid faces")
        
        print(f"   Total: {cells_with_solids} cells with solid faces, {total_solid_faces} solid faces")
        
    except Exception as e:
        print(f"   ERROR: {e}")
        return
    
    # Step 3: Compute interfaces
    print("\n3. Computing interface geometry...")
    try:
        interfaces = compute_interface_geometry(clipped_cells)
        
        solid_interfaces = [i for i in interfaces if i.get('is_solid', False)]
        fluid_interfaces = [i for i in interfaces if not i.get('is_solid', False)]
        
        print(f"   Success: {len(interfaces)} total interfaces")
        print(f"   - {len(solid_interfaces)} solid interfaces")
        print(f"   - {len(fluid_interfaces)} fluid interfaces")
        
        # Debug solid interfaces
        if solid_interfaces:
            print("\n   Solid interface details:")
            for i, interface in enumerate(solid_interfaces[:5]):  # Show first 5
                print(f"     [{i}] Cell {interface['cell_i']} -> solid, area={interface['area']:.3f}")
        else:
            print("   WARNING: No solid interfaces created!")
            
    except Exception as e:
        print(f"   ERROR: {e}")
        return
    
    # Visualization
    print("\n4. Creating visualization...")
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Setup
        ax1.scatter(positions[:, 0], positions[:, 1], c='blue', s=50, alpha=0.7, label='Particles')
        
        for i, segment in enumerate(solid_segments):
            start, end = segment['start'], segment['end']
            ax1.plot([start[0], end[0]], [start[1], end[1]], 'r-', linewidth=3, label='Solid' if i == 0 else "")
        
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_title('Test Setup')
        
        # Plot 2: Interfaces
        ax2.scatter(positions[:, 0], positions[:, 1], c='blue', s=50, alpha=0.7, label='Particles')
        
        # Draw solid segments
        for segment in solid_segments:
            start, end = segment['start'], segment['end']
            ax2.plot([start[0], end[0]], [start[1], end[1]], 'r-', linewidth=3, alpha=0.8)
        
        # Draw interfaces
        for interface in fluid_interfaces:
            mid = interface['midpoint']
            normal = interface['normal'] * 0.05
            ax2.arrow(mid[0], mid[1], normal[0], normal[1], 
                     head_width=0.02, head_length=0.01, fc='green', ec='green', alpha=0.6)
        
        for interface in solid_interfaces:
            mid = interface['midpoint']
            normal = interface['normal'] * 0.05
            ax2.arrow(mid[0], mid[1], normal[0], normal[1], 
                     head_width=0.02, head_length=0.01, fc='red', ec='red', alpha=0.8)
        
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        ax2.set_title(f'Interfaces (Green=Fluid, Red=Solid)\n{len(solid_interfaces)} solid interfaces')
        
        plt.tight_layout()
        plt.savefig('debug_interfaces.png', dpi=150, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"   Visualization ERROR: {e}")
    
    print("\n=== Debug Complete ===")
    
    # Return results for further analysis
    return {
        'positions': positions,
        'solid_segments': solid_segments,
        'clipped_cells': clipped_cells,
        'interfaces': interfaces,
        'solid_interfaces': solid_interfaces
    }

if __name__ == "__main__":
    results = debug_interface_creation()