#!/usr/bin/env python3
"""
Generate a simple oval/ellipse contour CSV for wind tunnel testing
Much simpler than the bunny but still aerodynamically interesting
"""

import numpy as np
import matplotlib.pyplot as plt

def create_simple_oval():
    """
    Create a simple oval shape - like an airplane wing cross-section
    This will be much easier to debug than the 981-point bunny
    """
    
    # Create an ellipse with slight asymmetry to make it more interesting
    n_points = 32  # Much simpler than 981 points!
    
    # Parameter for the ellipse
    t = np.linspace(0, 2*np.pi, n_points + 1)[:-1]  # Exclude last point (same as first)
    
    # Ellipse parameters
    a = 0.4  # Semi-major axis (width)
    b = 0.25  # Semi-minor axis (height)
    
    # Basic ellipse
    x = a * np.cos(t)
    y = b * np.sin(t)
    
    # Add slight asymmetry to make it more wing-like (thicker at front)
    # Shift the shape slightly to make it more aerodynamic
    for i, angle in enumerate(t):
        # Make it slightly more pointed at the back (like a teardrop/wing)
        if np.cos(angle) < 0:  # Back half
            scale_factor = 1.0 - 0.3 * abs(np.cos(angle))
            y[i] *= scale_factor
    
    # Ensure it's closed (first point = last point)
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    
    return np.column_stack([x, y])

def save_oval_csv(filename='simple_oval.csv'):
    """Save the oval contour to CSV file"""
    
    points = create_simple_oval()
    
    # Save to CSV
    with open(filename, 'w') as f:
        f.write('x,y\n')  # Header
        for i, (x, y) in enumerate(points):
            f.write(f'{x},{y}\n')
    
    print(f"Saved {len(points)} points to {filename}")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(points[:, 0], points[:, 1], 'b-o', linewidth=2, markersize=4, alpha=0.7)
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.title(f'Simple Oval Contour ({len(points)} points)\nPerfect for Wind Tunnel Testing')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Add some sample "wind" arrows to show the test concept
    wind_y = np.linspace(-0.4, 0.4, 5)
    for y in wind_y:
        plt.arrow(-0.8, y, 0.2, 0, head_width=0.03, head_length=0.05, 
                 fc='red', ec='red', alpha=0.6)
    
    plt.text(-0.9, 0, 'Wind\n→', ha='center', va='center', fontsize=12, color='red')
    
    plt.tight_layout()
    plt.savefig('simple_oval_preview.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return points

if __name__ == "__main__":
    print("Creating simple oval contour for wind tunnel testing...")
    points = save_oval_csv()
    
    print(f"\nOval specifications:")
    print(f"  Points: {len(points)}")
    print(f"  X range: [{np.min(points[:, 0]):.3f}, {np.max(points[:, 0]):.3f}]")
    print(f"  Y range: [{np.min(points[:, 1]):.3f}, {np.max(points[:, 1]):.3f}]")
    print(f"  Closed contour: {np.allclose(points[0], points[-1])}")
    
    print(f"\nThis shape is:")
    print(f"  ✓ Much simpler than bunny (32 vs 981 points)")
    print(f"  ✓ Still aerodynamically interesting")
    print(f"  ✓ Easy to debug")
    print(f"  ✓ Perfect for leakproof testing")
    
    print(f"\nTo use: Replace 'bunny_contour.csv' with 'simple_oval.csv'")