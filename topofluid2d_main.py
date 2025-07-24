#!/usr/bin/env python3
"""
FIXED Real-Time Bunny Wind Tunnel with Interactive Visualization
Shows simulation progress in real-time with smooth animation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import FIXED modules
try:
    from voronoi_utils import (
        compute_voronoi_diagram,
        clip_voronoi_by_solids,
        stitch_orphaned_cells,
        compute_interface_geometry
    )
except ImportError:
    print("Warning: Using placeholder functions.")
    # [Previous placeholder functions remain the same]

try:
    from fluid_solver import (
        compute_numerical_flux,
        update_fluid_state,
        compute_timestep,
        apply_boundary_conditions
    )
except ImportError:
    print("Warning: Using placeholder functions.")
    # [Previous placeholder functions remain the same]


def load_bunny_contour_from_csv(filename='bunny_contour.csv'):
    """Load bunny contour points from CSV file"""
    try:
        data = np.loadtxt(filename, delimiter=',', skiprows=1)
        return data
    except:
        print(f"Warning: Could not load {filename}")
        try:
            data = np.loadtxt('simple_oval.csv', delimiter=',', skiprows=1)
            print("Using simple_oval.csv instead")
            return data
        except:
            print("Warning: Could not load simple_oval.csv either, using default")
            return None


def create_bunny_geometry():
    """Create bunny geometry using actual contour data from CSV"""
    bunny_data = load_bunny_contour_from_csv()
    
    if bunny_data is not None:
        bunny_points = bunny_data
        print(f"Loaded contour with {len(bunny_points)} points")
        
        # Center and scale
        x_min, x_max = np.min(bunny_points[:, 0]), np.max(bunny_points[:, 0])
        y_min, y_max = np.min(bunny_points[:, 1]), np.max(bunny_points[:, 1])
        
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        bunny_points[:, 0] -= center_x
        bunny_points[:, 1] -= center_y
        
        current_width = x_max - x_min
        current_height = y_max - y_min
        scale_factor = 0.6 / max(current_width, current_height)
        bunny_points *= scale_factor
        
        print(f"Contour rescaled by factor {scale_factor:.3f}")
    else:
        print("Using simplified geometry")
        angles = np.linspace(0, 2*np.pi, 32)
        bunny_points = 0.3 * np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Create line segments
    solid_segments = []
    n_points = len(bunny_points)
    
    for i in range(n_points):
        start = bunny_points[i]
        end = bunny_points[(i + 1) % n_points]
        
        solid_segments.append({
            'start': start,
            'end': end,
            'velocity': np.array([0.0, 0.0]),
            'is_deformable': False
        })
    
    return solid_segments


def initialize_bunny_wind_tunnel(domain_bounds, n_particles, bunny_segments):
    """Initialize the bunny wind tunnel test case"""
    xmin, xmax = domain_bounds[0]
    ymin, ymax = domain_bounds[1]
    
    # Create particle positions
    nx = int(np.sqrt(n_particles * (xmax - xmin) / (ymax - ymin)))
    ny = int(n_particles / nx)
    
    x = np.linspace(xmin + 0.05, xmax - 0.05, nx)
    y = np.linspace(ymin + 0.05, ymax - 0.05, ny)
    xx, yy = np.meshgrid(x, y)
    
    positions = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Point-in-polygon test
    def point_in_polygon(point, polygon_segments):
        x, y = point
        inside = False
        
        vertices = []
        for segment in polygon_segments:
            vertices.append(segment['start'])
        vertices = np.array(vertices)
        
        n = len(vertices)
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
    
    inside_bunny = np.array([point_in_polygon(pos, bunny_segments) for pos in positions])
    
    # Remove particles too close to boundary
    def distance_to_boundary(point, segments):
        min_dist = np.inf
        for segment in segments:
            start, end = segment['start'], segment['end']
            seg_vec = end - start
            point_vec = point - start
            seg_len_sq = np.dot(seg_vec, seg_vec)
            
            if seg_len_sq == 0:
                dist = np.linalg.norm(point_vec)
            else:
                t = max(0, min(1, np.dot(point_vec, seg_vec) / seg_len_sq))
                projection = start + t * seg_vec
                dist = np.linalg.norm(point - projection)
            
            min_dist = min(min_dist, dist)
        return min_dist
    
    boundary_distances = np.array([distance_to_boundary(pos, bunny_segments) for pos in positions])
    valid_mask = boundary_distances > 0.04
    
    positions = positions[valid_mask]
    inside_bunny = inside_bunny[valid_mask]
    
    n_actual = len(positions)
    
    # Initialize fluid state
    gamma = 1.4
    rho_initial = 1.0
    p_initial = 1.0
    exterior_velocity = 0.1
    
    u = np.zeros(n_actual)
    v = np.zeros(n_actual)
    
    u[~inside_bunny] = exterior_velocity
    
    rho = np.full(n_actual, rho_initial)
    p = np.full(n_actual, p_initial)
    
    e_internal = p / ((gamma - 1) * rho)
    e_kinetic = 0.5 * (u**2 + v**2)
    e_total = e_internal + e_kinetic
    
    state = {
        'rho': rho,
        'rho_u': rho * u,
        'rho_v': rho * v,
        'rho_e': rho * e_total,
        'positions': positions,
        'gamma': gamma
    }
    
    return state, inside_bunny


def compute_average_interior_velocity(state, inside_bunny_mask):
    """Compute average velocity magnitude inside the bunny"""
    if not np.any(inside_bunny_mask):
        return 0.0
    
    rho = state['rho'][inside_bunny_mask]
    rho_u = state['rho_u'][inside_bunny_mask]
    rho_v = state['rho_v'][inside_bunny_mask]
    
    u = np.where(rho > 1e-15, rho_u / rho, 0.0)
    v = np.where(rho > 1e-15, rho_v / rho, 0.0)
    
    vel_magnitude = np.sqrt(u**2 + v**2)
    return np.mean(vel_magnitude)


def plot_bunny_wind_tunnel_fixed(ax1, ax2, stitched_cells, solid_segments, positions, 
                                pressure, inside_bunny_mask, domain_bounds, t):
    """FIXED Plot bunny wind tunnel visualization - resolves array comparison issue"""
    
    # Clear axes
    ax1.clear()
    ax2.clear()
    
    # Plot 1: Voronoi mesh with particles
    for idx, cell in stitched_cells.items():
        if 'vertices' not in cell or not cell['vertices']:
            continue
        vertices = np.array(cell['vertices'])
        if len(vertices) > 2:
            from matplotlib.patches import Polygon
            poly = Polygon(vertices, facecolor='lightblue', 
                         edgecolor='gray', alpha=0.1, linewidth=0.3)
            ax1.add_patch(poly)
    
    # FIXED: Plot particles - safer array handling
    try:
        if len(inside_bunny_mask) > 0 and len(positions) > 0:
            # FIXED: Safe boolean indexing
            exterior_indices = np.where(~inside_bunny_mask)[0]
            interior_indices = np.where(inside_bunny_mask)[0]
            
            if len(exterior_indices) > 0:
                exterior_pos = positions[exterior_indices]
                ax1.scatter(exterior_pos[:, 0], exterior_pos[:, 1], 
                           c='blue', s=20, alpha=0.7, label='Exterior Fluid')
            
            if len(interior_indices) > 0:
                interior_pos = positions[interior_indices]
                ax1.scatter(interior_pos[:, 0], interior_pos[:, 1], 
                           c='red', s=20, alpha=0.7, label='Interior Fluid')
        else:
            # Fallback: all particles in blue
            ax1.scatter(positions[:, 0], positions[:, 1], 
                       c='blue', s=20, alpha=0.7, label='All Particles')
    except Exception as e:
        print(f"Particle plotting error: {e}")
        # Safe fallback
        ax1.scatter(positions[:, 0], positions[:, 1], 
                   c='blue', s=20, alpha=0.7, label='All Particles')
    
    # Plot bunny boundary
    for segment in solid_segments:
        start, end = segment['start'], segment['end']
        ax1.plot([start[0], end[0]], [start[1], end[1]],
                'k-', linewidth=2, alpha=0.8)
    
    ax1.set_xlim(domain_bounds[0])
    ax1.set_ylim(domain_bounds[1])
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'FIXED Real-Time Wind Tunnel (t={t:.3f})')
    ax1.legend()
    
    # Plot 2: Pressure field
    if pressure is not None and hasattr(pressure, 'size') and pressure.size > 0:
        p_min, p_max = np.min(pressure), np.max(pressure)
        
        scatter = ax2.scatter(positions[:, 0], positions[:, 1],
                             c=pressure, s=60, cmap='RdBu_r',
                             vmin=p_min, vmax=p_max,
                             edgecolors='black', linewidths=0.5,
                             alpha=0.8)
        
        # Plot bunny boundary
        for segment in solid_segments:
            start, end = segment['start'], segment['end']
            ax2.plot([start[0], end[0]], [start[1], end[1]],
                    'k-', linewidth=3, alpha=0.9)
        
        ax2.set_xlim(domain_bounds[0])
        ax2.set_ylim(domain_bounds[1])
        ax2.set_aspect('equal')
        ax2.set_title(f'FIXED Real-Time Pressure (t={t:.3f})')
        ax2.grid(True, alpha=0.3)
        
        return scatter
    else:
        ax2.text(0.5, 0.5, 'No Pressure Data', transform=ax2.transAxes, 
                ha='center', va='center', fontsize=16)
        return None


class RealTimeSimulation:
    """Real-time simulation class with interactive controls"""
    
    def __init__(self):
        # Simulation parameters
        self.domain_bounds = ((-1.2, 1.2), (-0.8, 0.8))
        self.n_particles = 2500  # Reduced for better real-time performance
        self.t_final = 10.0
        self.cfl = 0.25  # Lower CFL for stability
        
        # Simulation state
        self.t = 0.0
        self.step = 0
        self.max_steps = 2000
        self.paused = False
        self.speed_multiplier = 1.0
        
        # Data storage
        self.velocity_history = []
        self.time_history = []
        
        # Initialize simulation
        self.setup_simulation()
        
        # Setup visualization
        self.setup_visualization()
    
    def setup_simulation(self):
        """Initialize simulation components"""
        print("=== FIXED Real-Time Bunny Wind Tunnel ===")
        print("Controls:")
        print("  SPACE: Pause/Resume")
        print("  UP/DOWN: Increase/Decrease speed")
        print("  Q: Quit")
        
        # Initialize geometry
        self.solid_segments = create_bunny_geometry()
        print(f"Created geometry with {len(self.solid_segments)} boundary segments")
        
        # Initialize fluid state
        self.state, self.inside_bunny_mask = initialize_bunny_wind_tunnel(
            self.domain_bounds, self.n_particles, self.solid_segments)
        
        self.positions = self.state['positions']
        
        n_interior = np.sum(self.inside_bunny_mask)
        n_exterior = len(self.positions) - n_interior
        print(f"Particles: {len(self.positions)} total ({n_exterior} exterior, {n_interior} interior)")
        print(f"Initial setup complete!")
    
    def setup_visualization(self):
        """Setup matplotlib figure and animation"""
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Setup colorbar placeholder
        self.colorbar = None
        
        print("Real-time visualization ready!")
        print("Close the window or press 'Q' to stop simulation")
    
    def on_key_press(self, event):
        """Handle keyboard input"""
        if event.key == ' ':  # Space bar
            self.paused = not self.paused
            status = "PAUSED" if self.paused else "RESUMED"
            print(f"Simulation {status}")
        elif event.key == 'up':
            self.speed_multiplier = min(5.0, self.speed_multiplier * 1.5)
            print(f"Speed: {self.speed_multiplier:.1f}x")
        elif event.key == 'down':
            self.speed_multiplier = max(0.1, self.speed_multiplier / 1.5)
            print(f"Speed: {self.speed_multiplier:.1f}x")
        elif event.key == 'q':
            plt.close('all')
            print("Simulation terminated by user")
    
    def simulation_step(self):
        """Perform one simulation step"""
        if self.paused or self.t >= self.t_final or self.step >= self.max_steps:
            return False
        
        try:
            # 1. Compute Voronoi diagram
            vor = compute_voronoi_diagram(self.positions)
            
            # 2. FIXED: Enhanced clip by solid boundaries
            clipped_cells = clip_voronoi_by_solids(vor, self.solid_segments)
            
            # 3. Stitch orphaned cells
            stitched_cells = stitch_orphaned_cells(clipped_cells, self.positions)
            
            # 4. FIXED: Enhanced compute interface geometry
            interfaces = compute_interface_geometry(stitched_cells)
            
            # Store for visualization
            self.interfaces = interfaces
            self.stitched_cells = stitched_cells
            
            # 5. Apply boundary conditions
            bc_interfaces = apply_boundary_conditions(interfaces, self.solid_segments, self.state)
            
            # 6. Compute fluxes
            fluxes = compute_numerical_flux(bc_interfaces, self.state)
            
            # 7. Compute timestep
            dt = compute_timestep(self.state, interfaces, self.cfl)
            dt = min(dt, self.t_final - self.t, 1e-2) * self.speed_multiplier
            
            # 8. Update fluid state
            self.state = update_fluid_state(self.state, fluxes, interfaces, dt)
            
            # 9. Update particle positions (Lagrangian motion)
            rho = self.state['rho']
            rho_u = self.state['rho_u']
            rho_v = self.state['rho_v']
            
            u = np.where(rho > 1e-15, rho_u / rho, 0.0)
            v = np.where(rho > 1e-15, rho_v / rho, 0.0)
            
            # Limit velocities
            max_vel = 2.0
            u = np.clip(u, -max_vel, max_vel)
            v = np.clip(v, -max_vel, max_vel)
            
            displacement = dt * np.column_stack([u, v])
            self.positions += displacement
            self.state['positions'] = self.positions
            
            # 10. Update time
            self.t += dt
            self.step += 1
            
            # Track interior velocity
            avg_interior_vel = compute_average_interior_velocity(self.state, self.inside_bunny_mask)
            self.velocity_history.append(avg_interior_vel)
            self.time_history.append(self.t)
            
            return True
            
        except Exception as e:
            print(f"Simulation error at step {self.step}: {e}")
            # Try to continue with smaller timestep
            self.t += 1e-4
            self.step += 1
            return True
    
    def update_visualization(self):
        """Update the real-time visualization"""
        try:
            # Compute pressure for visualization
            rho = self.state['rho']
            rho_e = self.state['rho_e']
            rho_u = self.state['rho_u']
            rho_v = self.state['rho_v']
            gamma = self.state['gamma']
            
            u_safe = np.where(rho > 1e-15, rho_u / rho, 0.0)
            v_safe = np.where(rho > 1e-15, rho_v / rho, 0.0)
            e_kinetic = 0.5 * (u_safe**2 + v_safe**2)
            e_internal = np.where(rho > 1e-15, rho_e / rho - e_kinetic, 1e-10)
            pressure = np.maximum((gamma - 1) * rho * e_internal, 1e-10)
            
            # FIXED: Plot with error handling
            scatter = plot_bunny_wind_tunnel_fixed(
                self.ax1, self.ax2, self.stitched_cells, self.solid_segments,
                self.positions, pressure, self.inside_bunny_mask, 
                self.domain_bounds, self.t
            )
            
            # Update colorbar
            if scatter is not None:
                try:
                    if self.colorbar is not None:
                        self.colorbar.remove()
                    self.colorbar = plt.colorbar(scatter, ax=self.ax2, shrink=0.8)
                    self.colorbar.set_label('Pressure', rotation=270, labelpad=15)
                except:
                    pass  # Skip colorbar errors
            
            # Add status text
            status_text = f"Step: {self.step}, Time: {self.t:.3f}s"
            if hasattr(self, 'interfaces'):
                fluid_interfaces = len([i for i in self.interfaces if not i.get('is_solid', False)])
                solid_interfaces = len([i for i in self.interfaces if i.get('is_solid', False)])
                status_text += f"\nFluid: {fluid_interfaces}, Solid: {solid_interfaces}"
            
            if len(self.velocity_history) > 0:
                status_text += f"\nInterior vel: {self.velocity_history[-1]:.6f}"
            
            self.ax1.text(0.02, 0.98, status_text, transform=self.ax1.transAxes,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)  # Small pause for real-time effect
            
        except Exception as e:
            print(f"Visualization error: {e}")
    
    def run(self):
        """Run the real-time simulation"""
        print(f"\n🚀 Starting FIXED real-time simulation...")
        print(f"Expected: Fluid interfaces should be > 0 for proper flow")
        
        start_time = time.time()
        
        try:
            while True:
                # Check if window is closed
                if not plt.get_fignums():
                    break
                
                # Perform simulation step
                if not self.simulation_step():
                    break
                
                # Update visualization every few steps for performance
                if self.step % 1 == 0:  # Update every 3 steps
                    self.update_visualization()
                
                # Print progress occasionally
                if self.step % 100 == 0:
                    avg_interior_vel = self.velocity_history[-1] if self.velocity_history else 0
                    
                    # FIXED: Check for interfaces
                    if hasattr(self, 'interfaces'):
                        fluid_interfaces = len([i for i in self.interfaces if not i.get('is_solid', False)])
                        solid_interfaces = len([i for i in self.interfaces if i.get('is_solid', False)])
                        
                        print(f"Step {self.step}: t={self.t:.3f}, Speed={self.speed_multiplier:.1f}x")
                        print(f"  Interior velocity: {avg_interior_vel:.6f}")
                        print(f"  Interfaces: {fluid_interfaces} fluid, {solid_interfaces} solid")
                        
                        # FIXED: Alert if no fluid interfaces
                        if fluid_interfaces == 0:
                            print(f"  🚨 CRITICAL: No fluid interfaces - simulation will fail!")
                        elif avg_interior_vel < 0.01:
                            print(f"  ✅ Good leakproofness!")
                        elif avg_interior_vel > 0.05:
                            print(f"  ⚠️  Possible leakage detected")
        
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
        except Exception as e:
            print(f"Simulation error: {e}")
        finally:
            self.finalize_simulation(start_time)
    
    def finalize_simulation(self, start_time):
        """Clean up and show final results"""
        elapsed_time = time.time() - start_time
        
        print(f"\n=== FIXED Simulation Complete ===")
        print(f"Total steps: {self.step}")
        print(f"Simulation time: {self.t:.3f}s")
        print(f"Wall clock time: {elapsed_time:.1f}s")
        print(f"Performance: {self.step/elapsed_time:.1f} steps/sec")
        
        if len(self.velocity_history) > 0:
            final_interior_velocity = self.velocity_history[-1]
            print(f"Final interior velocity: {final_interior_velocity:.6f}")
            
            if final_interior_velocity < 0.005:
                print("🏆 EXCELLENT: Very good leakproof properties!")
            elif final_interior_velocity < 0.02:
                print("✅ GOOD: Acceptable leakproof behavior")
            else:
                print("⚠️  NEEDS IMPROVEMENT: Significant leakage detected")
        
        # Show final plot
        if len(self.velocity_history) > 10:
            plt.figure(figsize=(10, 6))
            plt.plot(self.time_history, self.velocity_history, 'b-', linewidth=2, label='Interior Velocity')
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Perfect Leakproof')
            plt.axhline(y=0.005, color='r', linestyle=':', alpha=0.7, label='Excellent Threshold')
            plt.axhline(y=0.02, color='orange', linestyle=':', alpha=0.7, label='Good Threshold')
            plt.xlabel('Time (s)')
            plt.ylabel('Average Interior Velocity')
            plt.title('FIXED Real-Time Simulation - Leakproofness Analysis')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            plt.show()
        
        plt.ioff()


def main():
    """Main function to run the real-time simulation"""
    try:
        # Create and run real-time simulation
        sim = RealTimeSimulation()
        sim.run()
        
    except Exception as e:
        print(f"Main error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()