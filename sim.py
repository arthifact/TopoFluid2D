#!/usr/bin/env python3
"""
TopoFluid2D: Proper Wind Tunnel Simulation
Realistic compressible fluid wind tunnel with particle injection/removal
Based on proper CFD boundary conditions and thermodynamics
"""

import pygame
import numpy as np
import math
from scipy.spatial.distance import cdist
import time

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 800
FPS = 60

# Wind tunnel geometry
TUNNEL_X = 50
TUNNEL_Y = 200
INLET_WIDTH = 100
TEST_SECTION_WIDTH = 500
OUTLET_WIDTH = 150
TUNNEL_HEIGHT = 400

# Flow sections
INLET_X = TUNNEL_X
INLET_END = INLET_X + INLET_WIDTH
CONVERGENT_END = INLET_END + 100
TEST_START = CONVERGENT_END
TEST_END = TEST_START + TEST_SECTION_WIDTH
DIVERGENT_END = TEST_END + 100
OUTLET_X = DIVERGENT_END
OUTLET_END = OUTLET_X + OUTLET_WIDTH

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)
CYAN = (0, 255, 255)


class FluidParticle:
    """Fluid particle with compressible state"""

    def __init__(self, x, y, zone="ambient"):
        # Position
        self.x = x
        self.y = y

        # Conserved quantities (Euler equations)
        self.gamma = 1.4  # adiabatic index

        # Initialize based on zone
        if zone == "reservoir":
            # High pressure reservoir conditions
            self.rho = 2.0  # high density
            self.rho_u = 0.1  # small initial velocity
            self.rho_v = 0.0
            self.rho_E = 8.0  # high total energy
        elif zone == "test_section":
            # Test section design conditions
            self.rho = 1.2
            self.rho_u = 0.6  # design velocity
            self.rho_v = 0.0
            self.rho_E = 3.5
        else:  # ambient/outlet
            self.rho = 1.0
            self.rho_u = 0.2
            self.rho_v = 0.0
            self.rho_E = 2.5

        self.volume = 1.0  # Voronoi cell volume
        self.age = 0.0  # For particle lifecycle management

    @property
    def u(self):
        return self.rho_u / self.rho if self.rho > 1e-12 else 0.0

    @property
    def v(self):
        return self.rho_v / self.rho if self.rho > 1e-12 else 0.0

    @property
    def velocity_magnitude(self):
        return math.sqrt(self.u ** 2 + self.v ** 2)

    @property
    def kinetic_energy(self):
        return 0.5 * (self.u ** 2 + self.v ** 2)

    @property
    def internal_energy(self):
        return self.rho_E / self.rho - self.kinetic_energy if self.rho > 1e-12 else 0.0

    @property
    def pressure(self):
        return max(0.1, (self.gamma - 1) * self.rho * self.internal_energy)

    @property
    def sound_speed(self):
        return math.sqrt(self.gamma * self.pressure / self.rho) if self.rho > 1e-12 else 0.0

    @property
    def mach_number(self):
        return self.velocity_magnitude / self.sound_speed if self.sound_speed > 1e-6 else 0.0

    @property
    def total_pressure(self):
        """Stagnation pressure"""
        return self.pressure * (1 + 0.5 * (self.gamma - 1) * self.mach_number ** 2) ** (self.gamma / (self.gamma - 1))

    def get_color(self, mode="pressure"):
        """Get particle color based on different properties"""
        if mode == "pressure":
            p_norm = min(1.0, max(0.0, (self.pressure - 0.5) / 6.0))
            red = int(255 * p_norm)
            blue = int(255 * (1 - p_norm))
            green = int(50)
            return (red, green, blue)
        elif mode == "velocity":
            v_norm = min(1.0, self.velocity_magnitude / 3.0)
            green = int(255 * v_norm)
            red = int(128 * (1 - v_norm))
            blue = int(128 * (1 - v_norm))
            return (red, green, blue)
        elif mode == "mach":
            m_norm = min(1.0, self.mach_number / 2.0)
            if m_norm < 0.5:  # Subsonic: blue to green
                blue = int(255 * (1 - 2 * m_norm))
                green = int(255 * 2 * m_norm)
                red = 0
            else:  # Supersonic: green to red
                blue = 0
                green = int(255 * (2 - 2 * m_norm))
                red = int(255 * (2 * m_norm - 1))
            return (red, green, blue)


class WindTunnelSimulation:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("TopoFluid2D - Proper Wind Tunnel with Particle Flow")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

        # Simulation parameters
        self.dt = 0.003
        self.cfl = 0.3
        self.running = True
        self.paused = False

        # Wind tunnel parameters
        self.reservoir_pressure = 5.0  # Stagnation pressure
        self.ambient_pressure = 1.0  # Outlet pressure
        self.target_particles = 800  # Maintain this many particles

        # Visualization settings
        self.color_mode = "mach"  # "pressure", "velocity", "mach"
        self.show_vectors = True
        self.show_geometry = True
        self.particle_size = 3

        # Initialize particles
        self.particles = []
        self.initialize_flow_field()

        # Simulation state
        self.time = 0.0
        self.step_count = 0
        self.particles_injected = 0
        self.particles_removed = 0

        # Performance tracking
        self.frame_times = []

        print(f"Wind tunnel initialized with {len(self.particles)} particles")

    def get_tunnel_geometry(self, x):
        """Get tunnel height at position x (creates convergent-divergent nozzle)"""
        if x < INLET_END:
            # Inlet section - constant height
            return TUNNEL_HEIGHT
        elif x < CONVERGENT_END:
            # Convergent section - linear taper
            progress = (x - INLET_END) / (CONVERGENT_END - INLET_END)
            return TUNNEL_HEIGHT * (1.0 - 0.3 * progress)  # Contracts to 70%
        elif x < TEST_END:
            # Test section - constant height (throat)
            return TUNNEL_HEIGHT * 0.7
        elif x < DIVERGENT_END:
            # Divergent section - linear expansion
            progress = (x - TEST_END) / (DIVERGENT_END - TEST_END)
            return TUNNEL_HEIGHT * (0.7 + 0.2 * progress)  # Expands to 90%
        else:
            # Outlet section
            return TUNNEL_HEIGHT * 0.9

    def get_tunnel_bounds(self, x):
        """Get top and bottom tunnel bounds at position x"""
        height = self.get_tunnel_geometry(x)
        center_y = TUNNEL_Y + TUNNEL_HEIGHT / 2
        half_height = height / 2
        return center_y - half_height, center_y + half_height

    def initialize_flow_field(self):
        """Initialize particles throughout the tunnel"""
        self.particles = []

        # Create initial distribution
        x_positions = np.linspace(INLET_X + 20, OUTLET_END - 20, 40)

        for x in x_positions:
            y_min, y_max = self.get_tunnel_bounds(x)
            n_particles_column = int((y_max - y_min) / 25)  # Density control

            for i in range(n_particles_column):
                y = y_min + (i + 0.5) * (y_max - y_min) / n_particles_column
                y += np.random.uniform(-5, 5)  # Small random offset

                # Determine zone for initial conditions
                if x < CONVERGENT_END:
                    zone = "reservoir"
                elif x < DIVERGENT_END:
                    zone = "test_section"
                else:
                    zone = "ambient"

                particle = FluidParticle(x, y, zone)
                self.particles.append(particle)

    def inject_particles(self):
        """Inject new particles at inlet with proper distribution"""
        if len(self.particles) >= self.target_particles:
            return

        # Inject particles with realistic inlet profile
        particles_needed = max(1, int((self.target_particles - len(self.particles)) * 0.15))

        for _ in range(particles_needed):
            # Create better vertical distribution (avoid walls)
            y_min, y_max = self.get_tunnel_bounds(INLET_X + 10)
            tunnel_height = y_max - y_min

            # Use parabolic distribution (more particles in center, fewer near walls)
            # This creates a more realistic inlet velocity profile
            random_val = np.random.random()
            # Transform to parabolic distribution: more density in center
            y_normalized = 1 - 2 * abs(random_val - 0.5)  # Peak at center
            y_position = y_min + tunnel_height * 0.5 + (tunnel_height * 0.35) * y_normalized

            # Ensure we don't place particles too close to walls
            wall_buffer = 25
            y_position = max(y_min + wall_buffer, min(y_max - wall_buffer, y_position))

            # Create high-energy reservoir particle
            particle = FluidParticle(INLET_X + np.random.uniform(5, 15), y_position, "reservoir")

            # Set reservoir conditions with inlet velocity profile
            distance_from_center = abs(y_position - (y_min + y_max) * 0.5) / (tunnel_height * 0.5)
            velocity_factor = 1.0 - 0.3 * distance_from_center ** 2  # Parabolic velocity profile

            particle.rho = 1.6 + 0.4 * self.reservoir_pressure
            particle.rho_E = 1.8 * self.reservoir_pressure + 3.0
            particle.rho_u = particle.rho * 0.15 * velocity_factor  # Inlet velocity profile
            particle.rho_v = particle.rho * np.random.uniform(-0.02, 0.02)  # Small random component

            self.particles.append(particle)
            self.particles_injected += 1

    def remove_particles(self):
        """Remove particles that exit the tunnel or accumulate at walls"""
        particles_to_remove = []

        for i, particle in enumerate(self.particles):
            # Remove if past outlet
            if particle.x > OUTLET_END + 10:
                particles_to_remove.append(i)
            # Remove if goes upstream too far
            elif particle.x < INLET_X - 30:
                particles_to_remove.append(i)
            # Remove particles that stick to walls for too long
            elif particle.age > 5.0:  # Remove old stagnant particles
                y_min, y_max = self.get_tunnel_bounds(particle.x)
                distance_to_wall = min(particle.y - y_min, y_max - particle.y)
                if distance_to_wall < 20 and particle.velocity_magnitude < 0.1:
                    particles_to_remove.append(i)
            # Remove if way outside tunnel bounds
            else:
                y_min, y_max = self.get_tunnel_bounds(particle.x)
                if particle.y < y_min - 30 or particle.y > y_max + 30:
                    particles_to_remove.append(i)

        # Remove particles (reverse order to maintain indices)
        for i in reversed(particles_to_remove):
            self.particles_removed += 1
            del self.particles[i]

    def compute_voronoi_volumes(self):
        """Compute approximate Voronoi cell volumes"""
        if len(self.particles) < 2:
            return

        positions = np.array([[p.x, p.y] for p in self.particles])
        distances = cdist(positions, positions)

        for i, particle in enumerate(self.particles):
            neighbor_distances = distances[i]
            neighbor_distances[i] = np.inf

            k_nearest = min(6, len(neighbor_distances) - 1)
            if k_nearest > 0:
                nearest_distances = np.partition(neighbor_distances, k_nearest)[:k_nearest]
                avg_neighbor_dist = np.mean(nearest_distances)
                particle.volume = max(50, avg_neighbor_dist ** 2)

    def compute_fluxes(self):
        """Compute numerical fluxes between neighboring particles"""
        if len(self.particles) < 2:
            return [[0, 0, 0, 0] for _ in range(len(self.particles))]

        positions = np.array([[p.x, p.y] for p in self.particles])
        distances = cdist(positions, positions)

        flux_updates = [[0, 0, 0, 0] for _ in range(len(self.particles))]
        interaction_radius = 35.0
        flux_strength = 0.15

        for i, particle_i in enumerate(self.particles):
            neighbor_indices = np.where((distances[i] > 0) & (distances[i] < interaction_radius))[0]

            for j in neighbor_indices:
                particle_j = self.particles[j]

                dx = particle_j.x - particle_i.x
                dy = particle_j.y - particle_i.y
                distance = math.sqrt(dx * dx + dy * dy)

                if distance < 1e-6:
                    continue

                nx = dx / distance
                ny = dy / distance

                # Distance-weighted interface area
                interface_area = max(8.0, 40.0 - distance)

                # Compute flux
                flux = self.kurganov_tadmor_flux(particle_i, particle_j, nx, ny)

                weight = interface_area * flux_strength / max(particle_i.volume, particle_j.volume)

                # Apply conservation
                for k in range(4):
                    flux_updates[i][k] -= flux[k] * weight
                    flux_updates[j][k] += flux[k] * weight

        return flux_updates

    def kurganov_tadmor_flux(self, left, right, nx, ny):
        """Kurganov-Tadmor numerical flux computation"""
        u_L = left.u * nx + left.v * ny
        u_R = right.u * nx + right.v * ny

        c_L = left.sound_speed
        c_R = right.sound_speed
        a = max(abs(u_L) + c_L, abs(u_R) + c_R, 0.2)

        flux_L = self.euler_flux(left, nx, ny)
        flux_R = self.euler_flux(right, nx, ny)

        return [
            0.5 * (flux_L[0] + flux_R[0] - a * (right.rho - left.rho)),
            0.5 * (flux_L[1] + flux_R[1] - a * (right.rho_u - left.rho_u)),
            0.5 * (flux_L[2] + flux_R[2] - a * (right.rho_v - left.rho_v)),
            0.5 * (flux_L[3] + flux_R[3] - a * (right.rho_E - left.rho_E))
        ]

    def euler_flux(self, particle, nx, ny):
        """Compute Euler flux in normal direction"""
        u_n = particle.u * nx + particle.v * ny

        return [
            particle.rho * u_n,
            particle.rho * particle.u * u_n + particle.pressure * nx,
            particle.rho * particle.v * u_n + particle.pressure * ny,
            (particle.rho_E + particle.pressure) * u_n
        ]

    def apply_boundary_conditions(self):
        """Apply realistic wind tunnel boundary conditions"""
        for particle in self.particles:
            # Update particle age
            particle.age += self.dt

            # Inlet region - maintain reservoir conditions
            if particle.x < INLET_END:
                # Accelerate particles based on pressure gradient
                pressure_gradient = (self.reservoir_pressure - particle.pressure) * 0.02
                particle.rho_u += particle.rho * pressure_gradient

                # Maintain high stagnation conditions
                if particle.pressure < self.reservoir_pressure * 0.8:
                    energy_addition = (self.reservoir_pressure - particle.pressure) * 0.1
                    particle.rho_E += energy_addition

                # Prevent backflow
                if particle.rho_u < 0:
                    particle.rho_u = abs(particle.rho_u) * 0.1

            # Test section - maintain flow quality
            elif CONVERGENT_END < particle.x < TEST_END:
                # Small corrections to maintain steady flow
                target_velocity = 1.2  # Design test section velocity
                velocity_error = target_velocity - particle.u
                particle.rho_u += particle.rho * velocity_error * 0.005

            # Outlet region - create suction effect
            elif particle.x > DIVERGENT_END:
                # Pressure recovery in diffuser
                if particle.pressure > self.ambient_pressure:
                    pressure_diff = particle.pressure - self.ambient_pressure
                    particle.rho_E -= pressure_diff * 0.03
                    particle.rho_E = max(particle.rho_E, 1.5)

                # Maintain exit velocity
                if particle.rho_u < particle.rho * 0.2:
                    particle.rho_u += particle.rho * 0.01

            # Wall boundary conditions - proper no-slip treatment
            y_min, y_max = self.get_tunnel_bounds(particle.x)
            wall_margin = 15

            # Bottom wall
            if particle.y < y_min + wall_margin:
                # Apply no-slip condition and boundary layer effects
                distance_from_wall = particle.y - y_min
                wall_factor = max(0.1, distance_from_wall / wall_margin)  # Velocity reduction near wall

                particle.y = y_min + wall_margin  # Keep away from wall
                particle.rho_v = -abs(particle.rho_v) * 0.7  # Reflect with strong damping
                particle.rho_u *= wall_factor  # Reduce streamwise velocity (boundary layer effect)

                # Add wall friction drag
                particle.rho_u *= 0.995

            # Top wall
            elif particle.y > y_max - wall_margin:
                distance_from_wall = y_max - particle.y
                wall_factor = max(0.1, distance_from_wall / wall_margin)

                particle.y = y_max - wall_margin
                particle.rho_v = abs(particle.rho_v) * 0.7  # Reflect with strong damping
                particle.rho_u *= wall_factor  # Boundary layer effect

                # Add wall friction drag
                particle.rho_u *= 0.995

    def step_simulation(self):
        """Perform one simulation timestep"""
        if self.paused:
            return

        # Particle management
        self.inject_particles()
        self.remove_particles()

        if len(self.particles) == 0:
            return

        # Compute volumes and fluxes
        self.compute_voronoi_volumes()
        flux_updates = self.compute_fluxes()

        # Compute stable timestep
        dt = self.compute_timestep()

        # Update particle states
        for i, particle in enumerate(self.particles):
            particle.rho += flux_updates[i][0] * dt
            particle.rho_u += flux_updates[i][1] * dt
            particle.rho_v += flux_updates[i][2] * dt
            particle.rho_E += flux_updates[i][3] * dt

            # Ensure physical positivity
            particle.rho = max(particle.rho, 0.2)
            particle.rho_E = max(particle.rho_E, 1.0)

        # Apply boundary conditions
        self.apply_boundary_conditions()

        # Update particle positions (convection)
        position_scale = 80
        for particle in self.particles:
            particle.x += particle.u * dt * position_scale
            particle.y += particle.v * dt * position_scale

            # Add numerical diffusion for stability
            diffusion = 0.002
            particle.rho_u *= (1 - diffusion)
            particle.rho_v *= (1 - diffusion)

        self.time += dt
        self.step_count += 1

    def compute_timestep(self):
        """Compute stable timestep using CFL condition"""
        max_speed = 0.5

        for particle in self.particles:
            speed = particle.velocity_magnitude + particle.sound_speed
            max_speed = max(max_speed, speed)

        min_cell_size = 20  # Conservative estimate
        return min(self.dt, self.cfl * min_cell_size / max_speed)

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.initialize_flow_field()
                    self.time = 0
                    self.step_count = 0
                    self.particles_injected = 0
                    self.particles_removed = 0
                elif event.key == pygame.K_1:
                    self.color_mode = "pressure"
                elif event.key == pygame.K_2:
                    self.color_mode = "velocity"
                elif event.key == pygame.K_3:
                    self.color_mode = "mach"
                elif event.key == pygame.K_v:
                    self.show_vectors = not self.show_vectors
                elif event.key == pygame.K_g:
                    self.show_geometry = not self.show_geometry
                elif event.key == pygame.K_UP:
                    self.reservoir_pressure = min(8.0, self.reservoir_pressure + 0.3)
                elif event.key == pygame.K_DOWN:
                    self.reservoir_pressure = max(1.5, self.reservoir_pressure - 0.3)
                elif event.key == pygame.K_ESCAPE:
                    self.running = False

    def draw_tunnel_geometry(self):
        """Draw the wind tunnel geometry"""
        if not self.show_geometry:
            return

        # Draw tunnel walls
        wall_points_top = []
        wall_points_bottom = []

        for x in range(INLET_X, OUTLET_END, 5):
            y_min, y_max = self.get_tunnel_bounds(x)
            wall_points_top.append((x, y_min))
            wall_points_bottom.append((x, y_max))

        if len(wall_points_top) > 1:
            pygame.draw.lines(self.screen, GRAY, False, wall_points_top, 3)
            pygame.draw.lines(self.screen, GRAY, False, wall_points_bottom, 3)

        # Draw section labels and boundaries
        sections = [
            (INLET_X, "INLET", GREEN),
            (CONVERGENT_END, "THROAT", YELLOW),
            (TEST_END, "DIFFUSER", CYAN),
            (OUTLET_X, "OUTLET", BLUE)
        ]

        for x, label, color in sections:
            y_min, y_max = self.get_tunnel_bounds(x)
            pygame.draw.line(self.screen, color, (x, y_min - 20), (x, y_max + 20), 2)

            text = self.small_font.render(label, True, color)
            self.screen.blit(text, (x - 20, y_min - 40))

    def draw_particles(self):
        """Draw all particles with flow visualization"""
        for particle in self.particles:
            color = particle.get_color(self.color_mode)

            # Draw particle
            pygame.draw.circle(self.screen, color,
                               (int(particle.x), int(particle.y)), self.particle_size)

            # Draw velocity vectors
            if self.show_vectors and particle.velocity_magnitude > 0.2:
                scale = 15
                end_x = particle.x + particle.u * scale
                end_y = particle.y + particle.v * scale

                # Color vector by Mach number
                if particle.mach_number > 1.0:
                    vector_color = RED  # Supersonic
                elif particle.mach_number > 0.8:
                    vector_color = YELLOW  # Transonic
                else:
                    vector_color = WHITE  # Subsonic

                pygame.draw.line(self.screen, vector_color,
                                 (particle.x, particle.y), (end_x, end_y), 1)

    def draw_hud(self):
        """Draw heads-up display"""
        y = 10
        line_height = 22

        # Title and status
        title = self.font.render("TopoFluid2D - Proper Wind Tunnel", True, WHITE)
        self.screen.blit(title, (10, y))
        y += line_height

        status = "PAUSED" if self.paused else "RUNNING"
        status_text = self.small_font.render(
            f"Status: {status} | Particles: {len(self.particles)} | Time: {self.time:.1f}s", True, WHITE)
        self.screen.blit(status_text, (10, y))
        y += line_height

        # Flow statistics
        if len(self.particles) > 0:
            avg_pressure = sum(p.pressure for p in self.particles) / len(self.particles)
            avg_velocity = sum(p.velocity_magnitude for p in self.particles) / len(self.particles)
            max_mach = max(p.mach_number for p in self.particles)

            stats = [
                f"Reservoir P₀: {self.reservoir_pressure:.1f} (↑/↓ to adjust)",
                f"Avg Pressure: {avg_pressure:.2f}",
                f"Avg Velocity: {avg_velocity:.2f}",
                f"Max Mach: {max_mach:.2f}",
                f"Injected: {self.particles_injected} | Removed: {self.particles_removed}"
            ]

            for stat in stats:
                text = self.small_font.render(stat, True, WHITE)
                self.screen.blit(text, (10, y))
                y += 18

        # Controls
        y += 10
        controls = [
            "Controls:",
            "SPACE - Pause/Resume",
            "R - Reset tunnel",
            "1/2/3 - Color: Pressure/Velocity/Mach",
            "V - Toggle vectors | G - Toggle geometry",
            "↑/↓ - Reservoir pressure",
            "ESC - Exit"
        ]

        for i, control in enumerate(controls):
            color = YELLOW if i == 0 else WHITE
            text = self.small_font.render(control, True, color)
            self.screen.blit(text, (10, y))
            y += 16

        # Legend for color modes
        legend_x = SCREEN_WIDTH - 200
        legend_y = 10

        legend_title = self.small_font.render(f"Color: {self.color_mode.title()}", True, YELLOW)
        self.screen.blit(legend_title, (legend_x, legend_y))
        legend_y += 20

        if self.color_mode == "mach":
            legend_items = [
                ("Subsonic (M<1)", BLUE),
                ("Transonic (M≈1)", GREEN),
                ("Supersonic (M>1)", RED)
            ]
        elif self.color_mode == "pressure":
            legend_items = [
                ("Low Pressure", BLUE),
                ("High Pressure", RED)
            ]
        else:  # velocity
            legend_items = [
                ("Slow", RED),
                ("Fast", GREEN)
            ]

        for text, color in legend_items:
            pygame.draw.circle(self.screen, color, (legend_x, legend_y + 8), 6)
            text_surface = self.small_font.render(text, True, WHITE)
            self.screen.blit(text_surface, (legend_x + 15, legend_y))
            legend_y += 18

        # Performance
        if len(self.frame_times) > 0:
            avg_frame_time = sum(self.frame_times[-30:]) / len(self.frame_times[-30:])
            fps_actual = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            perf_text = self.small_font.render(f"FPS: {fps_actual:.1f}", True, WHITE)
            self.screen.blit(perf_text, (SCREEN_WIDTH - 100, SCREEN_HEIGHT - 30))

    def run(self):
        """Main simulation loop"""
        print("Starting proper wind tunnel simulation...")
        print("Watch particles flow from inlet to outlet!")
        print("Controls: SPACE=pause, ↑/↓=pressure, 1/2/3=colors, V=vectors")

        while self.running:
            frame_start = time.time()

            self.handle_events()
            self.step_simulation()

            # Draw everything
            self.screen.fill(BLACK)
            self.draw_tunnel_geometry()
            self.draw_particles()
            self.draw_hud()

            pygame.display.flip()

            # Performance tracking
            frame_time = time.time() - frame_start
            self.frame_times.append(frame_time)
            if len(self.frame_times) > 60:
                self.frame_times.pop(0)

            self.clock.tick(FPS)

        pygame.quit()


def main():
    """Run the wind tunnel simulation"""
    try:
        sim = WindTunnelSimulation()
        sim.run()
    except Exception as e:
        print(f"Error: {e}")
        pygame.quit()


if __name__ == "__main__":
    main()