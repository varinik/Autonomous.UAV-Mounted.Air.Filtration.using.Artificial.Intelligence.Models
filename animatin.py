import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# ========== CONFIG ==========
RADIUS = 10
CENTER = (15, 15)
Z_HEIGHT = 10
STEPS = 100

# Integer circle path in XY
def integer_circle_path(radius, center, steps):
    angles = np.linspace(0, 2 * np.pi, steps)
    x = np.round(center[0] + radius * np.cos(angles)).astype(int)
    y = np.round(center[1] + radius * np.sin(angles)).astype(int)
    coords = list(set(zip(x, y)))
    coords = sorted(coords, key=lambda p: np.arctan2(p[1]-center[1], p[0]-center[0]))
    return coords

# Source path
source_path_2d = integer_circle_path(RADIUS, CENTER, STEPS)
source_path = [(x, y, Z_HEIGHT) for (x, y) in source_path_2d]

# Predicted source path (lagged & noisy)
def predicted_source(i):
    lag = 5
    idx = max(0, i - lag)
    sx, sy, sz = source_path[idx]
    noise = np.random.uniform(-1.0, 1.0, size=3)
    return sx + noise[0], sy + noise[1], sz + noise[2]

# Drone motion in 3D (orbiting)
def drone_positions(i):
    t = i / 10.0
    drones = []
    for j in range(3):
        angle = t + (j * 2*np.pi / 3)
        dx = CENTER[0] + (RADIUS - 2) * np.cos(angle + j)
        dy = CENTER[1] + (RADIUS - 2) * np.sin(angle + j)
        dz = Z_HEIGHT + 3 * np.sin(angle * 0.5 + j)
        drones.append((dx, dy, dz))
    return drones

# ========== SETUP PLOT ==========
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0, 30)
ax.set_ylim(0, 30)
ax.set_zlim(0, 20)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Plot handles
true_src, = ax.plot([], [], [], 'ro', label='True Source')
pred_src, = ax.plot([], [], [], 'bo', label='Predicted Source')
drone_dots = [ax.plot([], [], [], 'gs')[0] for _ in range(3)]
ax.legend()

# ========== ANIMATION FUNCTIONS ==========
def init():
    true_src.set_data([], [])
    true_src.set_3d_properties([])
    pred_src.set_data([], [])
    pred_src.set_3d_properties([])
    for dot in drone_dots:
        dot.set_data([], [])
        dot.set_3d_properties([])
    return [true_src, pred_src] + drone_dots

def update(frame):
    if frame >= len(source_path):
        return

    # True source
    sx, sy, sz = source_path[frame]
    true_src.set_data(sx, sy)
    true_src.set_3d_properties(sz)

    # Predicted source
    px, py, pz = predicted_source(frame)
    pred_src.set_data(px, py)
    pred_src.set_3d_properties(pz)

    # Drones
    drones = drone_positions(frame)
    for dot, (dx, dy, dz) in zip(drone_dots, drones):
        dot.set_data(dx, dy)
        dot.set_3d_properties(dz)

    return [true_src, pred_src] + drone_dots

ani = FuncAnimation(fig, update, frames=len(source_path), init_func=init, blit=False, interval=200)
plt.show()
