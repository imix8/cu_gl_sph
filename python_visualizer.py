import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

df = pd.read_csv('positions.csv', header=None, names=['x','y','z','vmag','t','i'])
frames = int(df['t'].max()) + 1

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    ax.clear()
    pts = df[df['t'] == frame]
    x = np.clip(pts['x'], 0.0, 1.0)
    y = np.clip(pts['y'], 0.0, 1.0)
    z = np.clip(pts['z'], 0.0, 1.0)
    vmag = np.clip(pts['vmag'], 0.0, None)
    colors = plt.cm.plasma((vmag - vmag.min()) / (vmag.max() - vmag.min() + 1e-8)) # Normalize

    ax.scatter(x, y, z, c=colors, s=20)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_zlim(0,1)
    ax.set_title(f"Timestep {frame}")
    ax.view_init(elev=30, azim=-45) # ISO/ground as xy plane

ani = FuncAnimation(fig, update, frames=frames, interval=60)
plt.show()