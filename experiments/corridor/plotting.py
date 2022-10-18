import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


traj0 = np.load('optimized_trajectories_0.npy')
traj1 = np.load('optimized_trajectories_1.npy')

dt = 0.1
tfinal = len(traj0[0,:] + 1)

fig, ax = plt.subplots(1, 1, figsize = (6, 6))

def animate(i):
    ax.cla()                            # clear the previous image
    ax.plot(traj0[0,:i], traj0[1,:i])   # plot the traj for drone1
    ax.plot(traj1[0,:i], traj1[1,:i])   # plot the traj for drone2
    ax.set_xlim([0, 2])     # fix the x axis
    ax.set_ylim([0, 2])     # fix the y axis

anim = animation.FuncAnimation(fig, animate, frames = len(traj0[0,:]) + 1, interval = 1, blit = False)
plt.show()