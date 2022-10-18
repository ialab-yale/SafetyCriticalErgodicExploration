import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython import display


traj0 = np.load('optimized_trajectories_0.npy')
traj1 = np.load('optimized_trajectories_1.npy')

dt = 0.1
tfinal = len(traj0[0,:] + 1)

fig, ax = plt.subplots(1, 1, figsize = (6, 6))

def animate(i):
    ax.cla()                            # clear the previous image
    ax.plot(traj0[0,:i], traj0[1,:i], color='r')   # plot the traj for drone1
    ax.plot(traj1[0,:i], traj1[1,:i], color='g')   # plot the traj for drone2
    ax.scatter(traj0[0,i-1], traj0[1,i-1], edgecolors='r', s=10)   # plot the traj for drone1
    ax.scatter(traj1[0,i-1], traj1[1,i-1], edgecolors='g', s=10)   # plot the traj for drone2
    ax.set_xlim([0, 2])     # fix the x axis
    ax.set_ylim([0, 2])     # fix the y axis

anim = animation.FuncAnimation(fig, animate, frames = len(traj0[0,:]) + 1, interval = 1, blit = False)

writervideo = animation.FFMpegWriter(fps=30)
anim.save('corridor.mp4', writer=writervideo)

plt.show()