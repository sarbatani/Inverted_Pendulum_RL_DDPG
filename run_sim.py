
import single_pendulum_sim as tpsim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 100 seconds at h=0.01
cntr_f = 400
h = 1. / 100.
l = 1.0
mass = 1.0

q = np.array([[0.], [-l/2.0], [-np.pi/2.]])
qd = np.array([[0.], [0.], [0.]])

# Animation
fig = plt.figure(figsize=(8, 8), facecolor='w')
ax = fig.add_subplot(1, 1, 1)
plt.rcParams['font.size'] = 10
lns = []
t = 0.

# Recording variables
q1_h = np.zeros(cntr_f)
q2_h = np.zeros(cntr_f)
q3_h = np.zeros(cntr_f)

# Simulation
for i in range(cntr_f):
    q1_h[i,] = q[0][0]
    q2_h[i,] = q[1][0]
    q3_h[i,] = q[2][0]

    q,qd = tpsim.step_sim(-10.0,q,qd,mass,l,h)

def aniFunc(i):
    # plotting the string/chord
    x1 = q1_h[i,] - (l / 2.) * np.cos(q3_h[i,])
    x2 = q1_h[i,] + (l / 2.) * np.cos(q3_h[i,])
    y1 = q2_h[i,] - (l / 2.) * np.sin(q3_h[i,])
    y2 = q2_h[i,] + (l / 2.) * np.sin(q3_h[i,])
    ln, = ax.plot([x1, x2], [y1, y2], color='k', lw=2)

    # plotting the bob
    bob, = ax.plot(q1_h[i,] + (l/2.) * np.cos(q3_h[i,]), q2_h[i,] + (l/2.) * np.sin(q3_h[i,]), 'o', markersize=5, color='r')

    t = i * h
    tm = ax.text(-2.75, 2.5, 'Time = %.1fs' % t)
    lns = ([ln, bob, tm])

    return lns


ax.set_aspect('equal', 'datalim')

Writer = animation.writers['ffmpeg']
writer = Writer(fps=60, metadata=dict(artist='Siamak'), bitrate=1800)

# saving the animation
ani = animation.FuncAnimation(fig, aniFunc, frames=1000, interval=10, blit=True, save_count=50)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.autoscale(False)
plt.show()

# ani = animation.ArtistAnimation(fig, lns, interval=10)
# fn = 'Pendulum_Animation'
# ani.save(fn+'.mp4',writer=writer)
