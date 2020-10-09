import os
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import single_pendulum_sim as tpsim
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from os import path


class singlePendulumEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self, viewer=False):
        self.viewer = viewer
        self.max_force = 10.
        self.sim_fini = False
        self.nStep = 0
        # simulation parameters
        self.cntr_f = 400
        self.h = 1. / 100.
        self.l = 1.0
        self.mass = 1.0
        # initialize states
        self.q = np.array([[0.], [-self.l / 2.0], [-np.pi / 2.]])
        self.qd = np.array([[0.], [0.], [0.]])

        self.outValues = np.array([self.q[0][0], self.q[1][0], self.q[2][0],
                                   self.qd[0][0], self.qd[1][0], self.qd[2][0]], dtype=np.float32)
        # we have 1 action: f
        self.action_space = spaces.Box(low=-self.max_force, high=self.max_force, shape=(1,), dtype=np.float32)
        # we have 17 number of observations: all the system states
        self.observation_space = spaces.Box(low=np.array([-4., -0.5, -2.*np.pi, -100., -100., -100.]),
                                            high=np.array([4., 0.5, 2.*np.pi, 100., 100., 100.]), dtype=np.float32)

        # Recording variables
        self.q1_h = np.zeros(self.cntr_f)
        self.q2_h = np.zeros(self.cntr_f)
        self.q3_h = np.zeros(self.cntr_f)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        if self.viewer == True:
            self.q1_h[self.nStep,] = self.q[0][0]
            self.q2_h[self.nStep,] = self.q[1][0]
            self.q3_h[self.nStep,] = self.q[2][0]

        self.q,self.qd = tpsim.step_sim(u[0]*50.,self.q,self.qd,self.mass,self.l,self.h)

        np.array([self.q[0][0], self.q[1][0], self.q[2][0], self.qd[0][0], self.qd[1][0], self.qd[2][0]], dtype=np.float32)

        b = 0.
        if (self.q[0][0] > 4) or (self.q[0][0] < -4):
            b = 1.
            self.sim_fini = True

        costs = 4.*(np.sin(self.q[2][0]) - 1.0) - 0.1*self.q[0][0]**2 - 0.005*u[0]**2 - 100.*b

        self.nStep += 1

        if self.viewer == True and self.nStep == self.cntr_f:
            self._visualize()

        return self._get_obs(), costs, self.sim_fini, {}

    def reset(self):
        self.nStep = 0
        self.q = np.array([[0.], [-self.l / 2.0], [-np.pi / 2.]])
        self.qd = np.array([[0.], [0.], [0.]])
        self.sim_fini = False
        self.outValues = np.array([self.q[0][0], self.q[1][0], self.q[2][0],
                                   self.qd[0][0], self.qd[1][0], self.qd[2][0]], dtype=np.float32)
        return self._get_obs()

    def _get_obs(self):
        return np.array([self.q[0][0], self.q[1][0], (self.q[2][0]),
                         self.qd[0][0], self.qd[1][0], self.qd[2][0]], dtype=np.float32)

    def render(self, mode='human'):
        pass

    def _visualize(self):
        # Animation
        fig = plt.figure(figsize=(8, 8), facecolor='w')
        self.ax = fig.add_subplot(1, 1, 1)
        plt.rcParams['font.size'] = 10

        self.ax.set_aspect('equal', 'datalim')

        ani = animation.FuncAnimation(fig, self._aniFunc, frames=400, repeat=True, interval=0, blit=True, save_count=100)
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.autoscale(False)
        plt.show()

    def _aniFunc(self, i):
        # plotting the string
        x1 = self.q1_h[i,] - (self.l / 2.) * np.cos(self.q3_h[i,])
        x2 = self.q1_h[i,] + (self.l / 2.) * np.cos(self.q3_h[i,])
        y1 = self.q2_h[i,] - (self.l / 2.) * np.sin(self.q3_h[i,])
        y2 = self.q2_h[i,] + (self.l / 2.) * np.sin(self.q3_h[i,])
        ln1, = self.ax.plot([x1, x2], [y1, y2], color='k', lw=4)

        ln2, = self.ax.plot([-4.,4.],[0.,0.], color='k', lw=1)

        # plotting the bob
        bob, = self.ax.plot(self.q1_h[i,] - (self.l / 2.) * np.cos(self.q3_h[i,]), self.q2_h[i,] -
                              (self.l / 2.) * np.sin(self.q3_h[i,]), 'o', markersize=8, color='b')

        t = i * self.h
        tm = self.ax.text(-2.75, 2.5, 'Time = %.1fs' % t)
        lns = ([ln1, ln2, bob, tm])

        time.sleep(0.01)

        return lns

    def close(self):
        pass

