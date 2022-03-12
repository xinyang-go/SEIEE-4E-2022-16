import numpy as np
from scipy.linalg import expm


class MotionModel:
    def __init__(self):
        self.mass = 10
        self.At = np.array([
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0.],
        ])
        self.Bt = np.array([
            [0, 0],
            [1/self.mass, 0],
            [0, 0],
            [0, 1/self.mass],
        ])
        self.C = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0.],
        ])
        self.Qt = np.random.randn(4, 4) * 0.001
        self.Qt = self.Qt @ self.Qt.T
        self.A = None
        self.B = None
        self.Q = None
        self.R = np.ones([2, 1]) * 0.008

    def discretization(self, dt):
        n, m = self.Bt.shape
        tmp = np.hstack((self.At, self.Bt))
        tmp = np.vstack((tmp, np.zeros((m, n + m))))
        tmp = expm(tmp * dt)
        self.A = tmp[:n, :n]
        self.B = tmp[:n, n:n + m]
        self.Q = expm(self.At * dt) @ self.Qt @ expm(self.At * dt).T

    def step(self, x, u, dt, noise=False):
        self.discretization(dt)
        w_process_noise = np.random.multivariate_normal(np.zeros(4), self.Q, 1).T
        x = self.A @ x + self.B @ u + (w_process_noise if noise else 0)
        y = self.C @ x
        return x, y
