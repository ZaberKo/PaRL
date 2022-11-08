# Code in this file is copied and adapted from
# https://github.com/openai/evolution-strategies-starter.

import numpy as np


class Optimizer:
    def __init__(self, theta):
        self.theta = theta
        self.t = 0

    def update(self, grad):
        self.t += 1
        step = self._compute_step(grad)
        
        ratio = np.linalg.norm(step) / np.linalg.norm(self.theta)

        # inplace update
        self.theta += step

        return self.theta, ratio

    def _compute_step(self, grad):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, theta, stepsize, momentum=0.0):
        super().__init__(theta)
        self.v = np.zeros_like(self.theta, dtype=np.float32)
        self.stepsize = stepsize
        self.momentum = momentum

    def _compute_step(self, grad):
        self.v = self.momentum * self.v + (1.0 - self.momentum) * grad
        step = -self.stepsize * self.v
        return step


class Adam(Optimizer):
    def __init__(self, theta, stepsize, beta1=0.9, beta2=0.999, epsilon=1e-08):
        super().__init__(theta)
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros_like(self.theta, dtype=np.float32)
        self.v = np.zeros_like(self.theta, dtype=np.float32)

    def _compute_step(self, grad):
        a = self.stepsize * (
            np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        )
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad * grad)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step
