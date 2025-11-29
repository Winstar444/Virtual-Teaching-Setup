import math
import time

class OneEuroFilter:
    def __init__(self, freq, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = None
        self.last_time = time.time()

    def smoothing(self, alpha, x, x_prev):
        return alpha * x + (1 - alpha) * x_prev

    def compute_alpha(self, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x):
        now = time.time()
        dt = now - self.last_time
        if dt != 0:
            self.freq = 1 / dt
        self.last_time = now

        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0
            return x

        dx = (x - self.x_prev) * self.freq
        alpha_d = self.compute_alpha(self.dcutoff)
        dx_hat = self.smoothing(alpha_d, dx, self.dx_prev)

        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        alpha = self.compute_alpha(cutoff)
        x_hat = self.smoothing(alpha, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat
