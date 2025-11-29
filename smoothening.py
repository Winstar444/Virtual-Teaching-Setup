class Smoothener:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.x = None
        self.y = None

    def update(self, x_new, y_new):
        if self.x is None:
            self.x, self.y = x_new, y_new
            return self.x, self.y

        self.x = self.alpha * x_new + (1 - self.alpha) * self.x
        self.y = self.alpha * y_new + (1 - self.alpha) * self.y

        return self.x, self.y
