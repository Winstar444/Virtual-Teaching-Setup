import time

class GestureDetector:
    def __init__(self, buffer_len=6, velocity_threshold=600):
        self.buffer = []
        self.buffer_len = buffer_len
        self.velocity_threshold = velocity_threshold
        self.last_time = time.time()

    def update_and_check_swipe(self, wrist_y):
        t = time.time()
        dt = t - self.last_time
        self.last_time = t

        if len(self.buffer) >= self.buffer_len:
            self.buffer.pop(0)

        self.buffer.append((wrist_y, t))

        if len(self.buffer) < 3:
            return False

        y_old, t_old = self.buffer[0]
        y_new, t_new = self.buffer[-1]

        dy = y_old - y_new
        dt = max(t_new - t_old, 1e-3)

        velocity = dy / dt

        if velocity > self.velocity_threshold:
            self.buffer.clear()
            return True

        return False
