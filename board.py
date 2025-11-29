import numpy as np
import cv2

class Board:
    def __init__(self, width, height, bg_color=(255,255,255)):
        self.width = width
        self.height = height
        self.bg_color = bg_color
        self.canvas = np.full((height, width, 3), bg_color, dtype=np.uint8)
        self.offset_y = 0

    # Used only for dot drawing (not needed for writing anymore)
    def draw_point(self, pt, color, thickness):
        x, y = pt
        x = max(0, min(self.width - 1, x))
        y = max(0, min(self.height - 1, y))
        cv2.circle(self.canvas, (x, y), thickness // 2, color, -1)

    # 🔥 New full continuous writing line function
    def draw_line(self, start, end, color, thickness):
        if start is None or end is None:
            return

        x1, y1 = start
        x2, y2 = end

        # Clamp coordinates safely inside the canvas
        x1 = max(0, min(self.width - 1, x1))
        x2 = max(0, min(self.width - 1, x2))
        y1 = max(0, min(self.height - 1, y1))
        y2 = max(0, min(self.height - 1, y2))

        # Draw smooth connected line
        cv2.line(self.canvas, (x1, y1), (x2, y2), color, thickness)

    def scroll_up(self, step):
        self.offset_y += step
        max_offset = max(0, self.height - 540)
        self.offset_y = min(self.offset_y, max_offset)

    def get_view(self, w, h):
        top = self.offset_y
        if top + h > self.height:
            top = self.height - h
        return self.canvas[top:top + h, :w].copy()

    def save(self, path):
        cv2.imwrite(path, self.canvas)

    def clear(self):
        self.canvas[:] = self.bg_color
        self.offset_y = 0