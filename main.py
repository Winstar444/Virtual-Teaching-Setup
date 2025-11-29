import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector
from filterpy.kalman import KalmanFilter
from collections import deque

# -------------------- Config --------------------
USE_DSHOW = True
CAM_IDX = 0
DETECTION_CON = 0.55
MAX_HANDS = 1

BASE_THICK = 5
MAX_SPEED_EXTRA = 6
CATMULL_POINTS = 12
BUFFER_MAX = 48
ERASER_R = 36
COLOR_GESTURE_DIST = 40

PAGE_SWIPE_THRESHOLD = 100
PAGE_SWIPE_COOLDOWN = 0.8
UNDO_LIMIT = 25

# -------------------- Helpers --------------------
def open_cam(idx=CAM_IDX):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("❌ No camera detected")
    return cap

def create_kf():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([0.,0.,0.,0.])
    kf.F = np.array([[1,0,1,0],
                     [0,1,0,1],
                     [0,0,1,0],
                     [0,0,0,1]])
    kf.H = np.array([[1,0,0,0],[0,1,0,0]])
    kf.P *= 8
    kf.R *= 0.1
    kf.Q *= 0.001
    return kf

def catmull(p0,p1,p2,p3,count=CATMULL_POINTS):
    pts=[]
    for i in range(count):
        t=i/(count-1)
        t2=t*t
        t3=t2*t
        x=0.5*((2*p1[0])+(-p0[0]+p2[0])*t+(2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2+(-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3)
        y=0.5*((2*p1[1])+(-p0[1]+p2[1])*t+(2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2+(-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3)
        pts.append((int(x),int(y)))
    return pts

def draw_catmull(buf, canvas, color, base_thick):
    if len(buf) < 4:
        return
    p0,p1,p2,p3 = buf[-4:]
    smooth = catmull(p0,p1,p2,p3)
    for i in range(1, len(smooth)):
        x1,y1 = smooth[i-1]
        x2,y2 = smooth[i]
        speed = math.hypot(x2-x1, y2-y1)
        dyn = base_thick + int(min(speed / 6, MAX_SPEED_EXTRA))
        cv2.line(canvas, (x1,y1), (x2,y2), color, dyn, cv2.LINE_AA)

# -------------------- Page class --------------------
class Page:
    def __init__(self, h,w):
        self.canvas = np.zeros((h,w,3), dtype=np.uint8)
        self.undo = deque(maxlen=UNDO_LIMIT)
        self.redo = deque(maxlen=UNDO_LIMIT)

    def push(self):
        self.undo.append(self.canvas.copy())
        self.redo.clear()

    def undo_page(self):
        if self.undo:
            self.redo.append(self.canvas.copy())
            self.canvas = self.undo.pop()
            return True
        return False

    def redo_page(self):
        if self.redo:
            self.undo.append(self.canvas.copy())
            self.canvas = self.redo.pop()
            return True
        return False

# -------------------- Init --------------------
cap = open_cam()
ret, f0 = cap.read()
if not ret:
    raise RuntimeError("Camera failed")

f0 = cv2.flip(f0,1)
H,W = f0.shape[:2]

detector = HandDetector(maxHands=1, detectionCon=DETECTION_CON)
kf = create_kf()

pages = [Page(H,W)]
cur_page = 0

stroke_buf = []
current_color = (0,0,255)

palm_prev_x = None
palm_start_x = None
last_page_switch = time.time()

print("READY → Raise Index Finger to Write")

# -------------------- Main Loop --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame,1)
    frame_disp = frame.copy()

    # Debug draw ON so you see landmarks
    hands, _ = detector.findHands(frame, draw=True)

    index_pos = None
    write = False
    erase = False

    if hands:
        h = hands[0]
        lm = h["lmList"]
        fingers = detector.fingersUp(h)

        # Kalman smoothing
        ix, iy = lm[8][0], lm[8][1]
        mx, my = lm[12][0], lm[12][1]
        tx, ty = lm[4][0], lm[4][1]

        kf.predict()
        kf.update(np.array([ix, iy]))
        kx, ky = int(kf.x[0]), int(kf.x[1])
        index_pos = (kx, ky)

        print("Fingers:", fingers)  # DEBUG: See gesture in terminal

        # -------- write (easy mode) --------
        if fingers == [0,1,0,0,0]:
            write = True

        # -------- erase --------
        if math.hypot(ix - mx, iy - my) < 40:
            erase = True
            write = False

        # -------- color gestures --------
        if math.hypot(tx - lm[8][0], ty - lm[8][1]) < COLOR_GESTURE_DIST:
            current_color = (0,0,255)
        elif math.hypot(tx - lm[12][0], ty - lm[12][1]) < COLOR_GESTURE_DIST:
            current_color = (255,0,0)
        elif math.hypot(tx - lm[16][0], ty - lm[16][1]) < COLOR_GESTURE_DIST:
            current_color = (0,255,0)
        elif math.hypot(tx - lm[20][0], ty - lm[20][1]) < COLOR_GESTURE_DIST:
            current_color = (0,0,0)

        # -------- page swipe --------
        palm_x = int((lm[0][0] + lm[5][0] + lm[9][0] + lm[13][0] + lm[17][0]) / 5)

        if fingers == [1,1,1,1,1]:  # open palm
            if palm_prev_x is None:
                palm_prev_x = palm_x
                palm_start_x = palm_x
            else:
                if abs(palm_x - palm_start_x) > PAGE_SWIPE_THRESHOLD:
                    if (time.time() - last_page_switch) > PAGE_SWIPE_COOLDOWN:
                        if palm_x - palm_start_x > 0:
                            cur_page += 1
                            if cur_page >= len(pages):
                                pages.append(Page(H,W))
                            print("Page →",cur_page+1)
                        else:
                            if cur_page > 0:
                                cur_page -= 1
                                print("Page →",cur_page+1)
                        last_page_switch = time.time()
                    palm_prev_x = None
                    palm_start_x = None
        else:
            palm_prev_x = None
            palm_start_x = None

    # ------------------ DRAWING ------------------
    page = pages[cur_page]

    if write and index_pos:
        if len(stroke_buf) == 0:
            page.push()

        stroke_buf.append(index_pos)

        if len(stroke_buf) > BUFFER_MAX:
            stroke_buf = stroke_buf[-BUFFER_MAX:]

        if len(stroke_buf) >= 4:
            draw_catmull(stroke_buf, page.canvas, current_color, BASE_THICK)

    elif erase and index_pos:
        if len(stroke_buf) == 0:
            page.push()
        cv2.circle(page.canvas, index_pos, ERASER_R, (0,0,0), -1)
        stroke_buf = []

    else:
        if len(stroke_buf) >= 4:
            draw_catmull(stroke_buf, page.canvas, current_color, BASE_THICK)
        stroke_buf = []

    # ------------------ DISPLAY ------------------
    out = cv2.addWeighted(frame_disp, 1, page.canvas, 0.7, 0)
    cv2.putText(out, f"Page {cur_page+1}", (15,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255),1)

    cv2.imshow("Virtual Notebook (WORKING VERSION)", out)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('c'):
        page.push()
        page.canvas[:] = 0
    if key == ord('s'):
        cv2.imwrite(f"page_{cur_page+1}.png", page.canvas)
        print("Saved")

cap.release()
cv2.destroyAllWindows()
