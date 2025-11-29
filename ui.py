import cv2

def draw_hud(img, offset):
    cv2.putText(img, f"Offset: {offset}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(img, "Press S = Save  C = Clear  Q = Quit",
                (10, img.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0,0,0), 2)
