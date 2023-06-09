import cv2
import csv
import math
import numpy as np
import mediapipe as mp
import pyautogui as pag
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def get_dist_between_points(p1, p2):
  return math.sqrt((p1[0] - p1[0])**2 + (p1[1] - p2[1])**2)

DIAG_MOVE_TIME = 1
WIDTH, HEIGHT = pag.size()
DIAG_LEN = get_dist_between_points((0, 0), (WIDTH, HEIGHT))
CLICK_EPS = 35/DIAG_LEN # norm. pixel dist.
DRAG_TIME_EPS = 0.5 # seconds


NORM_F = DIAG_MOVE_TIME/DIAG_LEN
POSQ_LENGTH = 10
lastMouseClickTime = datetime(1970, 1, 1)
isDragging = False

Px, Ix, Dx = 0.6, 0.4, 0.2
Py, Iy, Dy = 0.6, 0.4, 0.2

print(f"DIAG_MOVE_TIME : {DIAG_MOVE_TIME}")
print(f"Width, Height : {WIDTH}, {HEIGHT}")
print(f"DIAG_LEN : {DIAG_LEN}")
print(f"CLICK_EPS : {CLICK_EPS}")


mhl_los = [([], [])]
labels = ['Index']
index_lm_series_X_time = []
posqx, posqy = [], []

# CSV writer stuff
fieldnames = ["timestamp", "index_tip_x", "index_tip_y"]
with open('mhl_data.csv', 'w') as mhl_csv_file:
   csv_writer = csv.DictWriter(mhl_csv_file, fieldnames=fieldnames, lineterminator='\n')
   csv_writer.writeheader()

def save_to_series(mhl_tup):
  mhl_los.append(mhl_tup.x)
  mhl_los.append(mhl_tup.y)
  index_lm_series_X_time.append(datetime.now())

def plot_mhl_series(frame):
  for idx, s in enumerate(mhl_los):
    plt.plot(index_lm_series_X_time, s[0], label=f'{labels[idx]} X')
    plt.plot(index_lm_series_X_time, s[1], label=f'{labels[idx]} Y')

  plt.legend(loc='upper left')

def save_series_to_csv(ts, index_x, index_y):
  with open('mhl_data.csv', 'a') as mhl_csv_file:
    csv_writer = csv.DictWriter(mhl_csv_file, fieldnames=fieldnames, lineterminator='\n')
    info = {"timestamp": ts,
            "index_tip_x": index_x,
            "index_tip_y": index_y}
    csv_writer.writerow(info)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cam = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:
  while cam.isOpened():
    success, image = cam.read()
    image = cv2.flip(image, 1)

    if not success:
        print("Ignoring empty camera frame.")
        continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        
        # print('*************')
        # print(hand_landmarks.landmark, type(hand_landmarks.landmark))
        # for idx, lm in enumerate(hand_landmarks.landmark):
        #   if idx==8:
        #     print(idx, lm.x, lm.y)

        # print('*************')

        index_tip = hand_landmarks.landmark[8]
        thumb_tip = hand_landmarks.landmark[4]
        thumb_index_dist = get_dist_between_points((thumb_tip.x, thumb_tip.y), 
                                                   (index_tip.x, index_tip.y))
        print(f"Dist {thumb_index_dist} CLICK_EPS {CLICK_EPS}")

        if thumb_index_dist <= CLICK_EPS:
          print(f"{datetime.now()} | Mouse Down | {thumb_index_dist}")
          pag.mouseDown()
          print(f"{datetime.now()} | After Mouse Down")


          # currMouseClickTime = datetime.now()
          # clickTimeDiff = (currMouseClickTime - lastMouseClickTime).total_seconds()
          # if clickTimeDiff <= DRAG_TIME_EPS:
          #   isDragging = True
          #   print(f"\t|- - -> DRAGGING : True | clickTimeDiff :{clickTimeDiff}")
          #   pag.mouseDown()

          # else:
          #   print(f"\t|-> DRAGGING : False | Left Mouse Click {currMouseClickTime}")
          #   isDragging = False
          #   pag.mouseUp()
          #   pag.click()

          # lastMouseClickTime = datetime.now()

        else:
          # isDragging = False
          print(f"Mouse Up | {thumb_index_dist}")
          pag.mouseUp()

        mouseX = int(index_tip.x*WIDTH)
        mouseY = int(index_tip.y*HEIGHT)

        # print(mouseX, mouseY)

        '''
        save_to_series(index_tip)
        
        if len(posqx) >= POSQ_LENGTH and len(posqy) >= POSQ_LENGTH:
           posqx.pop(0)
           posqy.pop(0)

        posqx.append(mouseX)
        posqy.append(mouseY)

        print(posqx)
        print(posqy)

        intg_mouseX = np.average(posqx)
        intg_mouseY = np.average(posqy)
        if len(posqx) >= 2 and len(posqy) >= 2:
           diff_mouseX = posqx[-1] - posqx[-2]
           diff_mouseY = posqy[-1] - posqy[-2]

        else:
           diff_mouseX, diff_mouseY = posqx[-1], posqy[-1]

        pid_mouseX = int(Px * mouseX + Ix * intg_mouseX + Dx * diff_mouseX)
        pid_mouseY = int(Py * mouseY + Iy * intg_mouseY + Dy * diff_mouseY)
        print(pid_mouseX, pid_mouseY)

        pag.moveTo(pid_mouseX, pid_mouseY)
        '''

        print(datetime.now(), index_tip.x, index_tip.y)
        currentMouseX, currentMouseY = pag.position()
        move_duration = get_dist_between_points((currentMouseX, currentMouseY), (mouseX, mouseY)) * NORM_F
        print(mouseX, mouseY, move_duration)
        
        if isDragging:
          pag.dragTo(mouseX, mouseY, 
                     duration=move_duration, 
                     tween=pag.easeInOutQuad, 
                     button='left')

        else:
          pag.moveTo(mouseX, mouseY, duration=move_duration, tween=pag.easeInOutQuad)
        # save_series_to_csv(datetime.now(), index_tip.x, index_tip.y)

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break

    k = cv2.waitKey(1)
    if cv2.waitKey(5) & 0xFF == 27:
        print("Escape hit, closing...")
        break

cam.release()
cv2.destroyAllWindows()


# with mp_hands.Hands(
#     model_complexity=0,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as hands:
#   while cam.isOpened():
#     success, image = cam.read()
#     if not success:
#       print("Ignoring empty camera frame.")
#       # If loading a video, use 'break' instead of 'continue'.
#       continue

#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = hands.process(image)

#     # Draw the hand annotations on the image.
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     if results.multi_hand_landmarks:
#       for hand_landmarks in results.multi_hand_landmarks:
#         mp_drawing.draw_landmarks(
#             image,
#             hand_landmarks,
#             mp_hands.HAND_CONNECTIONS,
#             mp_drawing_styles.get_default_hand_landmarks_style(),
#             mp_drawing_styles.get_default_hand_connections_style())
#     # Flip the image horizontally for a selfie-view display.
#     cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
#     if cv2.waitKey(5) & 0xFF == 27:
#       break
# cam.release()
