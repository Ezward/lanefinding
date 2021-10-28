# image_lane_lines.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import draw_lane_lines

# frame = cv2.imread('525_cam-image_array_.jpg')
# frame = cv2.imread('592_cam-image_array_.jpg')
# frame = cv2.imread('622_cam-image_array_.jpg')
# frame = cv2.imread('628_cam-image_array_.jpg')
# frame = cv2.imread('4370_cam-image_array_.jpg')
# frame = cv2.imread('4690_cam-image_array_.jpg')   # right turn
frame = cv2.imread('5885_cam-image_array_.jpg')
height, width, depth = frame.shape
lane_lines_frame = draw_lane_lines.draw_lane_lines(frame, np.array([[(0, height - 25), (0, height), (width, height), (width, height - 25), (width - 40, 45), (40, 45)]]))
# lane_lines_frame = draw_lane_lines.draw_lane_lines(frame, np.array([[(0, height - 35), (width, height - 35), (width - 40, 45), (40, 45)]]))
cv2.imshow("Image lane lines", lane_lines_frame)

while True:
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
