import cv2
import numpy as np
import matplotlib.pyplot as plt
import draw_lane_lines

#
# read frames from test video
#
video = cv2.VideoCapture("test2.mp4")
while(video.isOpened()):
    _, frame = video.read()
    if frame is not None:
        height, width, depth = frame.shape
        lane_lines_frame = draw_lane_lines.draw_lane_lines(frame, np.array([[(200, height), (1100, height), (550, 300)]]))
        cv2.imshow("video lane lines", lane_lines_frame)
    if (frame is None) or (cv2.waitKey(1) == ord('q')):
        break

video.release()
cv2.destroyAllWindows()
