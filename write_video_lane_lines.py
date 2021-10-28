import cv2
import numpy as np
import matplotlib.pyplot as plt
import draw_lane_lines

# Find OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
video = cv2.VideoCapture("test2.mp4")

out_video = None
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
if int(major_ver) < 3 :
    fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
else :
    fps = video.get(cv2.CAP_PROP_FPS)

frame_count = 0
while(video.isOpened()):
    _, frame = video.read()
    if frame is not None:
        height, width, depth = frame.shape
        lane_lines_frame = draw_lane_lines.draw_lane_lines(frame, np.array([[(200, height), (1100, height), (550, 300)]]))
        # cv2.imshow("video lane lines", lane_lines_frame)
        if out_video is None:
            out_video = cv2.VideoWriter('lane_lines.mp4',fourcc,fps,(width,height))
        out_video.write(lane_lines_frame)
        frame_count += 1
        print("frame ", frame_count)
    else:
        break

if out_video is not None:
    out_video.release()
video.release()
cv2.destroyAllWindows()
