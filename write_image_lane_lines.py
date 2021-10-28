import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import draw_lane_lines

# Find OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

#
# list files in the source directory and sort by digits in string, ascending
#
dir = "/Users/edmurphy/mycar20190118/tub20190420/tub"
fnames = [fname for fname in os.listdir(dir) if fname.endswith('.jpg')] # get names of all jpegs
fnames.sort(key=lambda f: int("".join(filter(str.isdigit, f))))         # sort by digits in name
print(fnames)

out_video = None
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
fps = 20

frame_count = 0
for fname in fnames:
    frame = cv2.imread(dir + '/' + fname)
    if frame is not None:
        height, width, depth = frame.shape
        lane_lines_frame = draw_lane_lines.draw_lane_lines(frame, np.array([[(0, height - 25), (0, height), (width, height), (width, height - 25), (width - 40, 45), (40, 45)]]))
        # cv2.imshow("video lane lines", lane_lines_frame)
        if out_video is None:
            out_video = cv2.VideoWriter('lane_lines_from_images.mp4',fourcc,fps,(width,height))
        out_video.write(lane_lines_frame)
        frame_count += 1
        print("frame ", frame_count)
    else:
        break

if out_video is not None:
    out_video.release()
cv2.destroyAllWindows()
