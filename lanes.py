import cv2
import numpy as np
import matplotlib.pyplot as plt

#
# 1. load the image
# 2. convert the image to greyscale
# 3. run canny edge detection - this will run a 5x5 Guassian blur convolution, so we don't have to do that
# 4. specify region of interest to limit computation
# 5. mask the image
# 6. apply Hough transform to detect straight lines
# 7 generate an image of the lane lines
# 8. combine the original image with the lines to show the lanes
# 10. average all the lines on each side to get a single line
# 11. draw the averaged lane lines
#


# 1. load the image
image = cv2.imread('test_image.jpg')
# cv2.imshow("Test Image", image)
# cv2.waitKey(0)

# 2. convert the image to greyscale
grey_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# blur_image = cv2.GaussianBlur(grey_image, (5, 5), 0)

# 3. run canny edge detection - this will run a 5x5 Guassian blur convolution, so we don't have to do that
edge_image = cv2.Canny(grey_image, 50, 150)
# plt.imshow(edge_image)
# plt.show() # show in a window (until window is closed)

# 4. specify region of interest to limit computation
height, width = edge_image.shape
region = np.array([[(200, height), (1100, height), (550, 250)]]) # list of polygons
mask = np.zeros_like(edge_image)
cv2.fillPoly(mask, region, 255)
# plt.imshow(mask)
# plt.show() # show in a window (until window is closed)

# 5. mask the image
masked_image = cv2.bitwise_and(edge_image, mask)
# plt.imshow(masked_image)
# plt.show() # show in a window (until window is closed)

# 6. apply Hough transform to detect straight lines; 2d array; list of list of 4 elements - endpoints of a line
hough_lines = cv2.HoughLinesP(masked_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# if hough_lines is not None:
#     for line in hough_lines:
#         print(line)

# 7 generate an image of the lane lines
line_image = np.zeros_like(image)
if hough_lines is not None:
    for line in hough_lines:
        x1, y1, x2, y2 = line.reshape(4)
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
# plt.imshow(line_image)
# plt.show() # show in a window (until window is closed)

# 8. combine the original image with the lines to show the lanes
lane_image = cv2.addWeighted(np.copy(image), 0.8, line_image, 1, 1)
# cv2.imshow("Lane Image", lane_image)
# cv2.waitKey(0)

# 9. sort lines into the left side or right side based on slope
left_lines = []
right_lines = []
for line in hough_lines:
    x1, y1, x2, y2 = line.reshape(4) # turn array of 4 elements into 4 elements
    parameters = np.polyfit((x1, x2), (y1, y2), 1) # calculate slope and intercept: fit first degree polynomial
    # print(parameters)
    slope = parameters[0]
    intercept = parameters[1]
    if slope < 0:
        left_lines.append((slope, intercept))
    else:
        right_lines.append((slope, intercept))

# 10. average all the lines on each side to get a single line
left_average = np.average(left_lines, axis = 0)
right_average = np.average(right_lines, axis = 0)
# print("left: ", left_average)
# print("right: ", right_average)

# 11. draw the averaged lane lines
average_lane_image = np.copy(image)
height = average_lane_image.shape[0]
average_lines = []

# left line
slope, intercept = left_average
y1 = height # bottom of image
y2 = int(y1*3/5) # arbitrarily stop 3/5 of the way up the image
x1 = int((y1 - intercept) / slope)
x2 = int((y2 - intercept) / slope)
cv2.line(average_lane_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

# right line
slope, intercept = right_average
y1 = height # bottom of image
y2 = int(y1*3/5) # arbitrarily stop 3/5 of the way up the image
x1 = int((y1 - intercept) / slope)
x2 = int((y2 - intercept) / slope)
cv2.line(average_lane_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
cv2.imshow("Average Lane Image", average_lane_image)
cv2.waitKey(0)
