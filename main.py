import numpy as np
import cv2 as cv

im = cv.imread('acon_ab.jpg')
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
imgray = cv.GaussianBlur(imgray, (5, 5), cv.BORDER_DEFAULT)
gray_filtered = cv.bilateralFilter(imgray, 1, 50, 50)
edges1 = cv.Canny(imgray, 40, 120)
edges2 = cv.Canny(gray_filtered, 50, 120)
edges = np.hstack((imgray, gray_filtered, edges1, edges2))
ret, thresh = cv.threshold(edges2, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

rects = map(lambda x : cv.boundingRect(x), contours)
areas = list(map(lambda x : x[2] * x[3], rects))

main_contour = contours[areas.index(max(areas))]
rotated_rect = cv.minAreaRect(main_contour)
box = cv.boxPoints(rotated_rect)
box = np.int0(box)
img = cv.drawContours(im, [box], 0, (0,255,0), 2)
print(rotated_rect[2])

cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()




# let rotatedRect = cv.minAreaRect(cnt);
# let vertices = cv.RotatedRect.points(rotatedRect);
# let contoursColor = new cv.Scalar(255, 255, 255);
# let rectangleColor = new cv.Scalar(255, 0, 0);
# cv.drawContours(dst, contours, 0, contoursColor, 1, 8, hierarchy, 100);
