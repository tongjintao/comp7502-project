import numpy as numpy
import cv2
#from matplotlib import pyplot as pyplot

img = cv2.imread('test_images/spade9.jpg')

# convert to grayscale and apply Gaussian filtering
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.GaussianBlur(img_gray, (7,7),0)

# threshold the image
ret, img_threshold = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)

# find contours in the image
ctrs, hier = cv2.findContours(img_threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for ctr in ctrs:
	print "contour"
	for point in ctr:
		contour_point = map(tuple,point)[0]
		cv2.circle(img, contour_point , 1, (127,255,0), 1)

# Get rectangles contains each contour 
rects = [cv2.boundingRect(ctr) for ctr in ctrs] 

for rect in rects:
	# draw the rectangles

	cv2.rectangle(img, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (127,255,0),1)

cv2.imshow('image', img)
cv2.imshow('image_gray', img_gray)
cv2.imshow('image_threshold', img_threshold)

cv2.waitKey(0)
cv2.destroyAllWindows()