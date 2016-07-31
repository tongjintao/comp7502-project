import numpy as np
import cv2
#from matplotlib import pyplot as pyplot

templateObjs = {}
#rank_list = ["two", "three", "five", "seven"]
rank_list = ["three"]

def highlight_contour(img, ctr):
	# highlight proceed contour in green
	img_highlight = img.copy()
	for point in ctr:
		contour_point = map(tuple,point)[0]
		cv2.circle(img_highlight, contour_point , 1, (127,255,127), 1)
	show_image('img_highlight', img_highlight)

def show_image(img_title, img):
	cv2.imshow(img_title,img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def crop_contour(img_threshold, ctr):
	# crop out contour 
	x,y,w,h = cv2.boundingRect(ctr)
	ctr_crop = img_threshold[y:y+h, x:x+w]
	show_image('img_threshold', ctr_crop)

def transform_list(numbers, desired_range):
	max_number = max(numbers)
	min_number = min(numbers)
	range_number = max_number - min_number

	transformed_numbers = []

	for number in numbers:
		transformed_number = (float(number) - min_number) / range_number * desired_range 
		transformed_numbers.append(transformed_number)

	return transformed_numbers

def make_histogram(contour):
	x_list = []
	y_list = []
	
	for contour_point in contour:	
		# convert numpy.arrayto tuple
		contour_point_tuple = map(tuple, contour_point)[0]
		x_list.append(contour_point_tuple[0])
		y_list.append(contour_point_tuple[1])

	# resize it into a 10-ranged list of number for easier histogram comparison
	size = 8
	x_list = transform_list(x_list, size)
	y_list = transform_list(y_list, size)
	
	x_histogram = np.histogram(x_list, bins=np.arange(size+1), density=True)
	y_histogram = np.histogram(y_list, bins=np.arange(size+1), density=True)
	return x_histogram, y_histogram

def load_template():
	for templateName in rank_list:

		template = cv2.imread("rank_template/" + templateName + ".png")
		
		# convert to grayscale and apply Gaussian filtering and thresholding
		template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
		template_gray = cv2.GaussianBlur(template_gray, (3,3),0)	
		_, template_gray = cv2.threshold(template_gray, 170, 255, cv2.THRESH_BINARY)

		#show_image('template_gray', template_gray)

		contours, _ = cv2.findContours(template_gray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	
		for contour in contours[:-1]:
			x_histogram, y_histogram = make_histogram(contour)

			# crop_contour(template_gray, contour)
			# highlight_contour(template_gray, contour)
		
		templateObjs[templateName] = (contours[0], x_histogram, y_histogram)

def compare_histogram_chisquare(hist1, hist2):
	# calculate difference in histogram by chisquare
	diff_chisquare = 0

	for i in range(len(hist1)):
		diff_chisquare += (hist1[i] - hist2[i]) * (hist1[i] - hist2[i]) / hist1[i]

	return diff_chisquare


def recognize_card(img):
	# convert to grayscale and apply Gaussian filtering
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img_gray = cv2.GaussianBlur(img_gray, (3,3),0)	

	# threshold the image
	ret, img_threshold = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)

	# find contours in the image
	ctrs, hier = cv2.findContours(img_threshold.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

	for rank, rank_value in templateObjs.items():
		rank_contour, rank_x_histogram, rank_y_histogram = rank_value 
		for ctr in ctrs:
			# screen out contour that are too small
			if len(ctr) < 20:
				continue

			x_histogram, y_histogram = make_histogram(ctr)

			print "rank " + rank
			print rank_x_histogram
			print rank_y_histogram

			print "contours to be compared"
			print x_histogram
			print y_histogram

			x_diff_chisquare = compare_histogram_chisquare(x_histogram[0].tolist(), rank_x_histogram[0].tolist())
			print("chisquare x difference: {0}").format(x_diff_chisquare)
			y_diff_chisquare = compare_histogram_chisquare(y_histogram[0].tolist(), rank_y_histogram[0].tolist())
			print("chisquare y difference: {0}").format(y_diff_chisquare)
						

			# try print matchSahpes difference
			retval = cv2.matchShapes(rank_contour, ctr, 2, 0)
			print retval
			
			# highlight proceed contour in green
			highlight_contour(img, ctr)

	# Get rectangles contains each contour 
	rects = [cv2.boundingRect(ctr) for ctr in ctrs] 

	for rect in rects:
		# draw the rectangles
		cv2.rectangle(img, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (127,255,0),1)

	# show image
	cv2.imshow('image', img)
	cv2.imshow('image_gray', img_gray)
	cv2.imshow('image_threshold', img_threshold)

	cv2.waitKey(0)
	cv2.destroyAllWindows()



load_template()

#img = cv2.imread('rank_template/three.png')
img = cv2.imread('playingcard/diamond3.jpg')
#img = cv2.imread('test_images/heart3.jpg')

recognize_card(img)






"""
For reference:

--------------------


for ctr in ctrs:
	print "contour"
	for point in ctr:
		contour_point = map(tuple,point)[0]
		cv2.circle(img, contour_point , 1, (127,255,0), 1)

"""