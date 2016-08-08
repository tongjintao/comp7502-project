import numpy as np
import cv2
import matplotlib.pyplot as plt

templateObjs = {}
#rank_list = ["ace", "two", "three", "four", "five", "six", "seven", "eight", "nine", "jack", "queen", "king"]
rank_list = ["seven" ,"queen"]

def highlight_contour(img, ctr, name='highlight'):
	# highlight proceed contour in green
	img_highlight = img.copy()
	for point in ctr:
		contour_point = map(tuple,point)[0]
		cv2.circle(img_highlight, contour_point , 1, (127,0,127), 1)
	show_image(name, img_highlight)

def show_image(img_title, img):
	cv2.imshow(img_title,img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def crop_contour(img_threshold, ctr):
	# crop out contour 
	x,y,w,h = cv2.boundingRect(ctr)
	ctr_crop = img_threshold[y:y+h, x:x+w]
	show_image('img_threshold', ctr_crop)

def crop_img(img_threshold, x, y, w, h):
	# crop out image
	ctr_crop = img_threshold[y:y+h, x:x+w]
	show_image('img_threshold', ctr_crop)
	return ctr_crop

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
	size = 1024
	x_list = transform_list(x_list, size)
	y_list = transform_list(y_list, size)

	n, bins, patches = plt.hist(x_list, normed=True)
	plt.show()
	n, bins, patches = plt.hist(y_list, normed=True)
	plt.show()
	
	x_histogram = np.histogram(x_list, bins=np.arange(size+1), density=True) 
	y_histogram = np.histogram(y_list, bins=np.arange(size+1), density=True)
	
	return x_histogram, y_histogram

def find_threshold(img_histogram):

	# find local maxima
	img_histogram = zip(img_histogram[0], img_histogram[1])
	local_maxima = img_histogram
	local_maxima_n = local_maxima 

	while not len(local_maxima_n) < 6:
		# repeat finding local maxima until only five local maxima left
		local_maxima_n = []
		for i in range(len(local_maxima)):
			if i == 0:
				if local_maxima[i][0] > local_maxima[i+1][0]:
					local_maxima_n.append(local_maxima[i])
			elif i == len(local_maxima)-1:
				if local_maxima[i][0] > local_maxima[i-1][0]:
					local_maxima_n.append(local_maxima[i])
			else:
				if local_maxima[i][0] > local_maxima[i-1][0] and  local_maxima[i][0] > local_maxima[i+1][0]:
					local_maxima_n.append(local_maxima[i])

		local_maxima = local_maxima_n

	#print local_maxima

	# sort it 
	local_maxima.sort(key = lambda d: d[0], reverse=True)
	#print local_maxima[0], local_maxima[1]

	return (local_maxima[0][1] + local_maxima[1][1])/ 2
	
def load_template():

	for templateName in rank_list:
		template = cv2.imread("rank_template/" + templateName + ".png")
		
		# convert to grayscale and apply Gaussian filtering and thresholding
		template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
		template_gray = cv2.GaussianBlur(template_gray, (3,3),0)	
		_, template_gray = cv2.threshold(template_gray, 170, 255, cv2.THRESH_BINARY)

		contours, hierarchy = cv2.findContours(template_gray.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

		for index in range(len(contours)):
			# skip outer contour
			if hierarchy[0][index][3] == -1:
				continue
			# if have children, include all sub-contour into contour_whole and break
			if hierarchy[0][index][2] != -1:
				child = [index]
				contour_whole = contours[index]
				for child_i in range(index+1, len(contours)):
					if hierarchy[0][child_i][3] == index:
						contour_whole = np.concatenate((contour_whole, contours[child_i]))
					else:
						break
				break
			# return the last contour as template
			if hierarchy[0][index][0] == -1:
				contour_whole = contours[index]
			#crop_contour(template_gray, contour)
		
		#highlight_contour(template_gray, contour_whole)

		x_histogram, y_histogram = make_histogram(contour_whole)
		# print x_histogram
		# print y_histogram
		# #plt.plot(x_histogram[0].tolist())

		# n, bins, patches = plt.hist(x_histogram[0],1)
		# plt.show()
		highlight_contour(template_gray, contour_whole)
		templateObjs[templateName] = (contour_whole, x_histogram, y_histogram)

def compare_histogram_chisquare(hist1, hist2):
	# calculate difference in histogram by chisquare
	diff_chisquare = 0

	for i in range(len(hist1)):
		# solving chisquare zero denomator issue
		if hist1[i] == 0:
			diff_chisquare += (hist1[i] - hist2[i]) * (hist1[i] - hist2[i]) / 0.5	
		else:
			diff_chisquare += (hist1[i] - hist2[i]) * (hist1[i] - hist2[i]) / hist1[i]

	return diff_chisquare

def recognize_card(img):
	# convert to grayscale and apply Gaussian filtering
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img_gray = cv2.GaussianBlur(img_gray, (3,3),0)	

	# # adaptive threshold the image
	# img_threshold = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3,2)

	# find threshold to best recognize image
	img_histogram = np.histogram(img_gray.flatten(), bins=np.arange(256), density=True) 
	threshold = find_threshold(img_histogram)
	
	# threshold the image
	ret, img_threshold = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)
	show_image('image', img_threshold)

	height, width = img_threshold.shape

	# find contours in the image
	ctrs, hier = cv2.findContours(img_threshold.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)	
	possible_ranks = {}

	for rank, rank_value in templateObjs.items():
		rank_contour, rank_x_histogram, rank_y_histogram = rank_value
		for index in range(len(ctrs)):
			# screen out contour that are too small
			if len(ctrs[index]) < 30:
				continue
			
			highlight_contour(img, ctrs[index], "contour_each")
			print hier[0][index]
			# if have children, include all sub-contour into contour_whole and break
			if hier[0][index][2] != -1:
				child = [index]
				contour_whole = ctrs[index]
				for child_i in range(index+1, len(ctrs)):
					if hier[0][child_i][3] == index:
						highlight_contour(img, ctrs[child_i], "contour_child")
						contour_whole = np.concatenate((contour_whole, ctrs[child_i]))
					else:
						break
			else:
				contour_whole = ctrs[index]

			highlight_contour(img, contour_whole, "contour_whole")
			x_histogram, y_histogram = make_histogram(contour_whole)

			## Print out value 
			# print "rank " + rank
			# print rank_x_histogram
			# print rank_y_histogram

			# print "contours to be compared"
			# print x_histogram
			# print y_histogram

			x_diff_chisquare = compare_histogram_chisquare(x_histogram[0].tolist(), rank_x_histogram[0].tolist())
			# print("chisquare x difference: {0}").format(x_diff_chisquare)
			y_diff_chisquare = compare_histogram_chisquare(y_histogram[0].tolist(), rank_y_histogram[0].tolist())
			# print("chisquare y difference: {0}").format(y_diff_chisquare)	

			# print("chisquare x * y difference: {0}".format(x_diff_chisquare * y_diff_chisquare))

			# # try print matchSahpes difference
			# retval = cv2.matchShapes(rank_contour, ctrs[index], 2, 0)
			# print retval
			
			# highlight proceed contour in green if value is lower enough
			x_y_diff = x_diff_chisquare * y_diff_chisquare 
			if x_y_diff < 0.05:
				if possible_ranks.has_key(rank) and possible_ranks.get(rank) > x_y_diff:
					possible_ranks[rank] = x_y_diff
				if not possible_ranks.has_key(rank):
					possible_ranks[rank] = x_y_diff

				# highlight_contour(img, ctrs[index])
				# print rank
				# print("chisquare x difference: {0}").format(x_diff_chisquare)
			 # 	print("chisquare y difference: {0}").format(y_diff_chisquare)	
			 # 	print("chisquare x * y difference: {0}".format(x_diff_chisquare * y_diff_chisquare))

	# for rank, x_y_diff in possible_ranks.items():
	# 	print rank, x_y_diff

	#print min(possible_ranks.items(), key=lambda x: x[1])[0]

	# Get rectangles contains each contour 
	rects = [cv2.boundingRect(ctr) for ctr in ctrs] 

	for rect in rects:
		# draw the rectangles
		cv2.rectangle(img, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (127,255,0),1)

	# show image
	show_image('image', img)


load_template()

#img = cv2.imread('rank_template/three.png')
#img = cv2.imread('playingcard/diamond3.jpg')
#img = cv2.imread('sift/club3.jpg')
#img = cv2.imread('test_images/heart3.jpg')

img = cv2.imread('rank_template/queen.png')

#img = cv2.imread('sift/club7.jpg')
#img = cv2.imread('test_images/club7.jpg')

#img = cv2.imread('playingcard/heartq.jpg')

#recognize_card(img)




"""
For reference:

--------------------


for ctr in ctrs:
	print "contour"
	for point in ctr:
		contour_point = map(tuple,point)[0]
		cv2.circle(img, contour_point , 1, (127,255,0), 1)

"""