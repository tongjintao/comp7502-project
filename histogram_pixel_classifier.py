import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

rank_templates = {}
suit_templates = {}
rank_list = ["ace", "two", "three", "four", "five", "six", "seven", "eight", "nine", "jack", "queen", "king"]
#rank_list = ["eight", "nine", "jack", "queen", "king"]
suit_list = ["heart", "spade", "club", "diamond"]
# suit_list = ["heart"]

def highlight_contour(img, ctr, name="highlight"):
	# highlight proceed contour in green
	img_highlight = img.copy()
	for point in ctr:
		contour_point = map(tuple,point)[0]
		cv2.circle(img_highlight, contour_point , 1, (127,0,127), 5)
	show_image(name, img_highlight)

def show_image(img_title, img):
	cv2.imshow(img_title,img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def crop_contour(img_threshold, ctr):
	# crop out contour 
	x,y,w,h = cv2.boundingRect(ctr)
	ctr_crop = img_threshold[y:y+h, x:x+w]
	#show_image('img_threshold', ctr_crop)

def crop_img(img_threshold, x, y, w, h):
	# crop out image
	ctr_crop = img_threshold[y:y+h, x:x+w]
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

def make_histogram_pixel(crop_img):
	x_list = []
	y_list = []
	
	height = crop_img.shape[0]
	width = crop_img.shape[1]

	size = 10

	# calculate pixel histogram
	for h in range(height):
		for w in range(width):
			if crop_img[h, w] < 200:
				y_list.append(h)

	for w in range(width):
		for h in range(height):
			if crop_img[h, w] < 200:
				x_list.append(w)

	n, bins, patches = plt.hist(x_list, normed=True)
	#plt.show()
	n, bins, patches = plt.hist(y_list, normed=True)
	#plt.show()


	x_histogram = np.histogram(x_list, bins=np.linspace(0, width, num=size), density=True) 
	y_histogram = np.histogram(y_list, bins=np.linspace(0, width, num=size), density=True) 

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
	# sort it 
	local_maxima.sort(key = lambda d: d[0], reverse=True)

	return (local_maxima[0][1] + local_maxima[1][1])/ 2

def load_template_suit():

	for suit in suit_list:

		template = cv2.imread("suit_template/" + suit + ".png")
		
		# convert to grayscale and apply Gaussian filtering and thresholding
		template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
		template_gray = cv2.GaussianBlur(template_gray, (3,3),0)	
		_, template_gray = cv2.threshold(template_gray, 170, 255, cv2.THRESH_BINARY)

		contours, hierarchy = cv2.findContours(template_gray.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

		# Get rectangles contains each contour 
		rect = cv2.boundingRect(contours[1])

		img_rec = template.copy()
		ctr_crop = template_gray[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
		x_histogram, y_histogram = make_histogram_pixel(ctr_crop)

		# draw the rectangles
		cv2.rectangle(img_rec, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (127,255,0),1)
		#show_image('img_rect_contour', img_rec)

		suit_templates[suit] = (contours[1], x_histogram, y_histogram)
	
def load_template_rank():

	for rank in rank_list:

		template = cv2.imread("rank_template/" + rank + ".png")
		
		# convert to grayscale and apply Gaussian filtering and thresholding
		template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
		template_gray = cv2.GaussianBlur(template_gray, (3,3),0)	
		_, template_gray = cv2.threshold(template_gray, 170, 255, cv2.THRESH_BINARY)

		contours, hierarchy = cv2.findContours(template_gray.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

		# Get rectangles contains each contour 
		rect = cv2.boundingRect(contours[1])

		img_rec = template.copy()
		ctr_crop = template_gray[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
		x_histogram, y_histogram = make_histogram_pixel(ctr_crop)

		# draw the rectangles
		cv2.rectangle(img_rec, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (127,255,0),1)
		#show_image('img_rect_contour', img_rec)

		rank_templates[rank] = (contours[1], x_histogram, y_histogram)

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

def recognize_rank(contours, hier):
	possible_ranks = {}
	for rank, rank_value in rank_templates.items():
		rank_contour, rank_x_histogram, rank_y_histogram = rank_value
		for index in range(len(contours)):
			# screen out contour that are too small
			if len(contours[index]) < 30:
				continue
			
			# if have children, include all sub-contour into contour_whole and break
			if hier[0][index][2] != -1:
				child = [index]
				contour_whole = contours[index]
				for child_i in range(index+1, len(contours)):
					if hier[0][child_i][3] == index:
						contour_whole = np.concatenate((contour_whole, contours[child_i]))
					else:
						break
			else:
				contour_whole = contours[index]

			# make histogram and compare difference with chisquare method
			x_histogram, y_histogram = make_histogram(contour_whole)
			x_diff_chisquare = compare_histogram_chisquare(x_histogram[0].tolist(), rank_x_histogram[0].tolist())
			y_diff_chisquare = compare_histogram_chisquare(y_histogram[0].tolist(), rank_y_histogram[0].tolist())
	
			# highlight proceed contour in green if value is lower enough
			x_y_diff = x_diff_chisquare * y_diff_chisquare 
			if x_y_diff < 0.05:
				if possible_ranks.has_key(rank) and possible_ranks.get(rank) > x_y_diff:
					possible_ranks[rank] = x_y_diff
				if not possible_ranks.has_key(rank):
					possible_ranks[rank] = x_y_diff
	return possible_ranks

def recognize_suit(img, contours, rects):
	possible_suits = {}
	for suit, suit_value in suit_templates.items():

		suit_contour, suit_x_histogram, suit_y_histogram = suit_value
		for index in range(len(contours)):
			# screen out contour that are too small
			if len(contours[index]) < 30:
				continue

			rect = rects[index]

			# make histogram and compare difference with chisquare method
			ctr_crop = img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
			x_histogram, y_histogram = make_histogram_pixel(ctr_crop)

			# compare histogram
			x_diff_chisquare = compare_histogram_chisquare(x_histogram[0].tolist(), suit_x_histogram[0].tolist())
			y_diff_chisquare = compare_histogram_chisquare(y_histogram[0].tolist(), suit_y_histogram[0].tolist())
			x_y_diff = x_diff_chisquare * y_diff_chisquare 

			if x_y_diff < 0.05:
				if possible_suits.has_key(suit) and possible_suits.get(suit) > x_y_diff:
					possible_suits[suit] = x_y_diff
				if not possible_suits.has_key(suit):
					possible_suits[suit] = x_y_diff

	return possible_suits

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
	#show_image('image', img_threshold)

	height, width = img_threshold.shape

	# crop out the top left corner
	img_threshold = crop_img(img_threshold, 0, 0, width/3, height/3)

	show_image('croped',img_threshold)
	# find contours in the image
	ctrs, hier = cv2.findContours(img_threshold.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)	

	rects = [cv2.boundingRect(contour) for contour in ctrs]

	possible_suits = recognize_suit(img_gray, ctrs, rects)
	suit = min(possible_suits.items(), key=lambda x: x[1])[0]

	# possible_ranks = recognize_rank(ctrs, hier)
	# rank = min(possible_ranks.items(), key=lambda x: x[1])[0]
	
	return suit #, rank


if __name__ == "__main__":
	load_template_rank()
	load_template_suit()

	directory = 'sift'

	imgs = os.listdir(os.getcwd()+'/' +directory)

	for img in imgs:

		if '.jpg' not in img:
			continue
		print img
		img = cv2.imread(directory+'/'+img)
		# recognize_card(img)
		#show_image('img',img)
		suit = recognize_card(img)
		print suit #, rank

