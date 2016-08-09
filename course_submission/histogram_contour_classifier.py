import numpy as np
import cv2
import os

rank_templates = {}
suit_templates = {}
rank_list = ["ace", "two", "three", "four", "five", "six", "seven", "eight", "nine", "jack", "queen", "king"]
suit_list = ["heart", "spade", "club", "diamond"]

def highlight_contour(img, ctr, name="highlight"):
	# highlight proceed contour in green
	img_highlight = img.copy()
	for point in ctr:
		contour_point = map(tuple,point)[0]
		cv2.circle(img_highlight, contour_point , 1, (127,0,127), 5)
	show_image(name, img_highlight)

def show_image(img_title, img):
	# for show image
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
	# transform a list of number to range 0 to len
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
		# break contour point in x-coordinate and y-coordinate
		contour_point_tuple = map(tuple, contour_point)[0]
		x_list.append(contour_point_tuple[0])
		y_list.append(contour_point_tuple[1])

	# resize it into a 8-ranged list of number for easier histogram comparison
	size = 8
	x_list = transform_list(x_list, size)
	y_list = transform_list(y_list, size)
	
	# make histogram
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
	# sort it 
	local_maxima.sort(key = lambda d: d[0], reverse=True)

	return (local_maxima[0][1] + local_maxima[1][1])/ 2

def load_template_suit():

	for suit in suit_list:

		# load template
		template = cv2.imread("suit_template/" + suit + ".png")
		
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

		x_histogram, y_histogram = make_histogram(contour_whole)
		suit_templates[suit] = (contour_whole, x_histogram, y_histogram)
	
def load_template_rank():

	for rank in rank_list:
		
		# load template
		template = cv2.imread("rank_template/" + rank + ".png")
		
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

		x_histogram, y_histogram = make_histogram(contour_whole)
		rank_templates[rank] = (contour_whole, x_histogram, y_histogram)

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

def recognize_suit(contours, hier):
	possible_suits = {}
	for suit, suit_value in suit_templates.items():

		suit_contour, suit_x_histogram, suit_y_histogram = suit_value
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

	# find threshold to best recognize image
	img_histogram = np.histogram(img_gray.flatten(), bins=np.arange(256), density=True) 
	#threshold = find_threshold(img_histogram)
	
	# threshold the image
	threshold, img_threshold = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	height, width = img_threshold.shape
	img_rotate = img_threshold.copy()

	# Find possible suit and rank by minimal histogram difference
	possible_suit = None
	possible_suit_score = 0.05
	possible_rank = None
	possible_rank_score = 0.05

	for rotation in range(0,4):
		# crop out the top left corner
		img_crop = crop_img(img_rotate, 0, 0, width/3, height/3)	
		
		ctrs, hier = cv2.findContours(img_crop.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)	
		possible_suits = recognize_suit(ctrs, hier)
		for suit, score in possible_suits.iteritems():
			if possible_suit is None or score < possible_suit_score:
				 possible_suit_score = score
				 possible_suit = suit
		possible_ranks = recognize_rank(ctrs, hier)
		for rank, score in possible_ranks.iteritems():
			if possible_rank is None or score < possible_rank_score:
				 possible_rank_score = score
				 possible_rank = rank
		
		img_rotate = np.rot90(img_rotate)
		
	
	if possible_suit == None or possible_rank == None:
		guess = None
	else: 
		guess = possible_suit+' '+possible_rank

	return guess

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
		guess = recognize_card(img)
		print guess
