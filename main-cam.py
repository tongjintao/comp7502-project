#Space for capture frame and recognize
#Esc for exit

import cv2
import numpy as np

def showImg(img, text = ""):
	cv2.putText(img, text, (20,60), cv2.FONT_HERSHEY_SIMPLEX, 2,(10,10,255),2)
	cv2.imshow("Image", img)
	cv2.waitKey(0)
	cv2.destroyWindow("Image")


def cardRecognize(card):
	cardGray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
	_, cardThreshold = cv2.threshold(cardGray, 100, 255, cv2.THRESH_BINARY)
	showImg(cardThreshold)

	contours, _ = cv2.findContours(cardThreshold, 1, 2)
	possibleSuit = None
	minMatchValue = 0.05

	for k, cardContour in enumerate(contours):
		k = k + 1
		if len(cardContour) < 20 or k == len(contours):
			#print "Skip small points/last round"
			continue

		hasGetNew = 0
		for suit,suitContour in templateObjs.items():	

			#Ref: http://docs.opencv.org/master/d5/d45/tutorial_py_contours_more_functions.html#gsc.tab=0
			ret = cv2.matchShapes(suitContour, cardContour, 3, 0.0)
			if ret < minMatchValue:
				minMatchValue = ret
				possibleSuit = suit
				hasGetNew = 1
				cv2.drawContours(card, [cardContour], 0, (0,255,0), -1)
				print "Round: " + str(k) + "/" + str(len(contours)) + ", count:" + str(len(cardContour)) + ", match: " + str(suit) + ", " + str(ret)	
			
	showImg(card, str(possibleSuit))
	print "Result: " + str(possibleSuit) + "\n"


#Load the suits
templateObjs = {}
for templateName in ["spade", "heart", "diamond", "club"]:

	template = cv2.imread(templateName + ".jpg")
	templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
	_, templateGray = cv2.threshold(templateGray, 170, 255, cv2.THRESH_BINARY)

	#showImg(templateGray)

	contours, _ = cv2.findContours(templateGray, 1, 2)		
	templateObjs[templateName] = contours[0]
	#cv2.drawContours(template, contours, 0, (0,255,0), -1)
	#showImg(template)


cap = cv2.VideoCapture(0)
while(True):
    _, frame = cap.read()
    cv2.imshow('Camera', frame)

    k = cv2.waitKey(20) & 0xFF
    if k == 27:    
    	print "Esc pressed, exiting ..."
        break
    elif k == ord(' '):  
    	cardRecognize(frame)

cap.release()
cv2.destroyAllWindows()
