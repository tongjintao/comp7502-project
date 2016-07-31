import numpy as np
import cv2
import glob

def cardRecognize(card):	
	_, des = sift.detectAndCompute(card, None)	#query image
	
	ret = ""
	maxGood = 10
	bestMatch = None

	for modelCardName,modelDes in modelCards.items():	
		matches = bf.knnMatch(des, modelDes, k=2)
		good = 0
		for m,n in matches:
			if m.distance < 0.5*n.distance:
				good = good + 1

		ret = ret + " " + modelCardName + ": " + str(good)
		if good >= maxGood:
			maxGood = good
			bestMatch = modelCardName

	text = str(bestMatch)
	cv2.putText(card, text, (20,60), cv2.FONT_HERSHEY_SIMPLEX, 2,(10,10,255),2)
	cv2.imshow('Camera', card)


#Init
sift = cv2.SIFT()
bf = cv2.BFMatcher()

#Load the models
modelCards = {}
for imagePath in glob.glob("*.jpg"):
	modelCard = cv2.imread(imagePath, 0)
	_, des = sift.detectAndCompute(modelCard, None)	
	cardName = imagePath.replace(".jpg", "")
	modelCards[cardName] = des

cap = cv2.VideoCapture(0)
while(True):
    _, frame = cap.read()
    cardRecognize(frame)

    k = cv2.waitKey(10) & 0xFF
    if k == 27: 
        break

cap.release()
cv2.destroyAllWindows()