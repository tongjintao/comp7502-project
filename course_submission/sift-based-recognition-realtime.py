import numpy as np
import cv2
#import sift
import glob

cvSift = cv2.SIFT()
bf = cv2.BFMatcher()
def getSiftFeatureValue(img):
	_, des = cvSift.detectAndCompute(img, None)
	#des = sift.getDescriptor(img)

	return des

def cardRecognize(card):	
	camSiftValues = getSiftFeatureValue(card)
	maxGood = 6
	bestMatch = None

	for modelCardName,modelDes in modelCards.items():	
		matches = bf.knnMatch(camSiftValues, modelDes, k=2)
		#matches = sift.match(camSiftValues, modelDes)
		good = 0
		for m,n in matches:
			if m.distance < 0.5 * n.distance:
				good = good + 1
		if good >= maxGood:
			maxGood = good
			bestMatch = modelCardName

	cv2.putText(card, str(bestMatch), (20,60), cv2.FONT_HERSHEY_SIMPLEX, 2,(10,10,255),2)
	cv2.imshow('Camera', card)

modelCards = {}
for imagePath in glob.glob("cardModels/*.jpg"):
	modelCard = cv2.imread(imagePath, 0)
	modelSiftValues = getSiftFeatureValue(modelCard)
	imagePath = imagePath.replace("cardModels/", "")
	cardName = imagePath.replace(".jpg", "")
	modelCards[cardName] = modelSiftValues

cap = cv2.VideoCapture(0)
while(True):
    _, frame = cap.read()
    cardRecognize(frame)
    if cv2.waitKey(10) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()