import numpy as np
import cv2
import os
import histogram_contour_classifier as hcc
import trapezoid_utils as tu
import wavelet_utils as wave
import siftcard

hcc.load_template_rank()
hcc.load_template_suit()

cap = cv2.VideoCapture(0)

while(True):
    _, frame = cap.read()
    cv2.imshow('Camera', frame)

    k = cv2.waitKey(20) & 0xFF
    if k == 27:    
    	print "Esc pressed, exiting ..."
        break
    elif k == ord(' '):  
    	cards = tu.get_trapezoids(frame)
    	for card in cards:
            hcc.show_image('cards', card)
            hcc_guess = hcc.recognize_card(card)
            print "histogram contour prediction: " 
            print hcc_guess
            guess = wave.recognize_card(card)
            print "wavelet-compression prediction: "    
            print hcc_guess
            print "sift prediction: "
            print siftcard.cardRecognize(card)



cap.release()
cv2.destroyAllWindows()