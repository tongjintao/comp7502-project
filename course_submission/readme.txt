(1) Installation

Linux / Mac OS environment is recommended.

(i)   Install Python 2.7.
(ii)  Install NumPy and SciPy.
(iii)  Install OpenCV 2.4 and all its dependencies.
(iv)  Install pywavelets with the command "pip install pywavelets".

(2) Running the video demo program

The command is as follows:

python 3-different-method-poker-recognition.py

Press Space Bar to make a webcam capture. The program would print the guesses from the three methods.

Since the non-SIFT-based methods are highly sensitive to the background, we assume that the cards are lying on a dark background.

(3) How to run realtime playing card recognition program?

You can launch the real-time playing card recognizer by:

python sift-based-recognition-realtime.py

This program is based on computer camera, and it utilizes all the JPG files in the "cardModels" folder to generate SIFT matching models. 
For test cases, only provided JPG cards could be recognized by the realtime camera.

(4) Demo video

https://www.youtube.com/watch?v=L_aVfjbd_JM
