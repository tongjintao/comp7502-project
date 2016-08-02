#!/usr/bin/python

import sys
import os
import argparse
import numpy as np
import scipy.spatial
import cv2
import pywt

parser = argparse.ArgumentParser()
parser.add_argument("--file", dest="filename", help="Input file name.")
parser.add_argument("--card", dest="card", default=None, help="Card name. If not provided, the input image will be matched against the database.")
parser.add_argument("--reconstruct", dest="reconstruct", action="store_true", help="Reconstruct the compressed card.")

options = parser.parse_args()

if options.filename is None:
    parser.print_help()
    exit(1)

np.set_printoptions(threshold=np.nan)

# Constants

FILTER_SIZE = 7

WAV = 'db8'

COMPRESSION_LEVEL = 3

DATA_DIRECTORY = "coeffs"

# Variables

imgname = options.filename

img = cv2.imread(imgname)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Gaussian Blur and Threshold

blurred_image = cv2.GaussianBlur(gray, (FILTER_SIZE, FILTER_SIZE), 6)
thresh, binary = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Wavelet decomposition

coefs = pywt.wavedec2(binary, WAV, level=COMPRESSION_LEVEL, mode='per')

# Discard details

if options.reconstruct:

    coefs = pywt.wavedec2(binary, WAV, level=COMPRESSION_LEVEL, mode='per')

    coefs2 = coefs.copy()

    for i in range(1, len(coefs)):
        cH, cV, cD = (np.zeros(coefs[i][0].shape), np.zeros(coefs[i][1].shape), np.zeros(coefs[i][2].shape))
        coefs2[i] = (cH, cV, cD)

    rec = pywt.waverec2(coefs2, WAV, mode='per')
    rec[rec >= 128] = 255
    rec[rec < 128] = 0

    cv2.imwrite("compressed.bmp", np.int0(rec))

# If card name is given, save the coefficients

if options.card is not None:

    try:
        os.makedirs(DATA_DIRECTORY)
    except OSError:
        pass

    with open(os.path.join(DATA_DIRECTORY, options.card), 'wb') as f:
        np.save(f, coefs[0])

# If card name is not given, give the best match against the database

else:

    try:

        possible_cards = os.listdir(DATA_DIRECTORY)

        guess = None
        max_corr = -1

        # Handle rotations

        for rotation in range(0, 4):

            for card in possible_cards:
                card_coefs = np.load(os.path.join(DATA_DIRECTORY, card))
                corr = np.corrcoef(coefs[0].reshape(-1), card_coefs.reshape(-1))
                corr = corr[0, 1]
#                print("%s: %f" % (card, corr), file=sys.stderr)
                if guess is None or corr > max_corr:
                    guess = card
                    max_corr = corr

            if rotation < 3:
                binary = np.rot90(binary)
                coefs = pywt.wavedec2(binary, WAV, level=COMPRESSION_LEVEL, mode='per')

        print("Best match: %s" % guess)

    except OSError:
        
        print("The database has no cards!", file=sys.stderr)

