#!/usr/bin/python

import sys
import os
import argparse
import numpy as np
import scipy.spatial
import cv2
import pywt
import copy

# Constants

FILTER_SIZE = 7

WAV = 'db8'

COMPRESSION_LEVEL = 3  # Shrink by a factor of 8x8=64 (2^3 = 8)

DATA_DIRECTORY = "coeffs"

# Variables

possible_cards = os.listdir(DATA_DIRECTORY)
possible_card_coefs = [np.load(os.path.join(DATA_DIRECTORY, card)) for card in possible_cards]

def recognize_card(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gaussian Blur and Threshold

    blurred_image = cv2.GaussianBlur(gray, (FILTER_SIZE, FILTER_SIZE), 6)
    thresh, binary = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Wavelet decomposition
    # Assume periodic signal (such that the artifacts at the boundaries can be minimized, for our purpose)

    coefs = pywt.wavedec2(binary, WAV, level=COMPRESSION_LEVEL, mode='per')

    # Retain only the first-level approximation
    # And zero all remaining coefficients

    coefs2 = copy.copy(coefs)

    for i in range(1, len(coefs)):
        cH, cV, cD = (np.zeros(coefs[i][0].shape), np.zeros(coefs[i][1].shape), np.zeros(coefs[i][2].shape))
        coefs2[i] = (cH, cV, cD)

    # Start with no guess

    guess = None
    max_corr = -1

    # Handle rotations (4 times, 90 degrees each)

    for rotation in range(0, 4):

        for (card, coef) in zip(possible_cards, possible_card_coefs):
            corr = np.corrcoef(coefs[0].reshape(-1), coef.reshape(-1))
            corr = corr[0, 1]
            # Update guess if better
            if guess is None or corr > max_corr:
                guess = card
                max_corr = corr

        if rotation < 3:
            binary = np.rot90(binary)
            coefs = pywt.wavedec2(binary, WAV, level=COMPRESSION_LEVEL, mode='per')

    return guess

