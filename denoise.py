#!/usr/bin/python

import argparse
import numpy as np
import scipy.spatial
import cv2


parser = argparse.ArgumentParser()
parser.add_argument("--file", dest="filename", help="Input file name.")

options = parser.parse_args()

if options.filename is None:
    parser.print_help()
    exit(1)

np.set_printoptions(threshold=np.nan)

# Constants

# Variables

imgname = options.filename

img = cv2.imread(imgname)

denoised = cv2.fastNlMeansDenoisingColored(img, None)

cv2.imwrite("out.bmp", denoised)

