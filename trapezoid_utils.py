#!/usr/bin/python

from __future__ import print_function
import sys
import subprocess
import argparse
import numpy as np
import scipy.spatial
import cv2

np.set_printoptions(threshold=np.nan)

# Constants

TOO_LARGE = 1024

DUMMY_VALUE = 0

FILTER_SIZE = 7

MIN_ASPECT_RATIO = 0.6

DELTA_DELTA = 0.001

OUTPUT_DIM = (512, 512)  # Notice the (x, y) ordering in OpenCV

# Blob detection params

# Note on blob detection: after studying the logic behind blob detection,
# we realized that it can be performed by standard 8-neighbour flood fill.
# Therefore we decided to use OpenCV's implementation.

params = cv2.SimpleBlobDetector_Params()

params.minThreshold = 0
params.maxThreshold = 127  # Darker blobs first

params.filterByArea = True
params.minArea = 20

params.filterByCircularity = False

params.filterByConvexity = False

params.filterByInertia = False

# Detector (for OpenCV 2 and 3.0+)

ver = (cv2.__version__).split('.')
if int(ver[0]) < 3:
    detector = cv2.SimpleBlobDetector(params)
else: 
    detector = cv2.SimpleBlobDetector_create(params)

def get_trapezoids(colored_image):

    # Preprocessing

    mean_shifted_image = cv2.pyrMeanShiftFiltering(colored_image, FILTER_SIZE, 32)
    image = cv2.cvtColor(mean_shifted_image, cv2.COLOR_BGR2GRAY)

    width = image.shape[1]
    height = image.shape[0]
    corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)

    # Gaussian blur and thresholding

    blurred_image = cv2.GaussianBlur(image, (FILTER_SIZE, FILTER_SIZE), 6)
    thresh, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    test_img = binary_image

    # Repeatedly detect and remove blobs

    mask = np.zeros((test_img.shape[0] + 2, test_img.shape[1] + 2), dtype=np.uint8)
    seen_points = set()

    while True:
        blobs = detector.detect(test_img)
        if len(blobs) <= 0:
            break
        all_seen = True
        for blob in blobs:
            coords = []
            for x in (0, 1):
                for y in (0, 1):
                    if test_img[int(blob.pt[1]) + y, int(blob.pt[0]) + x] == 0:
                        coords.append((int(blob.pt[1]) + y, int(blob.pt[0] + x)))
            for (x, y) in coords:
                cv2.floodFill(test_img, mask, (y, x), DUMMY_VALUE, flags = (8 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)))
            if blob.pt not in seen_points:
                all_seen = False
                seen_points.add(blob.pt)
        test_img = test_img | mask[1:-1, 1:-1]
        if all_seen:
            break

    # Select the (white) areas that look like a card

    contours, hierarchy = cv2.findContours(test_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    n_components = len(contours)

    min_xs = {}
    max_xs = {}
    min_ys = {}
    max_ys = {}
    counts = {}
    is_eligible = {}
    regions = []
    for i in range(0, n_components):
        is_eligible[i] = False
        regions.append(contours[i])
        min_xs[i] = np.min(contours[i][:, :, 0])
        max_xs[i] = np.max(contours[i][:, :, 0])
        min_ys[i] = np.min(contours[i][:, :, 1])
        max_ys[i] = np.max(contours[i][:, :, 1])
        # Too small
        if min_xs[i] == max_xs[i] or min_ys[i] == max_ys[i]:
            continue
        # No blobs were detected within the region
#        if np.sum((markers == i) * (mask[1:-1, 1:-1])) == 0:
#            continue
        # Aspect ratio fits our detection
        if np.float(max_xs[i] - min_xs[i]) / np.float(max_ys[i] - min_ys[i]) >= MIN_ASPECT_RATIO and np.float(max_xs[i] - min_xs[i]) / np.float(max_ys[i] - min_ys[i]) <= 1 / MIN_ASPECT_RATIO:
            is_eligible[i] = True

    eligible_regions = []
    for i in range(0, n_components):
        if is_eligible[i]:
            eligible_regions.append(i)

    # Find trapezoidal approximations for each region 

    approxs = []

    for region in eligible_regions:
        approx = regions[region].reshape((-1, regions[region].shape[2]))
        hull = scipy.spatial.ConvexHull(approx)
        approx = approx[hull.vertices]
        last_approx_shape = approx.shape[0]
        delta = DELTA_DELTA
        while approx.shape[0] > 4 and delta < 1:
            approx = cv2.approxPolyDP(approx, delta * cv2.arcLength(approx, True), True)
            if approx.shape[0] == last_approx_shape:
                delta += DELTA_DELTA
            last_approx_shape = approx.shape[0]
        if len(approx.shape) > 2:
            approx = approx.reshape((-1, approx.shape[2]))
        approxs.append(approx)

    # Delete if a box is fully inscribed within another

    to_keep = []
    for i in range(len(eligible_regions)):
        is_inscribed = False
        min_coords = np.min(approxs[i], axis=0)
        max_coords = np.max(approxs[i], axis=0)
        for j in range(len(eligible_regions)):
            if j == i:
                continue
            min_coords2 = np.min(approxs[j], axis=0)
            max_coords2 = np.max(approxs[j], axis=0)
            if np.all(min_coords >= min_coords2) and np.all(max_coords <= max_coords2):
                is_inscribed = True
                break
        if not is_inscribed:
            to_keep.append(i)

    approxs = [approxs[k] for k in to_keep]
    eligible_regions = [eligible_regions[k] for k in to_keep]

    # Perspective (and inverse) transform

    output_corners = np.array([[0, 0], [OUTPUT_DIM[0] - 1, 0], [OUTPUT_DIM[0] - 1, OUTPUT_DIM[1] - 1], [0, OUTPUT_DIM[1] - 1]], dtype=np.float32)

    output_images = []
    for (region, approx) in zip(eligible_regions, approxs):
        perspective = approx.astype(np.float32)
        transform = cv2.getPerspectiveTransform(perspective, output_corners)
        warp = cv2.warpPerspective(colored_image, transform, OUTPUT_DIM)
        output_images.append(warp)

    return output_images

