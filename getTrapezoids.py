#!/usr/bin/python

import numpy as np
import scipy.spatial
import cv2

np.set_printoptions(threshold=np.nan)

# Constants

DUMMY_VALUE = 0

SIZES = [2 * (2 ** x) - 1 for x in range(2, 3)]

MIN_ASPECT_RATIO = 0.6

DELTA_DELTA = 0.01

# Variables

imgname = "test_images/C9_1.jpg"

img = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)

# Preprocessing

# Gaussian Blur

blurred_images = [cv2.GaussianBlur(img, (s, s), 6) for s in SIZES]

# Thresholding

binary_images = []
for b in blurred_images:
    thresh, binary_image = cv2.threshold(b, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary_images.append(binary_image)

test_img = binary_images[0].copy()

# Blob detection

# Params

params = cv2.SimpleBlobDetector_Params()

params.minThreshold = 0
params.maxThreshold = 127  # Darker blobs first

params.filterByArea = True
params.minArea = 20

params.filterByCircularity = False

params.filterByConvexity = False

params.filterByInertia = False

# Detector (for OpenCV 3.0+)

detector = cv2.SimpleBlobDetector_create(params)

# Repeatedly detect and remove blobs

mask = np.zeros((test_img.shape[0] + 2, test_img.shape[1] + 2), dtype=np.uint8)
seen_points = set()

while True:
    blobs = detector.detect(test_img)
    if len(blobs) <= 0:
        break
    all_seen = True
    for blob in blobs:
        cv2.floodFill(test_img, mask, (int(blob.pt[0]), int(blob.pt[1])), DUMMY_VALUE, flags = (8 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)))
        if blob.pt not in seen_points:
            all_seen = False
            seen_points.add(blob.pt)
    test_img = test_img | mask[1:-1, 1:-1]
    if all_seen:
        break

# Perform a final dilation to remove dirt
# (median filter should work as well)

kernel = np.ones((SIZES[0], SIZES[0]), dtype=np.uint8)
dilated = cv2.dilate(test_img, kernel, iterations=1)

test_img = dilated

# Select the largest area that looks like a card

ret, markers = cv2.connectedComponents(test_img)
n_components = np.max(markers)

min_xs = {}
max_xs = {}
min_ys = {}
max_ys = {}
counts = {}
is_eligible = {}
for i in range(1, n_components + 1):
    counts[i] = np.sum(markers == i)
    is_eligible[i] = False
    any_x = np.sum(markers == i, axis=0)
    where_x = np.where(any_x > 0)[0]
    min_xs[i] = where_x[0]
    max_xs[i] = where_x[-1]
    any_y = np.sum(markers == i, axis=1)
    where_y = np.where(any_y > 0)[0]
    min_ys[i] = where_y[0]
    max_ys[i] = where_y[-1]
    # Too small
    if min_xs[i] == max_xs[i] or min_ys[i] == max_ys[i]:
        continue
    # No blobs were detected within the region
    if np.sum((markers == i) * (mask[1:-1, 1:-1])) == 0:
        continue
    # Aspect ratio fits our detection
    if (max_xs[i] - min_xs[i]) / (max_ys[i] - min_ys[i]) >= MIN_ASPECT_RATIO and (max_xs[i] - min_xs[i]) / (max_ys[i] - min_ys[i]) <= 1 / MIN_ASPECT_RATIO:
        is_eligible[i] = True

eligible_regions = []
for i in range(1, n_components + 1):
    if is_eligible[i]:
        eligible_regions.append(i)

# Annotate each region

approxs = []

for region in eligible_regions:
    # Convex Hull
    points = np.where(markers == region)
    points = np.vstack((points[1], points[0])).transpose()  # Transposed the coordinates to suit OpenCV's coordinate system
    hull = scipy.spatial.ConvexHull(points)
    vertices = points[hull.vertices]
    # Find bounding quadrilateral
    approx = vertices.copy()
    last_approx_shape = approx.shape[0]
    delta = DELTA_DELTA
    while approx.shape[0] > 4 and delta < 1:
        approx = cv2.approxPolyDP(approx, delta * cv2.arcLength(approx, True), True)
        if approx.shape[0] == last_approx_shape:
            delta += DELTA_DELTA
        last_approx_shape = approx.shape[0]
    approxs.append(approx)

img = cv2.imread(imgname)

for (region, approx) in zip(eligible_regions, approxs):
    cv2.drawContours(img, [approx], 0, (int(np.random.uniform(256)), int(np.random.uniform(256)), int(np.random.uniform(256))), 2)

cv2.imwrite("out.bmp", img)

exit(1)

# Unused

edges = cv2.Canny(test_img, 64, 192, apertureSize=3)

contour_image, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea)[::-1]

# out = cv2.drawKeypoints(img, blobs, np.array([]), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

