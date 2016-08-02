#!/usr/bin/python

from __future__ import print_function
import sys
import argparse
import numpy as np
import scipy.spatial
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--file", dest="filename", help="Input file name.")
parser.add_argument("--aspect", dest="aspect", default=1.0, help="Starting aspect ratio (default is 1).")
parser.add_argument("--binary", dest="binary", action="store_true", default=False, help="Produce binary image only")

options = parser.parse_args()

if options.filename is None:
    parser.print_help()
    exit(1)

np.set_printoptions(threshold=np.nan)

# Constants

TOO_LARGE = 1024

DUMMY_VALUE = 0

FILTER_SIZE = 7

MIN_ASPECT_RATIO = 0.6

DELTA_DELTA = 0.001

OUTPUT_DIM = (512, 512)  # Notice the (x, y) ordering in OpenCV

# Variables

image_name = options.filename

try:
    colored_image = cv2.imread(image_name)
    mean_shifted_image = cv2.pyrMeanShiftFiltering(colored_image, FILTER_SIZE, 32)
    image = cv2.cvtColor(mean_shifted_image, cv2.COLOR_BGR2GRAY)
except:
    image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)

width = image.shape[1]
height = image.shape[0]
corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)

current_aspect_ratio = options.aspect

# Blob detection params

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

# Trial-and-error

has_error = True

while has_error:

    try:

        has_error = False

        # Resize by aspect ratio

        new_width = int(width * current_aspect_ratio)
        new_height = int(height * current_aspect_ratio)
        new_corners = np.array([[0, 0], [new_width - 1, 0], [new_width - 1, new_height - 1], [0, new_height - 1]], dtype=np.float32)
        transform = cv2.getPerspectiveTransform(corners, new_corners)
        img = cv2.warpPerspective(image, transform, (new_width, new_height))

        # Preprocessing

        # Gaussian blur and thresholding

        blurred_image = cv2.GaussianBlur(img, (FILTER_SIZE, FILTER_SIZE), 6)
        thresh, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        if options.binary:
            cv2.imwrite("binary.bmp", binary_image)
            exit(0)

        test_img = binary_image.copy()

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

        # Perform a dilation to remove dirt

#        kernel = np.ones((FILTER_SIZE, FILTER_SIZE), dtype=np.uint8)
#        dilated = cv2.dilate(test_img, kernel, iterations=1)

#        test_img = dilated

        # Select the (white) areas that look like a card

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
            # Too large
            if max_xs[i] - min_xs[i] > TOO_LARGE or max_ys[i] - min_ys[i] > TOO_LARGE:
                raise Exception("The image is too large. Please resize for better detection.")
            # Too small
            if min_xs[i] == max_xs[i] or min_ys[i] == max_ys[i]:
                continue
            # No blobs were detected within the region
            if np.sum((markers == i) * (mask[1:-1, 1:-1])) == 0:
                continue
            # Aspect ratio fits our detection
            if np.float(max_xs[i] - min_xs[i]) / np.float(max_ys[i] - min_ys[i]) >= MIN_ASPECT_RATIO and np.float(max_xs[i] - min_xs[i]) / np.float(max_ys[i] - min_ys[i]) <= 1 / MIN_ASPECT_RATIO:
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

        print(approxs, file=sys.stderr)

    except Exception as e:

        print(e, file=sys.stderr)
        has_error = True
        current_aspect_ratio *= 0.5

# Output borders for verification

out = colored_image.copy()

for (region, approx) in zip(eligible_regions, approxs):
    print(approx / current_aspect_ratio, file=sys.stderr)
    cv2.drawContours(out, [np.int0(approx / current_aspect_ratio)], 0, (int(np.random.uniform(256)), int(np.random.uniform(256)), int(np.random.uniform(256))), int(2 / current_aspect_ratio))

cv2.imwrite("out.bmp", out)

# Perspective (and inverse) transform

output_corners = np.array([[0, 0], [OUTPUT_DIM[0] - 1, 0], [OUTPUT_DIM[0] - 1, OUTPUT_DIM[1] - 1], [0, OUTPUT_DIM[1] - 1]], dtype=np.float32)

output_images = []
for (region, approx) in zip(eligible_regions, approxs):
    perspective = approx.astype(np.float32)
    perspective /= current_aspect_ratio
    transform = cv2.getPerspectiveTransform(perspective, output_corners)
    warp = cv2.warpPerspective(colored_image, transform, OUTPUT_DIM)
    output_images.append(warp)

for (region, output_image) in zip(eligible_regions, output_images):
    cv2.imwrite("%d.bmp" % region, output_image)

