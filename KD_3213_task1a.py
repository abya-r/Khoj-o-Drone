import cv2
import numpy as np
import argparse
import os
import sys

# ANAMIKA'S PART
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    return p.parse_args()
    # add code

def load_image_and_validate(path):
    img = cv2.imread(path)

    if img is None:
        print("Image not found")
        sys.exit()

    return img
    # add code

def detect_aruco_markers(img):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    p = cv2.aruco.DetectorParameters()
    a = cv2.aruco.ArucoDetector(d, p)

    corners, ids, x = a.detectMarkers(img)

    if ids is None:
        print("No markers found")
        sys.exit()

    ids = ids.flatten()

    return ids, corners
    # add code

# ADITRI'S PART

def straighten_arena(img, marker_corners):
    # add code

def generate_grid_intersections():
    # add code

# ABYA'S PART

def isolate_survivors(rectified_img):
    # add code

def compute_centroid(contour):
    # add code

# KRITIKA'S PART

def render_debug_composite(rectified_img, intersections, red_data, yellow_data):
    # add code

# MAIN PIPELINE EXECUTION ENGINE

