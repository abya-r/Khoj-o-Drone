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

def extract_playing_field_corners(ids, marker_corners):
    # Ensure at least 4 markers are detected
    if ids is None or len(ids) < 4:
        print("Required markers not found")
        sys.exit()

    # Calculate center point for each detected marker
    centers = []
    for corner_set in marker_corners:
        pts = corner_set[0]
        center = np.mean(pts, axis=0)
        centers.append((center, pts))

    # Calculate overall centroid of all 4 markers combined
    all_centers = np.array([c[0] for c in centers])
    overall_center = np.mean(all_centers, axis=0)

    # Select the vertex from each marker closest to the center (the inner playing field corner)
    inner_corners = []
    for center, pts in centers:
        distances = [np.linalg.norm(pt - overall_center) for pt in pts]
        closest_idx = np.argmin(distances)
        inner_corners.append(pts[closest_idx])

    inner_corners = np.array(inner_corners, dtype=np.float32)

    # Sort inner corners in top-left, top-right, bottom-left, bottom-right order
    indices_y = np.argsort(inner_corners[:, 1])
    top_two = inner_corners[indices_y[:2]]
    bottom_two = inner_corners[indices_y[2:]]

    tl = top_two[np.argmin(top_two[:, 0])]
    tr = top_two[np.argmax(top_two[:, 0])]
    bl = bottom_two[np.argmin(bottom_two[:, 0])]
    br = bottom_two[np.argmax(bottom_two[:, 0])]

    return np.array([tl, tr, br, bl], dtype=np.float32)

def straighten_arena(img, ids, marker_corners, target_dim=900):
    # Extract the 4 inner playing field boundary points
    src_points = extract_playing_field_corners(ids, marker_corners)

    # Define destination coordinates for an exact 900x900 square canvas
    dst_points = np.array([
        [0, 0],
        [target_dim - 1, 0],
        [target_dim - 1, target_dim - 1],
        [0, target_dim - 1]
    ], dtype=np.float32)

    # Calculate homography matrix and warp image to top-down view
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    rectified_img = cv2.warpPerspective(img, matrix, (target_dim, target_dim))

    return rectified_img

# ABYA'S PART

def generate_grid_intersections():
    intersections = {}
    cell_size = 900.0 / 12.0  # 75.0 pixels per cell
    cols = [chr(ord('A') + i) for i in range(11)]  # Columns A to K

    for row_idx in range(1, 12):
        for col_idx in range(11):
            x = (col_idx + 1) * cell_size
            y = row_idx * cell_size
            label = f"{cols[col_idx]}{row_idx}"
            intersections[label] = (x, y)

    return intersections


def isolate_survivors(rectified_img):
    hsv = cv2.cvtColor(rectified_img, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    raw_red_contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_yellow_contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    red_triangles = []
    
    for cnt in raw_red_contours:
        area = cv2.contourArea(cnt)
        if area > 40:
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
            if len(approx) == 3:
                red_triangles.append(cnt)

    yellow_circles = []
    
    for cnt in raw_yellow_contours:
        area = cv2.contourArea(cnt)
        if area > 40:  # Noise threshold
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if circularity > 0.7:
                    yellow_circles.append(cnt)

    return red_triangles, yellow_circles

# KRITIKA'S PART

def render_debug_composite(rectified_img, intersections, red_data, yellow_data):


    result = rectified_img.copy()


    for point in intersections:
        x = int(point[0])
        y = int(point[1])

        cv2.circle(result, (x, y), 3, (255, 0, 0), -1)

    
    for point in red_data:
        x = int(point[0])
        y = int(point[1])

        cv2.circle(result, (x, y), 7, (0, 0, 255), 2)
        cv2.putText(result, "R", (x + 5, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 1)


    for point in yellow_data:
        x = int(point[0])
        y = int(point[1])

        cv2.circle(result, (x, y), 7, (0, 255, 255), 2)
        cv2.putText(result, "Y", (x + 5, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1)

    cv2.imshow("Debug Composite", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result

# MAIN PIPELINE EXECUTION ENGINE
