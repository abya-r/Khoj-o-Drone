import cv2
import numpy as np
import argparse
import os
import sys


ARENA_DIM = 900
GRID_DIVISIONS = 12

CORNER_MARKER_IDS = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument(
        "--show",
        action="store_true",
        help="Display the debug composite in a window (needs a display).",
    )
    return p.parse_args()


def load_image_and_validate(path):
    img = cv2.imread(path)
    if img is None:
        print("Image not found")
        sys.exit()
    return img


def detect_aruco_markers(img):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    p = cv2.aruco.DetectorParameters()
    a = cv2.aruco.ArucoDetector(d, p)

    res = a.detectMarkers(img)
    corners, ids = res[0], res[1]

    if ids is None:
        print("No markers found")
        sys.exit()

    ids = ids.flatten()
    return ids, corners


def extract_playing_field_corners(ids, marker_corners, corner_ids=CORNER_MARKER_IDS):
    if ids is None:
        print("Required markers not found")
        sys.exit()

    if corner_ids is not None:
        keep = [i for i, marker_id in enumerate(ids) if marker_id in corner_ids]
        ids = [ids[i] for i in keep]
        marker_corners = [marker_corners[i] for i in keep]

    if len(ids) != 4:
        print(f"Expected exactly 4 corner markers, found {len(ids)}")
        sys.exit()

    centers = []
    for corner_set in marker_corners:
        pts = corner_set[0]
        center = np.mean(pts, axis=0)
        centers.append((center, pts))

    all_centers = np.array([c[0] for c in centers])
    overall_center = np.mean(all_centers, axis=0)

    inner_corners = []
    for center, pts in centers:
        distances = [np.linalg.norm(pt - overall_center) for pt in pts]
        closest_idx = np.argmin(distances)
        inner_corners.append(pts[closest_idx])

    inner_corners = np.array(inner_corners, dtype=np.float32)

    indices_y = np.argsort(inner_corners[:, 1])
    top_two = inner_corners[indices_y[:2]]
    bottom_two = inner_corners[indices_y[2:]]

    tl = top_two[np.argmin(top_two[:, 0])]
    tr = top_two[np.argmax(top_two[:, 0])]
    bl = bottom_two[np.argmin(bottom_two[:, 0])]
    br = bottom_two[np.argmax(bottom_two[:, 0])]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def straighten_arena(img, ids, marker_corners, target_dim=ARENA_DIM):
    src_points = extract_playing_field_corners(ids, marker_corners)

    dst_points = np.array([
        [0, 0],
        [target_dim - 1, 0],
        [target_dim - 1, target_dim - 1],
        [0, target_dim - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    rectified_img = cv2.warpPerspective(img, matrix, (target_dim, target_dim))

    return rectified_img


def generate_grid_intersections(target_dim=ARENA_DIM, divisions=GRID_DIVISIONS):
    intersections = {}
    cell_size = target_dim / float(divisions)
    num_lines = divisions - 1  # interior grid lines only
    cols = [chr(ord('A') + i) for i in range(num_lines)]

    for row_idx in range(1, divisions):
        for col_idx in range(num_lines):
            x = (col_idx + 1) * cell_size
            y = row_idx * cell_size
            label = f"{cols[col_idx]}{row_idx}"
            intersections[label] = (x, y)

    return intersections


def isolate_survivors(rectified_img):
    hsv = cv2.cvtColor(rectified_img, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    lower_yellow = np.array([15, 70, 50])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    raw_red_contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_yellow_contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    red_triangles = []
    for cnt in raw_red_contours:
        area = cv2.contourArea(cnt)
        if area > 30:
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
            if len(approx) == 3:
                red_triangles.append(cnt)

    yellow_circles = []
    for cnt in raw_yellow_contours:
        area = cv2.contourArea(cnt)
        if area > 30:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if circularity > 0.65:
                    yellow_circles.append(cnt)

    return red_triangles, yellow_circles


def compute_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])
    else:
        x, y, w, h = cv2.boundingRect(contour)
        cx = x + w / 2.0
        cy = y + h / 2.0
    return (cx, cy)


def match_to_nearest_intersection(centroid, intersections):
    cx, cy = centroid
    min_dist = float("inf")
    closest_label = ""

    for label, (ix, iy) in intersections.items():
        dist = np.hypot(cx - ix, cy - iy)
        if dist < min_dist:
            min_dist = dist
            closest_label = label

    return closest_label


def render_debug_composite(rectified_img, intersections, red_data, yellow_data):
    result = rectified_img.copy()

    for label, (x, y) in intersections.items():
        cv2.circle(result, (int(x), int(y)), 3, (255, 0, 0), -1)

    for pt_info in red_data:
        x, y = int(pt_info[0]), int(pt_info[1])
        cv2.circle(result, (x, y), 7, (0, 0, 255), 2)
        cv2.putText(result, "R", (x + 5, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 1)

    for pt_info in yellow_data:
        x, y = int(pt_info[0]), int(pt_info[1])
        cv2.circle(result, (x, y), 7, (0, 255, 255), 2)
        cv2.putText(result, "Y", (x + 5, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1)

    return result


def export_results(image_path, marker_ids, critical_list, stable_list):
    base_path = os.path.splitext(image_path)[0]
    out_file = f"{base_path}_results.txt"

    sorted_ids = sorted(int(i) for i in marker_ids)
    crit_str = ", ".join(critical_list)
    stab_str = ", ".join(stable_list)

    marker_line = f"Detected marker IDs: {sorted_ids}"
    critical_line = f"Critical Survivors: {crit_str}"
    stable_line = f"Stable Survivors: {stab_str}"

    with open(out_file, "w") as f:
        f.write(marker_line + "\n\n")
        f.write(critical_line + "\n")
        f.write(stable_line + "\n")

    return out_file


def main():
    args = parse_args()
    img = load_image_and_validate(args.image)
    ids, corners = detect_aruco_markers(img)

    target_dim = ARENA_DIM
    rectified_img = straighten_arena(img, ids, corners, target_dim=target_dim)
    intersections = generate_grid_intersections(target_dim=target_dim)
    red_contours, yellow_contours = isolate_survivors(rectified_img)

    critical_labels = []
    red_centers = []
    for cnt in red_contours:
        center = compute_centroid(cnt)
        label = match_to_nearest_intersection(center, intersections)
        critical_labels.append(label)
        red_centers.append(center)

    stable_labels = []
    yellow_centers = []
    for cnt in yellow_contours:
        center = compute_centroid(cnt)
        label = match_to_nearest_intersection(center, intersections)
        stable_labels.append(label)
        yellow_centers.append(center)

    debug_img = render_debug_composite(rectified_img, intersections, red_centers, yellow_centers)

    out_file = export_results(args.image, list(ids), critical_labels, stable_labels)
    print(f"Results written to {out_file}")

    debug_path = f"{os.path.splitext(args.image)[0]}_debug.png"
    cv2.imwrite(debug_path, debug_img)
    print(f"Debug image written to {debug_path}")

    if args.show:
        try:
            cv2.imshow("Debug Composite", debug_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error as e:
            print(f"Could not open a display window ({e}); see {debug_path} instead.")


if __name__ == "__main__":
    main()
