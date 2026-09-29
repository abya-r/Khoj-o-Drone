'''
# Team ID:          3213
# Theme:            Khoj-o-Drone
# Author List:      Aditri Khanna, Abya Rao, Anamika Kumari, Kritika Raj
# Filename:         task1a.py
# Functions:        detect_markers, order_tl_tr_br_bl, rectify, build_grid,
#                   colour_masks, find_contours, centre_of, nearest_name, main
# Global variables: REQUIRED_IDS, SIZE, CELLS, MIN_AREA
'''

import argparse
import os
import sys

import cv2
import numpy as np

REQUIRED_IDS = [80, 85, 90, 95]
SIZE = 900            # rectified canvas is SIZE x SIZE
CELLS = 12            # 12 x 12 cells -> 11 interior lines each way
MIN_AREA = 150        # px^2 on the 900x900 canvas; smaller blobs are noise


def detect_markers(img):
    '''
    Purpose:
    ---
    Detects all ArUco 4x4_250 markers in the image. Works on both the
    new (>= 4.7) and the legacy OpenCV ArUco API.

    Input Arguments:
    ---
    `img` :  [ numpy.ndarray ]
        BGR image loaded with cv2.imread

    Returns:
    ---
    `found` :  [ dict ]
        {marker_id: 4x2 float32 array of corner points}

    Example call:
    ---
    markers = detect_markers(img)
    '''
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    if hasattr(aruco, "ArucoDetector"):            # OpenCV >= 4.7
        detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(img)
    else:                                          # OpenCV < 4.7 (legacy API)
        params = (aruco.DetectorParameters_create()
                  if hasattr(aruco, "DetectorParameters_create")
                  else aruco.DetectorParameters())
        corners, ids, _ = aruco.detectMarkers(img, dictionary, parameters=params)
    found = {}
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            found[int(i)] = c.reshape(4, 2).astype(np.float32)
    return found


def order_tl_tr_br_bl(pts):
    '''
    Purpose:
    ---
    Orders four points as top-left, top-right, bottom-right, bottom-left
    (image coordinates, y pointing down).

    Input Arguments:
    ---
    `pts` :  [ array-like, shape (4, 2) ]
        four unordered (x, y) points

    Returns:
    ---
    `ordered` :  [ numpy.ndarray, shape (4, 2) ]
        the points in TL, TR, BR, BL order

    Example call:
    ---
    src = order_tl_tr_br_bl(inner_corners)
    '''
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    return np.array([pts[np.argmin(s)],   # TL: smallest x+y
                     pts[np.argmax(d)],   # TR: largest x-y
                     pts[np.argmax(s)],   # BR: largest x+y
                     pts[np.argmin(d)]],  # BL: smallest x-y
                    dtype=np.float32)


def rectify(img, markers):
    '''
    Purpose:
    ---
    Perspective-warps the playing field to a SIZE x SIZE top-down view,
    using each marker's inner corner (the one nearest the arena centre).

    Input Arguments:
    ---
    `img` :  [ numpy.ndarray ]
        original BGR image

    `markers` :  [ dict ]
        output of detect_markers containing all REQUIRED_IDS

    Returns:
    ---
    `rect` :  [ numpy.ndarray ]
        SIZE x SIZE rectified BGR image

    Example call:
    ---
    rect = rectify(img, markers)
    '''
    centres = np.array([markers[i].mean(axis=0) for i in REQUIRED_IDS])
    middle = centres.mean(axis=0)
    inner = []
    for i in REQUIRED_IDS:
        c = markers[i]
        inner.append(c[np.argmin(np.linalg.norm(c - middle, axis=1))])
    src = order_tl_tr_br_bl(inner)
    dst = np.array([[0, 0], [SIZE - 1, 0],
                    [SIZE - 1, SIZE - 1], [0, SIZE - 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, H, (SIZE, SIZE))


def build_grid():
    '''
    Purpose:
    ---
    Computes the 121 interior grid intersections of the rectified arena and
    their names (column letter A-K, row number 1-11, e.g. "C2").

    Input Arguments:
    ---
    None

    Returns:
    ---
    `coords` :  [ numpy.ndarray, shape (121, 2) ]
        (x, y) pixel position of each intersection

    `names` :  [ list of str ]
        matching labels, A1 (top-left) ... K11 (bottom-right)

    Example call:
    ---
    coords, names = build_grid()
    '''
    step = (SIZE - 1) / CELLS
    coords, names = [], []
    for row in range(1, CELLS):           # rows 1..11, top to bottom
        for col in range(1, CELLS):       # cols A..K, left to right
            coords.append((col * step, row * step))
            names.append(f"{chr(ord('A') + col - 1)}{row}")
    return np.array(coords, dtype=np.float32), names


def colour_masks(rect):
    '''
    Purpose:
    ---
    Builds cleaned binary masks for red and yellow objects using HSV
    thresholds (OpenCV hue range is 0-179; red wraps around 0).

    Input Arguments:
    ---
    `rect` :  [ numpy.ndarray ]
        rectified BGR image

    Returns:
    ---
    `masks` :  [ dict ]
        {"red": mask, "yellow": mask}

    Example call:
    ---
    masks = colour_masks(rect)
    '''
    hsv = cv2.cvtColor(cv2.GaussianBlur(rect, (5, 5), 0), cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 110, 70), (10, 255, 255)) | \
          cv2.inRange(hsv, (170, 110, 70), (180, 255, 255))
    yellow = cv2.inRange(hsv, (18, 110, 100), (38, 255, 255))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    out = {}
    for name, m in (("red", red), ("yellow", yellow)):
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        out[name] = m
    return out


def find_contours(mask):
    '''
    Purpose:
    ---
    Extracts outer contours from a mask, discarding blobs smaller than MIN_AREA.

    Input Arguments:
    ---
    `mask` :  [ numpy.ndarray ]
        single-channel binary mask

    Returns:
    ---
    `cnts` :  [ list ]
        contours, one per survivor

    Example call:
    ---
    cnts = find_contours(masks["red"])
    '''
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in cnts if cv2.contourArea(c) >= MIN_AREA]


def centre_of(cnt):
    '''
    Purpose:
    ---
    Returns the centroid of a contour from image moments. Falls back to the
    bounding-box centre for zero-area regions (m00 == 0).

    Input Arguments:
    ---
    `cnt` :  [ numpy.ndarray ]
        a single contour

    Returns:
    ---
    `cx, cy` :  [ float, float ]
        centre point in pixels

    Example call:
    ---
    cx, cy = centre_of(cnt)
    '''
    m = cv2.moments(cnt)
    if m["m00"] > 1e-9:
        return m["m10"] / m["m00"], m["m01"] / m["m00"]
    x, y, w, h = cv2.boundingRect(cnt)
    return x + w / 2.0, y + h / 2.0


def nearest_name(pt, coords, names):
    '''
    Purpose:
    ---
    Returns the name of the grid intersection nearest to a point. Always
    yields a valid label, even for points closer to the arena edge.

    Input Arguments:
    ---
    `pt` :  [ tuple ]
        (x, y) point in the rectified image

    `coords` :  [ numpy.ndarray ]
        intersection positions from build_grid

    `names` :  [ list of str ]
        intersection labels from build_grid

    Returns:
    ---
    `label` :  [ str ]
        e.g. "D2"

    Example call:
    ---
    label = nearest_name((cx, cy), coords, names)
    '''
    d = np.linalg.norm(coords - np.array(pt, dtype=np.float32), axis=1)
    return names[int(np.argmin(d))]


def main():
    '''
    Purpose:
    ---
    Runs the full pipeline: markers -> rectify -> grid -> colour regions ->
    centres -> intersection names, then writes <image>_results.txt next to
    the input image. Use --debug to also save a composite image.

    Input Arguments:
    ---
    None (reads --image [--debug] from the command line)

    Returns:
    ---
    None

    Example call:
    ---
    python3 task1a.py --image image_1.jpg
    '''
    ap = argparse.ArgumentParser(description="Khojo Drone arena survivor detector")
    ap.add_argument("--image", required=True, help="path to the arena image")
    ap.add_argument("--debug", action="store_true",
                    help="also save <image>_debug.png (composite of every step)")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"ERROR: could not load image '{args.image}'", file=sys.stderr)
        sys.exit(1)

    markers = detect_markers(img)
    print("Detected IDs:", sorted(markers))
    missing = [i for i in REQUIRED_IDS if i not in markers]
    if missing:
        print(f"ERROR: missing marker ID(s) {missing}; cannot continue.",
              file=sys.stderr)
        sys.exit(1)

    rect = rectify(img, markers)
    coords, names = build_grid()

    masks = colour_masks(rect)
    results = {"red": [], "yellow": []}
    debug = rect.copy()
    for (x, y) in coords:
        cv2.circle(debug, (int(round(x)), int(round(y))), 2, (0, 255, 0), -1)

    for colour in ("red", "yellow"):
        cnts = find_contours(masks[colour])
        print(f"{colour}: {len(cnts)} region(s)")
        for c in cnts:
            cx, cy = centre_of(c)
            label = nearest_name((cx, cy), coords, names)
            results[colour].append(label)
            print(f"  {colour} centre=({cx:.1f},{cy:.1f}) -> {label}")
            cv2.drawContours(debug, [c], -1, (255, 0, 0), 2)
            cv2.circle(debug, (int(round(cx)), int(round(cy))), 4, (255, 0, 255), -1)
            cv2.putText(debug, label, (int(cx) + 6, int(cy) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(debug, label, (int(cx) + 6, int(cy) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    base = os.path.splitext(args.image)[0]
    out_path = base + "_results.txt"
    with open(out_path, "w") as f:
        f.write(f"Detected marker IDs: {sorted(REQUIRED_IDS)}\n")
        f.write("\n")
        f.write("Critical Survivors: " + ", ".join(results["red"]) + "\n")
        f.write("Stable Survivors: " + ", ".join(results["yellow"]) + "\n")
    print("Wrote", out_path)

    if args.debug:
        dbg_path = base + "_debug.png"
        cv2.imwrite(dbg_path, debug)
        print("Wrote", dbg_path)


if __name__ == "__main__":
    main()
