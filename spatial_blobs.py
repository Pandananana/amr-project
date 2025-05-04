from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys import get_types_from_msg
from pathlib import Path
from bps_oculus.core import unpack_data_entry, polar_to_cart
import cv2
import numpy as np
from skimage.measure import label, regionprops
from random import randint

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

def matching_blobs(centroids, prev_centroids, max_distance=30):
    used_curr_idxs = set()
    used_prev_idxs = set()
    matches = []

    for i, curr in enumerate(centroids):
        min_dist = float('inf')
        match_idx = None

        for j, prev in enumerate(prev_centroids):
            if j in used_prev_idxs:
                continue
            
            dist = euclidean_distance(curr, prev)
            if dist < min_dist and dist < max_distance:
                min_dist = dist
                match_idx = j

        if match_idx is not None:
            matches.append((i, match_idx, min_dist))
            used_curr_idxs.add(i)
            used_prev_idxs.add(match_idx)

    return matches

def generate_color():
    return (randint(50, 255), randint(50, 255), randint(50, 255))

next_id = 0
prev_centroids = []
prev_ids = []
blob_tracks = {}  # blob_id -> (centroid, color)

# Custom message definition for RawData
msg_text = """
std_msgs/Header header
int8 DATA_OUT=0
int8 DATA_IN=1
int8 direction
uint8[] data
"""
add_types = {}
add_types.update(get_types_from_msg(msg_text, 'apl_msgs/msg/RawData'))

bag_path = Path('blue3_2025-03-19-13-58-48.bag')
typestore = get_typestore(Stores.ROS1_NOETIC)
typestore.register(add_types)

START_TIMESTAMP = 1742392985721249949

with Reader(bag_path) as reader:
    for connection, timestamp, rawdata in reader.messages():
        if timestamp < START_TIMESTAMP:
            continue
        if connection.topic != '/oculus/raw_data':
            continue

        msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
        _, polar_image_data, _ = unpack_data_entry(msg.data.tobytes())
        if polar_image_data is None:
            continue

        # Load image data
        img = polar_to_cart(polar_image_data).cart_image

        # Some image processing
        img_uint8 = np.uint8(img / img.max() * 255)
        blurred = cv2.GaussianBlur(img_uint8, (5, 5), 0)
        equalized = cv2.equalizeHist(blurred)
        _, thresh = cv2.threshold(equalized, 70, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        img_color = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

        
        labeled = label(cleaned)
        regions = regionprops(labeled)

        centroids = []
        for region in regions:
            if region.area > 50:
                y, x = region.centroid
                centroids.append((int(x), int(y)))

        ## BRIGHTEST BLOB ##
        # if regions:
        #     largest_region = max(regions, key=lambda r: r.area)
        #     y, x = largest_region.centroid
        #     centroids = [(int(x), int(y))]

        # First frame: initialize blob_tracks
        if not prev_centroids:
            for i, c in enumerate(centroids):
                blob_tracks[next_id] = (c, generate_color())
                next_id += 1
            prev_centroids = centroids
            prev_ids = list(blob_tracks.keys())
            continue

        matched_list = matching_blobs(centroids, prev_centroids)
        new_blob_tracks = {}
        used_prev_ids = set()

        for curr_idx, prev_idx, _ in matched_list:
            prev_id = prev_ids[prev_idx]
            color = blob_tracks[prev_id][1]
            new_blob_tracks[prev_id] = (centroids[curr_idx], color)

            p1 = blob_tracks[prev_id][0]
            p2 = centroids[curr_idx]
            cv2.line(img_color, p1, p2, color, 2)
            cv2.circle(img_color, p2, 4, color, -1)
            used_prev_ids.add(prev_idx)

        for i, c in enumerate(centroids):
            if i not in [m[0] for m in matched_list]:
                new_blob_tracks[next_id] = (c, generate_color())
                cv2.circle(img_color, c, 4, new_blob_tracks[next_id][1], -1)
                next_id += 1

        prev_centroids = [v[0] for v in new_blob_tracks.values()]
        prev_ids = list(new_blob_tracks.keys())
        blob_tracks = new_blob_tracks

        FRAME_RATE = 10
        FRAME_DELAY = int(1000 / FRAME_RATE)
        cv2.namedWindow("Tracked Sonar Features", cv2.WINDOW_NORMAL)
        cv2.imshow("Tracked Sonar Features", img_color)

        if cv2.waitKey(FRAME_DELAY) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
