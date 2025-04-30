from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys import get_types_from_msg
from pathlib import Path
from bps_oculus.core import unpack_data_entry, polar_to_cart
import cv2
from cfar import detect_peaks
import numpy as np
from skimage.measure import label, regionprops


### SETUP CUSTOM MESSAGE DEFINITION FOR RawData
# Read definitions to python strings.
msg_text = """# Directly inspired by WHOI's ds_core_msgs/RawData.msg
# https://bitbucket.org/whoidsl/ds_msgs/src/master/ds_core_msgs/msg/RawData.msg
#
# Used for logging raw bytes sent to/from hardware.

std_msgs/Header header

int8 DATA_OUT=0  # Data published by the driver
int8 DATA_IN=1  # Data received by the driver

int8 direction

uint8[] data
"""

# Plain dictionary to hold message definitions.
add_types = {}

# Add definitions from one msg file to the dict.
add_types.update(get_types_from_msg(msg_text, 'apl_msgs/msg/RawData'))

# Path to your ROS bag (set it correctly)
bag_path = Path('blue3_2025-03-19-13-58-48.bag')

# Create a typestore for the matching ROS release.
typestore = get_typestore(Stores.ROS1_NOETIC)
typestore.register(add_types)

START_TIMESTAMP = 1742392985721249949
# Create reader instance and open for reading.
with Reader(bag_path) as reader:
    # Topic and msgtype information is available on .connections list.
    for connection in reader.connections:
        print(connection.topic, connection.msgtype)

    # Iterate over messages.
    for connection, timestamp, rawdata in reader.messages():
        # Skip messages before our desired start time
        if timestamp < START_TIMESTAMP:
            continue

        if connection.topic == '/oculus/raw_data':
            msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
            _, polar_image_data, _ = unpack_data_entry(msg.data.tobytes()) 

            # Skip if no data
            if polar_image_data is None:
                continue
            img = polar_to_cart(polar_image_data).cart_image

            # Median filter to reduce noise
            img_median = cv2.medianBlur(img, 11)
            
            # Threshold the image
            _, binary_img = cv2.threshold(img_median, 40, 255, cv2.THRESH_BINARY)
            
            # Convert to format expected by skimage (binary boolean array)
            binary_image = binary_img.astype(bool)
            
            # Find all blobs with 8-connectivity
            label_image = label(binary_image, connectivity=2)
            regions = regionprops(label_image)
            
            # Create a colored output image for visualization
            # Convert grayscale to color image for overlay
            result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            # Paint each detected region with a different color
            for region in regions:
                # Skip very small regions (likely noise)
                if region.area < 20 or region.area > 100:
                    continue
                
                # Get coordinates of all pixels in this region
                coords = region.coords
                
                # Generate a random color for this region
                color = np.random.randint(0, 255, size=3).tolist()
                
                # Paint each pixel of the region with the chosen color
                for coord in coords:
                    y, x = coord
                    if 0 <= y < result.shape[0] and 0 <= x < result.shape[1]:
                        result[y, x] = color
                
                # Draw a circle at the centroid of the region
                cy, cx = region.centroid
                cv2.circle(result, (int(cx), int(cy)), 3, (0, 255, 0), -1)
            
            # Display the result with the colored overlay
            cv2.namedWindow("Sonar with wall detection", cv2.WINDOW_NORMAL)
            cv2.imshow("Sonar with wall detection", result)
            
            # Wait for key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()