from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys import get_types_from_msg
from pathlib import Path
from bps_oculus.core import unpack_data_entry, polar_to_cart
import cv2
from cfar import detect_peaks
import numpy as np


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

            # Filter data
            filtered = img > 50

            # Convert boolean array to uint8 for display (255 for True, 0 for False)
            display_img = np.uint8(filtered) * 255

            # Display the image
            cv2.imshow("backscatter", display_img)

            key = cv2.waitKey(10)
            if key & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()