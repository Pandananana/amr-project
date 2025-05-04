from pathlib import Path
import matplotlib.pyplot as plt

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

import scipy.integrate as it

bagpath = Path('blue3_2025-03-19-13-58-48.bag')
typestore = get_typestore(Stores.ROS1_NOETIC)

# Create storage
timestamps = []
frameIDs = []
orientations_x = []
orientations_y = []
orientations_z = []
orientations_w = []
angular_velocity_x = []
angular_velocity_y = []
angular_velocity_z = []
linear_acceleration_x = []
linear_acceleration_y = []
linear_acceleration_z = []

with AnyReader([bagpath], default_typestore=typestore) as reader:
    #reader.open()

    # for conn in reader.connections:
    #     print(f"{conn.topic} --> {conn.msgtype}")
    connections = [x for x in reader.connections if x.topic == '/bluerov2/mavros/imu/data']
    for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
        
            timestamps.append(timestamp)
            frameIDs.append(msg.header.frame_id)
            orientations_x.append(msg.orientation.x)
            orientations_y.append(msg.orientation.y)
            orientations_z.append(msg.orientation.z)
            orientations_w.append(msg.orientation.w)
            angular_velocity_x.append(msg.angular_velocity.x)
            angular_velocity_y.append(msg.angular_velocity.y)
            angular_velocity_z.append(msg.angular_velocity.z)
            linear_acceleration_x.append(msg.linear_acceleration.x)
            linear_acceleration_y.append(msg.linear_acceleration.y)
            linear_acceleration_z.append(msg.linear_acceleration.z)

    # Convert nanosecond timestamps to seconds, relative to first timestamp
    start_time = timestamps[0]
    time_seconds = [(t - start_time) * 1e-9 for t in timestamps]

    # Plot the results
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 2, 1)
    plt.plot(time_seconds, orientations_x, label='Orientation X')
    plt.xlabel('Time (s)')
    plt.ylabel('Orientation X')
    plt.title('Orientation X over Time')
    plt.legend()
    plt.grid()

    plt.subplot(3, 2, 2)
    plt.plot(time_seconds, orientations_y, label='Orientation Y')
    plt.xlabel('Time (s)')
    plt.ylabel('Orientation Y')
    plt.title('Orientation Y over Time')
    plt.legend()
    plt.grid()

    plt.subplot(3, 2, 3)
    plt.plot(time_seconds, orientations_z, label='Orientation Z')
    plt.xlabel('Time (s)')
    plt.ylabel('Orientation Z')
    plt.title('Orientation Z over Time')
    plt.legend()
    plt.grid()

    plt.subplot(3, 2, 4)
    plt.plot(time_seconds, orientations_w, label='Orientation W')
    plt.xlabel('Time (s)')
    plt.ylabel('Orientation W')
    plt.title('Orientation W over Time')
    plt.legend()
    plt.grid()

    plt.subplot(3, 2, 5)
    plt.plot(time_seconds, angular_velocity_x, label='Angular Velocity X')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity X (rad/s)')
    plt.title('Angular Velocity X over Time')
    plt.legend()
    plt.grid()

    plt.subplot(3, 2, 6)
    plt.plot(time_seconds, angular_velocity_y, label='Angular Velocity Y')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity Y (rad/s)')
    plt.title('Angular Velocity Y over Time')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

        
# Obtain velocity via integration
linear_velocity_x = it.cumtrapz(linear_acceleration_x, time_seconds, initial=0.0)
linear_velocity_y = it.cumtrapz(linear_acceleration_y, time_seconds, initial=0.0)
linear_velocity_z = it.cumtrapz(linear_acceleration_z, time_seconds, initial=0.0)

# Obtain position via integration
position_x = it.cumtrapz(linear_velocity_x, time_seconds, initial=0.0)
position_y = it.cumtrapz(linear_velocity_y, time_seconds, initial=0.0)
position_z = it.cumtrapz(linear_velocity_z, time_seconds, initial=0.0)

# Plot the results
plt.figure(figsize=(15, 10))

plt.subplot(3, 2, 1)
plt.plot(time_seconds, linear_velocity_x, label='Linear_V_X')
plt.xlabel('Time (s)')
plt.ylabel('Linear Velocity X')
plt.title('Linear Veocity X over Time')
plt.legend()
plt.grid()

plt.subplot(3, 2, 2)
plt.plot(time_seconds, linear_velocity_y, label='Linear_V_Y')
plt.xlabel('Time (s)')
plt.ylabel('Linear Velocity Y')
plt.title('Linear Veocity Y over Time')
plt.legend()
plt.grid()

plt.subplot(3, 2, 3)
plt.plot(time_seconds, linear_velocity_z, label='Linear_V_Z')
plt.xlabel('Time (s)')
plt.ylabel('Linear Velocity Z')
plt.title('Linear Veocity Z over Time')
plt.legend()
plt.grid()

plt.subplot(3, 2, 4)
plt.plot(time_seconds, position_x, label='Position X')
plt.xlabel('Time (s)')
plt.ylabel('Position X')
plt.title('Position X over Time')
plt.legend()
plt.grid()

plt.subplot(3, 2, 5)
plt.plot(time_seconds, position_y, label='Position Y')
plt.xlabel('Time (s)')
plt.ylabel('Position Y')
plt.title('Position Y over Time')
plt.legend()
plt.grid()

plt.subplot(3, 2, 6)
plt.plot(time_seconds, position_z, label='Position Z')
plt.xlabel('Time (s)')
plt.ylabel('Position Z')
plt.title('Position Z over Time')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

plt.figure()
plt.plot(position_x, position_y)
plt.xlabel('pos x')
plt.ylabel('pos y')
plt.title('2D motion plot')
plt.grid()
plt.show()
