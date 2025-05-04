from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation as R
import scipy.integrate as it
from scipy.signal import butter, filtfilt

def highpass(data, cutoff=0.01, fs=100, order=2):
    b, a = butter(order, cutoff / (0.5 * fs), btype='high', analog=False)
    return filtfilt(b, a, data, axis=0)

bagpath = Path('blue3_2025-03-19-13-58-48.bag')
typestore = get_typestore(Stores.ROS1_NOETIC)

timestamps = []
orientations_quat = []
linear_acceleration = []

with AnyReader([bagpath], default_typestore=typestore) as reader:
    # reader.open()
    connections = [x for x in reader.connections if x.topic == '/bluerov2/mavros/imu/data']
    
    for connection, timestamp, rawdata in reader.messages(connections=connections):
        msg = reader.deserialize(rawdata, connection.msgtype)
        timestamps.append(timestamp)
        orientations_quat.append([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        linear_acceleration.append([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])

# Convert nanoseconds to seconds relative to start
start_time = timestamps[0]
time_seconds = [(t - start_time) * 1e-9 for t in timestamps]

# Rotate acceleration into world frame and subtract gravity
acc_world = []
for acc, quat in zip(linear_acceleration, orientations_quat):
    r = R.from_quat(quat)
    acc_w = r.apply(acc)  # rotate to world frame
    acc_w[2] -= 9.81      # subtract gravity
    acc_world.append(acc_w)

acc_world = np.array(acc_world)
acc_world_filtered = highpass(acc_world, cutoff=0.01, fs=50)

# Integrate to get velocity
# linear_velocity = it.cumtrapz(acc_world, time_seconds, axis=0, initial=0.0)
linear_velocity = it.cumtrapz(acc_world_filtered, time_seconds, axis=0, initial=0.0)

# Integrate to get position
position = it.cumtrapz(linear_velocity, time_seconds, axis=0, initial=0.0)

# Plot
plt.figure(figsize=(15, 10))

plt.subplot(3, 2, 1)
plt.plot(time_seconds, linear_velocity[:, 0], label='Velocity X')
plt.xlabel('Time (s)')
plt.ylabel('Velocity X (m/s)')
plt.legend()
plt.grid()

plt.subplot(3, 2, 2)
plt.plot(time_seconds, linear_velocity[:, 1], label='Velocity Y')
plt.xlabel('Time (s)')
plt.ylabel('Velocity Y (m/s)')
plt.legend()
plt.grid()

plt.subplot(3, 2, 3)
plt.plot(time_seconds, linear_velocity[:, 2], label='Velocity Z')
plt.xlabel('Time (s)')
plt.ylabel('Velocity Z (m/s)')
plt.legend()
plt.grid()

plt.subplot(3, 2, 4)
plt.plot(time_seconds, position[:, 0], label='Position X')
plt.xlabel('Time (s)')
plt.ylabel('Position X (m)')
plt.legend()
plt.grid()

plt.subplot(3, 2, 5)
plt.plot(time_seconds, position[:, 1], label='Position Y')
plt.xlabel('Time (s)')
plt.ylabel('Position Y (m)')
plt.legend()
plt.grid()

plt.subplot(3, 2, 6)
plt.plot(time_seconds, position[:, 2], label='Position Z')
plt.xlabel('Time (s)')
plt.ylabel('Position Z (m)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

plt.figure()
plt.plot(position[:, 0], position[:, 1])
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('UUV Trajectory (Top-down View)')
plt.grid()
plt.axis('equal')
plt.show()
