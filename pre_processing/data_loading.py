import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import data
data = pd.read_json("/Users/moaadil/audi_a2dc/ingolstadt/camera_lidar/20190401_121727/bus/20190401121727_bus_signals.json")
def extract_signal_data(signal_name):
    for key,value in data.items():
        if key == signal_name:
            values = value['values']
            signal_data = [pair[1] for pair in values]
            timestamps = [pair[0] for pair in values]
            return signal_data, timestamps
        
# Extracting all required signals
long_x,timestamps_x = extract_signal_data('longitude_degree')   # degrees
lat_y, timestamps_y = extract_signal_data('latitude_degree')    # degrees
ax, timestamps_ax = extract_signal_data('acceleration_x')       # m/s^2
Vx, timestamps_Vx = extract_signal_data('vehicle_speed')        # km/hr
Vx = (np.array(Vx) * 1000) / 3600                               # Convert to m/s
delta, timestamps_delta = extract_signal_data('steering_angle_calculated')  # degrees 
delta = np.deg2rad(np.array(delta))  # Convert to radians
delta_sign, t_dsign = extract_signal_data('steering_angle_calculated_sign')  # 0 or 1
Vpsi, timestamps_Vpsi = extract_signal_data('angular_velocity_omega_z')  # degrees/s

# plot to verify the extracted signals

# fig, axs = plt.subplots(3,2, figsize=(12, 10))
# axs[0, 0].plot(np.linspace(0, len(long_x), len(long_x)), long_x, color='b')
# axs[0, 0].set_title('Longitude')
# axs[0, 1].plot(np.linspace(0, len(lat_y), len(lat_y)), lat_y, color='g')
# axs[0, 1].set_title('Latitude')
# axs[1, 0].plot(np.linspace(0, len(ax), len(ax)), ax, color='r')
# axs[1, 0].set_title('Acceleration X')           
# axs[1, 1].plot(np.linspace(0, len(Vx), len(Vx)), Vx, color='c')
# axs[1, 1].set_title('Vehicle Speed')
# axs[2, 0].plot(np.linspace(0, len(delta), len(delta)), delta, color='m')
# axs[2, 0].set_title('Steering Angle')
# axs[2, 1].plot(np.linspace(0, len(Vpsi), len(Vpsi)), Vpsi, color='y')
# axs[2, 1].set_title('Angular Velocity Z')
# plt.tight_layout()
# plt.suptitle('Extracted Signals direct from CAN bus')
# plt.show()

#####################################################################################################################
# 1. Converting unix timestamp to the standard date time format
# AS the timestamp is in microseconds, so dividing by 1e6
# utc stands for the Universal Time Coordinated, which is used for the microcontrollers

timestamps_longx = [datetime.datetime.fromtimestamp(ts/1e6, datetime.timezone.utc) for ts in timestamps_x]
timestamps_lat = [datetime.datetime.fromtimestamp(ts/1e6, datetime.timezone.utc) for ts in timestamps_y]
timestamps_ax = [datetime.datetime.fromtimestamp(ts/1e6, datetime.timezone.utc) for ts in timestamps_ax]
timestamps_Vx = [datetime.datetime.fromtimestamp(ts/1e6, datetime.timezone.utc) for ts in timestamps_Vx]
timestamps_delta = [datetime.datetime.fromtimestamp(ts/1e6, datetime.timezone.utc) for ts in timestamps_delta]
timestamps_Vpsi = [datetime.datetime.fromtimestamp(ts/1e6, datetime.timezone.utc) for ts in timestamps_Vpsi]


# assigning the ref time, min and max timestamp. As the min and max timestamps can be found from the acceleration data, beacuse it has the highest sampling frequency or highest number of data points
t_min = timestamps_ax[0]
t_max = timestamps_ax[-1]
# Generating a time vector in sec
def generate_time_vector(timestamps):
    """
    Generate a time vector in seconds from a list of datetime objects.
    
    Parameters:
    timestamps (list of datetime): List of datetime objects representing timestamps. In datetime.datetime(10,44,23,12,17,44,693201,tzinfo=datetime.timezone.utc), the last part tzinfo=datetime.timezone.utc indicates that the time is in Coordinated Universal Time (UTC).
    
    Returns:
    np.ndarray: Array of time values in seconds.
    
    """
    t = np.zeros(np.size(timestamps))  # Initialize an empty array to store time values
    for i, dt in enumerate(timestamps):
        hour = dt.hour
        minute = dt.minute
        second = dt.second
        microsecond = dt.microsecond

        # For the first timestamp, calculate elapsed time from t_min
        if i == 0:
            if timestamps[0] == t_min and timestamps[-1] == t_max:
                t_current = 0.0
            else:
                elapsed_time = (hour - t_min.hour) * 3600 + (minute - t_min.minute) * 60 + (second - t_min.second) + (microsecond - t_min.microsecond) / 1e6
                t_current = elapsed_time
        else:
            elapsed_time = (hour - t_min.hour) * 3600 + (minute - t_min.minute) * 60 + (second - t_min.second) + (microsecond - t_min.microsecond) / 1e6        # Elapsed time in seconds from t_min
            t_current = elapsed_time  # Update t with the new elapsed time in seconds
        
        # Assign the calculated time to the array
        t[i] = t_current

        # Raise negative value error
        if t[0] < 0:
            raise ValueError("Negative time value found at index {}: {}".format(i, t[0]))
        
        # Raise smaller previous time value error
        if i > 0 and t[i] < t[i - 1]:
            raise ValueError("Time value at index {} is smaller than previous value: {} < {}".format(i, t[0], t[i - 1]))

    return t

# Generating time vector for each signal based on their timestamps (NON-UNIFORM SAMPLING TIME)
t_longx = generate_time_vector(timestamps_longx)
t_laty = generate_time_vector(timestamps_lat)
t_ax= generate_time_vector(timestamps_ax)
t_Vx= generate_time_vector(timestamps_Vx)
t_delta= generate_time_vector(timestamps_delta)
t_Vpsi = generate_time_vector(timestamps_Vpsi)

# Check if the size of all time vectors are not equal
# if np.size(t_longx) != np.size(t_laty) or np.size(t_longx) != np.size(t_ax) or np.size(t_longx) != np.size(t_Vx) or np.size(t_longx) != np.size(t_delta):
#     print("The size of time vectors are not equal. Hence, the data is collected at different sampling frequencies.")   
# 
######################################################################################################################  
# MAking a common_time frame

# Calculate time differences for each signal
diff_tx = np.diff(t_longx)
diff_ty = np.diff(t_laty)
diff_ta = np.diff(t_ax)
diff_tv = np.diff(t_Vx)
diff_td = np.diff(t_delta)
diff_tVpsi = np.diff(t_Vpsi)

# Choose a common time frame (e.g., based on the highest frequency sensor)
max_freq = max(1/np.mean(diff_tx), 1/np.mean(diff_ty), 1/np.mean(diff_ta), 1/np.mean(diff_tv), 1/np.mean(diff_td))
print("Max frequency (Hz):", max_freq)      # The highest frequency is 200 Hz of the Acceleration sensor
min_freq = min(1/np.mean(diff_tx), 1/np.mean(diff_ty), 1/np.mean(diff_ta), 1/np.mean(diff_tv), 1/np.mean(diff_td))
print("Min frequency (Hz):", min_freq)    # The lowest frequency is 5 Hz of the GPS sensor (longitude and latitude measurement)
common_freq = 200  # Choose a common frequency (e.g., based on the highest frequency sensor)
common_dt = 1 / common_freq  # Time step for the common frequency
max_time = np.max([np.max(t_longx), np.max(t_laty), np.max(t_ax), np.max(t_Vx), np.max(t_delta)])
min_time = np.min([np.min(t_longx), np.min(t_laty), np.min(t_ax), np.min(t_Vx), np.min(t_delta)])

# np.rint round off to nearest integer or use int only, you will get an value error solve it later on
num_time = np.rint((max_time - min_time)/common_dt)
common_time = np.arange(min_time, 
                        max_time, 
                        common_dt)

if np.shape(common_time)[0] != num_time:
    print("Shape mismatch in common_time array.")

#######################################################################################################################
# 
# 2. Converting the longitude and latitude from degrees to meters
# WGS84 constants for more correct ECEF conversion

a = 6378137.0            # semi-major axis, meters
f = 1/298.257223563      # flattening
e2 = f * (2 - f)         # eccentricity squared

def geodetic_to_ecef(lat_rad, lon_rad, h=0.0):
    """
    Convert geodetic coordinates (radians, meters) to ECEF (meters) using WGS84.
    Instead of angles, the location is expressed as distances (meters) from the center of the Earth along 3 perpendicular axes.
    Inputs can be scalars or numpy arrays.
    angles --> meters
    """
    slat = np.sin(lat_rad)
    clat = np.cos(lat_rad)
    N = a / np.sqrt(1 - e2 * slat**2)   # prime vertical radius
    X = (N + h) * clat * np.cos(lon_rad)
    Y = (N + h) * clat * np.sin(lon_rad)
    Z = (N * (1 - e2) + h) * slat
    return X, Y, Z

def ecef_to_enu_matrix(lat0_rad, lon0_rad):
    slat = np.sin(lat0_rad); clat = np.cos(lat0_rad)
    slon = np.sin(lon0_rad); clon = np.cos(lon0_rad)
    R = np.array([
        [-slon,           clon,            0.0],
        [-slat*clon, -slat*slon,  clat],
        [ clat*clon,  clat*slon,  slat]
    ])
    return R

def latlon_to_enu(lat_deg, lon_deg, lat0_deg=None, lon0_deg=None, h=0.0, h0=0.0):
    """
    Convert latitude and longitude in degrees to local ENU coordinates in meters.
    lat_deg, lon_deg: arrays or scalars of points
    lat0_deg, lon0_deg: reference point in degrees. If None, use first point.
    h, h0: altitudes in meters (can be scalar)
    Returns east, north, up arrays in meters
    """
    lat = np.deg2rad(np.array(lat_deg))
    lon = np.deg2rad(np.array(lon_deg))
    if lat0_deg is None: lat0_deg = np.rad2deg(lat[0])
    if lon0_deg is None: lon0_deg = np.rad2deg(lon[0])
    lat0 = np.deg2rad(lat0_deg)
    lon0 = np.deg2rad(lon0_deg)

    # ECEF coordinates
    X, Y, Z = geodetic_to_ecef(lat, lon, h)
    X0, Y0, Z0 = geodetic_to_ecef(lat0, lon0, h0)

    # Delta in ECEF
    dX = X - X0
    dY = Y - Y0
    dZ = Z - Z0
    d = np.vstack((dX, dY, dZ))   # shape (3, N)

    # Rotate to ENU
    R = ecef_to_enu_matrix(lat0, lon0)
    enu = R.dot(d)   # shape (3, N)
    east = enu[0, :]
    north = enu[1, :]
    up = enu[2, :]
    return east, north, up   

# Apply the conversion to the latitude and longitude degrees
x_long, y_lat, z_alt = latlon_to_enu(lat_y, long_x)
# As the vehicle is moving on the north side all the time, the longitudinal value is negative. So, taking its absolute value
x_long = np.abs(x_long)

#######################################################################################################################

# 3. Adjusting the steering angle
for i in range(len(delta_sign)):
    if delta_sign[i] == 0:
        delta_sign[i] = 1       # Positive sign or taking the right turn
    else:
        delta_sign[i] = -1    # Negative sign or taking the left turn

delta_sign_ch = delta * delta_sign

#########################################################################################################################

# 4. Changing the angular velocity from degrees/s to rad or psi_dot --> psi (should I have to convert first into the radian and then divided by sec or have to do vice versa)

# I Think heading anglew also need to rotate in ENU frame!!!!
Vpsi_rad = np.deg2rad(Vpsi)  # or Vpsi * np.pi/180
def diff_func(time_array, heading_velocity):
    diff_result = np.zeros_like(time_array)
    psi = np.zeros_like(time_array)
    for i in range(1, len(time_array)):
        if i == 0:
            diff_result[i] = 0.0
            psi[i] = 0.0
        else:
            diff_result[i] = time_array[i] - time_array[i-1]

            # If you only covert the cuurrent psi and do not add the previous psi value, then it will give wrong results. 
            # (You’re only getting the instantaneous small change, not the total accumulated orientation.)
            # You can also use the trapazoidal integration method to get more accurate results.
            psi[i] = psi[i-1] + (heading_velocity[i] * diff_result[i])   # rad
    return diff_result, psi

diff_Vpsi, psi= diff_func(t_Vpsi, Vpsi_rad)

# plotting psi and Vpsi to verify
# fig, axs = plt.subplots(1,2, figsize=(10,4))
# axs[0].plot(np.linspace(0, len(Vpsi), len(Vpsi)), Vpsi, color='b')
# axs[0].set_title('Angular Velocity Z - degrees/s')
# axs[1].plot(np.linspace(0, len(psi), len(psi)), psi, color='r')
# axs[1].set_title('Heading angle - rad')
# plt.tight_layout()
# plt.show()

#####################################################################################################################################
# Event Trigger classes for each sensor (Fort IMU the event trigger is not needed as it is the highest frequency sensor, 
# based of which the common time is defined) (It is also known as Zero Order Hold (ZOH) method)

# Solution: Using a Time tolerance--> the tolerance, ϵ (epsilon), defines the acceptable time window around common time step 
# for a sensor event to be considered "current."

# Trigger if: |t_sensor - t_common| <= ϵ
#####################################################################################################################################
delta = common_time[1] - common_time[0]
epsilon = delta / 2  # half the time step as tolerance

def event_triggered(time_sensor, global_time):
    # If the common time is less than or equal to the time of the GPS measurement, then the event is triggered
    if np.abs(time_sensor - global_time) <= epsilon:
        cond = 1
    elif np.abs(time_sensor - global_time) > epsilon:
        cond = 0
    else:
        print("Error in event-triggered condition.")
    return cond


# xlong_event = np.zeros((len(common_time),1))
# j = 0
# for i in range(len(t_longx)):
#     for j in range(len(common_time)):
#         if event_triggered(t_longx[i], common_time[j]) == 1:
#             xlong_event[j] = x_long[i]

def zoh_event_triggered(t_sensor, global_time, sensor_data):
    """
    Zero-Order Hold (ZOH) event triggering for sensor data.
    Hold the last valid measurement until a new measurement 
    is available within a specified time tolerance.

    Parameters:
    t_sensor (np.ndarray): Timestamps of the sensor measurements.
    global_time (np.ndarray): Common time frame for synchronization.
    sensor_data (np.ndarray): Sensor measurements corresponding to t_sensor.

    Returns:
    np.ndarray: ZOH synchronized sensor data aligned with global_time.
    """
    delta = global_time[1] - global_time[0]
    epsilon = delta / 2  # half the time step as tolerance

    indices = np.zeros_like(global_time, dtype=int)
    event = np.zeros_like(global_time, dtype=int)
    j = 0
    for i, t in enumerate(global_time):
        while j < len(t_sensor) and t_sensor[j] <= t + epsilon:
            j += 1
            event[i] = 1
        indices[i] = j


        # Adjust index to get the last valid measurement within the tolerance
        if indices[i] == 0:
            indices[i] = 0
        else:
            indices[i] -= 1    # This is how python indices work
    indices[indices < 0] = 0    # Ensure no negative indices
    zoh_array = sensor_data[indices]

    if np.shape(zoh_array)[0] != np.shape(global_time)[0]:
        raise ValueError("Shape mismatch in ZOH array.")
    
    if zoh_array[-1] != sensor_data[-1]:
        raise ValueError("ZOH array last element does not match sensor data last element, indicating potential indices error.")
    return zoh_array, indices, event


xlong_event_array, xlong_indices, xlong_event = zoh_event_triggered(t_longx, common_time, x_long)
ylat_event_array, ylat_indices, ylat_event = zoh_event_triggered(t_laty, common_time, y_lat)
psi_event_array = np.hstack([psi, psi[-1]]) # to match the length (Do not use zero order hold as the common time is based on the highest frequency sensor)
vx_event_array, vx_indices, vx_event = zoh_event_triggered(t_Vx, common_time, Vx)
delta_event_array, delta_indices, delta_event = zoh_event_triggered(t_delta, common_time, delta_sign_ch)
ax_event_array = np.hstack([ax, ax[-1]]) # to match the length (Do not use zero order hold as the common time is based on the highest frequency sensor)
psi_event = np.ones_like(common_time)  # since psi is at highest frequency, we can set all to 1
ax_event = np.ones_like(common_time)  # since ax is at highest frequency, we can set all to 1

########################. Just for checking the event-triggered arrays ########################
cond1 = np.zeros_like(common_time)
cond2 = np.zeros_like(common_time)
cond3 = np.zeros_like(common_time)
cond4 = np.zeros_like(common_time)
cond5 = np.zeros_like(common_time)
cond6 = np.zeros_like(common_time)
cond7 = np.zeros_like(common_time)
cond8 = np.zeros_like(common_time)
for i in range(len(common_time)):
    # GPS update (x, y)
    if (xlong_event[i] == 1 or ylat_event[i] == 1) and vx_event[i] == 0 and psi_event[i] == 0:
        cond1[i] = 1

    # Yaw update (psi)
    elif psi_event[i] == 1 and (xlong_event[i] == 0 and ylat_event[i] == 0) and vx_event[i] == 0:
        cond2[i] = 1

    # Velocity update (Vx)
    elif vx_event[i] == 1 and (xlong_event[i] == 0 and ylat_event[i] == 0) and psi_event[i] == 0:
        cond3[i] = 1
    
    # GPS + Yaw update
    elif (xlong_event[i] == 1 or ylat_event[i] == 1) and psi_event[i] == 1 and vx_event[i] == 0:
        cond4[i] = 1

    # GPS + Velocity update
    elif (xlong_event[i] == 1 or ylat_event[i] == 1) and vx_event[i] == 1 and psi_event[i] == 0:
        cond5[i] = 1

    # Yaw + Velocity update
    elif psi_event[i] == 1 and vx_event[i] == 1 and (xlong_event[i] == 0 and ylat_event[i] == 0):
        cond6[i] = 1

    # GPS + Yaw + Velocity update
    elif (xlong_event[i] == 1 or ylat_event[i] == 1) and psi_event[i] == 1 and vx_event[i] == 1:
        cond7[i] = 1

    else:
        cond8[i] = 1  # No measurement


## Plotting the event-triggered measurements
# plt.figure()
# plt.plot(common_time, cond6, label='Event-triggered measurement conditions', drawstyle='steps-post')
# plt.xlabel('Time (s)')
# plt.ylabel('Condition Code')
# plt.title('Event-triggered Measurement Conditions Over Time')
# plt.yticks([0, 1], ['No Measurement', 'Measurement Taken'])
# plt.legend()
# plt.grid()
# plt.show()


#########################################################################################################################
# Making the dictinary for the pre-processed data
# psi is not heading angle (rad)and the omega_z is the angular velocity around z-axis (rad/sec)

pre_processed_data = {
    'common_time': common_time,
    'dt': common_dt,
    'time_ax': t_ax,
    'time_Vx': t_Vx,
    'time_delta': t_delta,
    'time_Vpsi': t_Vpsi,
    'time_longx': t_longx,
    'time_laty': t_laty,
    'xlong_event_array': xlong_event_array,
    'ylat_event_array': ylat_event_array,
    'ax_event_array': ax,
    'vx_event_array': vx_event_array,
    'delta_event_array': delta_event_array,
    'omega_z_event_array': Vpsi_rad,
    'psi_event_array': psi
}