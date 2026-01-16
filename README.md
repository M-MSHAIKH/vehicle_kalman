# Vehicle State Estimation via Sensor Fusion: Implementation on the Audi A2D2 Dataset Using Various Estimation Algorithms

This project implements various state estimation algorithms to estimate the trajectory  (x,y) of a vehicle based on noisy GNSS (Global Navigation Satellite System) measurements. At its core, the system utilizes a Kinematic Bicycle Model to predict the vehicle's motion and fuses these predictions with real-time positional data to produce a smooth, accurate path estimation. For validation the Audi A2D2 opensource dataset is used. Apart from the accurate path estimation the other goal of this project is to estimate the underlying vehicle dynamic quantity, such as slip angle which is hard to measure via sensors. 

## Data Processing
As the actual CAN Bus raw data from the AUDI A2D2 dataset available are asynchronous. That means each sensor send a signal at it's own frequency. To use the dataset for other application such as for testing the MPC(Model Predictive Controller) or other controller requires a synchronous dataset. Same for the state estimation the control input would be same size. Moreover the timestamp given in a dataset is a **unix timestamps**. Longitudinal and Lateral coordinates are also in the global frame. 

### Converting CAN bus unix timestamp to the standard date time format

The CAN (Controller Area Network) bus is a message-based protocol used for communication between electronic control units (ECUs) in a vehicle. Unlike a system with a single clock that triggers all sensors simultaneously, a CAN bus operates asynchronously. This means each ECU or sensor broadcasts its data on the bus whenever it has an updated value to send, not at a predetermined, synchronized interval. 

To eradicate this problem, the most common solution is to implement the master or common clock for all signals. This common clock should have a equal time step (eg. at each 0.005 sec) for uniform sampling. In other word, the common clock must have a frequency equal or higher than a highest frequency sensor signal (eg. IMU). Then it will include all the sensor signals accurately. Otherwise the aliasing of signals will occur (**Nyquist-Shannon Sampling Theorem**). 

In python, the datetime class contain a method called fromtimestamp. This method directly convert the unix timestamps of the micro-controller into the UTC (Universal Time Coordinates). The UTC will give a Hour, Minute, Second, Microsecond. This can further be converted into the common_time using a method called generate_time_vector. Which is written in the data_loading.py file.

### Converting the longitude and latitude from degrees to meters

In the datasets, the longitudinal and latitude coordinate directly obtains from a GPS (Global Positioning System). This GPS usually gives a coordinate into the WGS 84 (World Geodetic System 1984) in radians. The WGS 84, defines an Earth-centered, Earth-fixed coordinate system and a geodetic datum [1].

Converting a vehicle's longitude and latitude (from a GPS, in the WGS 84 datum) into local Cartesian x and y state coordinates can be done by defining the coordinates first into the local frame, such as ENU (East-North-Up). Where the x-axis points east, y-axis axis south and the up direction indicates the z-axis. This transformation accounts for the curvature of the Earth and the Earth's shape (ellipsoid model, like WGS 84). Moreover, this local ENU frame gives the world fixed vehicle position. As this ENU frame is anchored to a specific spot on the Earth's surface and does not move or rotate when the car moves. To eliminate the difference in orientation between the world-fixed frame (eg. ENU) and a vehicle, **rotation matrix** can be use.

### Event Trigger System 

## Result

The EKF estimation along with the actual GNSS measurement shown below.

![Comparision With Vehicle Estimated Path with GNSS Measurement](EKF_estimated_vehicle_path.gif)





## References

1. https://en.wikipedia.org/wiki/World_Geodetic_System


