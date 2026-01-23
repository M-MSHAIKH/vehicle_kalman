# Vehicle State Estimation via Sensor Fusion: Implementation on the Audi A2D2 Dataset Using Various Estimation Algorithms
This project implements various state estimation algorithms to accurately estimate a vehicle's trajectory (x,y) by fusing noisy GNSS (Global Navigation Satellite System) measurements with a Kinematic Bicycle Model **[Sensor Fusion]**.

**The Challenge: Asynchronous & Noisy Data**

In real-world scenarios, such as those found in the Audi A2D2 dataset, raw CAN bus data is often noisy and asynchronous. This inconsistency makes it unsuitable for high-level control tasks or deep learning models without significant preprocessing.

**The Solution: State Estimation**

To bridge this gap, this project utilizes an Extended Kalman Filter (EKF) to:

Generate Synchronous Data: Produce a uniform, time-aligned dataset where all sensor information is synchronized.

Estimate Hidden States: Predict underlying vehicle dynamics that are difficult to measure directly with hardware sensors, such as the vehicle slip angle (β) **[State Estimation]**.

**Applications for Autonomous Driving**

The resulting high-fidelity, synchronous dataset provides a "clean" ground truth that can be used to train or test advanced autonomous driving modules, including:

Motion Planning: Model Predictive Control (MPC).

End-to-End Learning: Transformer-based trajectory prediction and behavior modeling.




## Project Structure

```
vehicle_kalman/
├── ekf/
│   ├── __init__.py
│   └── ekf.py                          # EKF implementation, Jacobians, noise matrices
├── vehicle_model/
│   ├── __init__.py
│   └── kinematic_bicycle_model.py      # Vehicle Kinematic model
├── pre_processing/
│   ├── __init__.py
│   └── data_loading.py                 # CAN bus data extraction & preprocessing
├── img/                                # Result visualizations
├── ekf_main.py                         # Main execution script
└── README.md
```


## Data Processing
As the actual CAN Bus raw data from the AUDI A2D2 dataset available are asynchronous. That means each sensor send a signal at it's own frequency. To use the dataset for other application such as for testing the MPC or other controller requires a synchronous dataset. Same for the state estimation the control input would be same size. Moreover the timestamp given in a dataset is a **unix timestamps**. Longitudinal and Lateral coordinates are also in the global frame. 

### Converting CAN bus unix timestamp to the standard date time format

The CAN (Controller Area Network) bus is a message-based protocol used for communication between electronic control units (ECUs) in a vehicle. Unlike a system with a single clock that triggers all sensors simultaneously, a CAN bus operates asynchronously. This means each ECU or sensor broadcasts its data on the bus whenever it has an updated value to send, not at a predetermined, synchronized interval. 

To eradicate this problem, the most common solution is to implement the master or common clock for all signals. This common clock should have a equal time step (eg. at each 0.005 sec) for uniform sampling. In other word, the common clock must have a frequency equal or higher than a highest frequency sensor signal (eg. IMU). Then it will include all the sensor signals accurately. Otherwise the aliasing of signals will occur (**Nyquist-Shannon Sampling Theorem**). 

In python, the datetime class contain a method called fromtimestamp. This method directly convert the unix timestamps of the micro-controller into the UTC (Universal Time Coordinates). The UTC will give a Hour, Minute, Second, Microsecond. This can further be converted into the common_time using a method called generate_time_vector. Which is written in the data_loading.py file.

### Converting the longitude and latitude from degrees to meters

In the datasets, the longitudinal and latitude coordinate directly obtains from a GPS (Global Positioning System). This GPS usually gives a coordinate into the WGS 84 (World Geodetic System 1984) in radians. The WGS 84, defines an Earth-centered, Earth-fixed coordinate system and a geodetic datum [1].

Converting a vehicle's longitude and latitude (from a GPS, in the WGS 84 datum) into local Cartesian x and y state coordinates can be done by defining the coordinates first into the local frame, such as ENU (East-North-Up). Where the x-axis points east, y-axis axis south and the up direction indicates the z-axis. This transformation accounts for the curvature of the Earth and the Earth's shape (ellipsoid model, like WGS 84). Moreover, this local ENU frame gives the world fixed vehicle position. As this ENU frame is anchored to a specific spot on the Earth's surface and does not move or rotate when the car moves. To eliminate the difference in orientation between the world-fixed frame (eg. ENU) and a vehicle, **rotation matrix** can be use.

### Event Trigger System 

## How to Use

You can copy paste the file path in the data_loading.py, which is accessible under the pre_processing folder. The file path should be the path of an actual unzipped CAN bus .json file. Now, run the ekf_main.py to obtain all the result including the estimated yaw angle.
## Result

The EKF estimation along with the actual GNSS measurement shown below.

![Comparision With Vehicle Estimated Path with GNSS Measurement](img/EKF_estimated_vehicle_path_Ingolstadt.gif)
![](img/EKF_estimated_vehicle_path_Munich.gif)
![](img/EKF_estimated_vehicle_path_Gaimerscheim.gif)

### Yaw (Heading) Angle Estimation
The figure below shows the relationship between the IMU-measured yaw rate and the yaw (heading) angle estimated using an Extended Kalman Filter (EKF). The yaw rate corresponds to the angular velocity about the vehicle’s vertical (z) axis measured by the IMU gyroscope. Here, Peaks in the signal represent turning maneuvers, while values close to zero indicate straight-line driving. The yaw angle($\psi$) is estimated by the EKF using a kinematic process model driven primarily by the IMU yaw rate.
The yaw angle is intentionally unwrapped, allowing continuous heading evolution beyond ($\pm\pi$).

![](img/Yaw_Rate_and_Yaw_Angle_Ingolstadt.png)

Periods where the yaw rate is close to zero correspond to nearly constant yaw angle.
Sustained positive or negative yaw-rate excursions result in smooth increases or decreases in the estimated yaw angle.
Sharp yaw-rate peaks are reflected as steeper slopes in the yaw-angle trajectory.
The estimated yaw angle follows the integrated trend of the IMU yaw rate, while remaining smoother due to EKF filtering.

This comparison demonstrates that the EKF yaw estimate is physically consistent with the IMU yaw-rate measurements, indicating correct sign conventions, proper integration behavior, and stable filter tuning. The smoothness of the yaw angle relative to the raw yaw rate further confirms effective noise attenuation by the EKF.






## References

1. https://en.wikipedia.org/wiki/World_Geodetic_System
2. https://www.a2d2.audi/en/
3. Teoh, T.S., Em, P.P., Ab Aziz, N.A.B., 2023. Vehicle Localization Based On IMU, OBD2, and GNSS Sensor Fusion Using Extended Kalman Filter. International Journal of Technology. Volume 14(6), pp. 1237-1246


