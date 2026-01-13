from ekf.ekf import ExtendedKalmanFilter
from ekf.ekf import JacobianofKinematicBicycleModel
from ekf.ekf import matrices_definition
from vehicle_model.kinematic_bicycle_model import KinematicBicycleModel
import numpy as np
import matplotlib.pyplot as plt
from pre_processing.data_loading import pre_processed_data

# Load pre-processed data
omega_z = pre_processed_data['omega_z_event_array']
vx_event_array = pre_processed_data['vx_event_array']
xlong_event_array = pre_processed_data['xlong_event_array']
ylat_event_array = pre_processed_data['ylat_event_array']
common_time = pre_processed_data['common_time']
dt = pre_processed_data['dt']
# Example usage of the Extended Kalman Filter
n = len(omega_z)
num_states = 3  # [x, y, psi]
num_control = 2  # [Vx, omega_z]
model = KinematicBicycleModel(dt, lr=2.5)  # Example wheelbase length
jacobian = JacobianofKinematicBicycleModel(dt)
Q = matrices_definition().process_noise_covariance(0.3,0.3) # (0.3,0.3)
R = matrices_definition().measurement_noise_covariance(25,30) #(25,30)
X_pred = np.zeros((n, num_states, 1))
X_upd = np.zeros((n, num_states, 1))
y_tilde = np.zeros((n, 2, 1))
jacobian_var = np.zeros((n, num_states, 1))
P_pred = np.zeros((n, num_states, num_states))
P_upd = np.zeros((n, num_states, num_states)) 
S = np.zeros((n, 2, 2))
K = np.zeros((n, num_states, num_control))  
U = np.zeros((n, num_control, 1))
F = np.zeros((n, num_states, num_states))
L = np.zeros((n, num_states, num_control))
diff = np.zeros((n,2,1))
H = jacobian.H_jacobian()
M = jacobian.M_jacobian()
I = np.eye(num_states)
Y = np.array([[1, 0, 0],
              [0, 1, 0]])  # Measurement matrix

for i in range(n):
    # Prediction Step
    U[i] = np.array([[vx_event_array[i]], [omega_z[i]]])  # Control input at time step (i - 1)
    if i == 0:
        X_upd[i] = np.array([[1.8], [0.87], [-1.35*10e-5]])  # Initial state
        P_upd[i] = np.eye(num_states)  # Initial covariance
        continue
        
    # Here should be U[i]
    X_pred[i], jacobian_var[i-1], diff[i] = model.state_transition(X_upd[i-1], U[i-1])
    F[i-1] = jacobian.F_jacobian(jacobian_var[i-1])
    L[i-1] = jacobian.L_jacobian(jacobian_var[i-1])
    P_pred[i] = (F[i-1] @ P_upd[i-1] @ F[i-1].T) + (L[i-1] @ Q @ L[i-1].T)

    # Innovation covariance only for the plotting purpose
    S[i] = (H @ P_pred[i] @ H.T) + R

    # Update Step
    mat_inv = (H @ P_pred[i] @ H.T) + (M @ R @ M.T)
    K[i] = (P_pred[i] @ H.T) @ np.linalg.inv(mat_inv)



    # Innovation or measurement residual
    y_tilde[i] = np.array([[xlong_event_array[i]], [ylat_event_array[i]]]) - np.array([[X_pred[i][0,0]], [X_pred[i][1,0]]])
    # Y_h = Y @ X_pred[i]  # Measurement vector
    X_upd[i] = X_pred[i] + (K[i] @ y_tilde[i])
    P_upd[i] = (I - (K[i] @ H)) @ P_pred[i]


# Plotting the results
# plottinfg the predicted x dot and y dot over time
# ax1 = plt.subplot(211)
# ax1.plot(common_time[:-1], diff[:,0,0], label='predicted x dot')
# ax1.set_xlabel('Time (s)')
# ax1.set_ylabel('X dot (m/s)')
# ax1.set_title('Predicted X dot over Time')
# ax1.grid()
# ax2 = plt.subplot(212)
# ax2.plot(common_time[:-1], diff[:,1,0], label='predicted y dot', color='orange')
# ax2.set_xlabel('Time (s)')
# ax2.set_ylabel('Y dot (m/s)')
# ax2.set_title('Predicted Y dotover Time')
# ax2.grid()
# plt.show()

# Plotting the estimated path vs GNSS measurements
plt.figure()
plt.plot(X_upd[:n,0,0], X_upd[:n,1,0], label='EKF Estimated Path')
plt.plot(xlong_event_array[1:n], ylat_event_array[1:n], color='red', label='GNSS Measurements')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Vehicle Path Estimation using Extended Kalman Filter')
plt.legend()
plt.axis('equal')
plt.show() 

# # potting the slip angle beta over time
# plt.figure()
# plt.plot(common_time[:-1], jacobian_var[:,0,0], label='Slip Angle Beta')
# plt.xlabel('Time (s)')
# plt.ylabel('Slip Angle Beta (rad)')
# plt.title('Slip Angle Beta over Time')
# plt.legend()
# plt.grid()
# plt.show()

##########################################################################################################################
# Interesting plots to visualize the EKF behaviour

# plotting innovation residueal over time
# plt.figure()
# plt.plot(common_time[0:n], y_tilde[:,0,0], label='Innovation Residual X')
# plt.plot(common_time[0:n], y_tilde[:,1,0], label='Innovation Residual Y')
# plt.xlabel('Time (s)')
# plt.ylabel('Innovation Residual (m)')
# plt.title('Innovation Residual over Time')
# plt.legend()
# plt.grid()
# plt.show()

# Innovation vs. 3σ Bounds (The Consistency Test)
S_x = S[:,0,0]
S_y = S[:,1,1]
upper_bound_x = 3 * np.sqrt(S_x)
lower_bound_x = -3 * np.sqrt(S_x)
upper_bound_y = 3 * np.sqrt(S_y)
lower_bound_y = -3 * np.sqrt(S_y)

plt.figure()
plt.plot(common_time[0:n], y_tilde[:,0,0], label='Innovation Residual X')
plt.plot(common_time[0:n], upper_bound_x, 'r--', label='3σ Upper Bound X')
plt.plot(common_time[0:n], lower_bound_x, 'r--', label='3σ Lower Bound X')
plt.xlabel('Time (s)')
plt.ylabel('Innovation Residual X (m)')
plt.title('Innovation Residual X vs. 3σ Bounds')
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(common_time[0:n], y_tilde[:,1,0], label='Innovation Residual Y')
plt.plot(common_time[0:n], upper_bound_y, 'r--', label='3σ Upper Bound Y')
plt.plot(common_time[0:n], lower_bound_y, 'r--', label='3σ Lower Bound Y')
plt.xlabel('Time (s)')          
plt.ylabel('Innovation Residual Y (m)')
plt.title('Innovation Residual Y vs. 3σ Bounds')
plt.legend()
plt.grid()
plt.show()



print("EKF processing completed.")
