import numpy as np
from vehicle_model.kinematic_bicycle_model import KinematicBicycleModel

class ExtendedKalmanFilter:
   def __init__(self,dt):
      self.dt = dt

   def prediction(self, state, P, control_input, F, L, Q_u):
       
      # Predict the next state
      model = KinematicBicycleModel(self.dt,L)
      state_pred,_ = model.state_transition(state, control_input)
       
      # Predict the error covariance
      P_pred = (F @ P @ F.T) + (L @ Q_u @ L.T)

      # Through an error if any element is nan
      if np.isnan(state_pred).any():
         raise ValueError("Prediction step contains NaN values.")

      return state_pred, P_pred
 
   def update(self, state_pred, P_pred, H, M, y, h, R):
       
       inv_term = (H @ P_pred @ H.T) + (M @ R @ M.T)
       K = P_pred @ H.T @ np.linalg.inv(inv_term)

       state_upd = (state_pred) + (K @ (y - h))
       P_upd = (np.eye(len(P_pred)) - (K @ H)) @ P_pred

       # Through an error if any element is nan
       if np.isnan(state_upd).any() or np.isnan(P_upd).any() or np.isnan(K).any():
           raise ValueError("Update step contains NaN values.")

       return state_upd, P_upd  
   
class JacobianofKinematicBicycleModel:

    def __init__(self, dt):
        self.dt = dt

    def F_jacobian(self, jacobian_var):
        beta, psi, Vx = jacobian_var[0], jacobian_var[1], jacobian_var[2]
        beta = float(beta)
        psi = float(psi)
        Vx = float(Vx)
        F = np.array([[1, 0, -Vx * np.sin(beta + psi) * self.dt],
                      [0, 1,  Vx * np.cos(beta + psi) * self.dt],
                      [0, 0, 1]])
        
        # Through an error if any element is nan
        if np.isnan(F).any():
            raise ValueError("Jacobian F contains NaN values.")
        return F
    
    def L_jacobian(self, jacobian_var):
        beta, psi, _ = jacobian_var[0], jacobian_var[1], jacobian_var[2]
        beta = float(beta)
        psi = float(psi)
        L = np.array([[np.cos(beta + psi) * self.dt, 0],
                      [np.sin(beta + psi) * self.dt, 0],
                      [0,                            self.dt]])
        
        # Through an error if any element is nan
        if np.isnan(L).any():
            raise ValueError("Jacobian L contains NaN values.")
        return L
    
    # Measurement model jacobian
    def H_jacobian(self):
        return np.array([[1, 0, 0], 
                         [0, 1, 0]])
    
    def M_jacobian(self):
        return np.array([[1, 0],
                         [0, 1]])
    
class matrices_definition:
    def __init__(self):
        pass

    def process_noise_covariance(self, Vx_variance, omega_z_variance):
        Q = np.array([[Vx_variance, 0],
                        [0, omega_z_variance]])
        return Q

    def measurement_noise_covariance(self, x_pos_variance, y_pos_variance):
        R = np.array([[x_pos_variance, 0],
                      [0, y_pos_variance]])
        return R