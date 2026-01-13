import numpy as np

class KinematicBicycleModel:
    def __init__(self, dt,lr):
        self.dt = dt
        self.lr = lr
        
    def state_transition(self, state, control_input):
        # here the input states should be the previous states at time step (i - 1)
        x, y, psi = state[0,0], state[1,0], state[2,0]
        Vx, omega_z = control_input[0,0], control_input[1,0]

        # Here we calcculate the rate of change of state variables for the previous step (i - 1)
        beta = np.arcsin((omega_z * self.lr) / Vx) 
        if np.isnan(beta):
            beta = 1e-5  # Handle NaN values when Vx is very small
        psi_new = psi + (omega_z * self.dt)
        x_dot = Vx * np.cos(beta + psi)
        y_dot = Vx * np.sin(beta + psi)
        # You can also write this equation in th ematrix form as in the eq. 16 and add procee noise W(i-1)
        x_new = x + (x_dot * self.dt)
        y_new = y + (y_dot * self.dt)
        state_pred = np.array([[x_new], [y_new], [psi_new]])
        jacobian_var = np.array([[beta], [psi], [Vx]])
        diff = np.array([[x_dot], [y_dot]])

        # Through an error if any element is nan
        if np.isnan(state_pred).any():
            raise ValueError("Predicted state contains NaN values [x, y, psi].")
        
        if np.isnan(jacobian_var).any():
            raise ValueError("Jacobian variables contain NaN values [beta, psi, Vx].")
        return state_pred, jacobian_var, diff