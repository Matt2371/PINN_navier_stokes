import numpy as np
import scipy

def load_cylinder_wake():
    """
    Load and reshape cylinder wake DNS data.
    Returns:
    x, y, t, u, v, p - np.array() of shape (N*T, 1), i.e. every data point is represented for T timesteps in a row
    (N, T) - tuple of dimensions (# of grid cells, timesteps)
    """
    # Read raw data
    data = scipy.io.loadmat('data/cylinder_wake.mat')
    U_star = data['U_star']  # N x 2 x T
    P_star = data['p_star']  # N x T
    t_star = data['t']  # T x 1
    X_star = data['X_star']  # N x 2

    # Reshape data (copy array T times to create time axis)
    N, T = X_star.shape[0], t_star.shape[0]
    XX = np.tile(X_star[:, [0]], (1, T))  # N x T
    YY = np.tile(X_star[:, [1]], (1, T))  # N x T
    TT = np.tile(t_star, (1, N)).T  # N x T
    UU = U_star[:, 0, :]  # N x T
    VV = U_star[:, 1, :]  # N x T
    PP = P_star  # N x T
    
    # Flatten and return results
    x_all = XX.flatten().reshape(-1, 1)  # NT x 1
    y_all = YY.flatten().reshape(-1, 1)  # NT x 1
    t_all = TT.flatten().reshape(-1, 1)  # NT x 1
    u_all = UU.flatten().reshape(-1, 1)  # NT x 1
    v_all = VV.flatten().reshape(-1, 1) # NT x 1
    p_all = PP.flatten().reshape(-1, 1) # NT x 1

    return x_all, y_all, t_all, u_all, v_all, p_all, (N, T)

