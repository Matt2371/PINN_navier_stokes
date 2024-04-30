### TRAIN MODEL ON CLYNDER WAKE DATA ###
from src.model import NavierStokesPINN1, NavierStokesPINNLoss1
from src.model import NavierStokesPINN2, NavierStokesPINNLoss2
from src.load_cynlinder_wake import load_cylinder_wake
import torch.nn as nn
import torch
import numpy as np
import scipy
from tqdm import tqdm

def boundary_indices(N, T):
    """ 
    Returns a boolean mask marking the boundary condition for all timesteps
    Params:
    N - # of data samples (space locations) in problem
    T - # of timesteps
    Return:
    nd-array of shape (N*T, )
    """
    # Create grid for one timestep
    grid_t_0 = np.zeros((50, 100))

    # Set boundary to 1
    grid_t_0[0, :] = 1
    grid_t_0[:, 0] = 1
    grid_t_0[-1, :] = 1
    grid_t_0[:, -1] = 1

    # Flatten and propagate over T timesteps
    grid_t_all = np.tile(grid_t_0.reshape(-1, 1), (1, T)) # (N, T)
    boundary_positions = grid_t_all.astype(bool).flatten()

    # For example:
    # boundary_positions.reshape(N, T)[:, [t]].reshape(50, 100) is the same as grid_t_0
    return boundary_positions

def main(hidden_size, num_layers, epochs, model, train_selection):
    """ 
    Params:
    hidden_size - int, # of hidden units for each neural network layer
    num_layers - int, # of neural network layers
    epochs - int, # of training epochs
    model - int, whether to use model 1 (Raissi 2019) or model 2 (continuity PDE)
    train_selection - float, frac of all data (N*T) to select for training OR 
                      'BC', select the boundary conditions for training (all timesteps)
    """
    # Load flattened cynlinder wake data
    x_all, y_all, t_all, u_all, v_all, p_all, (N, T) = load_cylinder_wake() # (NT, 1)

    # Select training data by random selecting DNS data
    if train_selection == 'BC':
        idx = boundary_indices(N, T)
    else:
        samples = int(round(N * T * train_selection))
        np.random.seed(0)
        idx = np.random.choice(x_all.shape[0], samples, replace=False)

    x_train = x_all[idx, :]
    y_train = y_all[idx, :]
    t_train = t_all[idx, :]
    u_train = u_all[idx, :]
    v_train = v_all[idx, :]
    p_train = p_all[idx, :]


    # Instantiate model, criterion, and optimizer
    nu = 0.01
    rho = 1

    torch.manual_seed(0)
    if model == 1:
        PINN_model = NavierStokesPINN1(hidden_size=hidden_size, num_layers=num_layers, nu=nu, rho=rho)
        criterion = NavierStokesPINNLoss1()
    elif model == 2:
        PINN_model = NavierStokesPINN2(hidden_size=hidden_size, num_layers=num_layers, nu=nu, rho=rho)
        criterion = NavierStokesPINNLoss2()

    optimizer = torch.optim.LBFGS(PINN_model.parameters(), line_search_fn='strong_wolfe')

    # Training loop
    def closure():
        """Define closure function to use with LBFGS optimizer"""
        optimizer.zero_grad()   # Clear gradients from previous iteration

        if model == 1:
            u_pred, v_pred, p_pred, f, g = PINN_model(x_train, y_train, t_train)
            loss = criterion(u_train, u_pred, v_train, v_pred, f, g)
        elif model == 2:
            u_pred, v_pred, p_pred, f, g, h = PINN_model(x_train, y_train, t_train)
            loss = criterion(u_train, u_pred, v_train, v_pred, p_train, p_pred, f, g, h)
    
        loss.backward() # Backprogation
        return loss 

    def training_loop(epochs):
        """Run full training loop"""
        for i in tqdm(range(epochs), desc='Training epochs: '):
            optimizer.step(closure)

    training_loop(epochs=epochs)

    # Save trained model
    torch.save(PINN_model.state_dict(), f'data/model{model}_{num_layers}l_{hidden_size}h_{epochs}e_{train_selection}d.pt')
    return

if __name__ == '__main__':
    main(hidden_size=30, num_layers=5, epochs=200, model=2, train_selection=0.005)