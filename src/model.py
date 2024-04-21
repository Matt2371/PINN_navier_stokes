### DEFINES PINN MODEL (AND LOSS FUNCTION) FOR SOLVING NAVIER STOKES ###

import numpy as np
import torch
import torch.nn as nn


class NavierStokesPINN1(nn.Module):
    def __init__(self, hidden_size, num_layers, nu, rho):
        """
        Model 1 conserves mass through latent psi function, only velocity and not
        pressure is used in training. Architecture from Raissi (2019)
        Params:
        hidden_size - int, number of hidden units per layer
        num_layers - int, number of feedforward layers
        nu - float, kinematic viscosity
        rho - float, density
        """
        self.nu = nu
        self.rho = rho

        super(NavierStokesPINN1, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Input layer: input size = 3 (x, y, t)
        layer_list = [nn.Linear(3, self.hidden_size)] 

        for _ in range(self.num_layers - 2):
            layer_list.append(nn.Linear(self.hidden_size, self.hidden_size)) # Hidden layers
        
        # Output layer: output size = 2 (psi, p)
        layer_list.append(nn.Linear(self.hidden_size, 2)) 
        self.layers = nn.ModuleList(layer_list) # Save as Module List

    def forward(self, x, y, t):
        """ 
        Params:
        x - nd array of shape (N, 1), input x coordinates
        y - nd array of shape (N, 1), input y coordinates
        t - nd array of shape (N, 1), input time coordinate
        Returns:
        u - tensor of shape (N, 1), output x-velocity
        v - tensor of shape (N, 1), output y-velocity
        p - tensor of shape (N, 1), output pressure
        f - x-momentum PDE evaluation of shape (N, 1)
        g - y-momentum PDE evaluation of shape (N, 1)

        """
        # Convert input data to tensor
        x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        y = torch.tensor(y, dtype=torch.float32, requires_grad=True)
        t = torch.tensor(t, dtype=torch.float32, requires_grad=True)

        # Feed-forward to calculate pressure and latent psi 
        input_data = torch.hstack([x, y, t]) # (N, 3)
        self.N = input_data.shape[0]
        out = input_data # Initialize feed-forward
        for layer in self.layers[:-1]:
            out = torch.relu(layer(out)) # No activation for the last layer
        
        out = self.layers[-1](out) # Final layer, (N, 2)

        # Seperate psi and pressure p
        psi, p = out[:, [0]], out[:, [1]] # (N, 1) each

        
        # Differentiate psi to find velocities (create_graph = True is useful for taking higher-order derivatives)
        u = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi), create_graph=True)[0] # (N, 1)
        v = -1 * torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0] # (N, 1)

        # Take space and time derivatives according to momentum equation to find PDE loss
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]

        # Evaluate momentum PDE's
        f = u_t + u * u_x + v * u_y + (1 / self.rho) * p_x - self.nu * (u_xx + u_yy) # (N, 1)
        g = v_t + u * v_x + v * v_y + (1 / self.rho) * p_y - self.nu * (v_xx + v_yy) # (N, 1)
        

        return u, v, p, f, g
    
class NavierStokesPINNLoss1(nn.Module):
    """
    Loss function for Model 1
    Implement PINN loss function for Navier-Stokes PINN
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss() # implement mean square error
        
    def forward(self, u, u_pred, v, v_pred, f, g):
        u = torch.tensor(u, dtype=torch.float32)
        v = torch.tensor(v, dtype=torch.float32)
        # Loss due to data
        L_data = self.mse(u, u_pred) + self.mse(v, v_pred)
        # Loss from momentum PDE's
        L_pde = self.mse(f, torch.zeros(f.shape[0], 1)) + self.mse(g, torch.zeros(g.shape[0], 1))

        return L_data + L_pde
    

class NavierStokesPINN2(nn.Module):
    def __init__(self, hidden_size, num_layers, nu, rho):
        """
        Model 2 conserves mass via continuity PDE. Both pressure and velocity used in training.
        Params:
        hidden_size - int, number of hidden units per layer
        num_layers - int, number of feedforward layers
        nu - float, kinematic viscosity
        rho - float, density
        """
        self.nu = nu
        self.rho = rho

        super(NavierStokesPINN2, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Input layer: input size = 3 (x, y, t)
        layer_list = [nn.Linear(3, self.hidden_size)] 

        for _ in range(self.num_layers - 2):
            layer_list.append(nn.Linear(self.hidden_size, self.hidden_size)) # Hidden layers
        
        # Output layer: output size = 3 (u, v, p)
        layer_list.append(nn.Linear(self.hidden_size, 3)) 
        self.layers = nn.ModuleList(layer_list) # Save as Module List

    def forward(self, x, y, t):
        """ 
        Params:
        x - nd array of shape (N, 1), input x coordinates
        y - nd array of shape (N, 1), input y coordinates
        t - nd array of shape (N, 1), input time coordinate
        Returns:
        u - tensor of shape (N, 1), output x-velocity
        v - tensor of shape (N, 1), output y-velocity
        p - tensor of shape (N, 1), output pressure
        f - x-momentum PDE evaluation of shape (N, 1)
        g - y-momentum PDE evaluation of shape (N, 1)

        """
        # Convert input data to tensor
        x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        y = torch.tensor(y, dtype=torch.float32, requires_grad=True)
        t = torch.tensor(t, dtype=torch.float32, requires_grad=True)

        # Feed-forward to calculate pressure and latent psi 
        input_data = torch.hstack([x, y, t]) # (N, 3)
        self.N = input_data.shape[0]
        out = input_data # Initialize feed-forward
        for layer in self.layers[:-1]:
            out = torch.relu(layer(out)) # No activation for the last layer
        
        out = self.layers[-1](out) # Final layer, (N, 3)

        # Seperate psi and pressure p
        u, v, p = out[:, [0]], out[:, [1]], out[:, [2]] # (N, 1) each

        # Take space and time derivatives according to find PDE loss
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]

        # Evaluate momentum PDE's
        f = u_t + u * u_x + v * u_y + (1 / self.rho) * p_x - self.nu * (u_xx + u_yy) # (N, 1)
        g = v_t + u * v_x + v * v_y + (1 / self.rho) * p_y - self.nu * (v_xx + v_yy) # (N, 1)
        # Evaluate continuity PDE
        h = u_x + v_y
        

        return u, v, p, f, g, h


class NavierStokesPINNLoss2(nn.Module):
    """
    Loss function for Model 2
    Implement PINN loss function for Navier-Stokes PINN
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss() # implement mean square error
        
    def forward(self, u, u_pred, v, v_pred, p, p_pred, f, g, h):
        u = torch.tensor(u, dtype=torch.float32)
        v = torch.tensor(v, dtype=torch.float32)
        p = torch.tensor(p, dtype=torch.float32)
        # Loss due to data
        L_data = self.mse(u, u_pred) + self.mse(v, v_pred) + self.mse(p, p_pred)
        # Loss from momentum PDE's
        L_pde = (self.mse(f, torch.zeros_like(f)) + self.mse(g, torch.zeros_like(g)) +
                 self.mse(h, torch.zeros_like(h)))

        return L_data + L_pde