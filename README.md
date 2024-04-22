# Flow Reconstruction using Physics Informed Machine Learning
## Overview
Using physics inspired neural networks (PINN) to solve turbulent flows using the Navier-Stokes equations. Specifically, given sparse observations, we reconstruct the entire flow field, i.e., flow reconstruction. We train and validate our results using direct numerical simulation (DNS) data from (Raissi et al., 2019), which models the vortex shedding in the wake past a cylindrical column at $Re=100$.

## Background
Neural networks, when their loss functions are modified to conserve physical laws, have shown promise in solving non-linear partial differential equations in physics (Raissi et al., 2019). This has broad application for computational fluid mechanics, such as solving the Navier-Stokes equations, with demonstrated success in a variety of flow scenarios (Cai et al., 2021; Eivazi et al., 2022). Rather than a pure data-driven approach of a statistical model, physics-inspired neural networks (PINN’s) take a hybrid approach by enforcing physics-based knowledge (PDE’s), while also optimizing the loss according to the data. For example, the incompressible 2D-Navier-Stokes equation gives

$x$-momentum:
$$f:= \frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} + \frac{1}{\rho}\frac{\partial p}{\partial x} -
\nu\left(\frac{\partial^2u}{\partial x^2} + \frac{\partial^2u}{\partial y^2}\right) = 0$$

$y$-momentum:
$$g:= \frac{\partial v}{\partial t} + u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y} + \frac{1}{\rho}\frac{\partial p}{\partial x} -
\nu\left(\frac{\partial^2v}{\partial x^2} + \frac{\partial^2v}{\partial y^2}\right) = 0$$

Continuity:
$$h:= \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$$

where $\rho$ is density and $\nu$ is kinematic viscosity. 

The neural network predicts the x-velocity $u\left(x,y,t\right)$, the y-velocity $v\left(x,y,t\right)$ and the pressure $p\left(x,y,t\right)$ fields. To train the model, we minimize the loss function, L, with components from the data $L_{data}$ and the governing equations $L_{PDE}$ such that

$$L = L_{data} + L_{PDE}$$

The “data” could include the boundary conditions, and/or other known data points. In our example, we reconstruct the entire flow field using sparse measurements, as represented by randomly sampled data points from the cylinder wake dataset. We want to find a solution that reduces the MSE between the prediction and these known data points. For a sample size of $n$, we have

$$L_data = \frac{1}{n}\sum_{i=1}^n\left((\hat{u}_i-u_i)^2 + (\hat{v}_i-v_i)^2 + (\hat{p}_i - p_i)^2\right)$$

To find $L_{PDE}$, we differentiate the neural network outputs with respect to position and time as needed, and evaluate the LHS of the partial differential equations $f$, $g$, and $h$. This is done by taking advantage of automatic differentiation (Paszke et al., 2019). The PDE's themselves are enforced when we minimize $L_{PDE}$ to be as close to 0 as possible.

## Repository Organization
src/model.py:

Defines PINN models in PyTorch. 
NavierStokesPINN1 (Model 1) implements a PINN as described in (Raissi et al., 2019), with associated loss function NavierStokesPINNLoss1. Continuity is enforced by predicting a latent function $\psi(x,y,t)$ and setting $u=\partial\psi/\partial y$ and $v=-\partial\psi/\partial x$, and pressure is not include in the data MSE.

NavierStokesPINN2 (Model 2) implements a PINN described by the equations above, with associated loss function NavierStokesPINNLoss2. We enforce continuity by using another PDE instead of predicting a latent function, i.e. the neural network predicts u and v directly. Pressure is included in the data MSE.

data/:

contains the cylinder wake data and the saved parameters of trained models with naming convention "model{1 or 2}_{# layers}l_{hidden size}h_{# epochs}e.pt"

train_model.py:

Defines the feed-forward architecture (number of layers, number of hidden units per layer), collects a random sample (by default, 0.5%) of training data from the full cylinder wake data, and trains the model for a specified number of epochs.

evaluate_model.ipynb:

Notebook comparing the predicted and DNS reference flows, including animations of the flow field with time.

PINN_ODE_demo.ipynb:

Notebook demonstrating how to use a PINN to solve an ordinary differential equation (ODE)

## References
Cai, S., Mao, Z., Wang, Z., Yin, M., & Karniadakis, G. E. (2021). Physics-informed neural networks (PINNs) for fluid mechanics: a review. Acta Mechanica Sinica, 37(12), 1727–1738. https://doi.org/10.1007/s10409-021-01148-1

Eivazi, H., Tahani, M., Schlatter, P., & Vinuesa, R. (2022). Physics-informed neural networks for solving Reynolds-averaged Navier-Stokes equations. Physics of Fluids, 34(7). https://doi.org/10.1063/5.0095270

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., … Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. 33rd Conference on Neural Information Processing Systems (NeurIPS 2019), 1–12. http://arxiv.org/abs/1912.01703

Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686–707. https://doi.org/10.1016/j.jcp.2018.10.045


