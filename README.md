# Flow Reconstruction using Physics Informed Machine Learning
## Overview
Using physics inspired neural networks (PINN) to solve turbulent flows using the Navier-Stokes equations. Specifically, given sparse observations, we reconstruct the entire flow field (inverse problem). 

## Background
Neural networks, when their loss functions are modified to conserve physical laws, have shown promise in solving non-linear partial differential equations in physics (Raissi et al., 2019). This has broad application for computational fluid mechanics, such as solving the Navier-Stokes equations, with demonstrated success in a variety of flow scenarios (Cai et al., 2021; Eivazi et al., 2022). Rather than a pure data-driven approach of a statistical model, physics-inspired neural networks (PINN’s) take a hybrid approach by enforcing physics-based knowledge (PDE’s), while also optimizing the loss according to the data. For example, the incompressible momentum 2D-Navier-Stokes equation gives:

x-momentum:
$$ 
\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} + \frac{1}{\rho}\frac{\partial p}{\partial x} -
\nu\left(\frac{\partial^2u}{\partial x^2} + \frac{\partial^2u}{\partial y^2}\right) = 0
$$

