# Quadcopter System Identification with SINDy

With the increasing prevalence and importance of UAS (unmanned aerial systems) in various applications such as surveillance & monitoring, search & rescue, leisure activities and others, there is also an increasing need to accurately model the UAV (unmanned aerial vehicle) within the system to develop robust and high performing controllers for it. A UAV can be modelled from first principles, relying on expert knowledge in rotor-aerodynamics and powertrain systems. However, even with high fidelity models, there will always be a certain degree of plant-model mismatch. This may be a result of unaccounted for physics.

Data-driven identification methods such as SINDy allow us to identify a system directly from data. The learned system may allow us to correct for any mismatches in the derived model, or used to aid in developing a robust controller for the system. However, there are also a number of common challenges that we will seek to address in this tutorial:
* Real sensors are noisy, the data must be pre-processed using data smoothing methods demonstrated in section 1 of the tutorial.
* Real systems often contain messy dynamics that are not easily defined as linear combination of nonlinear functions. For example, the equation may contain reciprocals, exponents, in-function parameters etc. To address this, we must embed known physics of the system into the learning process by carefully selecting the appropriate library function, constraining known coefficients, and enforcing known sparsity.
* Not all flight data is useful, we will learn very little of the lateral dynamics of an aircraft if it only executes longitudinal manoeuvre. The inputs given in a flight test must be carefully designed to excite all relevant dynamics.

The system identification process here can also be applied to other dynamical systems. For instance, in non-cooperative target tracking of orbital vehicles, it follows known orbital mechanics that can be enforced, but with unknown thrust characteristics that must be learned.


## The Quadcopter System

![Parrot Minidrone Quadcopter](./Parrot%20Minidrone.jpeg)

Here, we will briefly introduce the quadcopter system simply as a dynamical system. For a full explanation of quadcopter flight dynamics, the reader should refer to resources such as [Small Unmanned Aircraft: Theory and Practice by R. W. Beard and T. W. McLain](https://github.com/byu-magicc/mavsim_public) which provides a concise introduction to aircraft coordinate system and kinematics in chapters 2 and 3 respectively.

The dynamics of a quadcopter are governed by a number of states:

| Variable | Description |
| :--- | :--- |
| $\phi$ | Roll Euler angle [rad] |
| $\theta$ | Pitch Euler angle [rad] |
| $\psi$ | Yaw Euler angle [rad] |
| $v_x$ | x velocity in body axes [m/s] |
| $v_y$ | y velocity in body axes [m/s] |
| $v_z$ | z velocity in body axes [m/s] |
| $p$ | Angular velocity around x axis of body axes [rad/s] |
| $q$ | Angular velocity around y axis of body axes [rad/s] |
| $r$ | Angular velocity around z axis of body axes [rad/s] |

These states are controller by 4 motors with 4 associated inputs denoted as $u_1$, $u_2$, $u_3$ and $u_4$. Under simplified assumptions, each motor generates a force and a moment along its z axis of the body frame of reference. The resultant force $F$ and moment $M$ acting on the vehicle in the body frame of reference originates from a combination of all 4 motors and others such as aerodynamic drag. The system itself can be defined as

$$
\begin{aligned}
    \dot{\phi} &= p + \tan(\theta) \left[ q \sin(\theta) + r \cos(\phi) \right], \\
    \dot{\theta} &= q \cos(\phi) - r \sin(\phi), \\
    \dot{\psi} &= \sec(\theta) \left[ q \sin(\theta) + r \cos(\phi) \right], \\
    \dot{v}_x &= (r v_y - q v_z) - g \sin(\theta) + F_x / m, \\
    \dot{v}_y &= (p v_z - r v_x) + g \cos(\theta) \sin(\phi) + F_y / m, \\
    \dot{v}_z &= (q v_x - p v_y) + g \cos(\phi) \cos(\theta) + F_z / m, \\
    \dot{p} &= \left[ M_x + qr(I_{yy} - I_{zz}) \right] / I_{xx}, \\
    \dot{q} &= \left[ M_y + pr(I_{zz} - I_{xx}) \right] / I_{yy}, \\
    \dot{r} &= \left[ M_z + pq(I_{xx} - I_{yy}) \right] / I_{zz}. \\
\end{aligned}
$$

One should note that certain states are intrinsically coupled. Any changes with one state will affect the dynamics of other states. The governing equations are relatively complicated and not easily represented in Galerkin form. However, many of the terms are a result from fundementals of mechanics. For instance, the cross product terms in the time derivatives of $v_x$, $v_y$ and $v_z$ are a result of the [transport theorem](https://en.wikipedia.org/wiki/Transport_theorem) which defines the linear acceleration within a rotating frame, or known as the coriolis effect. Whereas, the trigonometric functions appear as a result of applying rotational transformation between body axes to the global frame of reference. The key unknowns are the resultant forces and moments acting on the vehicle that must be discovered.


## Quadcopter Simulation

In this example application, we will utilize Mathwork's [Parrot Minidrone Simulink model](https://uk.mathworks.com/help/simulink/supportpkg/parrot_ug/fly-a-parrot-minidrone-using-the-hover-simulink-model.html) to generate suitable flight trajectories. The Parrot Minidrone is a lightweight quadcopter equipped with onboard IMU, ultrasound, barometer and camera. The Simulink model models both the vehicle dynamics and its state estimation. The Simulink model is based on the works of [(Pounds et al, 2006)](https://openresearch-repository.anu.edu.au/items/d053d123-3099-4969-bd9c-185fa3412018) and [(Riether, 2016)](https://dspace.mit.edu/handle/1721.1/106777) where higher fidelity physics are captured such as rotor pitch and roll damping, blade flapping due to difference in velocities of the advancing and retreating blades and others. Whilst these are effects that cannot be fully captured by the ODEs specified above, the ODEs do capture the key dynamics of the vehicle. Additionally, due to factors such as sensor noise, we will see that using a regularizer such as the one employed by SINDy is critical in discovering a sparse representation that captures only the key physics and does not overfit to the provided data.

A number of sample flight trajectories are provided in the [data](./data/) folder. If more flight trajectories need to be generated, one should run [flight_simulations.m](./flight_simulations.m) script. To run this script, it requires:
* MATLAB 2025b or newer
* Simulink
* Computer vision toolbox
* Aerospace blockset