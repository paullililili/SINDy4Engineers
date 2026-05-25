# The Thermosyphon Heat Exchanger

A thermosyphon is a passive heat exchanger that relies on the natural convection of the flow to transport heat around the geometry. Many past studies on the annulus configuration of a thermosyphon showed that it exhibits stability bifurcation as well as chaotic behaviors. At low Rayleigh numbers (which is a non-dimensional parameter defined by the ratio between the buoyant driving forces against the viscous forces), the flow tends to settle to a static conducting state, where the fluid remains motionless. Past a certain critical Rayleigh number, the flow loses its stability in the conducting state and begins to circulate. At even higher Rayleigh numbers, the flow begins to exhibit chaotic flow reversals.

![Chaotic Thermosyphon Flow](../S3%20Thermosyphon%20Heat%20Exchanger/Chaotic%20Thermosyphon.gif)

In the works of [(Huang et al, 2023)](https://arxiv.org/abs/2307.13146), the authors derived a low-dimensional ODE system (using Fourier expansion similar to Edward Lorenz's original derivation on the Lorenz system from atmospheric convection), that explained such bifurcation behaviors. On the other hand, [(Loiseau, 2020)](https://link.springer.com/article/10.1007/s00162-020-00536-w) discovered a low-dimensional ODE using the flow field data with SINDy-DMD. In this tutorial, we will cover multiple data-driven learning approaches, demonstrating how SINDy can be used to:
1. Identify the governing PDEs of the flow field
2. Learn a parameterized low-dimensional ODE that exhibits the same bifurcation characteristics
3. Learn a low-dimensional ODE using DMD for dimension reduction