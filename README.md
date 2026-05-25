# SINDy4Engineers

SINDy4Engineers is a companion Github repository for An Introduction to Sparse Identification of Non-Linear Dynamics for Engineering Applications which aims to provide a guided and hands-on approach to learning the Sparse Identification of Nonlinear Dynamics (SINDy) method and its relevant extensions. SINDy is a nonlinear dynamical system identification method that identifies the governing ODE(s) or PDE(s) from time-series data. The figure below provides an overview of the SINDy methodology. The method was first published by [(Brunton et al, 2016)](https://www.pnas.org/doi/abs/10.1073/pnas.1517384113), and have seen multiple extensions developed by various research groups across the globe. We have included the most relevant extensions to tackle common challenges faced in various engineering applications.

![Work flow of employing the SINDy method, starting with 1) collecting the time-series data and computing its time derivatives, 2) constructing a candidate function library that contains all possible terms, and 3) identifying the core dynamics via sparse regression.](./SINDy%20Architecture.png)

This tutorial aims to build fundamental knowledge of SINDy and its relevant extensions from the ground up. Knowledge of SINDy's Python package [PySINDy](https://github.com/dynamicslab/pysindy/tree/main) is not necessary. This tutorial should help to demystify the selected methods implemented in PySINDy.


## Installation and Setup

To run this repository, first clone or download this repository. Then, install the python package manager [uv](https://github.com/astral-sh/uv) and simply run

```shell
uv sync
```

Alternatively, the required Python packages can also be installed using pip with

```shell
pip install -e .
```

This should automatically download and install all required Python packages to the local working directory.

When running the Jupyter notebooks, please ensure that **the Jupyter root folder is set to the root folder of the repository**, and select the Python interpreter installed within the working directory.

> Online binder notebook version coming soon!



## Tutorial Structure

The tutorial itself can be found in [Tutorials](./Tutorials/) in the form of interactive Jupyter notebooks. Each Jupyter notebook tutorial contains mathematical explanation of SINDy and its various related methods. The implementation of these methods can be found in the [sindy](./sindy/) module. Although one can recover the same results using PySINDy, the design justification of PySINDy's implementation may not be very obvious. The goal of our implementation is to provide a standalone version that disentangles the method from its software design choices, and provide a clear explanation of both the math and the code. All methods are fully documented and should provide a useful learning guide.

After an introduction to SINDy and its methods in section 1 of the tutorial, we provide two separate example applications in the next two sections:
1. System identification of a UAV system governed by ODEs.
2. Identification of a chaotic flow-field in a thermosyphon heat exchanger.