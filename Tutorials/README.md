# SINDy Tutorials

Welcome to SINDy tutorial! The tutorials are written in [Jupyter](https://jupyter.org/) notebooks. Jupyter provides an interactive method to document and run Python code. A notebook can contain multiple cells, each cell contains a block of Python code that can be ran as many times as required. You may use this feature to try out different hyperparameters, different datasets etc. After all, the best way to learn and familiarize yourself with the method is to simply try out different things! However, please run the cells in order at least once, as many cells require data/variables from the previous cells.

The tutorial is structured as follows:

* **§1 SINDy and its Extensions**
   * 1-1 Introduction to SINDy - A brief introduction to SINDy and how to run your first SINDy example.
   * 1-2 Ill-Conditioned Regression - Demonstrates commonly faced issues when applying SINDy and how to address them.
   * 1-3 Embedding Prior Knowledge - Presents how to bake in prior physics knowledge and constraints within SINDy.
   * 1-4 Choice of Coordinate Basis - Selection of coordinate basis with informed choices, or data-driven methods.
   * 1-5 Handling Noisy Data - How to deal with noisy datasets?
   * 1-6 SINDy and Controls - Introducing SINDy with controls (SINDyc) formulation and how it may integrated with model predictive control (MPC).
   * 1-7 Weak Form SINDy - Presents the weak formulation of SINDy which circumvents the need to compute the time derivative from noisy data. The method is applicable to both ODEs and PDEs.
   * 1-8 Ensembling Methods - Introduces SINDy with ensembling methods, and how one may employ it to quantify uncertainty and guide active learning strategies.
 * **§2 UAV System Identification**
   * 2-1 Data Preparation - Walks through a typical data pre-processing and clean up process.
   * 2-2 Learning Quadcopter Dynamics - Using the cleaned data and prior physics knowledge, we employ SINDy to discover the governing ODEs of a quadcopter.
 * **§3 Thermosyphon Heat Exchanger**
   * 3-1 Thermosyphon Flow - Introduces the thermosyphon flow and simulation, as well as generating the required training data.
   * 3-2 PDE Learning - Application of SINDy to identify the governing PDEs of the flow.
   * 3-3 Direct ODE Learning - Identification of low-dimensional ODE from data using informed choice of coordinate basis.
   * 3-4 Learning Parameterized ODE - Identification of a parameterized low-dimensional ODE to perform bifurcation analysis.
   * 3-5 Learning ODE with DMD - Learning a low-dimensional ODE using coordinate basis from DMD.

In §1, we will employ our own implementation of various SINDy methods that are written to be independent of any class dependencies. This should make the code base easier to read and understand. Whereas, in §2 and §3, we will apply SINDy and its various methods using the official PySINDy Python package.

There is also an additional §0, which contains a few notebooks that provide reference on various utility functions such as ones pertaining to:
* Data generation
* Error metrics
* Visualization methods

Each tutorial provides a detailed mathematical explanation, key algorithms will be shown using Jupyter notebook's `??` which prints out a copy of the function/class. For instance, the following cell would print out the function `stlsq` in full.

```python
stlsq??
```

However, the reader is highly encouraged to explore the provided [SINDy implementations](../sindy/) for themselves.


## How the tutorial is structured and written:

* Whilst the code for §1 is largely standalone and can be applied directly to the reader's own applications, it is written for clarity and may not be the most efficient implementation.
* Python type annotations are employed extensively to promote clarity of variables. However, unlike other type enforced languages, type annotations are purely 'informative'. For example, to denote that a variable may be a scalar real value, it may be annotated with `float` type, but may in fact be a `np.ndarray` type object with a shape of `(1,)`.
* In implemented classes, if a `_` prefix is placed in front of method names, it denotes that the method is private and should not be called externally directly in normal use cases.