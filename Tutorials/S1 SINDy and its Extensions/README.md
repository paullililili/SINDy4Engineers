# §1 SINDy and its Extensions

This section aims to introduce the reader to SINDy and its selected extensions. The tutorials will cover:

* 1-1 Introduction to SINDy - A brief introduction to SINDy and how to run your first SINDy example.
* 1-2 Ill-Conditioned Regression - Demonstrates commonly faced issues when applying SINDy and how to address them.
* 1-3 Embedding Prior Knowledge - Presents how to bake in prior physics knowledge and constraints within SINDy.
* 1-4 Choice of Coordinate Basis - Selection of coordinate basis with informed choices, or data-driven methods.
* 1-5 Handling Noisy Data - How to deal with noisy datasets?
* 1-6 SINDy and Controls - Introducing SINDy with controls (SINDyc) formulation and how it may integrated with model predictive control (MPC).
* 1-7 Weak Form SINDy - Presents the weak formulation of SINDy which circumvents the need to compute the time derivative from noisy data.
  * 1-7-1 Weak form for ODEs - Covers weak form extension for ODEs.
  * 1-7-2 Weak form for PDEs - Covers weak form extension for PDEs.
* 1-8 Ensembling Methods - Introduces SINDy with ensembling methods.
  * 1-8-1 Ensemble SINDy - Demonstrates the use of ensembling for noise mitigation and uncertainty quantification.
  * 1-8-2 Active SINDy - Demonstrates the use of ensembling to guide active learning.