# Weak Form SINDy

A common challenge in applying SINDy to real datasets is obtaining accurate and clean time derivatives. More often than not, the time derivatives of a state cannot be directly measured, but must be computed numerically using methods such as finite difference. However, by utilizing [weak derivatives](https://en.wikipedia.org/wiki/Weak_derivative), one does not require a function to be differentiable, but only integrable. Using the same motivating example from [1-5 Handling Noisy Data](../1-5%20Handling%20Noisy%20Data.ipynb) tutorial, if we assume the measured time-series data is corrupted with additive noise $\epsilon$ (composed of a single frequency for demonstration), then its derivative becomes

$$
\begin{aligned}
    y &= f(x) + A \sin(\omega_{\text{noise}} x), \\
    \frac{dy}{dx} &= f'(x) + A \omega_{\text{noise}} \cos(\omega_{\text{noise}} x).
\end{aligned}
$$

This has the negative effect of amplifying noise. On the contrary, if we integrate the measured data such that

$$
\begin{aligned}
    \int \frac{dy}{dx} dx &= \int f(x) dx - \frac{A}{\omega_{\rm noise}} \cos(\omega_{\rm noise} x) + C,
\end{aligned}
$$

we attenuate the noise. Hence, why numerical differentiation should be avoided whenever possible when handling noisy data.

> Although numerical truncation errors still exist for numerical integration, it is far less critical than errors from the amplification of noise.

In the next two tutorials, we will first introduce weak form SINDy for ODEs that operates only on the time domain. Then, we will further generalize the method to apply it to any PDEs that exist in a spatiotemporal domain of N-dimensions.